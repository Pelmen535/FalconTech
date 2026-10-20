"""Single October runtime for RPC/CLI: explicit encoder, math and raw-feature store."""
from pathlib import Path
import argparse
import json
import sys
import tempfile
import time
import traceback


class OctoberRuntime:
    def __init__(self, root, spec, cache, device, gate):
        import numpy as np
        from .october_math import OctoberMath
        from .october_features import RawFeatureStore
        self.root, self.spec, self.device = Path(root), spec, device
        self.gate = gate
        self.cache = Path(cache);self.cache.mkdir(parents=True,exist_ok=True)
        self.math = OctoberMath(root,context_identity={
            'model':spec['identity'],'backend':device,
            'embedding':spec.get('backend_embedding_sha256',{}).get(device,spec['identity']['embedding_sha256']),
            'artifact_pins':spec['files']})
        self.references_only=spec.get('storage_contract')=='raw-pair-le-v2'
        if self.references_only:
            from .october_reference import RawReferenceStore
            RawFeatureStore=RawReferenceStore
        self.raw = RawFeatureStore(cache,spec.get('backend_embedding_sha256',{}).get(device,spec['identity']['embedding_sha256']))
        self.native = None
        self.initial_start_reserved = True
        self.gallery_signature = None
        self.gallery_refs = None
        self.np = np

    def check_live(self):
        from .robust import ResourceUnavailableError
        reasons,snapshot=self.gate.live_reasons()
        if reasons:
            raise ResourceUnavailableError('; '.join(reasons))

    def extractor(self):
        if self.native is None:
            if not self.initial_start_reserved:
                from .robust import ResourceUnavailableError
                reasons,snapshot=self.gate.start_reasons()
                if reasons:raise ResourceUnavailableError('; '.join(reasons))
            self.initial_start_reserved=False
            from .october_native import create_native
            logs = self.cache/'logs';logs.mkdir(exist_ok=True)
            folder = Path(tempfile.mkdtemp(prefix='native-',dir=logs))
            if self.device=='cpu':
                from .october_cpu import CPUFeatures
                self.native = CPUFeatures(self.root,self.spec,folder,1,self.check_live)
            else:
                self.native = create_native(self.root,self.spec,folder,16,self.check_live)
        return self.native

    def extract(self, records, batch_size=16, workers=4):
        native = self.extractor();native.batch_size=batch_size;native.set_workers(workers)
        # Strip all labels/camera/source metadata before either encoder.
        rows = [{k:r[k] for k in ('key','path','bbox') if k in r} for r in records]
        return native.extract(rows)

    def encode(self, message):
        records=[{**r,'key':str(i)} for i,r in enumerate(message['rows'])]
        persist=message.get('persist',False);result=[None]*len(records);pending=[];input_keys={}
        for i,row in enumerate(records):
            if persist:
                key=self.raw.input_key(row['path'],row.get('bbox'));input_keys[i]=key
                cached=self.raw.find_input(key)
                if cached is not None:
                    ref,values=cached
                    result[i]={'feature_ref':ref,'native_loaded':self.native is not None} if self.references_only else {'vector':values['base'].tolist(),'feature_ref':ref,'native_loaded':self.native is not None}
                    continue
            pending.append(i)
        if pending:
            features,timing=self.extract([records[i] for i in pending],16)
            vectors=None if self.references_only else self.math.teacher(features['emb'])
            for position,i in enumerate(pending):
                vector=None if self.references_only else vectors[position]
                ref=self.raw.put(features['emb'][position],features['specialist'][position],vector,persist=persist,explicit_ref=True)
                if persist:
                    ref,values=self.raw.remember_input(input_keys[i],ref);vector=None if self.references_only else values['base']
                result[i]={'feature_ref':ref,'native_loaded':self.native is not None} if self.references_only else {'vector':vector.tolist(),'feature_ref':ref,'native_loaded':self.native is not None}
        return result

    def gallery(self, message):
        from .october_features import snapshot_signature
        np=self.np;ids=message['ids'];refs=message.get('feature_refs')
        if self.references_only:
            from .october_reference import reference_signature
            signature=reference_signature(ids,refs,self.raw.identity,message['content_revisions'])
            pairs=[(ref,self.raw.get(ref,persisted=True)) for ref in refs]
        else:
            vectors=np.asarray(message['vectors'],dtype=np.float32)
            signature=snapshot_signature(vectors,ids,refs)
        if signature!=message['signature']:
            raise ValueError('Gallery registration signature mismatch')
        if self.references_only:
            pass
        elif refs is None:
            pairs=[self.raw.legacy(v,persisted=True) for v in vectors]
        else:
            pairs=[self.raw.legacy(v,persisted=True) if ref is None else (ref,self.raw.get(ref,persisted=True)) for v,ref in zip(vectors,refs)]
            if any(not np.array_equal(v,packet['base']) for v,(ref,packet) in zip(vectors,pairs)):
                raise ValueError('Observation vector and raw feature_ref disagree')
        pack={'keys':np.array(ids),'emb':np.stack([v['emb'] for ref,v in pairs]),
              'specialist':np.stack([v['specialist'] for ref,v in pairs])}
        generation=self.math.register(pack,observation_generation=signature)
        if self.references_only:generation=signature
        self.gallery_signature=signature;self.gallery_refs=[ref for ref,v in pairs]
        return {'signature':signature,'generation':generation,'feature_refs':self.gallery_refs}

    def rank(self, message):
        np=self.np
        if message['gallery_signature']!=self.gallery_signature:
            raise ValueError('Gallery must be registered after worker reset')
        # A persisted object is checked again on registration; retained gallery
        # arrays are detached and immutable for the duration of this generation.
        feature=self.raw.get(message['query_ref'])
        key=message.get('query_id')
        if key is None:
            key='__service_query__'
            while key.split('#')[0] in {k.split('#')[0] for k in self.math.gallery['keys']}:
                key+='_' 
        q={'keys':np.array([key]),'emb':feature['emb'][None,:],'specialist':feature['specialist'][None,:]}
        result=self.math.route(q,ann_size=message.get('ann_size'));prediction=getattr(self.math,'confidence_public',self.math.confidence)(result)
        order=np.argsort(-result['rank'][0],kind='stable')[:10]
        if not np.isfinite(result['rank'][0,order]).all():
            raise ValueError('Need ten eligible gallery observations outside query frame')
        ids=self.math.gallery['keys'];ranking=[{'image_id':str(ids[j]),'rank':i+1,
            'cosine':float(result['raw'][0,j]),'ranking_score':float(result['rank'][0,j])} for i,j in enumerate(order)]
        confidence=float(prediction['confidence'][0]);accepted=bool(prediction['accepted'][0])
        return {'ranking':ranking,'accepted':accepted,'candidate':{**ranking[0],'confidence':confidence} if accepted else None,
                'confidence':confidence,'threshold':self.math.boundary,'stage':'october_full_gallery',
                'supports_explain':bool(self.spec['capabilities']['explain']),'gallery_generation':self.gallery_signature if self.references_only else self.math.generation,**result['search_info'],
                'ann_index':self.math.ann.stats()}

    def explain(self,message):
        from .october_explain import explain_pair
        import base64
        result=explain_pair(self,message)
        for side in result['sides'].values():side['png_base64']=base64.b64encode(side.pop('png')).decode('ascii')
        return result

    def offline(self,message):
        import csv
        import numpy as np
        from vreid.submit import save_embeddings,write_submission
        output=Path(message['out'])
        self.gate.add_write_paths(output=output,output_staging=output.parent)
        self.check_live()
        query,qt=self.extract(message['query'],message.get('query_batch_size') or message.get('batch_size',16),message.get('workers',4))
        gallery,gt=self.extract(message['gallery'],message.get('batch_size',16),message.get('workers',4))
        # Offline scoring needs only detached host feature banks; release the GPU first.
        self.native.close();self.native=None
        result=self.math.route(query,gallery);prediction=getattr(self.math,'confidence_public',self.math.confidence)(result)
        final=Path(message['out'])
        if final.exists() or final.is_symlink():raise FileExistsError('Output already exists')
        final.parent.mkdir(parents=True,exist_ok=True)
        output=Path(tempfile.mkdtemp(prefix='october-output-',dir=final.parent))
        try:
            save_embeddings(np.concatenate([result['query'],result['gallery']]),output/'embeddings.npy')
            write_submission(query['keys'].tolist(),gallery['keys'].tolist(),result['rank'],output/'submission.csv',topk=10,exclude_self=False)
            with (output/'candidates.csv').open('w',newline='',encoding='utf-8') as stream:
                writer=csv.writer(stream);writer.writerow(['query_id','gallery_id','confidence'])
                for i,value in enumerate(prediction['confidence']):
                    if value>=self.math.boundary:
                        writer.writerow([str(query['keys'][i]),str(gallery['keys'][result['top'][i]]),format(float(value),'.17g')])
            receipt={'status':'COMPLETE_OCTOBER_UNIFIED_OFFLINE','profile':self.spec['profile'],
                'identity':self.spec['identity'],'query':qt,'gallery':gt,'batch_size':message.get('batch_size',16),
                'query_batch_size':message.get('query_batch_size') or message.get('batch_size',16),'device':self.device,'requested_workers':message.get('workers',4),'preprocess_workers':min(message.get('workers',4),4),'no_labels_enter_inference':True,'embedding_kind':self.spec['export_embedding_kind']}
            (output/'run_info.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf-8')
            self.gate.snapshot();self.check_live()
            from .publish_directory import publish_new_directory
            publish_new_directory(output,final)
            return receipt
        finally:
            if output.exists():
                if output.resolve().parent!=final.parent.resolve() or output.is_symlink() or getattr(output,'is_junction',lambda:False)() or not output.name.startswith('october-output-'):
                    raise RuntimeError('Unsafe private output cleanup')
                import shutil
                shutil.rmtree(output)

    def close(self):
        if self.native is not None:
            self.native.close();self.native=None


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True,type=Path)
    parser.add_argument('--manifest',required=True,type=Path);parser.add_argument('--cache',required=True,type=Path)
    parser.add_argument('--device',choices=['cpu','cuda'],required=True);args=parser.parse_args()
    if sys.platform == 'linux':
        from .vric_road_linux_worker import isolate_native_stdout
        control = isolate_native_stdout()
    else:
        control = sys.stdout
    sys.stdout = sys.stderr
    from .october_manifest import load_manifest
    from .october_resources import ResourceGate
    spec=load_manifest(args.manifest,args.root)
    if args.device=='cpu' and not spec['capabilities']['cpu']:
        raise ValueError('Selected October manifest has no qualified CPU backend')
    gate=ResourceGate(args.root,spec['operations'],args.device,write_paths={
        'data':args.cache.parent.parent,'raw_cache':args.cache});runtime=None
    def respond(response,message,started):
        print(json.dumps({'event':'runtime.operation','command':message.get('command'),
            'rpc_id':message.get('rpc_id'),'request_id':message.get('request_id'),
            'profile':spec['profile'],'model_sha256':spec['identity']['model_sha256'],
            'outcome':'ok' if response['ok'] else 'resource_wait' if response.get('resource_wait') else 'error',
            'duration_ms':round((time.perf_counter()-started)*1000,3),
            **({'resource_snapshot':gate._last_snapshot,'resource_limits':gate.limits} if response.get('resource_wait') else {})},allow_nan=False),file=sys.stderr,flush=True)
        control.write(json.dumps({**response,'rpc_id':message.get('rpc_id')},allow_nan=False)+'\n');control.flush()
    try:
        for line in sys.stdin:
            message={};closing=False;started=time.perf_counter()
            try:
                message=json.loads(line);command=message['command'];rpc_id=message['rpc_id']
                if command=='close':
                    if runtime is not None:runtime.close();runtime=None
                    response={'ok':True,'result':True};closing=True
                else:
                    if runtime is None:
                        reasons,snapshot=gate.start_reasons()
                        if reasons:
                            response={'ok':False,'resource_wait':True,'error':'; '.join(reasons)}
                            respond(response,message,started);continue
                        runtime=OctoberRuntime(args.root,spec,args.cache,args.device,gate)
                    reasons,snapshot=gate.live_reasons()
                    if reasons:
                        response={'ok':False,'resource_wait':True,'error':'; '.join(reasons)}
                    elif command in ('encode','gallery','rank','offline','explain'):
                        response={'ok':True,'result':getattr(runtime,command)(message)}
                    else:raise ValueError('Unknown October runtime command')
            except Exception as exc:
                print(json.dumps({'event':'runtime.error','error_type':type(exc).__name__,
                    'rpc_id':message.get('rpc_id'),'request_id':message.get('request_id')}),file=sys.stderr,flush=True)
                print(''.join(traceback.format_tb(exc.__traceback__)),file=sys.stderr,flush=True)
                from .robust import ResourceUnavailableError
                response={'ok':False,'error_kind':'input' if isinstance(exc,ValueError) else 'internal',
                          'resource_wait':isinstance(exc,ResourceUnavailableError),'error':str(exc)}
            respond(response,message,started)
            if closing:break
    finally:
        if runtime is not None:runtime.close()

if __name__=='__main__':main()
