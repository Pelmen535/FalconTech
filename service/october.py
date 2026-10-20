"""One manifest/constructor shared by October HTTP and offline CLI."""
from pathlib import Path
import hashlib
import json
import sys
import threading
import numpy as np

from .october_manifest import load_manifest,digest
from .october_features import FeatureVector
from .worker_rpc import WorkerRPC
from .october_features import snapshot_signature


class OctoberAdapter:
    dimension=512
    supports_ann=True
    manages_ann=True
    supports_explain=False
    confidence_scale='calibrated_acceptance_score'

    def __init__(self, core_dir, data_dir, device='cuda', *, manifest=None):
        self.root=Path(core_dir).resolve();self.data_dir=Path(data_dir).resolve()
        default=self.root/'release_current/runtime-manifest.json'
        if not default.is_file():
            default=self.root/'var/october-v14-compatibility-20261019/runtime-manifest.json'
        self.manifest=Path(manifest or default).resolve()
        self.spec=load_manifest(self.manifest,self.root)
        self.manifest_sha256=digest(self.manifest)
        if device not in ('cuda','cpu') or (device=='cpu' and not self.spec['capabilities']['cpu']):
            raise ValueError('Selected October manifest has no qualified '+str(device)+' backend')
        self.supports_ann=bool(self.spec['capabilities']['ann'])
        self.supports_explain=bool(self.spec['capabilities']['explain'])
        self.supports_correspondence=bool(self.spec['capabilities'].get('correspondence',False))
        self.device=device
        self.references_only=self.spec.get('storage_contract')=='raw-pair-le-v2'
        identity=self.spec['identity']
        self.model_sha256=identity['model_sha256'];self.recipe_sha256=identity['recipe_sha256']
        self.embedding_sha256=self.spec.get('backend_embedding_sha256',{}).get(device,identity['embedding_sha256'])
        self.cache_dir=self.data_dir/'raw-features'/self.embedding_sha256
        from .disk_reserve import WriteDiskReserve,DiskReserveError
        import tempfile
        try:
            WriteDiskReserve({'data':self.data_dir,'raw_cache':self.cache_dir,'temporary':tempfile.gettempdir()},
                             self.spec['operations']['start_C_disk_free_GiB']).require()
        except DiskReserveError as exc:
            from .robust import ResourceUnavailableError
            raise ResourceUnavailableError(str(exc)) from exc
        self.cache_dir.mkdir(parents=True,exist_ok=True)
        self.rpc=WorkerRPC([sys.executable,'-u','-X','utf8','-B','-m','service.october_worker',
            '--root',self.root,'--manifest',self.manifest,'--cache',self.cache_dir,'--device',device],
            cwd=self.root,stderr_path=self.data_dir/'october-worker.log',
            timeout=self.spec['operations']['worker_timeout_seconds'],startup_timeout=self.spec['operations']['startup_timeout_seconds'])
        self._lock=threading.RLock();self._registered=None;self._gallery_paths=[];self._gallery_stamp=None
        self._loaded=False;self._worker_generation=None;self._index_stats=None
        self._reference_descriptor=None;self._reference_signature=None;self.reference_signature_computations=0

    def metadata(self):
        return {'model_sha256':self.model_sha256,'recipe_sha256':self.recipe_sha256,
            'embedding_sha256':self.embedding_sha256,'runtime_manifest_SHA256':self.manifest_sha256,
            'dimension':512,'device':self.device,'loaded':self._loaded and self.rpc.running,
            'threshold':self.spec['public_boundary'],'confidence_kind':self.confidence_scale,
            'confidence_scale':self.confidence_scale,'bbox_format':'xywh','backend':'october',
            'rerank':'current_query_reciprocity_joint25','refusal_profile':'vric-road-verifier-v1',
            'acceptance_guard':self.spec.get('decision_policy'),
            'storage_embedding_kind':self.spec['storage_embedding_kind'],
            'storage_contract':self.spec.get('storage_contract','legacy_vector_ref'),
            'storage_vector_dimension':None if self.references_only else 512,
            'export_embedding_kind':self.spec['export_embedding_kind'],
            'supports_ann':self.supports_ann,'supports_explain':self.supports_explain,
            'supports_correspondence':self.supports_correspondence,
            'raw_feature_cache_required':True,'gallery_style_routing':False,
            'worker_deadline_seconds':self.rpc.timeout,'resource_telemetry_max_age_seconds':self.spec['operations'].get('telemetry_max_age_seconds',.25),'qualification':self.spec['qualification']}

    def correspondences(self,query_path,query_bbox,candidate_path,candidate_bbox):
        from .correspondence import run_isolated
        if not self.supports_correspondence:
            raise ValueError('Experimental correspondence is unavailable')
        if digest(self.manifest)!=self.manifest_sha256:
            raise RuntimeError('October manifest changed; restart runtime explicitly')
        with self._lock:
            return run_isolated(self.root,self.spec,query_path,query_bbox,candidate_path,candidate_bbox)

    def _request(self,message,**kwargs):
        if digest(self.manifest)!=self.manifest_sha256:
            raise RuntimeError('October manifest changed; restart runtime explicitly')
        from .observability import REQUEST_ID
        message={**message,'request_id':REQUEST_ID.get()}
        try:
            result=self.rpc.request(message,**kwargs)
        except BaseException:
            self._registered=None;self._gallery_stamp=None;self._loaded=False
            raise
        if self._worker_generation!=self.rpc.generation:
            self._registered=None;self._gallery_stamp=None
            self._worker_generation=self.rpc.generation
        return result

    def _encode(self,rows,persist):
        with self._lock:
            result=self._request({'command':'encode','rows':rows,'persist':persist},
                timeout=self.spec['operations']['bulk_timeout_seconds'] if len(rows)>1 else None)
            self._loaded=any(r.get('native_loaded',False) for r in result)
            if self.references_only:
                from .october_reference import FeatureReference
                return [FeatureReference(r['feature_ref'],self.embedding_sha256,self.rpc.generation) for r in result]
            return [FeatureVector(r['vector'],r['feature_ref'],self.rpc.generation) for r in result]

    def encode(self,path,bbox_xywh):
        return self._encode([{'path':str(Path(path).resolve()),'bbox':list(bbox_xywh)}],False)[0]

    def encode_gallery(self,path,bbox_xywh):
        return self._encode([{'path':str(Path(path).resolve()),'bbox':list(bbox_xywh)}],True)[0]

    def encode_gallery_many(self,rows):
        return self._encode(rows,True)

    @staticmethod
    def _stamp(paths):
        return tuple((str(p),p.stat().st_size,p.stat().st_mtime_ns) for p in paths)

    def rank(self,query_vector,gallery_vectors,gallery_ids,query_id=None,mode="exact"):
        with self._lock:
            if mode not in ('exact','ann') or (mode=='ann' and not self.supports_ann):raise ValueError('Unsupported October search mode')
            query_ref=getattr(query_vector,'feature_ref',None)
            if not query_ref:
                raise ValueError('Query requires its October feature_ref; encode the query with this runtime')
            if getattr(query_vector,'worker_generation',None)!=self.rpc.generation:
                raise RuntimeError('Worker reset after query encoding; re-encode the query')
            signature=self._ensure_gallery(gallery_vectors,gallery_ids)
            result=self._request({'command':'rank','query_ref':query_ref,'gallery_signature':signature,'query_id':query_id,
                                  'ann_size':200 if mode=='ann' else None})
            self._index_stats=result.pop('ann_index',None)
            return result

    def _ensure_gallery(self,gallery_vectors,gallery_ids):
        refs=getattr(gallery_vectors,'feature_refs',None)
        ids=list(gallery_ids)
        if self.references_only:
            from .october_reference import GalleryReferences,reference_signature
            if not isinstance(gallery_vectors,GalleryReferences) or gallery_vectors.namespace!=self.embedding_sha256:
                raise ValueError('Gallery requires reference-only extractor snapshot')
            descriptor=(tuple(ids),refs,gallery_vectors.content_revisions,self.embedding_sha256)
            if descriptor!=self._reference_descriptor:
                signature=reference_signature(ids,refs,self.embedding_sha256,gallery_vectors.content_revisions)
                self._reference_descriptor=descriptor;self._reference_signature=signature
                self.reference_signature_computations+=1
            else:
                signature=self._reference_signature
            message={'command':'gallery','ids':ids,'signature':signature,'feature_refs':refs,'content_revisions':gallery_vectors.content_revisions}
        else:
            vectors=np.asarray(gallery_vectors,dtype=np.float32)
            signature=snapshot_signature(vectors,ids,refs)
            message={'command':'gallery','vectors':vectors.tolist(),'ids':ids,'signature':signature,'feature_refs':refs}
        stamp=self._stamp(self._gallery_paths) if self._gallery_paths else None
        if self._registered!=signature or stamp!=self._gallery_stamp:
            result=self._request(message,
                timeout=self.spec['operations']['bulk_timeout_seconds'])
            if result['signature']!=signature:
                raise RuntimeError('Worker registered a different gallery')
            self._registered=signature
            self._gallery_paths=[self.cache_dir/'objects'/(ref+'.npz') for ref in result['feature_refs']]
            self._gallery_stamp=self._stamp(self._gallery_paths)
        return signature

    def explain(self,query_path,query_bbox,candidate_path,candidate_bbox,regions=3,*,gallery_vectors,gallery_ids,gallery_id):
        if not self.supports_explain:raise ValueError('Explanation is not enabled in the manifest')
        with self._lock:
            signature=self._ensure_gallery(gallery_vectors,gallery_ids)
            result=self._request({'command':'explain','query_path':str(query_path),'query_bbox':list(query_bbox),
                'candidate_path':str(candidate_path),'candidate_bbox':list(candidate_bbox),
                'gallery_signature':signature,'gallery_id':gallery_id,'regions':regions},
                timeout=self.spec['operations']['bulk_timeout_seconds'])
            import base64
            for side in result['sides'].values():side['png']=base64.b64decode(side['png_base64']);del side['png_base64']
            return result

    def index_stats(self):
        from .ann import ShortlistIndex
        stats=ShortlistIndex().stats() if self._index_stats is None else dict(self._index_stats)
        return {**stats,'space':'gallery-adapted DINO+CNN joint1024','full_gallery_context':True,
                'scope':'HNSW shortlist before query-insertion reciprocity; full static graph context and confidence statistics retained; conservative exact completion may expand the pool',
                'available':self.supports_ann and stats['available']}

    def offline(self,query,gallery,out,batch_size=16,workers=4,query_batch_size=None):
        with self._lock:
            return self._request({'command':'offline','query':query,'gallery':gallery,
                                  'out':str(Path(out).resolve()),'batch_size':batch_size,'workers':workers,'query_batch_size':query_batch_size},
                                 timeout=self.spec['operations']['bulk_timeout_seconds'])

    def close(self):
        with self._lock:
            self.rpc.close();self._registered=None;self._loaded=False
