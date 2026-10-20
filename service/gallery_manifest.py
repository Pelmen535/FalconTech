"""Validate a portable raw-gallery runtime description without executing it.

This checks consistency, not availability/qualification of the model on this
host. Runtime startup still verifies actual artifacts and environment.
"""
import json,re
from .storage import GalleryIdentityError
from .october_manifest import PROFILE,BASELINE_SHA256

_HEX=re.compile('[0-9a-f]{64}')

def _fingerprint(value):
    return isinstance(value,str) and _HEX.fullmatch(value) is not None

def validate_raw_manifest(data,identity):
    try:
        spec=json.loads(data.decode('utf-8-sig'))
        if not isinstance(spec,dict):raise ValueError('object required')
        if spec.get('schema')!=1 or spec.get('profile')!=PROFILE or spec.get('baseline_manifest_SHA256')!=BASELINE_SHA256:
            raise ValueError('unsupported runtime schema/profile')
        model=spec['identity']
        if not isinstance(model,dict) or any(not _fingerprint(model.get(k)) for k in ('model_sha256','recipe_sha256','embedding_sha256')):
            raise ValueError('invalid runtime fingerprints')
        if model['model_sha256']!=identity['model_sha256']:raise ValueError('model identity mismatch')
        if spec.get('dimension')!=identity['dimension'] or spec['dimension']!=512:raise ValueError('dimension mismatch')
        backends=spec['backend_embedding_sha256']
        if not isinstance(backends,dict) or not backends or set(backends)-{'cpu','cuda'} or any(not _fingerprint(v) for v in backends.values()):
            raise ValueError('invalid backend identities')
        selected=[name for name,value in backends.items() if value==identity['recipe_sha256']]
        if len(selected)!=1:raise ValueError('backend embedding identity mismatch or ambiguous')
        refs=identity.get('representation')=='raw-pair-le-v2'
        if refs:
            if spec.get('storage_contract')!='raw-pair-le-v2' or spec.get('storage_embedding_kind')!='feature_ref/raw-pair-le-v2':
                raise ValueError('raw-store format mismatch')
            if not isinstance(spec.get('reference_schema'),str) or not spec['reference_schema']:
                raise ValueError('missing raw-store schema')
        elif spec.get('storage_contract')=='raw-pair-le-v2' or spec.get('storage_embedding_kind')!='pre_gallery_dino_projection':
            raise ValueError('legacy raw-store format mismatch; explicit migration required')
        if spec.get('raw_feature_kinds')!={'dino':{'shape':[2048],'dtype':'float32'},'cnn':{'shape':[2048],'dtype':'float16'}}:
            raise ValueError('raw feature format mismatch')
        files=spec['files'];weights=spec['weight_files']
        if not isinstance(files,dict) or not isinstance(weights,list) or len(weights)!=16 or len(set(weights))!=16:
            raise ValueError('invalid runtime weight inventory')
        total=0
        for name in weights:
            record=files[name]
            if not isinstance(name,str) or not isinstance(record,dict) or not _fingerprint(record.get('SHA256')):
                raise ValueError('invalid weight artifact')
            size=record['bytes']
            if type(size) is not int or size<=0:raise ValueError('invalid weight size')
            total+=size
        if spec.get('weight_budget_bytes')!=2_000_000_000 or spec.get('weights_bytes')!=total or total>2_000_000_000:
            raise ValueError('invalid runtime weight byte budget')
        timeout=spec['operations']['worker_timeout_seconds']
        import math
        if isinstance(timeout,bool) or not isinstance(timeout,(int,float)) or not math.isfinite(timeout) or timeout<=0:
            raise ValueError('invalid worker deadline')
        reserve=spec['operations']['start_C_disk_free_GiB']
        if isinstance(reserve,bool) or not isinstance(reserve,(int,float)) or not math.isfinite(reserve) or reserve<=0:
            raise ValueError('invalid runtime write disk reserve')
    except (ValueError,KeyError,TypeError,AttributeError,UnicodeError) as exc:
        raise GalleryIdentityError('Raw gallery runtime manifest is missing/malformed/incompatible: '+str(exc)) from exc
    return {'schema':2,'format':'october-raw-pair-le-v2' if refs else 'october-raw-with-compatibility-vector-v1',
            'backend':selected[0],'runtime_model_sha256':model['model_sha256'],
            'runtime_embedding_sha256':backends[selected[0]],'consistency_verified':True}
