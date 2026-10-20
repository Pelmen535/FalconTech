"""Lossless October-v1 -> reference-v2 migration; source DB/cache are read-only."""
from pathlib import Path
import hashlib,json,os,shutil,tempfile
from .gallery_bundle import ReadOnlySQLite,snapshot_rows
from .storage import GalleryStore
from .october_features import RawFeatureStore
from .october_reference import RawReferenceStore,FeatureReference
from .october_manifest import load_manifest


def migrate_references(source,target,source_cache,root,manifest,device='cuda'):
    source=Path(source).resolve();target=Path(target).resolve();source_cache=Path(source_cache).resolve()
    if target.exists() or target.is_relative_to(source) or source.is_relative_to(target):
        raise ValueError('Migration requires a new separate target')
    spec=load_manifest(manifest,root)
    if spec.get('storage_contract')!='raw-pair-le-v2':raise ValueError('Target is not reference-v2')
    original=ReadOnlySQLite(source)
    expected=spec['extractor_backend_identity_before_reference_schema'][device]
    if original.model_sha256!=spec['identity']['model_sha256'] or original.recipe_sha256!=expected:
        raise ValueError('Source model/extractor mismatch; use image re-encoding migration')
    if original.references_only:raise ValueError('Source is already reference-only')
    old_raw=RawFeatureStore(source_cache,expected)
    target.parent.mkdir(parents=True,exist_ok=True)
    staging=Path(tempfile.mkdtemp(prefix='reference-migration-',dir=target.parent))
    namespace=spec['backend_embedding_sha256'][device]
    try:
        raw=RawReferenceStore(staging/'raw-features'/namespace,namespace)
        store=GalleryStore(staging,original.model_sha256,namespace,512,references_only=True)
        ids=[];lineage=[]
        with snapshot_rows(original) as rows:
            for row in rows:
                ref=row.pop('feature_ref')
                if not ref:raise ValueError('Source lacks explicit raw references; re-encode a copy first')
                packet=old_raw.get(ref,persisted=True)
                import numpy as np
                if not np.array_equal(row.pop('embedding'),packet['base']):raise ValueError('Source vector/raw mismatch')
                new_ref=raw.put(packet['emb'],packet['specialist'],persist=True)
                identity={'schema':1,'embedding_identity':namespace,'image_SHA256':hashlib.sha256(row['image_bytes']).hexdigest(),'bbox':[float(x) for x in row['bbox_xywh']]}
                input_key=hashlib.sha256(json.dumps(identity,sort_keys=True,allow_nan=False).encode()).hexdigest()
                raw.remember_input(input_key,new_ref)
                store.add_many([{**row,'embedding':FeatureReference(new_ref,namespace)}])
                with store._connection() as conn:
                    with conn:conn.execute('UPDATE gallery_items SET created_at=? WHERE image_id=?',(row['created_at'],row['image_id']))
                assert store.image(row['image_id'])==row['image_bytes']
                ids.append(row['image_id']);lineage.append({'observation':row['image_id'],'old_ref':ref,'new_ref':new_ref})
        assert store.snapshot()[0]==ids
        receipt={'status':'COMPLETE_READ_ONLY_RAW_PAIR_MIGRATION','observations':len(ids),'encoder_forward_count':0,'source':str(source),'target':str(target),'source_embedding_identity':expected,'target_namespace':namespace,'lineage':lineage,'source_DB_and_raw_cache_unchanged':True,'activation':'Use target data-dir with reference-v2 manifest; rollback uses original directory and original manifest.'}
        (staging/'migration.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf-8')
        os.rename(staging,target)
        return receipt
    finally:
        if staging.exists():
            if staging.is_symlink() or staging.parent!=target.parent or not staging.name.startswith('reference-migration-'):raise RuntimeError('Unsafe migration staging')
            shutil.rmtree(staging)
