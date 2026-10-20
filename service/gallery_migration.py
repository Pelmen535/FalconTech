"""Re-encode a read-only gallery into a separate, verified model version."""
from pathlib import Path
import hashlib,json,os,shutil,tempfile
import numpy as np
from .gallery_bundle import ReadOnlySQLite,snapshot_rows
from .storage import GalleryStore


def migrate_gallery(source,target,core_factory,*,batch_size=16):
    source=Path(source).resolve();target=Path(target).resolve()
    if source==target or target.is_relative_to(source) or source.is_relative_to(target):raise ValueError('Migration needs separate gallery directories')
    if target.exists():raise FileExistsError('Migration target must be new')
    if batch_size<1 or batch_size>32:raise ValueError('Invalid migration batch size')
    return _migrate_store(ReadOnlySQLite(source),target,core_factory,batch_size=batch_size,source_description=str(source))


def migrate_postgres_gallery(source_dsn,target,core_factory,*,batch_size=16):
    from .gallery_postgres_admin import ReadOnlyPostgres
    target=Path(target).resolve()
    if target.exists():raise FileExistsError('Migration target must be new')
    if batch_size<1 or batch_size>32:raise ValueError('Invalid migration batch size')
    return _migrate_store(ReadOnlyPostgres(source_dsn),target,core_factory,batch_size=batch_size,source_description='Read-only PostgreSQL source; DSN omitted')


def _migrate_store(original,target,core_factory,*,batch_size,source_description):
    target.parent.mkdir(parents=True,exist_ok=True)
    staging=Path(tempfile.mkdtemp(prefix='gallery-migration-',dir=target.parent));core=None
    try:
        core=core_factory(staging)
        references_only=getattr(core,'references_only',False)
        store=GalleryStore(staging,core.model_sha256,core.embedding_sha256,core.dimension,references_only=references_only)
        source_rows=[];hashes=[]
        def insert(rows):
            with tempfile.TemporaryDirectory(prefix='source-images-',dir=staging) as folder:
                records=[]
                for i,row in enumerate(rows):
                    path=Path(folder)/(str(i)+'.img');path.write_bytes(row['image_bytes'])
                    records.append({'path':str(path),'bbox':row['bbox_xywh']})
                vectors=core.encode_gallery_many(records)
                if len(vectors)!=len(rows):raise ValueError('Migration encoder count mismatch')
                store.add_many([{**row,'embedding':vector} for row,vector in zip(rows,vectors)])
                # Preserve timestamps as well as observation IDs, frames and boxes.
                with store._connection() as conn:
                    with conn:conn.executemany('UPDATE gallery_items SET created_at=? WHERE image_id=?',[(r['created_at'],r['image_id']) for r in rows])
        with snapshot_rows(original) as rows:
            pending=[]
            for row in rows:
                # Old vectors and refs never enter the new encoder or target DB.
                row.pop('embedding');row.pop('feature_ref',None)
                hashes.append({'image_id':row['image_id'],'image_SHA256':hashlib.sha256(row['image_bytes']).hexdigest(),'bbox_xywh':row['bbox_xywh'],'created_at':row['created_at']})
                pending.append(row)
                if len(pending)>=batch_size:insert(pending);pending=[]
            if pending:insert(pending)
        ids,vectors=store.snapshot()
        if ids!=[r['image_id'] for r in hashes] or len(vectors)!=len(ids):raise ValueError('Migration order/count mismatch')
        if not references_only and vectors.shape!=(len(ids),core.dimension):raise ValueError('Migration shape mismatch')
        from .october_features import RawFeatureStore
        if references_only:
            from .october_reference import RawReferenceStore
            RawFeatureStore=RawReferenceStore
        raw=RawFeatureStore(core.cache_dir,core.embedding_sha256)
        for i,row in enumerate(hashes):
            item=store.get_item(row['image_id'])
            if hashlib.sha256(store.image(row['image_id'])).hexdigest()!=row['image_SHA256'] or item['bbox_xywh']!=row['bbox_xywh'] or item['created_at']!=row['created_at']:
                raise ValueError('Migration changed source observation')
            packet=raw.get(vectors.feature_refs[i],persisted=True)
            if not references_only and not np.array_equal(packet['base'],vectors[i]):raise ValueError('Migration raw reference mismatch')
        model_metadata=core.metadata();core.close();core=None
        receipt={'status':'COMPLETE_REENCODED_GALLERY_VERSION','observations':len(ids),'source':source_description,'target':str(target),
                 'source_identity':{'model_sha256':original.model_sha256,'embedding_sha256':original.recipe_sha256,'dimension':original.dimension},
                 'target_identity':model_metadata,'source_observations':hashes,'old_gallery_untouched':True,'activation':'Point --data-dir at the target; rollback points it back at the original. No in-place model switch.'}
        (staging/'migration.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf-8')
        os.rename(staging,target)
        return receipt
    finally:
        if core is not None:core.close()
        if staging.exists():
            if staging.resolve().parent!=target.parent or not staging.name.startswith('gallery-migration-') or staging.is_symlink():raise RuntimeError('Unsafe migration staging cleanup')
            shutil.rmtree(staging)
