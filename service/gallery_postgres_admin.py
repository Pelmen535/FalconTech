"""Read-only PostgreSQL backup and cold restore to a fresh target namespace.

No existing gallery/table is cleared. DDL, identity and all target observations
commit in one PostgreSQL transaction. A verified local snapshot is staged first;
if DB publication fails it remains available for recovery with a failure receipt.
No DSN/password is written to receipts.
"""
from contextlib import contextmanager
from pathlib import Path
import json
import numpy as np
from .gallery_bundle import validated_bundle,restore_bundle,snapshot_rows,ReadOnlySQLite,file_sha
from .storage import PostgresGalleryStore,GalleryStore,GalleryIdentityError


class ReadOnlyPostgres:
    backend='postgres'
    def __init__(self,dsn):
        import psycopg
        self.dsn=dsn;self._psycopg=psycopg
        with self._connection() as conn:
            meta=dict(conn.execute('SELECT key,value FROM gallery_metadata').fetchall())
            columns={row[0] for row in conn.execute("SELECT column_name FROM information_schema.columns WHERE table_schema=current_schema() AND table_name='gallery_items'")}
        self.model_sha256=meta['model_sha256'];self.recipe_sha256=meta['recipe_sha256'];self.dimension=int(meta['dimension'])
        self.references_only=meta.get('representation')=='raw-pair-le-v2';self.has_feature_ref='feature_ref' in columns
        if self.references_only and not self.has_feature_ref:raise GalleryIdentityError('Reference gallery has no feature_ref column')

    @contextmanager
    def _connection(self):
        conn=self._psycopg.connect(self.dsn,autocommit=True,connect_timeout=15)
        try:
            # Preserve caller search_path/options; make every subsequent
            # transaction on this source connection read-only at the server.
            conn.execute('SET default_transaction_read_only=on')
            yield conn
        finally:conn.close()


def _require_fresh_schema(conn):
    rows=conn.execute('SELECT tablename FROM pg_catalog.pg_tables WHERE schemaname=current_schema()').fetchall()
    if rows:raise GalleryIdentityError('PostgreSQL restore requires a fresh empty target schema/database; existing tables are preserved')


def _restore_rows(store,source,conn,batch_size):
    pending=[];timestamps=[]
    def flush():
        if not pending:return
        store.add_many(pending)
        conn.cursor().executemany('UPDATE gallery_items SET created_at=%s WHERE image_id=%s',timestamps)
        pending.clear();timestamps.clear()
    with snapshot_rows(source) as rows:
        for row in rows:
            if not isinstance(row['created_at'],str) or not row['created_at']:raise ValueError('Invalid snapshot timestamp')
            if store.references_only:
                from .october_reference import FeatureReference
                row['embedding']=FeatureReference(row['feature_ref'],store.recipe_sha256)
            pending.append(row);timestamps.append((row['created_at'],row['image_id']))
            if len(pending)>=batch_size:flush()
        flush()


def _verify_rows(source,target):
    count=0
    with snapshot_rows(source) as left,snapshot_rows(target) as right:
        for a,b in zip(left,right,strict=True):
            for key in ('image_id','image_bytes','bbox_xywh','created_at','feature_ref'):
                if a[key]!=b[key]:raise ValueError('PostgreSQL restore changed observation '+key)
            if a['embedding'] is None:
                if b['embedding'] is not None:raise ValueError('Reference-only restore invented a vector')
            elif not np.array_equal(a['embedding'],b['embedding']):raise ValueError('Restored vector differs')
            count+=1
    return count


def restore_bundle_postgres(path,target,dsn,*,expected_identity=None,batch_size=128):
    import psycopg
    target=Path(target).resolve();path=Path(path).resolve()
    if target.exists():raise FileExistsError('Restore requires a new data directory')
    if not 1<=batch_size<=4096:raise ValueError('Invalid restore batch size')
    # Validate bytes and model identity before changing either target resource.
    with validated_bundle(path) as (_,header):
        identity=header['identity'].copy();expected_count=len(header['observations'])
        runtime_contract=header.get('validated_runtime_contract')
    if expected_identity is not None and identity!=expected_identity:raise GalleryIdentityError('Backup model/embedding identity mismatch')
    with psycopg.connect(dsn,autocommit=True,connect_timeout=15) as conn:
        with conn.transaction():
            conn.execute('SET TRANSACTION READ ONLY');_require_fresh_schema(conn)
    restore_bundle(path,target,expected_identity=identity)
    source=ReadOnlySQLite(target);references_only=identity.get('representation')=='raw-pair-le-v2'
    try:
        with psycopg.connect(dsn,autocommit=True,connect_timeout=15) as conn:
            with conn.transaction():
                conn.execute('SELECT pg_advisory_xact_lock(%s)',(0x46414C434F4E,))
                _require_fresh_schema(conn)
                store=PostgresGalleryStore(dsn,identity['model_sha256'],identity['recipe_sha256'],identity['dimension'],references_only=references_only,transaction_connection=conn)
                _restore_rows(store,source,conn,batch_size)
        published=PostgresGalleryStore(dsn,identity['model_sha256'],identity['recipe_sha256'],identity['dimension'],references_only=references_only)
        count=_verify_rows(source,published)
        if count!=expected_count:raise ValueError('PostgreSQL restored count mismatch')
        # Snapshot bytes/rawrefs were verified by restore_bundle; SQL bytes,
        # order, boxes, timestamps and optional vectors were verified above.
        removed=True
        for name in ['gallery.sqlite3-wal','gallery.sqlite3-shm','gallery.sqlite3']:
            candidate=target/name
            if candidate.exists():
                if candidate.resolve().parent!=target or candidate.is_symlink():raise RuntimeError('Unsafe validation snapshot cleanup')
                try:candidate.unlink()
                except OSError:removed=False
        receipt={'status':'COMPLETE_COLD_POSTGRES_RESTORE','observations':count,'identity':identity,'target_data_dir':str(target),'source_backup_SHA256':file_sha(path),'target_DDL_identity_and_rows_one_transaction':True,'order_images_boxes_timestamps_refs_vectors_verified':True,'validation_SQLite_copy_removed':removed,'source_or_existing_target_tables_overwritten':False,'activation':'Use this new data directory together with the target PostgreSQL DSN; rollback uses the prior directory and DB. DSN is never saved here.'}
        legacy_raw=runtime_contract is not None and runtime_contract['format']!='october-raw-pair-le-v2'
        if legacy_raw:receipt['status']='COMPLETE_LEGACY_POSTGRES_DATA_RESTORE_REQUIRES_RUNTIME_MIGRATION'
        receipt.update(runtime_consistency_verified=runtime_contract is not None,
                       legacy_runtime_not_validated=runtime_contract is None,
                       current_refonly_runtime_requires_migration=legacy_raw)
        (target/'postgres-restore.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf-8')
        return receipt
    except BaseException as exc:
        (target/'postgres-restore-failed.json').write_text(json.dumps({'status':'POSTGRES_RESTORE_NOT_ACCEPTED','error_type':type(exc).__name__,'verified_local_snapshot_retained':True,'automatic_user_table_deletion':False,'note':'No success receipt. If publication failed inside its SQL transaction, target DDL and rows rolled back. A postcommit verification failure requires inspection; do not activate this target.'},indent=2),encoding='utf-8')
        raise
