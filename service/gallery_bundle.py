"""Portable consistent gallery backup. Images and explicit raw features are included.

A backup is a DB snapshot, never a filesystem copy of a live SQLite WAL.
Restore uses a new directory and publishes it only after complete validation.
"""
from contextlib import contextmanager
from pathlib import Path,PurePosixPath
import hashlib,json,os,re,shutil,sqlite3,tempfile,zipfile
import numpy as np
from .storage import GalleryStore,GalleryIdentityError
from .disk_reserve import WriteDiskReserve,manifest_reserve_gib
from .publish_directory import publish_new_directory


def sha(data):return hashlib.sha256(data).hexdigest()

def file_sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


class ReadOnlySQLite:
    backend='sqlite'
    def __init__(self,data_dir):
        self.data_dir=Path(data_dir).resolve();self.db_path=self.data_dir/'gallery.sqlite3'
        if not self.db_path.is_file():raise ValueError('Source gallery database not found')
        with self._connection() as conn:
            meta=dict(conn.execute('SELECT key,value FROM gallery_metadata'))
        self.model_sha256=meta['model_sha256'];self.recipe_sha256=meta['recipe_sha256'];self.dimension=int(meta['dimension'])
        self.references_only=meta.get('representation')=='raw-pair-le-v2'
    @contextmanager
    def _connection(self):
        conn=sqlite3.connect(self.db_path.as_uri()+'?mode=ro',uri=True,timeout=30)
        try:yield conn
        finally:conn.close()


@contextmanager
def snapshot_rows(store):
    with store._connection() as conn:
        if store.backend=='sqlite':
            conn.execute('BEGIN')
            columns={r[1] for r in conn.execute('PRAGMA table_info(gallery_items)')}
            ref='feature_ref' if 'feature_ref' in columns else 'NULL'
            cursor=conn.execute('SELECT image_id,image_bytes,bbox_xywh,embedding,created_at,'+ref+' FROM gallery_items ORDER BY sequence')
            def rows():
                for row in cursor:
                    yield {'image_id':row[0],'image_bytes':bytes(row[1]),'bbox_xywh':json.loads(row[2]),
                           'embedding':None if getattr(store,'references_only',False) else np.frombuffer(row[3],dtype='<f4').copy(),'created_at':row[4],'feature_ref':row[5]}
            yield rows()
        else:
            with conn.transaction():
                conn.execute('SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY')
                with conn.cursor(name='falcon_backup_snapshot') as cursor:
                    cursor.itersize=2
                    ref='feature_ref' if getattr(store,'has_feature_ref',True) else 'NULL AS feature_ref'
                    cursor.execute('SELECT image_id,image_bytes,bbox_xywh,embedding::text,created_at,'+ref+' FROM gallery_items ORDER BY sequence')
                    def rows():
                        for row in cursor:
                            yield {'image_id':row[0],'image_bytes':bytes(row[1]),'bbox_xywh':row[2],
                                   'embedding':None if getattr(store,'references_only',False) else np.asarray(json.loads(row[3]),np.float32),'created_at':row[4],'feature_ref':row[5]}
                    yield rows()


def create_bundle(store,destination,*,manifest=None,cache_dir=None):
    destination=Path(destination).resolve()
    if destination.exists():raise FileExistsError(destination)
    identity={'model_sha256':store.model_sha256,'recipe_sha256':store.recipe_sha256,'dimension':store.dimension}
    if getattr(store,'references_only',False):identity['representation']='raw-pair-le-v2'
    runtime_data=None;runtime_contract=None
    if cache_dir is not None or getattr(store,'references_only',False):
        if manifest is None:raise GalleryIdentityError('Raw-feature backup requires a matching runtime manifest')
        runtime_data=Path(manifest).read_bytes()
        from .gallery_manifest import validate_raw_manifest
        runtime_contract=validate_raw_manifest(runtime_data,identity)
        if cache_dir is None:raise GalleryIdentityError('Raw-feature backup requires its complete raw cache')
    elif manifest is not None:
        runtime_data=Path(manifest).read_bytes()
    WriteDiskReserve({'backup':destination.parent,'backup_staging':destination.parent},
                     manifest_reserve_gib(runtime_data) if runtime_contract else 0).require()
    destination.parent.mkdir(parents=True,exist_ok=True)
    fd,name=tempfile.mkstemp(prefix='gallery-backup-',suffix='.writing',dir=destination.parent);os.close(fd);temporary=Path(name)
    inventory={};entries=[];refs=set();cache=Path(cache_dir) if cache_dir else None
    def put(archive,name,data):
        if name in inventory:return
        archive.writestr(name,data);inventory[name]={'bytes':len(data),'SHA256':sha(data)}
    try:
        with zipfile.ZipFile(temporary,'w',compression=zipfile.ZIP_STORED,allowZip64=True) as archive:
            with snapshot_rows(store) as rows:
                for i,row in enumerate(rows):
                    image='images/'+str(i)+'.bin';put(archive,image,row.pop('image_bytes'))
                    references_only=getattr(store,'references_only',False)
                    value=row.pop('embedding')
                    vector=None if references_only else np.asarray(value,np.float32)
                    if references_only:
                        if value is not None:raise ValueError('Reference-only snapshot contains vector')
                    elif vector.shape!=(store.dimension,) or not np.isfinite(vector).all():raise ValueError('Malformed snapshot vector')
                    ref=row['feature_ref']
                    if cache is not None:
                        if ref is None:raise ValueError('Raw-feature gallery requires explicit references before backup')
                        from .october_features import RawFeatureStore
                        if references_only:
                            from .october_reference import RawReferenceStore
                            RawFeatureStore=RawReferenceStore
                        RawFeatureStore(cache,store.recipe_sha256).get(ref,persisted=True)
                        put(archive,'raw/objects/'+ref+'.npz',(cache/'objects'/(ref+'.npz')).read_bytes());refs.add(ref)
                    elif ref is not None:raise ValueError('Backup needs the raw cache for explicit feature_refs')
                    entries.append({**row,'image':image,'embedding':None if references_only else vector.tolist()})
            if cache is not None:
                for folder in ['gallery-input-refs','legacy-vector-refs']:
                    for path in sorted((cache/folder).glob('*.json')):
                        record=json.loads(path.read_text(encoding='utf-8'))
                        if record.get('feature_ref') in refs:put(archive,'raw/'+folder+'/'+path.name,path.read_bytes())
                # Recreate the old read-only rollback lookup from verified raw objects.
                for path in sorted((cache/'legacy-vector-refs').glob('*.json')):
                    record=json.loads(path.read_text(encoding='utf-8'))
                    if record.get('feature_ref') in refs and not record.get('ambiguous'):
                        legacy=cache/(path.stem+'.npz')
                        if legacy.is_file():put(archive,'raw/'+legacy.name,legacy.read_bytes())
            if runtime_data is not None:put(archive,'runtime-manifest.json',runtime_data)
            header={'schema':2 if runtime_contract else 1,'identity':identity,
                    'source_backend':store.backend,'observations':entries,'files':inventory,'raw_cache_included':cache is not None}
            if runtime_contract:header['runtime_contract']=runtime_contract
            else:header['legacy_format']='vectors-v1-runtime-not-validated'
            archive.writestr('backup.json',json.dumps(header,ensure_ascii=False,allow_nan=False,separators=(',',':')))
        # Re-open and verify every member before publishing a backup.
        with validated_bundle(temporary):pass
        # Atomic no-overwrite publication on both NTFS and POSIX.
        os.link(temporary,destination);temporary.unlink()
        return {'status':'COMPLETE_GALLERY_SNAPSHOT','observations':len(entries),'files':len(inventory),'bytes':destination.stat().st_size,'SHA256':file_sha(destination),'identity':header['identity'],
                'backup_schema':header['schema'],'runtime_consistency_verified':runtime_contract is not None,
                'runtime_contract':runtime_contract,'legacy_runtime_not_validated':runtime_contract is None}
    finally:
        if temporary.exists():temporary.unlink()


@contextmanager
def validated_bundle(path):
    with zipfile.ZipFile(path) as archive:
        members=archive.infolist();names=[r.filename for r in members]
        if len(names)!=len(set(names)):raise ValueError('Duplicate backup members')
        allowed=re.compile(r'(backup\.json|runtime-manifest\.json|images/[0-9]+\.bin|raw/(objects/[a-f0-9]{64}\.npz|[a-f0-9]{64}\.npz|(gallery-input-refs|legacy-vector-refs)/[a-f0-9]{64}\.json))')
        for info in members:
            if not allowed.fullmatch(info.filename):raise ValueError('Unexpected backup member')
            name=PurePosixPath(info.filename)
            if name.is_absolute() or '..' in name.parts or '\\' in info.filename or (info.external_attr>>16)&0o170000==0o120000:
                raise ValueError('Unsafe backup member')
        if 'backup.json' not in names or archive.getinfo('backup.json').file_size>256*1024**2:raise ValueError('Missing or oversized backup metadata')
        header=json.loads(archive.read('backup.json'))
        identity=header['identity']
        if type(header.get('schema')) is not int or type(header.get('raw_cache_included')) is not bool:
            raise ValueError('Invalid backup schema/cache declaration')
        if any(not isinstance(identity.get(k),str) or not re.fullmatch('[a-f0-9]{64}',identity[k]) for k in ('model_sha256','recipe_sha256')):
            raise ValueError('Invalid backup identity')
        if isinstance(identity['dimension'],bool) or not isinstance(identity['dimension'],int) or not 1<=identity['dimension']<=16384:
            raise ValueError('Invalid backup dimension')
        if header['schema'] not in (1,2) or set(names)!=set(header['files'])|{'backup.json'}:raise ValueError('Backup inventory mismatch')
        for name,record in header['files'].items():
            info=archive.getinfo(name)
            if info.file_size!=record['bytes']:raise ValueError('Backup size mismatch: '+name)
            digest=hashlib.sha256()
            with archive.open(name) as stream:
                for chunk in iter(lambda:stream.read(1024*1024),b''):digest.update(chunk)
            if digest.hexdigest()!=record['SHA256']:raise ValueError('Backup checksum mismatch: '+name)
        if header.get('raw_cache_included') or identity.get('representation')=='raw-pair-le-v2':
            if not header.get('raw_cache_included'):raise GalleryIdentityError('Reference-only backup has no raw cache')
            if 'runtime-manifest.json' not in names:raise GalleryIdentityError('Raw-feature backup has no runtime manifest; explicit legacy migration required')
            from .gallery_manifest import validate_raw_manifest
            contract=validate_raw_manifest(archive.read('runtime-manifest.json'),identity)
            if header['schema']==2 and header.get('runtime_contract')!=contract:
                raise GalleryIdentityError('Backup runtime contract mismatch')
            # Schema1 may be read only when its actual runtime is consistent.
            # A subsequent raw-store/backend change still requires migration.
            header['validated_runtime_contract']=contract
        elif header['schema']==2:
            raise ValueError('Version2 raw-feature bundle has no raw cache')
        yield archive,header


def restore_bundle(path,target,*,expected_identity=None):
    target=Path(target).resolve()
    if target.exists():raise FileExistsError('Restore requires a new data directory')
    staging=None
    try:
        with validated_bundle(path) as (archive,header):
            identity=header['identity']
            if expected_identity is not None and identity!=expected_identity:raise GalleryIdentityError('Backup model/embedding identity mismatch')
            minimum=manifest_reserve_gib(archive.read('runtime-manifest.json')) if header.get('validated_runtime_contract') else 0
            WriteDiskReserve({'data':target,'raw_cache':target/'raw-features','restore_staging':target.parent},minimum).require()
            target.parent.mkdir(parents=True,exist_ok=True)
            staging=Path(tempfile.mkdtemp(prefix='gallery-restore-',dir=target.parent))
            references_only=identity.get('representation')=='raw-pair-le-v2'
            store=GalleryStore(staging,identity['model_sha256'],identity['recipe_sha256'],int(identity['dimension']),references_only=references_only)
            with store._connection() as conn:
                with conn:
                    for item in header['observations']:
                        row={**item,'image_bytes':archive.read(item['image'])}
                        if references_only:
                            from .october_reference import FeatureReference
                            row['embedding']=FeatureReference(item['feature_ref'],identity['recipe_sha256'])
                        prepared=store._prepare(row);prepared=(*prepared[:4],item['created_at'],*prepared[5:])
                        statement='INSERT INTO gallery_items ' + ('(image_id,image_bytes,bbox_xywh,embedding,created_at,feature_ref,content_revision) VALUES (?,?,?,?,?,?,?)' if references_only else '(image_id,image_bytes,bbox_xywh,embedding,created_at,feature_ref) VALUES (?,?,?,?,?,?)')
                        conn.execute(statement,prepared)
            for name in header['files']:
                if name.startswith('raw/'):
                    output=staging/'raw-features'/identity['recipe_sha256']/name[4:]
                elif name=='runtime-manifest.json':output=staging/'backup-runtime-manifest.json'
                else:continue
                output.parent.mkdir(parents=True,exist_ok=True)
                with archive.open(name) as src,output.open('xb') as dst:shutil.copyfileobj(src,dst,1024*1024)
            ids,vectors=store.snapshot()
            if ids!=[r['image_id'] for r in header['observations']]:raise ValueError('Restored observation order mismatch')
            if header['raw_cache_included']:
                from .october_features import RawFeatureStore
                if references_only:
                    from .october_reference import RawReferenceStore
                    RawFeatureStore=RawReferenceStore
                raw=RawFeatureStore(staging/'raw-features'/identity['recipe_sha256'],identity['recipe_sha256'])
                if references_only:
                    for ref in vectors.feature_refs:raw.get(ref,persisted=True)
                else:
                    for vector,ref in zip(vectors,vectors.feature_refs):
                        if not np.array_equal(vector,raw.get(ref,persisted=True)['base']):raise ValueError('Restored raw reference mismatch')
            # Closed connections, complete files and verified order precede activation.
            publish_new_directory(staging,target)
            runtime=header.get('validated_runtime_contract')
            legacy_raw=runtime is not None and runtime['format']!='october-raw-pair-le-v2'
            return {'status':'COMPLETE_LEGACY_DATA_RESTORE_REQUIRES_RUNTIME_MIGRATION' if legacy_raw else 'COMPLETE_COLD_RESTORE',
                    'observations':len(ids),'identity':identity,'target':str(target),
                    'backup_schema':header['schema'],'runtime_consistency_verified':runtime is not None,
                    'legacy_runtime_not_validated':runtime is None,'current_refonly_runtime_requires_migration':legacy_raw}
    finally:
        if staging is not None and staging.exists():
            if staging.resolve().parent!=target.parent or not staging.name.startswith('gallery-restore-') or staging.is_symlink():raise RuntimeError('Unsafe restore staging cleanup')
            shutil.rmtree(staging)
