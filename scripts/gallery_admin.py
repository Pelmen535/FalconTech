"""Backup, cold restore and explicit gallery re-encoding. Never changes active data."""
from pathlib import Path
import argparse,json,sys,os
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def main():
    parser=argparse.ArgumentParser(description=__doc__);commands=parser.add_subparsers(dest='command',required=True)
    backup=commands.add_parser('backup');backup.add_argument('--data-dir',required=True,type=Path);backup.add_argument('--out',required=True,type=Path);backup.add_argument('--runtime-manifest',type=Path);backup.add_argument('--raw-cache',type=Path);backup.add_argument('--postgres',action='store_true',help='Read-only source from VREID_DATABASE_URL; data-dir identifies its local cache')
    restore=commands.add_parser('restore');restore.add_argument('--backup',required=True,type=Path);restore.add_argument('--target-dir',required=True,type=Path);restore.add_argument('--postgres',action='store_true',help='Restore into a fresh PostgreSQL schema from VREID_DATABASE_URL')
    migration=commands.add_parser('migrate');source=migration.add_mutually_exclusive_group(required=True);source.add_argument('--source-dir',type=Path);source.add_argument('--source-postgres',action='store_true',help='Read-only source from VREID_SOURCE_DATABASE_URL');migration.add_argument('--target-dir',required=True,type=Path);migration.add_argument('--runtime-manifest',required=True,type=Path);migration.add_argument('--device',choices=['cpu','cuda'],default='cuda');migration.add_argument('--batch-size',type=int,default=16)
    args=parser.parse_args()
    def required_dsn(name):
        value=os.environ.get(name,'').strip()
        if not value:parser.error(name+' is required for the selected PostgreSQL operation')
        return value
    from service.gallery_bundle import ReadOnlySQLite,create_bundle,restore_bundle
    if args.command=='backup':
        if args.postgres:
            from service.gallery_postgres_admin import ReadOnlyPostgres
            store=ReadOnlyPostgres(required_dsn('VREID_DATABASE_URL'))
        else:store=ReadOnlySQLite(args.data_dir)
        cache=args.raw_cache
        if cache is None and store.references_only:cache=args.data_dir/'raw-features'/store.recipe_sha256
        result=create_bundle(store,args.out,manifest=args.runtime_manifest,cache_dir=cache)
    elif args.command=='restore':
        if args.postgres:
            from service.gallery_postgres_admin import restore_bundle_postgres
            result=restore_bundle_postgres(args.backup,args.target_dir,required_dsn('VREID_DATABASE_URL'))
        else:result=restore_bundle(args.backup,args.target_dir)
    else:
        from service.october import OctoberAdapter
        from service.gallery_migration import migrate_gallery,migrate_postgres_gallery
        factory=lambda data:OctoberAdapter(ROOT,data,args.device,manifest=args.runtime_manifest)
        if args.source_postgres:
            result=migrate_postgres_gallery(required_dsn('VREID_SOURCE_DATABASE_URL'),args.target_dir,factory,batch_size=args.batch_size)
        else:result=migrate_gallery(args.source_dir,args.target_dir,factory,batch_size=args.batch_size)
    print(json.dumps(result,ensure_ascii=False,indent=2));return 0

if __name__=='__main__':raise SystemExit(main())
