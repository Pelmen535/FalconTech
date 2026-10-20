"""Compatible offline CLI using the same OctoberAdapter as HTTP."""
from pathlib import Path
import argparse
import json
import os
import sys
import numpy as np


def main(argv=None):
    parser=argparse.ArgumentParser(description='October ReID: прежние CLI-входы, единый runtime/manifest, top10 и отказ')
    parser.add_argument('--data',type=Path,help='папка images/, test_query.csv, test_gallery.csv; default /data')
    parser.add_argument('--release',type=Path,default=Path('release_current'),help='папка runtime-manifest.json')
    parser.add_argument('--runtime-manifest',type=Path)
    parser.add_argument('--manifest',type=Path,help='входной JSON query/gallery; не модельный manifest')
    parser.add_argument('--query',type=Path);parser.add_argument('--gallery',type=Path);parser.add_argument('--images',type=Path)
    parser.add_argument('--out',type=Path,default=Path('/out'))
    parser.add_argument('--batch-size',type=int,choices=[1,8,16,32],default=16)
    parser.add_argument('--query-batch-size',type=int,choices=[1,8,16,32],help='отдельный query batch для проверки HTTP B1/galleryB16')
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--device',choices=['cpu','cuda'],default=os.environ.get('VREID_DEVICE','cuda'))
    args=parser.parse_args(argv)
    if args.workers<0:parser.error('--workers must be nonnegative')
    root=Path(__file__).resolve().parents[1]
    manifest=args.runtime_manifest
    if manifest is None:
        manifest=args.release/'runtime-manifest.json'
        if not manifest.is_file():
            candidate=root/'var/october-v14-compatibility-20261019/runtime-manifest.json'
            if args.release in [Path('release'),Path('release_current')] and candidate.is_file():
                manifest=candidate
            else:parser.error('release must contain October runtime-manifest.json; no silent v1.4 fallback')
    try:
        if args.out.exists():raise ValueError('Output directory already exists; choose a new directory')
        if args.manifest:
            if any(v is not None for v in [args.query,args.gallery,args.images,args.data]):
                raise ValueError('Use --manifest OR CSV/data inputs')
            inputs=json.loads(args.manifest.read_text(encoding='utf-8-sig'))
        else:
            if any(v is not None for v in [args.query,args.gallery,args.images]):
                if not all(v is not None for v in [args.query,args.gallery,args.images]) or args.data:
                    raise ValueError('Supply all query/gallery/images OR --data')
                query,gallery,images=args.query,args.gallery,args.images
            else:
                data=args.data or Path('/data')
                query,gallery,images=data/'test_query.csv',data/'test_gallery.csv',data/'images'
            from vreid.hackathon_data import read_annotations
            inputs={side:[{'key':r.key,'path':str(Path(r.path).resolve()),'bbox':list(r.bbox)}
                for r in read_annotations(path,images,bbox_format='xywh',strict=True)]
                for side,path in [('query',query),('gallery',gallery)]}
        clean={}
        for side in ['query','gallery']:
            clean[side]=[]
            for row in inputs[side]:
                key=row['key'];path=Path(row['path']);box=row.get('bbox')
                if not isinstance(key,str) or not key or key.strip()!=key or '\x00' in key:
                    raise ValueError('Observation IDs must be nonempty strings')
                if not path.is_file():raise ValueError('Input image not found: '+str(path))
                if box is not None:
                    a=np.asarray(box,dtype=float)
                    if a.shape!=(4,) or not np.isfinite(a).all() or (a[:2]<0).any() or (a[2:]<=0).any():
                        raise ValueError('bbox must be finite XYWH with nonnegative origin and positive extent')
                clean[side].append({'key':key,'path':str(path.resolve()),'bbox':box})
            keys=[r['key'] for r in clean[side]]
            if not keys or len(set(keys))!=len(keys):raise ValueError('Empty or duplicate observation IDs in '+side)
        if len(clean['gallery'])<10:raise ValueError('At least ten distinct gallery observations required')
        from vreid.rerank import frame_block_mask
        gkeys=np.array([r['key'] for r in clean['gallery']]);qkeys=np.array([r['key'] for r in clean['query']])
        for off in range(0,len(qkeys),128):
            if np.any((~frame_block_mask(qkeys[off:off+128],gkeys)).sum(1)<10):
                raise ValueError('At least ten eligible gallery entries per query required')
        from .october import OctoberAdapter
        from .robust import ResourceUnavailableError
        # Separate delivery output from persistent HTTP galleries; no query cache is published.
        data_dir=args.out.with_name(args.out.name+'-runtime')
        core=OctoberAdapter(root,data_dir,args.device,manifest=manifest)
        print(f"October profile: {core.spec['profile']} | manifestSHA={core.manifest_sha256}",flush=True)
        print(f"Weights: {core.spec['weights_bytes']}bytes; confidence=calibrated acceptance score",flush=True)
        try:
            receipt=core.offline(clean['query'],clean['gallery'],args.out,args.batch_size,args.workers,args.query_batch_size)
        finally:core.close()
        print('COMPLETE',args.out,flush=True)
        return 0
    except ValueError as exc:
        parser.error(str(exc))
    except Exception as exc:
        from .robust import ResourceUnavailableError
        print(f'October runtime failed: {exc}',file=sys.stderr)
        return 77 if isinstance(exc,ResourceUnavailableError) else 2

if __name__=='__main__':raise SystemExit(main())
