"""Container commands all use the same October manifest and runtime."""
from pathlib import Path
import os,sys
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))


def main():
    args=sys.argv[1:];command=args.pop(0) if args else 'serve'
    manifest=ROOT/'release_current/runtime-manifest.json'
    if command=='predict':
        from service.october_cli import main as predict
        if not any(v=='--runtime-manifest' or v.startswith('--runtime-manifest=') for v in args):args=['--runtime-manifest',str(manifest),*args]
        return predict(args)
    if command=='serve':
        if '--backend' in args or any(v.startswith('--backend=') for v in args):raise SystemExit('Final assembly uses backend october')
        defaults=['--backend','october']
        if not any(v=='--runtime-manifest' or v.startswith('--runtime-manifest=') for v in args):defaults+=['--runtime-manifest',str(manifest)]
        os.execv(sys.executable,[sys.executable,'-u','-X','utf8','-B',str(ROOT/'scripts/run_service.py'),*defaults,*args])
    if command=='gallery':os.execv(sys.executable,[sys.executable,'-u','-X','utf8','-B',str(ROOT/'scripts/gallery_admin.py'),*args])
    if command=='verify':
        from service.october_manifest import load_manifest
        spec=load_manifest(manifest,ROOT)
        import json
        print(json.dumps({'status':'PASS_ARTIFACTS_AND_ENVIRONMENT_ONLY','profile':spec['profile'],'weights_bytes':spec['weights_bytes'],'scope':'Integrity check; not an inference or quality test'}));return 0
    raise SystemExit('Use serve, predict, gallery or verify')

if __name__=='__main__':raise SystemExit(main())
