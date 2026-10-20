"""Download immutable v2.0 artifacts, verifying size and SHA-256 before publication."""
from pathlib import Path, PurePosixPath
import argparse,hashlib,json,os,tempfile,urllib.request
ROOT=Path(__file__).resolve().parents[1]

def digest(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
 return h.hexdigest()

def records(root):
 manifest=json.loads((root/'release/weights.json').read_text(encoding='utf-8'))
 if manifest.get('schema')!=2 or manifest.get('tag')!='v2.0':raise ValueError('Unsupported weights manifest')
 rows=manifest['files'];names=set();total=0
 for row in rows:
  name=row['path'];p=PurePosixPath(name)
  if not name or p.is_absolute() or '..' in p.parts or '\\' in name or ':' in name or name in names:raise ValueError('Unsafe or duplicate artifact path')
  names.add(name);target=(root/Path(*p.parts)).resolve()
  if not target.is_relative_to(root.resolve()):raise ValueError('Artifact escapes root')
  if row['bytes']<=0 or len(row['sha256'])!=64 or any(c not in '0123456789abcdef' for c in row['sha256']):raise ValueError('Invalid artifact pin')
  if row['url']!='https://github.com/Pelmen535/FalconTech/releases/download/v2.0/'+p.name:raise ValueError('Unexpected artifact source')
  total+=row['bytes']
 if len(rows)!=16 or total!=manifest['sum_bytes'] or total>manifest['limit_bytes'] or manifest['limit_bytes']!=2_000_000_000:raise ValueError('Weight count/budget mismatch')
 runtime=json.loads((root/'release_current/runtime-manifest.json').read_text(encoding='utf-8'))
 if set(runtime['weight_files'])!=names:raise ValueError('Runtime and download manifests differ')
 for r in rows:
  p=runtime['files'][r['path']]
  if (p['bytes'],p['SHA256'])!=(r['bytes'],r['sha256']):raise ValueError('Runtime and download checksums differ')
 return rows

def fetch(row,root,verify_only=False):
 target=root/row['path']
 if target.exists():
  if target.is_symlink() or target.stat().st_size!=row['bytes'] or digest(target)!=row['sha256']:raise ValueError('Existing artifact differs; preserve/rename it before retry: '+row['path'])
  print('VERIFIED',row['path'],flush=True);return
 if verify_only:raise FileNotFoundError(row['path'])
 target.parent.mkdir(parents=True,exist_ok=True)
 # Own adjacent temporary file, then an atomic no-overwrite publication.
 temporary=None
 try:
  request=urllib.request.Request(row['url'],headers={'User-Agent':'FalconTech-v2.0-weight-fetcher'})
  with tempfile.NamedTemporaryFile(prefix=target.name+'.',suffix='.download',dir=target.parent,delete=False) as f:
   temporary=Path(f.name);h=hashlib.sha256();count=0
   with urllib.request.urlopen(request,timeout=120) as response:
    for block in iter(lambda:response.read(4*1024*1024),b''):
     count+=len(block)
     if count>row['bytes']:raise ValueError('Artifact exceeds pinned size')
     f.write(block);h.update(block)
   f.flush();os.fsync(f.fileno())
  if count!=row['bytes'] or h.hexdigest()!=row['sha256']:raise ValueError('Download integrity failure: '+row['path'])
  os.link(temporary,target)
  print('DOWNLOADED_VERIFIED',row['path'],count,flush=True)
 finally:
  if temporary is not None:temporary.unlink(missing_ok=True)

def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--verify-only',action='store_true');parser.add_argument('--metadata-only',action='store_true');args=parser.parse_args()
 rows=records(ROOT)
 if not args.metadata_only:
  for row in rows:fetch(row,ROOT,args.verify_only)
 print('PASS_WEIGHTS_METADATA' if args.metadata_only else 'PASS_ALL_16_WEIGHT_FILES',sum(r['bytes'] for r in rows),'bytes')

if __name__=='__main__':main()
