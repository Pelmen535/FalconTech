"""Regression for native TensorRT messages corrupting October JSON RPC."""
from pathlib import Path
import subprocess,sys,json
ROOT=Path(__file__).resolve().parents[2]

source=r"""import sys,io,os
sys.path.insert(0,sys.argv[1])
from service import october_worker as worker
from service import october_manifest as manifest,october_resources as resources
manifest.load_manifest=lambda *args:{'capabilities':{'cpu':True},'operations':{},'profile':'probe','identity':{'model_sha256':'test'}}
class Gate:
 def __init__(self,*args,**kwargs):pass
 def start_reasons(self):return [],{}
 def live_reasons(self):return [],{}
resources.ResourceGate=Gate
class Runtime:
 def __init__(self,*args):os.write(1,b'native constructor warning\n')
 def encode(self,message):
  os.write(1,b'native CUDA warning\n');return {'features':True}
 def close(self):os.write(1,b'native close warning\n')
worker.OctoberRuntime=Runtime
sys.stdin=io.StringIO('{"command":"encode","rpc_id":"one"}\n{"command":"close","rpc_id":"two"}\n')
sys.argv=['worker','--root','/tmp/probe-root','--manifest','/tmp/probe-manifest','--cache','/tmp/probe/cache','--device','cuda']
worker.sys.platform='linux'
worker.main()
"""
def test_actual_october_rpc_separates_native_fd1():
 p=subprocess.run([sys.executable,'-u','-B','-c',source,str(ROOT)],capture_output=True,text=True)
 assert p.returncode==0,p.stderr
 rows=[json.loads(line) for line in p.stdout.splitlines()]
 assert rows==[{'ok':True,'result':{'features':True},'rpc_id':'one'},{'ok':True,'result':True,'rpc_id':'two'}]
 for warning in ['native constructor warning','native CUDA warning','native close warning']:
  assert warning in p.stderr and warning not in p.stdout
