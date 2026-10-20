"""One new DINO forward supplies global+local outputs; original CNN stays exact."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import shared_memory
import ctypes,subprocess,struct,json,contextlib
import numpy as np

def core_class(original,spec):
 import joint_runtime as source
 class SingleCore:
  def __init__(self,logs):
   self.pool=None;self.mem=None;self.arrays=[];self.err=None;self.p=None
   assert source.digest(Path(spec['local_engine']))==spec['files'][spec['local_engine']]
   proof=json.loads((source.E/'output-proof.json').read_text());assert source.digest(source.E/'model.engine')==proof['engine_sha256'];assert source.digest(source.NORMALIZER)=='8999ad14040324f6becc591454f96ed57fc77b07de6aa9863587314fb92c5673'
   try:
    self.dll=ctypes.CDLL(str(source.NORMALIZER));self.normalize=self.dll.normalize_rgb;self.normalize.argtypes=[ctypes.POINTER(ctypes.c_uint8),ctypes.POINTER(ctypes.c_float),ctypes.c_size_t];self.normalize.restype=None;self.pool=ThreadPoolExecutor(max_workers=4);n0=32*3*384*384;self.mem=shared_memory.SharedMemory(create=True,size=(n0+32*3*256*256)*4);self.arrays=[np.ndarray((32,3,384,384),np.float32,buffer=self.mem.buf),np.ndarray((32,3,256,256),np.float32,buffer=self.mem.buf,offset=n0*4)];root=Path(__file__).resolve().parents[1];self.err=(Path(logs)/'single-local-stderr.log').open('x',encoding='utf-8');self.p=subprocess.Popen([str(source.N/'python312/tools/python.exe'),'-X','utf8','-B',str(root/'service/single_local_entry.py'),self.mem.name,spec['local_engine'],spec['files'][spec['local_engine']]],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.err,creationflags=subprocess.CREATE_NO_WINDOW);assert self.exact(4)==b'RDY!'
   except BaseException:
    with contextlib.suppress(Exception):self.close()
    raise
  def exact(self,n):
   data=bytearray()
   while len(data)<n:
    part=self.p.stdout.read(n-len(data))
    if not part:raise RuntimeError('Single local worker exited; inspect stderr')
    data.extend(part)
   return bytes(data)
  def embed(self,batch):
   x,cf=batch;n=len(x);assert 1<=n<=32;self.arrays[0][:n]=x;self.arrays[1][:n]=cf.numpy();self.p.stdin.write(struct.pack('<I',n));self.p.stdin.flush();raw=np.frombuffer(self.exact(n*4096*4),np.float32).reshape(n,4096).copy();local=np.frombuffer(self.exact(n*36*1024*2),np.float16).reshape(n,36,1024).copy();return {'emb':raw[:,:2048],'specialist':raw[:,2048:].astype(np.float16),'local36':local}
  def close(self):
   try:
    if self.pool is not None:self.pool.shutdown(wait=True)
    if self.p is not None:
     if self.p.poll() is None:self.p.stdin.write(struct.pack('<I',0));self.p.stdin.flush();self.p.wait(timeout=30)
     assert self.p.returncode==0
   finally:
    if self.err is not None:self.err.close()
    self.arrays.clear()
    if self.mem is not None:self.mem.close();self.mem.unlink()
 return SingleCore
