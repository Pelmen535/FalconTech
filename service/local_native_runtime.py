"""Keep the exact old global/CNN worker; add one isolated local DINO pass."""
from pathlib import Path
import subprocess,struct
from multiprocessing import shared_memory
import numpy as np

def core_class(original,spec):
 class DualCore:
  def __init__(self,logs):
   self.base=None;self.mem=None;self.input=None;self.error=None;self.worker=None
   try:
    folder=Path(logs)/'original-global';folder.mkdir();self.base=original(folder);self.pool=self.base.pool;self.normalize=self.base.normalize;self.mem=shared_memory.SharedMemory(create=True,size=32*3*384*384*4);self.input=np.ndarray((32,3,384,384),np.float32,buffer=self.mem.buf);self.error=(Path(logs)/'local-token-stderr.log').open('x',encoding='utf-8');root=Path(__file__).resolve().parents[1];python=root/'var/nive-night-transfer-20261009/python312/tools/python.exe';self.worker=subprocess.Popen([str(python),'-X','utf8','-B',str(root/'service/local_token_worker.py'),self.mem.name,spec['local_engine'],spec['files'][spec['local_engine']]],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.error,creationflags=subprocess.CREATE_NO_WINDOW)
    if self.exact(4)!=b'RDY!':raise RuntimeError('Local token worker startup failed')
   except BaseException:
    import contextlib
    with contextlib.suppress(Exception):self.close()
    raise
  def exact(self,n):
   data=bytearray()
   while len(data)<n:
    part=self.worker.stdout.read(n-len(data))
    if not part:raise RuntimeError('Local token worker exited; inspect local-token-stderr.log')
    data.extend(part)
   return bytes(data)
  def embed(self,batch):
   out=self.base.embed(batch);x=batch[0];n=len(x);self.input[:n]=x;self.worker.stdin.write(struct.pack('<I',n));self.worker.stdin.flush();out['local36']=np.frombuffer(self.exact(n*36*1024*2),np.float16).reshape(n,36,1024).copy();return out
  def close(self):
   try:
    if self.worker is not None:
     if self.worker.poll() is None:self.worker.stdin.write(struct.pack('<I',0));self.worker.stdin.flush();self.worker.wait(timeout=30)
     assert self.worker.returncode==0
   finally:
    try:
     if self.error is not None:self.error.close()
     self.input=None
     if self.mem is not None:self.mem.close();self.mem.unlink()
    finally:
     if self.base is not None:self.base.close()
 return DualCore
