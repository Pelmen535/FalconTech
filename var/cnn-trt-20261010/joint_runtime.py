from context import *
from runtime_padded_decode import FeaturesPadded
from runtime_features import Features
from trt_runtime import NORMALIZER
from multiprocessing import shared_memory
from concurrent.futures import ThreadPoolExecutor
import ctypes,subprocess,struct
class Core:
 def __init__(self,logs):
  assert digest(B/'export/model.engine')=='68b18e2ff7cc9e23cbe2dd8143fd49eec5fe0ac1c2b1e9fc46dc2994eec0e786';proof=json.loads((E/'output-proof.json').read_text());assert proof['engine_sha256']==digest(E/'model.engine') and proof['repeat_maxdiff']<=1e-6 and proof['norm_maxdiff']<=.001;assert digest(NORMALIZER)=='8999ad14040324f6becc591454f96ed57fc77b07de6aa9863587314fb92c5673';self.dll=ctypes.CDLL(str(NORMALIZER));self.normalize=self.dll.normalize_rgb;self.normalize.argtypes=[ctypes.POINTER(ctypes.c_uint8),ctypes.POINTER(ctypes.c_float),ctypes.c_size_t];self.normalize.restype=None;self.pool=ThreadPoolExecutor(max_workers=4);n0=32*3*384*384;self.mem=shared_memory.SharedMemory(create=True,size=(n0+32*3*256*256)*4);self.arrays=[np.ndarray((32,3,384,384),np.float32,buffer=self.mem.buf),np.ndarray((32,3,256,256),np.float32,buffer=self.mem.buf,offset=n0*4)];self.err=(Path(logs)/'joint-trt-stderr.log').open('x',encoding='utf-8');self.p=subprocess.Popen([str(N/'python312/tools/python.exe'),'-X','utf8','-B',str(J/'joint_worker.py'),self.mem.name],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.err,creationflags=subprocess.CREATE_NO_WINDOW);assert self.exact(4)==b'RDY!'
 def exact(self,n):
  data=bytearray()
  while len(data)<n:
   part=self.p.stdout.read(n-len(data))
   if not part:raise RuntimeError('Joint worker exited, inspect log')
   data.extend(part)
  return bytes(data)
 def embed(self,batch):
  x,cf=batch;size=len(x);assert 1<=size<=32;self.arrays[0][:size]=x;self.arrays[1][:size]=cf.numpy();self.p.stdin.write(struct.pack('<I',size));self.p.stdin.flush();blob=np.frombuffer(self.exact(size*4096*4),np.float32).reshape(size,4096).copy();return {'emb':blob[:,:2048],'specialist':blob[:,2048:].astype(np.float16)}
 def close(self):
  self.pool.shutdown(wait=True)
  if self.p.poll() is None:self.p.stdin.write(struct.pack('<I',0));self.p.stdin.flush();self.p.wait(timeout=30)
  self.err.close();self.arrays.clear();self.mem.close();self.mem.unlink();assert self.p.returncode==0
class JointFeatures:
 prepare_dual=FeaturesPadded.prepare_dual
 prepare=FeaturesPadded.prepare
 pack=Features.pack
 extract=Features.extract
 def __init__(self,logs,batch_size=16):
  torch.set_num_threads(2);self.core=Core(logs);self.cnn=True;self.batch_size=batch_size;self.cfg=json.loads((W/'fusion-selection.json').read_text())['config'];self.reused_core=True;self.engine=B/'export/model.engine';self.producer=ThreadPoolExecutor(max_workers=1)
 def embed(self,batch):return self.core.embed(batch)
 def close(self):self.producer.shutdown(wait=True);self.core.close()
