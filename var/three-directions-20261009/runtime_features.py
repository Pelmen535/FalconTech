from env import *
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import shared_memory
import importlib.util,ctypes,subprocess,struct
from trt_runtime import TRTRuntime,NORMALIZER

class NewCore(TRTRuntime):
 def __init__(self,logs,batch_size=16,allow_distinct=False):
  proof=json.loads((W/'export/output-proof.json').read_text());self.spec=json.loads((W/'export/spec.json').read_text());engine=W/'export/model.engine';assert proof['status'] in ['PASS','FAIL_NUMERICAL_PARITY'] and proof['repeat_maxdiff']==0 and proof['norm_maxdiff']<=.001 and digest(engine)==proof['engine_sha256'];assert digest(NORMALIZER)=='8999ad14040324f6becc591454f96ed57fc77b07de6aa9863587314fb92c5673';assert self.spec['selection_SHA']==digest(W/'selection.json')
  if not allow_distinct:
   admission=json.loads((W/'backend-admission.json').read_text());assert admission['status']=='PASS_DISTINCT_BACKEND' and admission['engine_SHA']==digest(engine)
  assert batch_size in [1,4,8,16,32];self.batch_size=batch_size;self.dll=ctypes.CDLL(str(NORMALIZER));self.normalize=self.dll.normalize_rgb;self.normalize.argtypes=[ctypes.POINTER(ctypes.c_uint8),ctypes.POINTER(ctypes.c_float),ctypes.c_size_t];self.normalize.restype=None
  self.pool=ThreadPoolExecutor(max_workers=4);self.producer=ThreadPoolExecutor(max_workers=1);self.mem=shared_memory.SharedMemory(create=True,size=32*3*384*384*4);self.shared=np.ndarray((32,3,384,384),np.float32,buffer=self.mem.buf);self.err=(Path(logs)/'trt-stderr.log').open('x',encoding='utf-8');self.p=subprocess.Popen([str(N/'python312/tools/python.exe'),'-X','utf8','-B',str(W/'trt_worker.py'),self.mem.name],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.err,creationflags=subprocess.CREATE_NO_WINDOW)
  if self.exact(4)!=b'RDY!':self.close();raise RuntimeError('NewTRTstartupfail')
 def embed(self,batch):
  a=np.asarray(batch,dtype=np.float32);assert a.ndim==4 and a.shape[1:]==(3,384,384) and 1<=len(a)<=32;self.shared[:len(a)]=a;self.p.stdin.write(struct.pack('<I',len(a)));self.p.stdin.flush();return np.frombuffer(self.exact(len(a)*self.spec['size']*4),np.float32).reshape(len(a),self.spec['size']).copy()
 def close(self):
  try:super().close()
  finally:del self.shared;self.mem.close();self.mem.unlink()

class Features:
 def __init__(self,logs,batch_size=16,allow_distinct=False,cfg_override=None):
  torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;assert cfg_override is None or allow_distinct,'override restricted to exploratory quality-free runtime pilot';self.cfg=cfg_override or json.loads((W/'selection.json').read_text())['config'];self.batch_size=batch_size;self.reused_core=self.cfg['kind']=='base' and not any(self.cfg.get(x) for x in ['local','pair','token'])
  if not allow_distinct:
   admission=json.loads((W/'backend-admission.json').read_text());assert admission['status']=='PASS_DISTINCT_BACKEND' and admission['selection_SHA']==digest(W/'selection.json')
  if self.reused_core:
   spec=importlib.util.spec_from_file_location('protected_incumbent_core',B/'trt_runtime_new.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);self.core=mod.Runtime(logs,batch_size);self.spec={'components':[{'name':'raw','start':0,'groups':1,'width':2048}],'size':2048};self.engine=B/'export/model.engine'
  else:self.core=NewCore(logs,batch_size,allow_distinct);self.spec=self.core.spec;self.engine=W/'export/model.engine'
  self.producer=ThreadPoolExecutor(max_workers=1);self.graphs={};self.cnn=None
  if self.cfg.get('cnn'):
   ck=torch.load(W/self.cfg['cnn_head'],map_location='cpu',weights_only=True);self.cnn=load_cnn();self.cnn.load_state_dict(ck['model']);self.cnn.eval()
 def prepare(self,rows):
  x=self.core.prepare(rows);cf=torch.stack(list(self.core.pool.map(cnn_preprocess,rows))) if self.cnn is not None else None;return x,cf
 def embed(self,batch):
  x,cf=batch;blob=self.core.embed(x);out={}
  for c in self.spec['components']:
   a=blob[:,c['start']:c['start']+c['groups']*c['width']];out['emb' if c['name']=='raw' else c['name']]=a if c['name']=='raw' else a.astype(np.float16).reshape(len(a),64,64) if c['name']=='tokens' else a.astype(np.float16)
  if cf is not None:
   size=len(cf)
   if size not in self.graphs:
    inp=cf.cuda();warm=torch.cuda.Stream();warm.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warm),torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
     for _ in range(5):self.cnn(inp)
    torch.cuda.current_stream().wait_stream(warm);graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph),torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):result=self.cnn(inp)
    self.graphs[size]=(inp,result,graph)
   inp,result,graph=self.graphs[size];inp.copy_(cf);graph.replay();out['specialist']=result.cpu().numpy().astype(np.float16)
  return out
 def pack(self,rows,features):
  n=len(rows);p={'keys':np.array([r['key'] for r in rows]),'vids':np.array([r.get('vid',0) for r in rows]),'cams':np.array([r.get('cam',0) for r in rows]),**features};p.setdefault('parts',np.zeros((n,1),np.float16));p.setdefault('cnn',np.zeros((n,1),np.float16));p.setdefault('specialist',p['cnn']);p.setdefault('tokens',np.zeros((n,1),np.float16));return p
 def extract(self,rows):
  assert rows;start=time.perf_counter();b=self.batch_size;pending=self.producer.submit(self.prepare,rows[:b]);allout=[]
  for off in range(0,len(rows),b):
   prepared=pending.result()
   if off+b<len(rows):pending=self.producer.submit(self.prepare,rows[off+b:off+2*b])
   allout.append(self.embed(prepared))
  elapsed=time.perf_counter()-start;out={k:np.concatenate([x[k] for x in allout]) for k in allout[0]};assert all(np.isfinite(v).all() for v in out.values());return self.pack(rows,out),{'count':len(rows),'seconds':elapsed,'FPS':len(rows)/elapsed,'scope':'all JPEG reads/decodes/crops/preprocess + main TRT + optional CNNgraph + component transfer/quantization; projections/ranking timed separately for final speed','reused_core':self.reused_core}
 def close(self):
  self.producer.shutdown(wait=True);self.graphs.clear();self.core.close()
