"""Optional direct GPU telemetry, preserving every existing resource gate.

Only the resource module's subprocess dependency is substituted. Other users of
subprocess retain their normal implementation. NVML failure delegates to the
original nvidia-smi call; snapshots are never cached or skipped.
"""
import atexit,ctypes,json,os,subprocess,sys,threading
from pathlib import Path

GPU_ARGS=['nvidia-smi','--query-gpu=memory.total,memory.used,utilization.gpu','--format=csv,noheader,nounits']
class Memory(ctypes.Structure):_fields_=[('total',ctypes.c_ulonglong),('free',ctypes.c_ulonglong),('used',ctypes.c_ulonglong)]
class Util(ctypes.Structure):_fields_=[('gpu',ctypes.c_uint),('memory',ctypes.c_uint)]

class NVML:
 def __init__(self):
  self.lib=ctypes.CDLL(str(Path(os.environ['WINDIR'])/'System32/nvml.dll') if os.name=='nt' else 'libnvidia-ml.so.1');self.lock=threading.Lock();self.closed=False
  for name in ['nvmlInit_v2','nvmlShutdown','nvmlDeviceGetHandleByIndex_v2','nvmlDeviceGetMemoryInfo','nvmlDeviceGetUtilizationRates']:getattr(self.lib,name).restype=ctypes.c_int
  self.lib.nvmlDeviceGetHandleByIndex_v2.argtypes=[ctypes.c_uint,ctypes.POINTER(ctypes.c_void_p)];self.lib.nvmlDeviceGetMemoryInfo.argtypes=[ctypes.c_void_p,ctypes.POINTER(Memory)];self.lib.nvmlDeviceGetUtilizationRates.argtypes=[ctypes.c_void_p,ctypes.POINTER(Util)]
  if self.lib.nvmlInit_v2()!=0:raise RuntimeError('NVML initialization failed')
  self.handle=ctypes.c_void_p()
  if self.lib.nvmlDeviceGetHandleByIndex_v2(0,ctypes.byref(self.handle))!=0:self.close();raise RuntimeError('NVML GPU0 unavailable')
 def query(self):
  with self.lock:
   if self.closed:raise RuntimeError('NVML closed')
   m=Memory();u=Util()
   if self.lib.nvmlDeviceGetMemoryInfo(self.handle,ctypes.byref(m))!=0 or self.lib.nvmlDeviceGetUtilizationRates(self.handle,ctypes.byref(u))!=0:raise RuntimeError('NVML GPU measurement failed')
   if not 0<=m.free<=m.total or not 0<=m.used<=m.total or not 0<=u.gpu<=100:raise RuntimeError('Invalid NVML measurements')
   # V1 used includes reserved driver memory on this driver. Encode a free
   # reserve no larger than either physical free or total-used; old gate then
   # subtracts used as before. Never overstate memory through integer rounding.
   free=min(m.free,m.total-m.used);return m.total/2**20,(m.total-free)/2**20,int(u.gpu)
 def close(self):
  if not self.closed:self.closed=True;self.lib.nvmlShutdown()

class GPUProxy:
 def __init__(self,original,provider):self.original=original;self.provider=provider;self.fast_calls=0;self.fallback_calls=0
 def __getattr__(self,name):return getattr(self.original,name)
 def run(self,args,*extra,**kwargs):
  if not isinstance(args,(list,tuple)) or list(args)!=GPU_ARGS:return self.original.run(args,*extra,**kwargs)
  try:
   total,used,util=self.provider.query();stdout=f'{total:.9f}, {used:.9f}, {util}\n';self.fast_calls+=1;return subprocess.CompletedProcess(args,0,stdout=stdout,stderr='')
  except Exception:
   self.fallback_calls+=1;return self.original.run(args,*extra,**kwargs)

def install(resource_module,log=None):
 log=log or sys.stderr
 if getattr(resource_module.subprocess,'_native_nvml_proxy',False):return resource_module.subprocess
 try:provider=NVML()
 except Exception as exc:
  print(json.dumps({'event':'resource.provider','active':'nvidia-smi-fallback','error':str(exc)}),file=log,flush=True);return None
 original=resource_module.subprocess;proxy=GPUProxy(original,provider);proxy._native_nvml_proxy=True;resource_module.subprocess=proxy
 print(json.dumps({'event':'resource.provider','active':'nvml','no_threshold_rate_or_other_snapshot_fields_changed':True}),file=log,flush=True)
 def finish():
  resource_module.subprocess=original;provider.close();print(json.dumps({'event':'resource.provider.summary','NVML_calls':proxy.fast_calls,'SMI_fallback_calls':proxy.fallback_calls}),file=log,flush=True)
 atexit.register(finish);return proxy
