"""Optional telemetry preserves gates, process isolation and conservative memory."""
import ctypes,io,json,subprocess,threading
from types import SimpleNamespace
import pytest
from service import nvml_resources as nr

class Original:
 def __init__(self):self.calls=[]
 def run(self,*args,**kwargs):self.calls.append((args,kwargs));return subprocess.CompletedProcess(args[0],0,stdout='original',stderr='')
class Provider:
 def query(self):return 2048.,1024.,37
 def close(self):self.closed=True

def test_only_exact_resource_gpu_command_is_intercepted():
 original=Original();proxy=nr.GPUProxy(original,Provider());standard=subprocess.run
 out=proxy.run(nr.GPU_ARGS,capture_output=True,text=True,timeout=4);assert out.returncode==0 and list(map(float,out.stdout.strip().split(',')))==[2048,1024,37];assert not original.calls
 proxy.run(['nvidia-smi','-L'],text=True);assert original.calls==[((['nvidia-smi','-L'],),{'text':True})]
 assert subprocess.run is standard and proxy.fast_calls==1 and proxy.fallback_calls==0

def test_query_failure_uses_original_resource_call_without_inventing_headroom():
 class Failure:
  def query(self):raise RuntimeError('measurement unavailable')
 original=Original();proxy=nr.GPUProxy(original,Failure());kwargs={'text':True,'timeout':4,'check':True};out=proxy.run(nr.GPU_ARGS,**kwargs)
 assert out.stdout=='original' and original.calls==[((nr.GPU_ARGS,),kwargs)] and proxy.fallback_calls==1 and proxy.fast_calls==0

def test_unavailable_nvml_does_not_change_existing_snapshot_or_provider(monkeypatch):
 def unavailable():raise OSError('driver DLL absent')
 monkeypatch.setattr(nr,'NVML',unavailable);original=Original();snapshot=lambda:{'RAM_free_GiB':2,'GPU_free_GiB':None};resource=SimpleNamespace(subprocess=original,snapshot=snapshot);log=io.StringIO()
 assert nr.install(resource,log) is None and resource.subprocess is original and resource.snapshot is snapshot
 assert json.loads(log.getvalue())['active']=='nvidia-smi-fallback'

def test_install_is_local_idempotent_and_restores_after_close(monkeypatch):
 callbacks=[];provider=Provider();monkeypatch.setattr(nr,'NVML',lambda:provider);monkeypatch.setattr(nr.atexit,'register',callbacks.append);original=Original();snapshot=lambda:None;resource=SimpleNamespace(subprocess=original,snapshot=snapshot);log=io.StringIO();standard=subprocess.run
 proxy=nr.install(resource,log);assert nr.install(resource,log) is proxy and len(callbacks)==1
 assert resource.snapshot is snapshot and subprocess.run is standard
 resource.subprocess.run(nr.GPU_ARGS,text=True);callbacks[0]();assert resource.subprocess is original and provider.closed
 events=[json.loads(x) for x in log.getvalue().splitlines()];assert events[0]['no_threshold_rate_or_other_snapshot_fields_changed'] and events[1]['NVML_calls']==1

def test_memory_accounting_never_exceeds_physical_free_or_total_minus_used():
 class Library:
  def nvmlDeviceGetMemoryInfo(self,handle,pointer):
   m=ctypes.cast(pointer,ctypes.POINTER(nr.Memory)).contents;m.total=2*2**30;m.free=2**30;m.used=2**29;return 0
  def nvmlDeviceGetUtilizationRates(self,handle,pointer):ctypes.cast(pointer,ctypes.POINTER(nr.Util)).contents.gpu=42;return 0
 p=nr.NVML.__new__(nr.NVML);p.lib=Library();p.lock=threading.Lock();p.closed=False;p.handle=ctypes.c_void_p(1);total,used,util=p.query();assert total-used==1024 and util==42
