"""Small stdlib-only resource gate; never adjusts or stops unrelated processes."""
import ctypes,json,os,shutil,subprocess,time,tempfile
from pathlib import Path
HERE=Path(__file__).resolve().parent
OUT=Path(tempfile.gettempdir())/'FalconTech-teacher-projection-20261011'
GiB=1024**3
_previous_cpu=None
class Mem(ctypes.Structure):
 _fields_=[('length',ctypes.c_ulong),('load',ctypes.c_ulong),*[(n,ctypes.c_ulonglong) for n in ('total','available','commit_total','commit_available','virtual_total','virtual_available','extended')]]
def cpu_percent():
 global _previous_cpu
 values=[]
 for _ in range(3):values.append(ctypes.c_ulonglong())
 ok=ctypes.windll.kernel32.GetSystemTimes(*[ctypes.byref(v) for v in values])
 if not ok:return None
 idle,kernel,user=[v.value for v in values];now=(idle,kernel+user);previous=_previous_cpu;_previous_cpu=now
 if previous is None or now[1]<=previous[1]:return None
 return max(0,min(100,100*(1-(now[0]-previous[0])/(now[1]-previous[1]))))
def snapshot():
 m=Mem();m.length=ctypes.sizeof(m);assert ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
 s={'time':time.time(),'RAM_free_GiB':m.available/GiB,'RAM_total_GiB':m.total/GiB,'CPU_percent':cpu_percent(),'C_free_GiB':shutil.disk_usage(OUT.parent).free/GiB,'GPU_free_GiB':None,'GPU_util_percent':None}
 try:
  r=subprocess.run(['nvidia-smi','--query-gpu=memory.total,memory.used,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=4,creationflags=subprocess.CREATE_NO_WINDOW)
  assert r.returncode==0
  total,used,util=[float(v.strip()) for v in r.stdout.strip().splitlines()[0].split(',')];s.update(GPU_free_GiB=(total-used)/1024,GPU_util_percent=util)
 except (OSError,ValueError,AssertionError,subprocess.TimeoutExpired):pass
 jobs=map_processes();s['priority_apps']=jobs;s['map_editor_active']=jobs is None or any(v['name'].lower() not in ['sc2_x64.exe','echochasm-win64-shipping.exe'] for v in jobs)
 return s

def start_reasons(s,kind):
 reasons=[]
 limits={'RAM_free_GiB':1.5 if kind=='gpu' else 2.5,'C_free_GiB':10 if kind=='gpu' else 9}
 for name,minimum in limits.items():
  if s.get(name) is None or s[name]<minimum:reasons.append(f'{name} below {minimum}')
 if s.get('CPU_percent') is None or s['CPU_percent']>(60 if kind=='gpu' else 70):reasons.append('CPU not idle enough or unknown')
 if kind=='gpu':
  if s.get('GPU_free_GiB') is None or s['GPU_free_GiB']<9:reasons.append('GPU free below9GiB or unknown')
  limit=20 if s.get('map_editor_active',True) else 40
  if s.get('GPU_util_percent') is None or s['GPU_util_percent']>limit:reasons.append('GPU busy or unknown; limit '+str(limit))
 return reasons

def yield_reasons(s,kind='gpu'):
 r=[]
 if s['RAM_free_GiB']<1.5:r.append('RAM reserve')
 if s['C_free_GiB']<8:r.append('C disk reserve')
 if s.get('CPU_percent') is not None and s['CPU_percent']>85:r.append('CPU pressure')
 if kind=='gpu' and (s.get('GPU_free_GiB') is None or s['GPU_free_GiB']<3):r.append('GPU reserve or unknown telemetry')
 return r

def restrict_current_process():
 # Only this worker: lower scheduling priority and two allowed logical CPUs.
 k=ctypes.windll.kernel32;k.GetCurrentProcess.restype=ctypes.c_void_p;p=k.GetCurrentProcess()
 k.SetPriorityClass.argtypes=[ctypes.c_void_p,ctypes.c_ulong];assert k.SetPriorityClass(p,0x4000)
 proc=ctypes.c_size_t();system=ctypes.c_size_t();k.GetProcessAffinityMask.argtypes=[ctypes.c_void_p,ctypes.POINTER(ctypes.c_size_t),ctypes.POINTER(ctypes.c_size_t)];assert k.GetProcessAffinityMask(p,ctypes.byref(proc),ctypes.byref(system))
 bits=[1<<i for i in range(ctypes.sizeof(ctypes.c_size_t)*8) if proc.value&(1<<i)];mask=sum(bits[:2]);k.SetProcessAffinityMask.argtypes=[ctypes.c_void_p,ctypes.c_size_t];assert k.SetProcessAffinityMask(p,mask)
 for name in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[name]='1'
 return mask

def write(name,obj):
 OUT.mkdir(exist_ok=True);p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.writing');tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8');tmp.replace(p)

def guard_before_import(kind):
 restrict_current_process();snapshot();time.sleep(.15);s=snapshot();reasons=start_reasons(s,kind)
 if reasons:write('worker-deferred.json',{'status':'WAITING_RESOURCES','reasons':reasons,'snapshot':s});raise SystemExit(77)
 return s

def requested_yield():return (OUT/'yield.request').exists() or (OUT/'STOP').exists()

_map_cache=(0,None)
class ProcessEntry(ctypes.Structure):
 _fields_=[('size',ctypes.c_ulong),('usage',ctypes.c_ulong),('pid',ctypes.c_ulong),('heap',ctypes.c_size_t),('module',ctypes.c_ulong),('threads',ctypes.c_ulong),('parent',ctypes.c_ulong),('priority',ctypes.c_long),('flags',ctypes.c_ulong),('exe',ctypes.c_wchar*260)]
def map_processes():
 global _map_cache
 now=time.monotonic()
 if now-_map_cache[0]<3:return _map_cache[1]
 k=ctypes.windll.kernel32;k.CreateToolhelp32Snapshot.argtypes=[ctypes.c_ulong,ctypes.c_ulong];k.CreateToolhelp32Snapshot.restype=ctypes.c_void_p;handle=k.CreateToolhelp32Snapshot(2,0)
 if handle in (None,ctypes.c_void_p(-1).value):return None
 names={'unrealeditor.exe','unrealeditor-cmd.exe','shadercompileworker.exe','unrealpak.exe','blender.exe','ue4editor.exe','sc2_x64.exe','echochasm-win64-shipping.exe'};found=[]
 try:
  entry=ProcessEntry();entry.size=ctypes.sizeof(entry)
  for name in ['Process32FirstW','Process32NextW']:getattr(k,name).argtypes=[ctypes.c_void_p,ctypes.POINTER(ProcessEntry)]
  ok=k.Process32FirstW(handle,ctypes.byref(entry))
  if not ok:return None
  while ok:
   if entry.exe.lower() in names:found.append({'name':entry.exe,'pid':entry.pid})
   ok=k.Process32NextW(handle,ctypes.byref(entry))
 finally:
  k.CloseHandle.argtypes=[ctypes.c_void_p];k.CloseHandle(handle)
 _map_cache=(now,found);return found
