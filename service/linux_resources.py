"""Linux resource measurements; no Windows commit counter is fabricated."""
from pathlib import Path
import os,shutil,subprocess,time

GiB=1024**3
_previous_cpu=None

def restrict_current_process():
    for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
        os.environ[name]='2'
    # This VM is dedicated to inference; keep its available CPU affinity.

def memory_headroom_gib():
    info={line.split(':')[0]:int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines()}
    available=info['MemAvailable']
    limit=Path('/sys/fs/cgroup/memory.max');used=Path('/sys/fs/cgroup/memory.current')
    if limit.exists() and used.exists() and limit.read_text().strip()!='max':
        available=min(available,max(0,int(limit.read_text())-int(used.read_text())))
    return available/GiB

def snapshot():
    global _previous_cpu
    values=list(map(int,Path('/proc/stat').read_text().splitlines()[0].split()[1:9]))
    now=(sum(values),values[3]+values[4]);cpu=None
    if _previous_cpu and now[0]>_previous_cpu[0]:
        cpu=100*(1-(now[1]-_previous_cpu[1])/(now[0]-_previous_cpu[0]))
    _previous_cpu=now
    result={'time':time.time(),'RAM_free_GiB':memory_headroom_gib(),
            'C_free_GiB':shutil.disk_usage(Path(__file__).resolve().parents[1]).free/GiB,
            'CPU_percent':cpu,'GPU_free_GiB':None,'GPU_util_percent':None,
            'memory_measurement':'Linux MemAvailable capped by cgroup headroom; not Windows commit'}
    try:
        p=subprocess.run(['nvidia-smi','--query-gpu=memory.total,memory.used,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=4,check=True)
        total,used,util=map(float,p.stdout.strip().splitlines()[0].split(','))
        result.update(GPU_free_GiB=(total-used)/1024,GPU_util_percent=util)
    except (OSError,ValueError,subprocess.SubprocessError):
        pass
    return result

def start_reasons(s,kind):
    reasons=[]
    for name,minimum in [('RAM_free_GiB',1.5),('C_free_GiB',10)]:
        if s.get(name) is None or s[name]<minimum:reasons.append(f'{name} below {minimum}')
    if s.get('CPU_percent') is None or s['CPU_percent']>85:reasons.append('CPU busy or unknown')
    if s.get('GPU_free_GiB') is None or s['GPU_free_GiB']<9:reasons.append('GPU free below9GiB or unknown')
    if s.get('GPU_util_percent') is None or s['GPU_util_percent']>40:reasons.append('GPU busy or unknown')
    return reasons
