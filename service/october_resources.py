"""Measured resource guards; reuse project telemetry without changing its globals."""
from pathlib import Path
import ctypes
import importlib.util
import os
import sys
import time
import math
import tempfile


class ResourceGate:
    def __init__(self, root, limits, device, *, write_paths=None):
        self.limits, self.device = limits, device
        from .disk_reserve import WriteDiskReserve
        self.write_disks=WriteDiskReserve({'temporary':tempfile.gettempdir(),**(write_paths or {})},limits['start_C_disk_free_GiB'])
        if sys.platform=='win32' and device=='cuda':
            # Configure the later-loaded Intel/OpenMP runtime before torch
            # import. Keep the original number of threads and CPU affinity.
            os.environ['OMP_WAIT_POLICY']='PASSIVE'
            os.environ['KMP_BLOCKTIME']='0'
        self.max_age=float(limits.get('telemetry_max_age_seconds',.25))
        if not math.isfinite(self.max_age) or not 0<=self.max_age<=.5:
            raise ValueError('Resource telemetry age must be within0..0.5seconds')
        self._last_snapshot=None;self._last_measured=-math.inf
        if sys.platform == 'win32':
            path = Path(root)/'var/teacher-projection-20261011/resources.py'
            spec = importlib.util.spec_from_file_location('_falcon_october_resources',path)
            self.provider = importlib.util.module_from_spec(spec);spec.loader.exec_module(self.provider)
        else:
            from . import linux_resources
            self.provider = linux_resources
        from .nvml_resources import install
        self.telemetry_proxy=install(self.provider)
        self.provider.restrict_current_process()
        for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
            os.environ[key] = '2'

    def _measure(self):
        value = self.provider.snapshot()
        if sys.platform == 'win32':
            memory = self.provider.Mem();memory.length = ctypes.sizeof(memory)
            if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory)):
                raise RuntimeError('Windows memory telemetry unavailable')
            value['commit_free_GiB'] = memory.commit_available/2**30
        else:
            value['memory_headroom_GiB'] = self.provider.memory_headroom_gib()
        value['write_filesystems']=self.write_disks.snapshot()
        return value

    def add_write_paths(self,**paths):
        self.write_disks.add(paths)
        self._last_snapshot=None;self._last_measured=-math.inf

    def snapshot(self):
        # Fresh startup probes remain synchronous. Serial live checks reuse only
        # a bounded-age measurement, avoiding several nvidia-smi processes for
        # the same query while retaining all memory/disk/VRAM safety floors.
        value=self._measure()
        self._last_snapshot=value;self._last_measured=time.monotonic()
        return value

    def start_reasons(self):
        self.snapshot();time.sleep(.2);value = self.snapshot()
        if self.device == 'cuda':
            reasons = [r for r in self.provider.start_reasons(value,'gpu')
                       if not r.startswith('C_free_GiB below') and r!='GPU free below9GiB or unknown']
        else:
            reasons = []
            if value.get('CPU_percent') is None or value['CPU_percent']>70:
                reasons.append('CPU busy or telemetry unavailable')
        def below(key, minimum):
            return value.get(key) is None or value[key]<minimum
        if below('C_free_GiB',self.limits['start_C_disk_free_GiB']):
            reasons.append('Insufficient measured runtime disk reserve')
        reasons.extend(value.get('write_filesystems',{}).get('reasons',()))
        if below('RAM_free_GiB',self.limits['start_RAM_free_GiB']):
            reasons.append('Insufficient measured RAM reserve')
        key = 'commit_free_GiB' if sys.platform=='win32' else 'memory_headroom_GiB'
        if below(key,self.limits['start_commit_free_GiB']):
            reasons.append('Insufficient measured memory headroom')
        if self.device=='cuda' and below('GPU_free_GiB',self.limits['start_GPU_free_GiB']):
            reasons.append('Insufficient measured VRAM reserve')
        return reasons,value

    def live_reasons(self):
        value=self._last_snapshot
        if value is None or time.monotonic()-self._last_measured>=self.max_age:
            value=self.snapshot()
        reasons=[]
        reasons.extend(value.get('write_filesystems',{}).get('reasons',()))
        if value['RAM_free_GiB']<1.5 or value['C_free_GiB']<self.limits['start_C_disk_free_GiB']:
            reasons.append('Insufficient live RAM/disk reserve')
        if sys.platform=='win32' and value['commit_free_GiB']<1.5:
            reasons.append('Insufficient live commit reserve')
        if self.device=='cuda' and (value.get('GPU_free_GiB') is None or value['GPU_free_GiB']<3):
            reasons.append('Insufficient live VRAM reserve')
        return reasons,value
