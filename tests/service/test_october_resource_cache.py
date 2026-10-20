"""Live checks cannot reuse a stale safe resource sample; startup always probes."""
from types import SimpleNamespace
from service.october_resources import ResourceGate
import service.october_resources as module


def gate(monkeypatch):
    clock=[0.];measurements=[];value={'RAM_free_GiB':10.,'C_free_GiB':10.,'commit_free_GiB':10.,'memory_headroom_GiB':10.,'GPU_free_GiB':10.,'CPU_percent':10.}
    obj=ResourceGate.__new__(ResourceGate);obj.max_age=.25;obj._last_snapshot=None;obj._last_measured=-float('inf');obj.device='cuda'
    obj.limits={'start_C_disk_free_GiB':5,'start_RAM_free_GiB':4.35,'start_commit_free_GiB':7,'start_GPU_free_GiB':7}
    obj.provider=SimpleNamespace(start_reasons=lambda v,k:[])
    def measure():measurements.append(clock[0]);return dict(value)
    obj._measure=measure
    monkeypatch.setattr(module.time,'monotonic',lambda:clock[0])
    monkeypatch.setattr(module.time,'sleep',lambda duration:clock.__setitem__(0,clock[0]+duration))
    return obj,clock,measurements,value


def test_live_reuse_expires_before_low_vram_is_accepted(monkeypatch):
    obj,clock,calls,value=gate(monkeypatch)
    assert obj.live_reasons()[0]==[] and len(calls)==1
    clock[0]=.1;value['GPU_free_GiB']=2.
    assert obj.live_reasons()[0]==[] and len(calls)==1
    clock[0]=.25
    assert any('VRAM' in r for r in obj.live_reasons()[0]) and len(calls)==2


def test_startup_takes_two_new_samples_even_with_fresh_live_cache(monkeypatch):
    obj,clock,calls,value=gate(monkeypatch)
    obj.live_reasons();assert len(calls)==1
    assert obj.start_reasons()[0]==[] and len(calls)==3
    assert obj.live_reasons()[0]==[] and len(calls)==3
