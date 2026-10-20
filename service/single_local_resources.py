"""Disk reserve for bounded inference outputs; preserve all other resource gates."""
import math

def install(resources,minimum_gib=5.0):
 if not math.isfinite(minimum_gib) or minimum_gib<5:raise ValueError('Inference disk reserve must be at least5GiB')
 if getattr(resources.start_reasons,'_single_disk_reserve',None) is not None:raise ValueError('Disk reserve already installed')
 original_start=resources.start_reasons;original_yield=resources.yield_reasons
 def low(snapshot):
  value=snapshot.get('C_free_GiB')
  return value is None or not math.isfinite(value) or value<minimum_gib
 def start(snapshot,kind):
  reasons=[r for r in original_start(snapshot,kind) if not r.startswith('C_free_GiB below ')]
  if low(snapshot):reasons.append(f'C_free_GiB below {minimum_gib:g} or unknown')
  return reasons
 def yielding(snapshot,kind='gpu'):
  reasons=[r for r in original_yield(snapshot,kind) if r!='C disk reserve']
  if low(snapshot):reasons.append('C disk reserve')
  return reasons
 start._single_disk_reserve=minimum_gib;resources.start_reasons=start;resources.yield_reasons=yielding
 def restore():resources.start_reasons=original_start;resources.yield_reasons=original_yield
 return restore

def install_windows(root):
 import sys
 sys.path.insert(0,str(root/'var/teacher-projection-20261011'))
 import resources
 return install(resources)
