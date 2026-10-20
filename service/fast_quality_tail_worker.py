"""Direct resource telemetry, then the unchanged frozen quality-tail worker."""
from pathlib import Path
import os,sys
def main():
 from .fast_quality_tail import verified_selection
 root=Path(__file__).resolve().parents[1];verified_selection(root)
 if os.name=='nt':
  sys.path.insert(0,str(root/'var/teacher-projection-20261011'));import resources
 else:from . import linux_resources as resources
 from .nvml_resources import install
 install(resources)
 from .quality_tail_worker import main as run
 run()
if __name__=='__main__':main()
