from pathlib import Path
import sys
J=Path(__file__).resolve().parent;R=J.parents[1];sys.path.insert(0,str(R/'var/three-directions-20261009'))
from env import *
M=Path(__file__).resolve().parent;J=R/'var/cnn-trt-20261010';E=J/'export';sys.path.insert(0,str(J))
def put(n,x):
 p=M/(n+'.writing');p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8');p.replace(M/n)
