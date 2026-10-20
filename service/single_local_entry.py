from pathlib import Path
import sys,runpy
ROOT=Path(__file__).resolve().parents[1];J=ROOT/'var/cnn-trt-20261010';N=ROOT/'var/nive-night-transfer-20261009';T=Path('C:/Users/Sheka/AppData/Local/Temp/vehicle-reid-comparison-20261005-01');sys.path[:0]=[str(N/'packages'),str(T/'tensorrt-install-20261006-01/venv/Lib/site-packages')];runpy.run_path(str(ROOT/'service/single_local_server_v2.py'),run_name='__main__')
