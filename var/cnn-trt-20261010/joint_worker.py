from pathlib import Path
import sys,runpy
J=Path(__file__).resolve().parent;N=J.parent/'nive-night-transfer-20261009';T=Path('C:/Users/Sheka/AppData/Local/Temp/vehicle-reid-comparison-20261005-01');sys.path[:0]=[str(N/'packages'),str(T/'tensorrt-install-20261006-01/venv/Lib/site-packages')];runpy.run_path(str(J/'joint_server.py'),run_name='__main__')
