"""Relocate only Windows TensorRT dependency paths; model operations stay intact."""
from pathlib import Path
import os
import sys

ROOT=Path(__file__).resolve().parents[1]
site=Path(os.environ['VREID_TRT_SITE']).resolve()
sys.path[:0]=[str(ROOT/'var/nive-night-transfer-20261009/packages'),str(site)]
server=ROOT/'var/cnn-trt-20261010/joint_server.py'
source=server.read_text(encoding='utf-8-sig')
old="T/'tensorrt-install-20261006-01/venv/Lib/site-packages/tensorrt_libs',Path('C:/Users/Sheka/AppData/Roaming/Python/Python314/site-packages/torch/lib')"
new="Path(os.environ['VREID_TRT_SITE'])/'tensorrt_libs',Path(os.environ['VREID_CUDA_DLL_DIR'])"
if source.count(old)!=1:
    raise RuntimeError('Unexpected GPU worker source; cannot relocate dependencies safely')
source=source.replace(old,new)
exec(compile(source,str(server),'exec'),dict(__name__='__main__',__file__=str(server)))
