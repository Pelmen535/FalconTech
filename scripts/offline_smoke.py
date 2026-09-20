"""CPU CLI smoke with Python socket connection/name-resolution attempts denied.

This checks this process; it is not a Docker --network none or GPU test.
"""
import os
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',VREID_COMPILE='0')
def guard(event,args):
    if event in ('socket.connect','socket.getaddrinfo'):
        raise RuntimeError('Network access attempted during offline smoke: '+event)
sys.addaudithook(guard)
from vreid.predict import main
import json
def write_json(path,data):path.write_text(json.dumps(data,indent=2),'utf-8')
if __name__=='__main__':
    main();out=sys.argv[sys.argv.index('--out')+1];result=json.loads((Path(out)/'run_info.json').read_text('utf-8'))
    write_json(Path(out)/'offline_smoke.json',dict(passed=True,model_sha256=result['model_sha256'],
        recipe_sha256=result['recipe_sha256'],scope='Python socket guard, native CPU, no container or A5000 verification'))
