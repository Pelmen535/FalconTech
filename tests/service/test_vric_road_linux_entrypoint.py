"""Public v2 entrypoint routing, output preservation, and native stdout separation."""
from pathlib import Path
import os,runpy,sys
import pytest
ROOT=Path(__file__).resolve().parents[2]

@pytest.mark.parametrize('mode',['predict','serve'])
def test_delivery_dispatch(monkeypatch,mode):
 captured=[]
 class Dispatched(Exception):pass
 if mode=='predict':
  from service import october_cli
  def predict(args):captured.append(args);raise Dispatched
  monkeypatch.setattr(october_cli,'main',predict)
  arguments=['predict','--query','/input/q.csv']
 else:
  def execute(executable,args):captured.append(args);raise Dispatched
  monkeypatch.setattr(os,'execv',execute)
  arguments=['serve','--threads','2']
 monkeypatch.setattr(sys,'argv',['entrypoint',*arguments])
 with pytest.raises(Dispatched):runpy.run_path(str(ROOT/'scripts/october_entrypoint.py'),run_name='__main__')
 assert captured[0][-2:]==arguments[-2:]
 assert '--runtime-manifest' in captured[0]
 assert str(ROOT/'release_current/runtime-manifest.json') in captured[0]
 if mode=='serve':
  assert str(ROOT/'scripts/run_service.py') in captured[0]
  assert captured[0][captured[0].index('--backend')+1]=='october'

def test_linux_cli_preserves_existing_output(monkeypatch,tmp_path):
 output=tmp_path/'existing';output.mkdir();sentinel=output/'user.txt';sentinel.write_text('keep')
 module=runpy.run_path(str(ROOT/'scripts/predict_october.py'),run_name='import_for_test')
 with pytest.raises(SystemExit) as error:module['main'](['--out',str(output)])
 assert error.value.code==2 and sentinel.read_text()=='keep'

def test_native_stdout_cannot_corrupt_rpc():
    import subprocess,json
    source=r'''from service.vric_road_linux_worker import isolate_native_stdout
import os
stream=isolate_native_stdout()
os.write(1,b'native diagnostic\n')
stream.write('{"ok": true}\n')
stream.flush()
'''
    child=subprocess.run([sys.executable,"-X","utf8","-c",source],cwd=ROOT,capture_output=True,text=True,check=True)
    assert json.loads(child.stdout)=={"ok":True}
    assert "native diagnostic" in child.stderr
