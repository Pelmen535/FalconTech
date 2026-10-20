"""Real HTTP download integrity and preservation checks for v2 release assets."""
from pathlib import Path
import hashlib,importlib.util,threading
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from contextlib import contextmanager
import pytest
spec=importlib.util.spec_from_file_location('fetch_v2',Path(__file__).resolve().parents[1]/'scripts/fetch_weights.py')
fetcher=importlib.util.module_from_spec(spec);spec.loader.exec_module(fetcher)

@contextmanager
def served(payload):
 class Handler(BaseHTTPRequestHandler):
  def do_GET(self):
   self.send_response(200);self.end_headers();self.wfile.write(payload)
  def log_message(self,*args):pass
 server=ThreadingHTTPServer(('127.0.0.1',0),Handler);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
 try:yield 'http://127.0.0.1:'+str(server.server_port)+'/artifact'
 finally:server.shutdown();thread.join();server.server_close()

def row(data,url):return {'path':'artifacts/model.bin','bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),'url':url}

def test_download_verified_then_idempotent(tmp_path):
 data=b'verified model bytes'
 with served(data) as url:
  record=row(data,url);fetcher.fetch(record,tmp_path);assert (tmp_path/record['path']).read_bytes()==data
  fetcher.fetch(record,tmp_path)
 assert not list((tmp_path/'artifacts').glob('*.download'))

@pytest.mark.parametrize('bad',[b'corrupted model byt',b'x',b'oversized model bytes'*10])
def test_bad_download_never_publishes(tmp_path,bad):
 data=b'verified model bytes'
 with served(bad) as url:
  with pytest.raises(ValueError):fetcher.fetch(row(data,url),tmp_path)
 assert not (tmp_path/'artifacts/model.bin').exists()
 assert not list((tmp_path/'artifacts').glob('*.download'))

def test_existing_user_file_is_preserved(tmp_path):
 target=tmp_path/'artifacts/model.bin';target.parent.mkdir();target.write_bytes(b'user work')
 with pytest.raises(ValueError):fetcher.fetch(row(b'verified model bytes','https://unreachable.invalid/'),tmp_path)
 assert target.read_bytes()==b'user work'

def test_release_download_and_runtime_manifests_match():
 records=fetcher.records(Path(__file__).resolve().parents[1]);assert len(records)==16
