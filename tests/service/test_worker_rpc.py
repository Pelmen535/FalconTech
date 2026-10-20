"""Actual child processes: bounded pipe operations, failure and restart isolation."""
import json
import sys
import time
import threading

import pytest

from service.worker_rpc import WorkerRPC, WorkerProtocolError, WorkerTimeoutError

WORKER = r"""
import json,sys,time
for line in sys.stdin:
 m=json.loads(line)
 if m['command']=='hang':time.sleep(30)
 if m['command']=='bad_id':
  print(json.dumps({'ok':True,'rpc_id':'wrong','result':0}),flush=True);continue
 if m['command']=='fail':
  print(json.dumps({'ok':False,'rpc_id':m['rpc_id'],'error_kind':'internal','error':'private value'}),flush=True);continue
 print(json.dumps({'ok':True,'rpc_id':m['rpc_id'],'result':m.get('value',True)}),flush=True)
 if m['command']=='close':break
"""

def rpc(tmp_path, code=WORKER):
 return WorkerRPC([sys.executable,'-u','-c',code],cwd=tmp_path,stderr_path=tmp_path/'worker.log',timeout=.3,startup_timeout=5)


def test_response_and_graceful_close(tmp_path):
 client=rpc(tmp_path)
 assert client.request({'command':'echo','value':42})==42
 process=client.process
 client.close();assert process.poll() is not None
 with pytest.raises(RuntimeError,match='closed'):client.request({'command':'echo'})


def test_live_read_hang_has_deadline_and_next_request_recovers(tmp_path):
 client=rpc(tmp_path)
 try:
  assert client.request({'command':'echo','value':'ready'})=='ready'
  old=client.process;generation=client.generation;start=time.monotonic()
  with pytest.raises(WorkerTimeoutError):client.request({'command':'hang'})
  assert time.monotonic()-start<5 and old.poll() is not None
  assert client.generation>generation and client.request({'command':'echo','value':7})==7
 finally:client.close()


def test_live_worker_not_reading_stdin_bounds_large_pipe_write(tmp_path):
 client=rpc(tmp_path,'import time;time.sleep(30)')
 try:
  start=time.monotonic()
  with pytest.raises(WorkerTimeoutError):client.request({'command':'echo','payload':'x'*2_000_000},timeout=.3)
  assert time.monotonic()-start<5 and not client.running
 finally:client.close()


def test_protocol_mismatch_resets_process_without_accepting_wrong_reply(tmp_path):
 client=rpc(tmp_path)
 try:
  with pytest.raises(WorkerProtocolError,match='identity'):client.request({'command':'bad_id'})
  assert not client.running
  assert client.request({'command':'echo','value':3})==3
 finally:client.close()


def test_internal_worker_failure_restarts_and_does_not_leak_payload(tmp_path):
 client=rpc(tmp_path)
 try:
  with pytest.raises(RuntimeError) as failure:client.request({'command':'fail'})
  assert 'private value' not in str(failure.value)
  assert not client.running
  assert client.request({'command':'echo','value':8})==8
 finally:client.close()


def test_foreign_process_and_no_rpc_thread_leak(tmp_path):
 import subprocess
 foreign=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)'])
 client=rpc(tmp_path)
 try:
  client.request({'command':'echo'})
  with pytest.raises(WorkerTimeoutError):client.request({'command':'hang'})
  assert foreign.poll() is None
  client.close()
  assert not any(t.name in ('falcon-worker-read','falcon-worker-write') for t in threading.enumerate())
 finally:
  client.close();foreign.terminate();foreign.wait(timeout=5)
