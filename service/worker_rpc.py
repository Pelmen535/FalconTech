"""Deadline-bound JSON RPC; only this object's subprocess tree may be stopped."""
from __future__ import annotations
import json
import os
from pathlib import Path
import queue
import subprocess
import threading
import time
from uuid import uuid4

from .robust import ResourceUnavailableError


class WorkerTimeoutError(RuntimeError):
    pass


class WorkerProtocolError(RuntimeError):
    pass


class WorkerRPC:
    def __init__(self, argv, *, cwd, stderr_path, env=None, timeout=120., startup_timeout=180.):
        if timeout <= 0 or startup_timeout <= 0:
            raise ValueError('Worker deadlines must be positive')
        self.argv = [str(x) for x in argv]
        self.cwd, self.stderr_path = Path(cwd), Path(stderr_path)
        self.env = dict(os.environ if env is None else env)
        self.timeout, self.startup_timeout = float(timeout), float(startup_timeout)
        self.process = None
        self._job = None
        self.generation = 0
        self.closed = False
        self._lock = threading.RLock()
        self._reader = self._writer = None
        self._stderr = None
        self._stop_event = threading.Event()
        self._responses = queue.Queue(maxsize=2)

    @property
    def running(self):
        return self.process is not None and self.process.poll() is None

    def _publish(self, value):
        while not self._stop_event.is_set():
            try:
                self._responses.put(value, timeout=.1)
                return
            except queue.Full:
                continue

    def _read(self, process, job):
        try:
            while not self._stop_event.is_set():
                line = process.stdout.readline(16 * 1024 * 1024 + 1)
                if not line:
                    self._publish(('error', WorkerProtocolError('Worker closed its response stream')))
                    return
                if len(line) > 16 * 1024 * 1024 or not line.endswith('\n'):
                    self._publish(('error', WorkerProtocolError('Invalid or excessive worker response')))
                    return
                self._publish(('line', line))
        except (OSError, ValueError) as exc:
            self._publish(('error', WorkerProtocolError(str(exc))))
        finally:
            # Captured job belongs to this reader's worker generation only.
            # EOF from a dead parent cannot leave its native GPU child orphaned.
            if job is not None:job.close()

    def _start(self):
        self.stderr_path.parent.mkdir(parents=True, exist_ok=True)
        self._responses = queue.Queue(maxsize=2)
        self._stop_event = threading.Event()
        self._stderr = self.stderr_path.open('ab')
        try:
            self.process = subprocess.Popen(self.argv, cwd=self.cwd, env=self.env,
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._stderr,
                text=True, encoding='utf-8', bufsize=1,
                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            from .child_lifetime import worker_job
            self._job=worker_job(self.process)
        except BaseException:
            if self.process is not None:
                if self.process.poll() is None:self.process.kill()
                self.process.wait(timeout=3)
                for stream in (self.process.stdin,self.process.stdout):
                    if stream is not None:stream.close()
                self.process=None
            if self._job is not None:self._job.close();self._job=None
            self._stderr.close()
            self._stderr = None
            raise
        self.generation += 1
        self._reader = threading.Thread(target=self._read, args=(self.process,self._job), daemon=True,
                                        name='falcon-worker-read')
        self._reader.start()

    def _stop(self):
        process = self.process
        self._stop_event.set()
        if self._job is not None:
            self._job.close();self._job=None
        if process is not None:
            # The subprocess handle identifies the process we created. Do not
            # inspect/stop unrelated GPU applications or another service.
            if process.poll() is None:
                import psutil
                try:
                    parent = psutil.Process(process.pid)
                    children = parent.children(recursive=True)
                except psutil.NoSuchProcess:
                    children = []
                for child in reversed(children):
                    try:
                        child.kill()
                    except psutil.NoSuchProcess:
                        pass
                process.kill()
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    pass
            for thread in (self._writer, self._reader):
                if thread is not None and thread is not threading.current_thread():
                    thread.join(timeout=1)
            for stream in (process.stdin, process.stdout):
                if stream is not None:
                    try:
                        stream.close()
                    except OSError:
                        # A dead child can leave a broken Windows pipe whose
                        # flush fails during close. Preserve the RPC failure
                        # and still clear process/generation state.
                        pass
        if self._stderr is not None:
            self._stderr.close()
        self.process = self._stderr = self._reader = self._writer = None
        self.generation += 1

    def _write(self, process, line):
        try:
            process.stdin.write(line)
            process.stdin.flush()
        except (OSError, ValueError) as exc:
            self._publish(('error', WorkerProtocolError(str(exc))))

    def request(self, message, *, timeout=None):
        with self._lock:
            if self.closed:
                raise RuntimeError('Worker RPC is closed')
            first = not self.running
            if self.process is not None and not self.running:
                self._stop()
            if not self.running:
                self._start()
            deadline = float(timeout if timeout is not None else
                             self.startup_timeout if first else self.timeout)
            if deadline <= 0:
                raise ValueError('Worker deadline must be positive')
            request_id = uuid4().hex
            line = json.dumps({**message, 'rpc_id': request_id}, allow_nan=False) + '\n'
            # Pipe writes can also block when a live worker stops reading.
            # Both write and read are covered by the same wall-clock deadline.
            self._writer = threading.Thread(target=self._write, args=(self.process, line),
                                           daemon=True, name='falcon-worker-write')
            self._writer.start()
            try:
                kind, payload = self._responses.get(timeout=deadline)
                if kind == 'error':
                    raise payload
                try:
                    payload = json.loads(payload)
                except (ValueError, TypeError) as exc:
                    raise WorkerProtocolError('Malformed worker JSON') from exc
                if not isinstance(payload, dict) or payload.get('rpc_id') != request_id or not isinstance(payload.get('ok'), bool):
                    raise WorkerProtocolError('Worker response identity mismatch')
                if not payload['ok']:
                    if payload.get('resource_wait'):
                        raise ResourceUnavailableError(payload.get('error', 'Insufficient worker resources'))
                    if payload.get('error_kind') == 'input':
                        raise ValueError(payload.get('error', 'Invalid worker input'))
                    raise RuntimeError('Worker operation failed; inspect worker log')
                self._writer.join(timeout=1)
                return payload.get('result')
            except queue.Empty as exc:
                self._stop()
                raise WorkerTimeoutError(f'Worker exceeded {deadline:g}s deadline; runtime reset') from exc
            except BaseException:
                self._stop()
                raise

    def close(self):
        with self._lock:
            if self.closed:
                return
            if self.running:
                try:
                    self.request({'command': 'close'}, timeout=5.)
                except Exception:
                    pass
            self._stop()
            self.closed = True
