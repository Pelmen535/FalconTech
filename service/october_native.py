"""Explicit native dependency construction; frozen preprocessing/worker math reused."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import ctypes
import importlib
from multiprocessing import shared_memory
import os
import struct
import subprocess
import sys
import numpy as np

from .october_manifest import artifact, digest


class InlinePool:
    def map(self,fn,rows):
        return map(fn,rows)
    def shutdown(self,wait=True):
        pass


class NativeCore:
    def __init__(self, root, spec, logs, workers=4):
        if sys.platform != 'win32':
            raise ValueError('Windows native core selected on another platform')
        self.root = Path(root)
        envspec = spec['native_environment']['windows']
        normalizer = Path(envspec['normalizer']).resolve()
        if digest(normalizer) != envspec['normalizer_SHA256']:
            raise ValueError('Native normalizer identity mismatch')
        self.dll = ctypes.CDLL(str(normalizer))
        self.normalize = self.dll.normalize_rgb
        self.normalize.argtypes = [ctypes.POINTER(ctypes.c_uint8), ctypes.POINTER(ctypes.c_float), ctypes.c_size_t]
        self.normalize.restype = None
        self.pool = ThreadPoolExecutor(max_workers=workers) if workers else InlinePool()
        n0 = 32 * 3 * 384 * 384
        self.memory = shared_memory.SharedMemory(create=True, size=(n0+32*3*256*256)*4)
        self.arrays = [np.ndarray((32,3,384,384),np.float32,buffer=self.memory.buf),
                       np.ndarray((32,3,256,256),np.float32,buffer=self.memory.buf,offset=n0*4)]
        self.stderr = (Path(logs)/'native-stderr.log').open('xb')
        env = {**os.environ, 'VREID_TRT_SITE':envspec['trt_site'], 'VREID_CUDA_DLL_DIR':envspec['cuda_dll_dir']}
        self.process = subprocess.Popen([str(artifact(root,envspec['python'])),'-X','utf8','-B',
            str(artifact(root,envspec['entry'])),self.memory.name],cwd=root,env=env,
            stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.stderr,
            creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        if self.exact(4) != b'RDY!':
            raise RuntimeError('Unexpected native worker handshake')

    def exact(self, count):
        data = bytearray()
        while len(data)<count:
            part = self.process.stdout.read(count-len(data))
            if not part:
                raise RuntimeError('Native encoder exited; inspect native-stderr.log')
            data.extend(part)
        return bytes(data)

    def embed(self, batch):
        x, cnn = batch
        count = len(x)
        if not 1<=count<=32:
            raise ValueError('Native batch must be in1..32')
        self.arrays[0][:count] = x
        self.arrays[1][:count] = cnn.numpy()
        self.process.stdin.write(struct.pack('<I',count));self.process.stdin.flush()
        blob = np.frombuffer(self.exact(count*4096*4),np.float32).reshape(count,4096).copy()
        return {'emb':blob[:,:2048], 'specialist':blob[:,2048:].astype(np.float16)}

    def close(self):
        self.pool.shutdown(wait=True)
        if self.process.poll() is None:
            self.process.stdin.write(struct.pack('<I',0));self.process.stdin.flush()
            self.process.wait(timeout=30)
        self.stderr.close()
        self.process.stdin.close();self.process.stdout.close()
        self.arrays.clear();self.memory.close();self.memory.unlink()
        if self.process.returncode!=0:
            raise RuntimeError('Native encoder failed while closing')


class NativeFeatures:
    def __init__(self,root,spec,logs,batch_size=16,live_check=None,workers=4):
        import torch
        torch.set_num_threads(2)
        core_class=NativeCore
        if sys.platform=='linux':
            from .october_linux import LinuxNativeCore
            core_class=LinuxNativeCore
        self.torch=torch;self.workers=workers;self.core=core_class(root,spec,logs,workers)
        self.batch_size=batch_size;self.live_check=live_check
        self.producer=ThreadPoolExecutor(max_workers=1)

    def set_workers(self,requested):
        workers=min(int(requested),4)
        if workers<0:
            raise ValueError('Preprocessing workers must be nonnegative')
        if workers!=self.workers:
            self.core.pool.shutdown(wait=True)
            self.core.pool=ThreadPoolExecutor(max_workers=workers) if workers else InlinePool()
            self.workers=workers

    def prepare_dual(self,row):
        from PIL import Image
        from .october_images import crop_image
        crop=crop_image(row)
        rgb=np.ascontiguousarray(np.asarray(crop.resize((384,384),Image.Resampling.BICUBIC),dtype=np.uint8))
        output=np.empty((3,384,384),np.float32)
        self.core.normalize(rgb.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
                            output.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),384*384)
        cnn=np.array(crop.resize((256,256),Image.Resampling.BICUBIC),dtype=np.float32,copy=True)
        return output,self.torch.from_numpy(cnn.transpose(2,0,1).copy())

    def prepare(self,rows):
        pairs=list(self.core.pool.map(self.prepare_dual,rows))
        return np.stack([p[0] for p in pairs]),self.torch.stack([p[1] for p in pairs])

    def embed(self,batch):
        if self.live_check is not None:
            self.live_check()
        return self.core.embed(batch)

    def extract(self,rows):
        import time
        if not rows:
            raise ValueError('Cannot encode an empty observation set')
        start=time.perf_counter();size=self.batch_size
        pending=self.producer.submit(self.prepare,rows[:size]);outputs=[]
        for offset in range(0,len(rows),size):
            prepared=pending.result()
            if offset+size<len(rows):
                pending=self.producer.submit(self.prepare,rows[offset+size:offset+2*size])
            outputs.append(self.embed(prepared))
        elapsed=time.perf_counter()-start
        features={k:np.concatenate([part[k] for part in outputs]) for k in outputs[0]}
        if not all(np.isfinite(v).all() for v in features.values()):
            raise ValueError('Nonfinite native encoder output')
        return {'keys':np.array([r['key'] for r in rows]),**features}, {
            'count':len(rows),'seconds':elapsed,'FPS':len(rows)/elapsed,
            'scope':'JPEG/crop/padded352/preprocess/native DINO+CNN/transfer; includes live resource guard',
            'reused_core':True}

    def close(self):
        self.producer.shutdown(wait=True);self.core.close()


def create_native(root,spec,logs,batch_size=16,live_check=None):
    if batch_size not in (1,8,16,32):
        raise ValueError('Unsupported native batch size')
    return NativeFeatures(root,spec,logs,batch_size,live_check)
