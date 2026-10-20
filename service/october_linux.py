"""Explicit Linux TensorRT construction, reusing qualified CUDA graph operations."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import ctypes,subprocess,sys
from .linux_runtime import LinuxCore,Cuda
from .october_manifest import artifact,digest
from .october_native import InlinePool
from .october_engine import deserialize_engine


class LinuxNativeCore(LinuxCore):
    def __init__(self,root,spec,logs,workers=4):
        if sys.platform!='linux':raise ValueError('Linux native backend selected on another platform')
        env=spec['native_environment']['linux']
        import tensorrt as trt
        if trt.__version__!=env['tensorrt_version']:raise ValueError('TensorRT version does not match serialized engines')
        actual=subprocess.run(['nvidia-smi','--query-gpu=compute_cap','--format=csv,noheader'],capture_output=True,text=True,check=True,timeout=5).stdout.splitlines()[0].strip()
        if actual!=env['compute_capability']:raise ValueError('TensorRT engines require GPU compute capability '+env['compute_capability'])
        self.cuda=Cuda();P=ctypes.c_void_p
        self.logger=trt.Logger(trt.Logger.WARNING);self.runtime=trt.Runtime(self.logger)
        self.engines=[deserialize_engine(self.runtime,artifact(root,p)) for p in env['engines']]
        if not all(self.engines):raise ValueError('Cannot deserialize selected Linux engines')
        normalizer=artifact(root,env['normalizer'])
        if digest(normalizer)!=env['normalizer_SHA256']:raise ValueError('Linux normalizer mismatch')
        self.dll=ctypes.CDLL(str(normalizer));self.normalize=self.dll.normalize_rgb
        self.normalize.argtypes=[ctypes.POINTER(ctypes.c_uint8),ctypes.POINTER(ctypes.c_float),ctypes.c_size_t];self.normalize.restype=None
        self.pool=ThreadPoolExecutor(max_workers=workers) if workers else InlinePool()
        self.din=[P(),P()];self.dout=[P(),P()];self.stream=P();self.graphs={};self.closed=False
        for i,size in enumerate([384,256]):
            self.cuda.call('cudaMalloc',ctypes.byref(self.din[i]),32*3*size*size*4)
            self.cuda.call('cudaMalloc',ctypes.byref(self.dout[i]),32*2048*4)
        self.cuda.call('cudaStreamCreateWithFlags',ctypes.byref(self.stream),1)
