"""Linux/A5000 execution of unchanged ONNX; frozen image/ranking math is reused."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import ctypes,ctypes.util,hashlib,json,os,sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
ARTIFACTS=ROOT/'var/linux-a5000-20261017/runtime'

def digest(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def verified_manifest(root=ROOT):
    root=Path(root);path=root/'var/linux-a5000-20261017/runtime-manifest.json'
    m=json.loads(path.read_text(encoding='utf-8'))
    if m['source_manifest_SHA']!=digest(root/'var/cached-feature-post-20261010/candidate-manifest.json'):
        raise ValueError('Linux runtime source manifest changed')
    if m['selection_SHA']!=digest(root/'var/general-refusal-final-20261017/selection.json'):
        raise ValueError('Linux runtime refusal selection changed')
    for item in m['files']:
        path=(root/item['path']).resolve()
        if not path.is_relative_to(root.resolve()) or digest(path)!=item['SHA256']:
            raise ValueError('Linux artifact checksum mismatch: '+item['path'])
    return m

class Cuda:
    def __init__(self):
        candidates=['libcudart.so.12',ctypes.util.find_library('cudart')]
        candidates += [str(p) for base in sys.path if base for p in Path(base).glob('nvidia/cuda_runtime/lib/libcudart.so.12')]
        errors=[]
        for name in candidates:
            if not name:continue
            try:self.lib=ctypes.CDLL(name);break
            except OSError as e:errors.append(str(e))
        else:raise RuntimeError('CUDA runtime library missing: '+'; '.join(errors))
        P=ctypes.c_void_p;I=ctypes.c_int;S=ctypes.c_size_t
        signatures={'cudaMalloc':[ctypes.POINTER(P),S],'cudaFree':[P],
          'cudaMemcpyAsync':[P,P,S,I,P],'cudaStreamCreateWithFlags':[ctypes.POINTER(P),ctypes.c_uint],
          'cudaStreamSynchronize':[P],'cudaStreamBeginCapture':[P,I],
          'cudaStreamEndCapture':[P,ctypes.POINTER(P)],
          'cudaGraphInstantiateWithFlags':[ctypes.POINTER(P),P,ctypes.c_ulonglong],
          'cudaGraphLaunch':[P,P],'cudaGraphDestroy':[P],'cudaGraphExecDestroy':[P],
          'cudaStreamDestroy':[P],'cudaDeviceSynchronize':[]}
        for name,args in signatures.items():
            f=getattr(self.lib,name);f.argtypes=args;f.restype=I
    def call(self,name,*args):
        code=getattr(self.lib,name)(*args)
        if code:raise RuntimeError(f'{name}: CUDA error {code}')
    def synchronize(self):self.call('cudaDeviceSynchronize')

class LinuxCore:
    def __init__(self,logs=None,verify=True):
        if sys.platform!='linux':raise RuntimeError('Linux TensorRT backend requires Linux')
        if verify:verified_manifest()
        import tensorrt as trt
        if trt.__version__!='10.9.0.34':raise RuntimeError('Expected TensorRT10.9.0.34')
        self.cuda=Cuda();P=ctypes.c_void_p
        self.logger=trt.Logger(trt.Logger.WARNING);self.runtime=trt.Runtime(self.logger)
        self.engines=[self.runtime.deserialize_cuda_engine((ARTIFACTS/(n+'.engine')).read_bytes()) for n in ['dino','cnn']]
        if not all(self.engines):raise RuntimeError('Cannot deserialize Linux/A5000 engines')
        self.dll=ctypes.CDLL(str(ARTIFACTS/'normalize.so'))
        self.normalize=self.dll.normalize_rgb
        self.normalize.argtypes=[ctypes.POINTER(ctypes.c_uint8),ctypes.POINTER(ctypes.c_float),ctypes.c_size_t]
        self.normalize.restype=None
        self.pool=ThreadPoolExecutor(max_workers=4)
        self.din=[P(),P()];self.dout=[P(),P()];self.stream=P();self.graphs={};self.closed=False
        for i,size in enumerate([384,256]):
            self.cuda.call('cudaMalloc',ctypes.byref(self.din[i]),32*3*size*size*4)
            self.cuda.call('cudaMalloc',ctypes.byref(self.dout[i]),32*2048*4)
        self.cuda.call('cudaStreamCreateWithFlags',ctypes.byref(self.stream),1)
    def infer(self,inputs):
        P=ctypes.c_void_p;xs=[np.ascontiguousarray(x,dtype=np.float32) for x in inputs]
        size=len(xs[0])
        if not 1<=size<=32 or xs[0].shape!=(size,3,384,384) or xs[1].shape!=(size,3,256,256):
            raise ValueError('Invalid dual model inputs')
        ys=[np.empty((size,2048),np.float32),np.empty((size,2048),np.float32)]
        for i in [0,1]:self.cuda.call('cudaMemcpyAsync',self.din[i],P(xs[i].ctypes.data),xs[i].nbytes,1,self.stream)
        if size not in self.graphs:
            from .october_context_memory import create_contexts
            import tensorrt as trt
            def allocate(count):
                pointer=P();self.cuda.call('cudaMalloc',ctypes.byref(pointer),count);return pointer
            if not hasattr(self,'workspaces'):
                from .october_workspace import Workspaces
                self.workspaces=Workspaces(self.engines,allocate,lambda p:self.cuda.call('cudaFree',p))
            self.workspaces.begin_contexts()
            contexts=create_contexts(self.engines,[x.shape for x in xs],self.din,self.dout,
                lambda e:e.create_execution_context(trt.ExecutionContextAllocationStrategy.USER_MANAGED),
                self.workspaces.acquire,lambda p:None)
            for c in contexts:
                if not c.execute_async_v3(self.stream.value):raise RuntimeError('TensorRT warmup failed')
            self.cuda.call('cudaStreamSynchronize',self.stream)
            graph=P();exe=P();self.cuda.call('cudaStreamBeginCapture',self.stream,0)
            for c in contexts:
                if not c.execute_async_v3(self.stream.value):raise RuntimeError('TensorRT graph capture failed')
            self.cuda.call('cudaStreamEndCapture',self.stream,ctypes.byref(graph))
            self.cuda.call('cudaGraphInstantiateWithFlags',ctypes.byref(exe),graph,0)
            self.graphs[size]=(contexts,graph,exe)
        self.cuda.call('cudaGraphLaunch',self.graphs[size][2],self.stream)
        for i in [0,1]:self.cuda.call('cudaMemcpyAsync',P(ys[i].ctypes.data),self.dout[i],ys[i].nbytes,2,self.stream)
        self.cuda.call('cudaStreamSynchronize',self.stream)
        if not all(np.isfinite(y).all() for y in ys):raise RuntimeError('Nonfinite engine output')
        return ys
    def embed(self,batch):
        x,cf=batch;cnn=cf.numpy() if hasattr(cf,'numpy') else cf
        a,b=self.infer([x,cnn]);return {'emb':a,'specialist':b.astype(np.float16)}
    def close(self):
        if self.closed:return
        self.closed=True;self.pool.shutdown(wait=True)
        self.cuda.call('cudaStreamSynchronize',self.stream)
        for contexts,graph,exe in self.graphs.values():
            self.cuda.call('cudaGraphExecDestroy',exe);self.cuda.call('cudaGraphDestroy',graph)
            contexts.clear()
        self.graphs.clear()
        if hasattr(self,'workspaces'):self.workspaces.close()
        for p in self.din+self.dout:self.cuda.call('cudaFree',p)
        self.cuda.call('cudaStreamDestroy',self.stream)
        self.engines.clear()

def joint_features_class():
    # Import only after the original cached context has established its paths.
    from joint_runtime import JointFeatures as FrozenJointFeatures
    import torch
    class LinuxJointFeatures(FrozenJointFeatures):
        def __init__(self,logs,batch_size=16):
            if batch_size not in [1,8,16,32]:raise ValueError('Unsupported batch')
            torch.set_num_threads(2)
            self.core=LinuxCore(logs);self.cnn=True;self.batch_size=batch_size
            self.cfg=json.loads((ROOT/'var/three-directions-20261009/fusion-selection.json').read_text())['config']
            self.reused_core=True;self.engine=ARTIFACTS/'dino.engine'
            self.producer=ThreadPoolExecutor(max_workers=1)
    return LinuxJointFeatures
