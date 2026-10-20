"""Isolated Windows TensorRT local-output pass; no model selection or query state."""
from pathlib import Path
import os,sys,ctypes,struct,hashlib
from multiprocessing import shared_memory
ROOT=Path(__file__).resolve().parents[1];N=ROOT/'var/nive-night-transfer-20261009';T=Path('C:/Users/Sheka/AppData/Local/Temp/vehicle-reid-comparison-20261005-01');sys.path[:0]=[str(N/'packages'),str(T/'tensorrt-install-20261006-01/venv/Lib/site-packages')];handles=[os.add_dll_directory(str(p)) for p in [T/'tensorrt-install-20261006-01/venv/Lib/site-packages/tensorrt_libs',Path('C:/Users/Sheka/AppData/Roaming/Python/Python314/site-packages/torch/lib')]]
import tensorrt as trt,numpy as np
path=Path(sys.argv[2]);assert hashlib.sha256(path.read_bytes()).hexdigest()==sys.argv[3]
cuda=ctypes.WinDLL('cudart64_12.dll');V=ctypes.c_void_p;I=ctypes.c_int;S=ctypes.c_size_t
for name,args in {'cudaMalloc':[ctypes.POINTER(V),S],'cudaFree':[V],'cudaMemcpyAsync':[V,V,S,I,V],'cudaStreamCreateWithFlags':[ctypes.POINTER(V),ctypes.c_uint],'cudaStreamSynchronize':[V],'cudaStreamBeginCapture':[V,I],'cudaStreamEndCapture':[V,ctypes.POINTER(V)],'cudaGraphInstantiateWithFlags':[ctypes.POINTER(V),V,ctypes.c_ulonglong],'cudaGraphLaunch':[V,V],'cudaGraphDestroy':[V],'cudaGraphExecDestroy':[V],'cudaStreamDestroy':[V]}.items():getattr(cuda,name).argtypes=args;getattr(cuda,name).restype=I
def ck(code):
 if code:raise RuntimeError('CUDA '+str(code))
runtime=trt.Runtime(trt.Logger(trt.Logger.WARNING));engine=runtime.deserialize_cuda_engine(path.read_bytes());assert engine;mem=shared_memory.SharedMemory(name=sys.argv[1]);inputs=np.ndarray((32,3,384,384),np.float32,buffer=mem.buf);din=V();dglobal=V();dlocal=V();stream=V();ck(cuda.cudaMalloc(ctypes.byref(din),inputs.nbytes));ck(cuda.cudaMalloc(ctypes.byref(dglobal),32*2048*4));ck(cuda.cudaMalloc(ctypes.byref(dlocal),32*36*1024*2));ck(cuda.cudaStreamCreateWithFlags(ctypes.byref(stream),1));graphs={}
try:
 sys.stdout.buffer.write(b'RDY!');sys.stdout.buffer.flush()
 while True:
  header=sys.stdin.buffer.read(4)
  if len(header)!=4:raise EOFError('Incomplete token command')
  n=struct.unpack('<I',header)[0]
  if n==0:break
  assert 1<=n<=32;x=inputs[:n];y=np.empty((n,36,1024),np.float16);ck(cuda.cudaMemcpyAsync(din,V(x.ctypes.data),x.nbytes,1,stream))
  if n not in graphs:
   context=engine.create_execution_context();assert context and context.set_input_shape('images',x.shape);context.set_tensor_address('images',din.value);context.set_tensor_address('embeddings',dglobal.value);context.set_tensor_address('local36',dlocal.value);assert context.execute_async_v3(stream.value);ck(cuda.cudaStreamSynchronize(stream));graph=V();exe=V();ck(cuda.cudaStreamBeginCapture(stream,0));assert context.execute_async_v3(stream.value);ck(cuda.cudaStreamEndCapture(stream,ctypes.byref(graph)));ck(cuda.cudaGraphInstantiateWithFlags(ctypes.byref(exe),graph,0));graphs[n]=(context,graph,exe)
  ck(cuda.cudaGraphLaunch(graphs[n][2],stream));ck(cuda.cudaMemcpyAsync(V(y.ctypes.data),dlocal,y.nbytes,2,stream));ck(cuda.cudaStreamSynchronize(stream));assert np.isfinite(y).all();sys.stdout.buffer.write(y.tobytes());sys.stdout.buffer.flush()
finally:
 for context,graph,exe in graphs.values():ck(cuda.cudaGraphExecDestroy(exe));ck(cuda.cudaGraphDestroy(graph))
 graphs.clear()
 for pointer in [din,dglobal,dlocal]:ck(cuda.cudaFree(pointer))
 ck(cuda.cudaStreamDestroy(stream));del inputs;mem.close()
