import os,sys,ctypes,struct
from pathlib import Path
from multiprocessing import shared_memory
R=Path(__file__).resolve().parents[1];J=R/'var/cnn-trt-20261010';import hashlib;local_engine=Path(sys.argv[2]);assert hashlib.sha256(local_engine.read_bytes()).hexdigest()==sys.argv[3];T=Path(json.loads((WORK/'protocol.json').read_text())['store'])/'R27-native';T=Path('C:/Users/Sheka/AppData/Local/Temp/vehicle-reid-comparison-20261005-01');handles=[os.add_dll_directory(str(p)) for p in [T/'tensorrt-install-20261006-01/venv/Lib/site-packages/tensorrt_libs',Path('C:/Users/Sheka/AppData/Roaming/Python/Python314/site-packages/torch/lib')]]
import tensorrt as trt,numpy as np
cuda=ctypes.WinDLL('cudart64_12.dll');P=ctypes.c_void_p;I=ctypes.c_int;S=ctypes.c_size_t
signatures={'cudaMalloc':[ctypes.POINTER(P),S],'cudaFree':[P],'cudaMemcpyAsync':[P,P,S,I,P],'cudaStreamCreateWithFlags':[ctypes.POINTER(P),ctypes.c_uint],'cudaStreamSynchronize':[P],'cudaStreamBeginCapture':[P,I],'cudaStreamEndCapture':[P,ctypes.POINTER(P)],'cudaGraphInstantiateWithFlags':[ctypes.POINTER(P),P,ctypes.c_ulonglong],'cudaGraphLaunch':[P,P],'cudaGraphDestroy':[P],'cudaGraphExecDestroy':[P],'cudaStreamDestroy':[P]}
for name,args in signatures.items():getattr(cuda,name).argtypes=args;getattr(cuda,name).restype=I
def ck(code):
 if code:raise RuntimeError('CUDA '+str(code))
runtime=trt.Runtime(trt.Logger(trt.Logger.WARNING));engines=[runtime.deserialize_cuda_engine(p.read_bytes()) for p in [local_engine,J/'export/model.engine']];assert all(engines);mem=shared_memory.SharedMemory(name=sys.argv[1]);n0=32*3*384*384;inputs=[np.ndarray((32,3,384,384),np.float32,buffer=mem.buf),np.ndarray((32,3,256,256),np.float32,buffer=mem.buf,offset=n0*4)];din=[P(),P()];dout=[P(),P()];stream=P()
for i in [0,1]:ck(cuda.cudaMalloc(ctypes.byref(din[i]),inputs[i].nbytes));ck(cuda.cudaMalloc(ctypes.byref(dout[i]),32*2048*4))
dlocal=P();ck(cuda.cudaMalloc(ctypes.byref(dlocal),32*36*1024*2));ck(cuda.cudaStreamCreateWithFlags(ctypes.byref(stream),1));graphs={}
try:
 sys.stdout.buffer.write(b'RDY!');sys.stdout.buffer.flush()
 while True:
  header=sys.stdin.buffer.read(4)
  if len(header)!=4:raise EOFError('Incomplete command')
  size=struct.unpack('<I',header)[0]
  if size==0:break
  assert 1<=size<=32;xs=[a[:size] for a in inputs];ys=[np.empty((size,2048),np.float32),np.empty((size,2048),np.float32),np.empty((size,36,1024),np.float16)]
  for i in [0,1]:ck(cuda.cudaMemcpyAsync(din[i],P(xs[i].ctypes.data),xs[i].nbytes,1,stream))
  if size not in graphs:
   contexts=[e.create_execution_context() for e in engines]
   for i,c in enumerate(contexts):
    assert c and c.set_input_shape('images',xs[i].shape);c.set_tensor_address('images',din[i].value);c.set_tensor_address('embeddings',dout[i].value)
    if i==0:c.set_tensor_address('local36',dlocal.value)
    assert c.execute_async_v3(stream.value)
   ck(cuda.cudaStreamSynchronize(stream));graph=P();exe=P();ck(cuda.cudaStreamBeginCapture(stream,0))
   for c in contexts:assert c.execute_async_v3(stream.value)
   ck(cuda.cudaStreamEndCapture(stream,ctypes.byref(graph)));ck(cuda.cudaGraphInstantiateWithFlags(ctypes.byref(exe),graph,0));graphs[size]=(contexts,graph,exe);del contexts,c
  ck(cuda.cudaGraphLaunch(graphs[size][2],stream))
  for i in [0,1]:ck(cuda.cudaMemcpyAsync(P(ys[i].ctypes.data),dout[i],ys[i].nbytes,2,stream))
  ck(cuda.cudaMemcpyAsync(P(ys[2].ctypes.data),dlocal,ys[2].nbytes,2,stream));ck(cuda.cudaStreamSynchronize(stream));assert all(np.isfinite(y).all() for y in ys);sys.stdout.buffer.write(np.concatenate(ys[:2],axis=1).tobytes());sys.stdout.buffer.write(ys[2].tobytes());sys.stdout.buffer.flush()
  if size not in [1,16]:
   retired_contexts,retired_graph,retired_exe=graphs.pop(size);ck(cuda.cudaGraphExecDestroy(retired_exe));ck(cuda.cudaGraphDestroy(retired_graph));retired_contexts.clear();del retired_contexts
finally:
 for contexts,graph,exe in graphs.values():ck(cuda.cudaGraphExecDestroy(exe));ck(cuda.cudaGraphDestroy(graph))
 graphs.clear()
 for p in din+dout+[dlocal]:ck(cuda.cudaFree(p))
 ck(cuda.cudaStreamDestroy(stream));inputs.clear();mem.close()
