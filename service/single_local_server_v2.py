"""Fixed common batch graphs; transient rare contexts bound long-running VRAM."""
import os,sys,ctypes,struct,hashlib,json
from pathlib import Path
from multiprocessing import shared_memory
ROOT=Path(__file__).resolve().parents[1];CNN=ROOT/'var/cnn-trt-20261010';TEMP=Path('C:/Users/Sheka/AppData/Local/Temp/vehicle-reid-comparison-20261005-01');handles=[os.add_dll_directory(str(p)) for p in [TEMP/'tensorrt-install-20261006-01/venv/Lib/site-packages/tensorrt_libs',Path('C:/Users/Sheka/AppData/Roaming/Python/Python314/site-packages/torch/lib')]]
import tensorrt as trt,numpy as np
path=Path(sys.argv[2]);assert hashlib.sha256(path.read_bytes()).hexdigest()==sys.argv[3]
cuda=ctypes.WinDLL('cudart64_12.dll');P=ctypes.c_void_p;I=ctypes.c_int;S=ctypes.c_size_t
signatures={'cudaMalloc':[ctypes.POINTER(P),S],'cudaFree':[P],'cudaMemcpyAsync':[P,P,S,I,P],'cudaStreamCreateWithFlags':[ctypes.POINTER(P),ctypes.c_uint],'cudaStreamSynchronize':[P],'cudaStreamBeginCapture':[P,I],'cudaStreamEndCapture':[P,ctypes.POINTER(P)],'cudaGraphInstantiateWithFlags':[ctypes.POINTER(P),P,ctypes.c_ulonglong],'cudaGraphLaunch':[P,P],'cudaGraphDestroy':[P],'cudaGraphExecDestroy':[P],'cudaStreamDestroy':[P]}
for name,args in signatures.items():getattr(cuda,name).argtypes=args;getattr(cuda,name).restype=I
def ck(code):
 if code:raise RuntimeError('CUDA '+str(code))
logger=trt.Logger(trt.Logger.WARNING);runtime=trt.Runtime(logger);engines=[runtime.deserialize_cuda_engine(f.read_bytes()) for f in [path,CNN/'export/model.engine']];assert all(engines);mem=shared_memory.SharedMemory(name=sys.argv[1]);n0=32*3*384*384;inputs=[np.ndarray((32,3,384,384),np.float32,buffer=mem.buf),np.ndarray((32,3,256,256),np.float32,buffer=mem.buf,offset=n0*4)];din=[P(),P()];dout=[P(),P()];dlocal=P();stream=P();graphs={};common={1,8,16,32}
for i in [0,1]:ck(cuda.cudaMalloc(ctypes.byref(din[i]),inputs[i].nbytes));ck(cuda.cudaMalloc(ctypes.byref(dout[i]),32*2048*4))
ck(cuda.cudaMalloc(ctypes.byref(dlocal),32*36*1024*2));ck(cuda.cudaStreamCreateWithFlags(ctypes.byref(stream),1))
def contexts_for(size):
 result=[]
 for i,engine in enumerate(engines):
  context=engine.create_execution_context();assert context and context.set_input_shape('images',inputs[i][:size].shape);context.set_tensor_address('images',din[i].value);context.set_tensor_address('embeddings',dout[i].value)
  if i==0:context.set_tensor_address('local36',dlocal.value)
  result.append(context)
 return result
def launch(contexts):
 for context in contexts:assert context.execute_async_v3(stream.value)
try:
 sys.stdout.buffer.write(b'RDY!');sys.stdout.buffer.flush()
 while True:
  header=sys.stdin.buffer.read(4)
  if len(header)!=4:raise EOFError('Incomplete single-local command')
  size=struct.unpack('<I',header)[0]
  if size==0:break
  assert 1<=size<=32;ys=[np.empty((size,2048),np.float32),np.empty((size,2048),np.float32),np.empty((size,36,1024),np.float16)]
  for i in [0,1]:x=inputs[i][:size];ck(cuda.cudaMemcpyAsync(din[i],P(x.ctypes.data),x.nbytes,1,stream))
  transient=None
  if size in common:
   if size not in graphs:
    contexts=contexts_for(size);launch(contexts);ck(cuda.cudaStreamSynchronize(stream));graph=P();exe=P();ck(cuda.cudaStreamBeginCapture(stream,0));launch(contexts);ck(cuda.cudaStreamEndCapture(stream,ctypes.byref(graph)));ck(cuda.cudaGraphInstantiateWithFlags(ctypes.byref(exe),graph,0));graphs[size]=(contexts,graph,exe);del contexts
   ck(cuda.cudaGraphLaunch(graphs[size][2],stream))
  else:transient=contexts_for(size);launch(transient)
  for i in [0,1]:ck(cuda.cudaMemcpyAsync(P(ys[i].ctypes.data),dout[i],ys[i].nbytes,2,stream))
  ck(cuda.cudaMemcpyAsync(P(ys[2].ctypes.data),dlocal,ys[2].nbytes,2,stream));ck(cuda.cudaStreamSynchronize(stream));assert all(np.isfinite(y).all() for y in ys)
  if transient is not None:
   transient.clear();del transient;sys.stderr.write(json.dumps({'event':'native.context.transient_released','batch':size,'retained_common_batches':sorted(graphs)})+'\n');sys.stderr.flush()
  sys.stdout.buffer.write(np.concatenate(ys[:2],axis=1).tobytes());sys.stdout.buffer.write(ys[2].tobytes());sys.stdout.buffer.flush()
finally:
 for contexts,graph,exe in graphs.values():ck(cuda.cudaGraphExecDestroy(exe));ck(cuda.cudaGraphDestroy(graph));contexts.clear()
 graphs.clear()
 for pointer in din+dout+[dlocal]:ck(cuda.cudaFree(pointer))
 ck(cuda.cudaStreamDestroy(stream));inputs.clear();mem.close()
