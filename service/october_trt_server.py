"""Unchanged frozen joint TensorRT math, explicit local dependency paths."""

def main():
    import os,sys,ctypes,struct
    from pathlib import Path
    from multiprocessing import shared_memory
    import json,hashlib
    R=Path(__file__).resolve().parents[1];J=R/'var/cnn-trt-20261010'
    site=Path(os.environ['VREID_TRT_SITE']).resolve()
    sys.path.insert(0,str(R))
    from service.october_graph_cache import make_room
    from service.october_context_memory import create_contexts
    from service.october_workspace import Workspaces
    from service.october_engine import deserialize_engine
    sys.path[:0]=[str(R/'var/nive-night-transfer-20261009/packages'),str(site)]
    handles=[os.add_dll_directory(str(p)) for p in [site/'tensorrt_libs',Path(os.environ['VREID_CUDA_DLL_DIR'])]]
    import tensorrt as trt,numpy as np
    cuda=ctypes.WinDLL('cudart64_12.dll');P=ctypes.c_void_p;I=ctypes.c_int;S=ctypes.c_size_t
    signatures={'cudaMalloc':[ctypes.POINTER(P),S],'cudaFree':[P],'cudaMemcpyAsync':[P,P,S,I,P],'cudaStreamCreateWithFlags':[ctypes.POINTER(P),ctypes.c_uint],'cudaStreamSynchronize':[P],'cudaStreamBeginCapture':[P,I],'cudaStreamEndCapture':[P,ctypes.POINTER(P)],'cudaGraphInstantiateWithFlags':[ctypes.POINTER(P),P,ctypes.c_ulonglong],'cudaGraphLaunch':[P,P],'cudaGraphDestroy':[P],'cudaGraphExecDestroy':[P],'cudaStreamDestroy':[P]}
    for name,args in signatures.items():getattr(cuda,name).argtypes=args;getattr(cuda,name).restype=I
    def ck(code):
     if code:raise RuntimeError('CUDA '+str(code))
    # This native subprocess owns its CUDA context. Sleep while the same
    # stream completes instead of occupying one of the two assigned CPU cores.
    cuda.cudaSetDeviceFlags.argtypes=[ctypes.c_uint];cuda.cudaSetDeviceFlags.restype=I
    ck(cuda.cudaSetDeviceFlags(4)) # cudaDeviceScheduleBlockingSync
    runtime=trt.Runtime(trt.Logger(trt.Logger.WARNING));engines=[deserialize_engine(runtime,p) for p in [R/'var/backbone-night-finetune-20261009/export/model.engine',J/'export/model.engine']];assert all(engines);mem=shared_memory.SharedMemory(name=sys.argv[1]);n0=32*3*384*384;inputs=[np.ndarray((32,3,384,384),np.float32,buffer=mem.buf),np.ndarray((32,3,256,256),np.float32,buffer=mem.buf,offset=n0*4)];din=[P(),P()];dout=[P(),P()];stream=P()
    for i in [0,1]:ck(cuda.cudaMalloc(ctypes.byref(din[i]),inputs[i].nbytes));ck(cuda.cudaMalloc(ctypes.byref(dout[i]),32*2048*4))
    ck(cuda.cudaStreamCreateWithFlags(ctypes.byref(stream),1));graphs={};workspaces=None
    try:
     sys.stdout.buffer.write(b'RDY!');sys.stdout.buffer.flush()
     while True:
      header=sys.stdin.buffer.read(4)
      if len(header)!=4:raise EOFError('Incomplete command')
      size=struct.unpack('<I',header)[0]
      if size==0:break
      assert 1<=size<=32;xs=[a[:size] for a in inputs];ys=[np.empty((size,2048),np.float32),np.empty((size,2048),np.float32)]
      for i in [0,1]:ck(cuda.cudaMemcpyAsync(din[i],P(xs[i].ctypes.data),xs[i].nbytes,1,stream))
      if size not in graphs:
       def allocate(count):
        pointer=P();ck(cuda.cudaMalloc(ctypes.byref(pointer),count));return pointer
       if workspaces is None:workspaces=Workspaces(engines,allocate,lambda p:ck(cuda.cudaFree(p)))
       workspaces.begin_contexts()
       contexts=create_contexts(engines,[x.shape for x in xs],din,dout,lambda e:e.create_execution_context(trt.ExecutionContextAllocationStrategy.USER_MANAGED),workspaces.acquire,lambda p:None)
       print(json.dumps({'event':'native.workspace','batch':size,'engine_workspace_bounds':workspaces.sizes,'shape_requirements':contexts.required_bytes,'cached_shapes_after_insert':len(graphs)+1}),file=sys.stderr,flush=True)
       for c in contexts:assert c.execute_async_v3(stream.value)
       ck(cuda.cudaStreamSynchronize(stream));graph=P();exe=P();ck(cuda.cudaStreamBeginCapture(stream,0))
       for c in contexts:assert c.execute_async_v3(stream.value)
       ck(cuda.cudaStreamEndCapture(stream,ctypes.byref(graph)));ck(cuda.cudaGraphInstantiateWithFlags(ctypes.byref(exe),graph,0));graphs[size]=(contexts,graph,exe);del c
      ck(cuda.cudaGraphLaunch(graphs[size][2],stream))
      for i in [0,1]:ck(cuda.cudaMemcpyAsync(P(ys[i].ctypes.data),dout[i],ys[i].nbytes,2,stream))
      ck(cuda.cudaStreamSynchronize(stream));assert all(np.isfinite(y).all() for y in ys);sys.stdout.buffer.write(np.concatenate(ys,axis=1).tobytes());sys.stdout.buffer.flush()
    finally:
     for contexts,graph,exe in graphs.values():ck(cuda.cudaGraphExecDestroy(exe));ck(cuda.cudaGraphDestroy(graph));contexts.clear()
     graphs.clear()
     if workspaces is not None:workspaces.close()
     for p in din+dout:ck(cuda.cudaFree(p))
     ck(cuda.cudaStreamDestroy(stream));inputs.clear();mem.close()

if __name__=='__main__':main()
