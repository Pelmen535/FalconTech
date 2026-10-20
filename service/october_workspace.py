"""One activation allocation for serial DINO/CNN enqueue on the same stream.

The backend already shares its stream/input/output control and synchronizes
before replying. TensorRT10.9 main-stream ordering includes auxiliary-stream
joins at enqueue boundaries. Activation memory is not shared by parallel work.
All contexts/graphs must be destroyed before the allocation is released.
https://raw.githubusercontent.com/NVIDIA/TensorRT/release/10.9/python/docstrings/infer/pyCoreDoc.h
"""
class Workspaces:
    def __init__(self,engines,allocate,free):
        self.pointers=[];self.sizes=[];self.free=free;self.position=0;self.closed=False
        for engine in engines:
            size=int(engine.device_memory_size_v2)
            if size<0:raise RuntimeError('Invalid TensorRT workspace bound')
            self.sizes.append(size)
        maximum=max(self.sizes,default=0)
        if maximum:self.pointers.append(allocate(maximum))
    def acquire(self,size):
        if self.closed:raise RuntimeError('TensorRT workspace is closed')
        index=self.position
        if index>=len(self.sizes) or size<0 or size>self.sizes[index]:
            raise RuntimeError('TensorRT workspace requirement exceeds pinned engine bound')
        self.position+=1
        return self.pointers[0] if self.sizes[index] else None
    def begin_contexts(self):
        if self.closed:raise RuntimeError('TensorRT workspace is closed')
        self.position=0
    def close(self):
        while self.pointers:
            self.free(self.pointers[-1]);self.pointers.pop()
        self.closed=True
