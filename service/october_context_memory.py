"""TensorRT 10.9 user-managed activation memory for actual input shapes.

Each context owns a separate CUDA allocation. Nothing is shared concurrently.
CUDA graphs must be synchronized and destroyed before clear() releases contexts
and then their backing allocations. Engines/tactics/precision are unchanged.
"""

class Contexts(list):
    def __init__(self,free):
        super().__init__();self.allocations=[];self.free=free;self.required_bytes=[]

    def clear(self):
        super().clear()
        while self.allocations:
            pointer=self.allocations[-1]
            self.free(pointer)
            self.allocations.pop()


def create_contexts(engines,shapes,inputs,outputs,create,allocate,free):
    contexts=Contexts(free)
    try:
        for engine,shape,inp,out in zip(engines,shapes,inputs,outputs,strict=True):
            context=create(engine)
            if context is None:
                raise RuntimeError('TensorRT context construction failed')
            contexts.append(context)
            if not context.set_input_shape('images',shape):
                raise RuntimeError('TensorRT input shape rejected')
            required=context.update_device_memory_size_for_shapes()
            if required < 0:
                raise RuntimeError('Invalid activation memory requirement')
            contexts.required_bytes.append(required)
            pointer=allocate(required)
            if pointer is not None:contexts.allocations.append(pointer)
            context.device_memory=pointer.value if pointer is not None else 0
            if not context.set_tensor_address('images',inp.value) or not context.set_tensor_address('embeddings',out.value):
                raise RuntimeError('TensorRT tensor binding failed')
        return contexts
    except BaseException:
        # Drop the loop's final reference before freeing its memory.
        context=None
        contexts.clear()
        raise
