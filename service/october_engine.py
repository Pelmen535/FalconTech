"""Read-only mapped engine input: same serialized bytes, no Python-sized copy."""
from pathlib import Path
import mmap


def deserialize_engine(runtime,path):
    with Path(path).open('rb') as stream:
        with mmap.mmap(stream.fileno(),0,access=mmap.ACCESS_READ) as serialized:
            return runtime.deserialize_cuda_engine(serialized)
