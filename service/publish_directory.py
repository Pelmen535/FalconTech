"""Publish a new same-filesystem directory without replacing any existing path."""
from pathlib import Path
import ctypes,errno,os,sys

def publish_new_directory(staging,target):
    staging,target=Path(staging),Path(target)
    if staging.parent.resolve()!=target.parent.resolve() or not staging.is_dir() or staging.is_symlink():
        raise ValueError('Publication needs a private directory beside the destination')
    if os.name=='nt':
        os.rename(staging,target)
        return
    if sys.platform.startswith('linux'):
        libc=ctypes.CDLL(None,use_errno=True);rename=getattr(libc,'renameat2',None)
        if rename is None:raise OSError(errno.ENOTSUP,'Atomic no-replace directory publication unavailable')
        rename.argtypes=[ctypes.c_int,ctypes.c_char_p,ctypes.c_int,ctypes.c_char_p,ctypes.c_uint];rename.restype=ctypes.c_int
        if rename(-100,os.fsencode(staging),-100,os.fsencode(target),1)!=0:
            number=ctypes.get_errno();raise OSError(number,os.strerror(number),str(target))
        return
    raise OSError(errno.ENOTSUP,'Atomic no-replace publication unsupported on this platform')
