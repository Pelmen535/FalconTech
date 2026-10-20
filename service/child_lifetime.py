"""Windows job owns only one explicitly created RPC worker and its children.

No process-name/PID scan or breakaway flags. Attach before sending any RPC, so
no native child can be launched before job association. The unnamed handle is
not inherited. Last-handle close (including owner death) kills this job tree.
https://learn.microsoft.com/en-us/windows/win32/procthread/job-objects
"""
import os,ctypes,threading

class WorkerJob:
    def __init__(self,process):
        if os.name!='nt':raise RuntimeError('Windows job selected on another platform')
        from ctypes import wintypes as w
        class Basic(ctypes.Structure):
            _fields_=[('PerProcessUserTimeLimit',ctypes.c_int64),('PerJobUserTimeLimit',ctypes.c_int64),('LimitFlags',w.DWORD),('MinimumWorkingSetSize',ctypes.c_size_t),('MaximumWorkingSetSize',ctypes.c_size_t),('ActiveProcessLimit',w.DWORD),('Affinity',ctypes.c_size_t),('PriorityClass',w.DWORD),('SchedulingClass',w.DWORD)]
        class IO(ctypes.Structure):
            _fields_=[(name,ctypes.c_uint64) for name in ('ReadOperationCount','WriteOperationCount','OtherOperationCount','ReadTransferCount','WriteTransferCount','OtherTransferCount')]
        class Extended(ctypes.Structure):
            _fields_=[('BasicLimitInformation',Basic),('IoInfo',IO),('ProcessMemoryLimit',ctypes.c_size_t),('JobMemoryLimit',ctypes.c_size_t),('PeakProcessMemoryUsed',ctypes.c_size_t),('PeakJobMemoryUsed',ctypes.c_size_t)]
        self.lock=threading.Lock();self.api=ctypes.WinDLL('kernel32',use_last_error=True);self.handle=None
        self.api.CreateJobObjectW.argtypes=[ctypes.c_void_p,w.LPCWSTR];self.api.CreateJobObjectW.restype=w.HANDLE
        self.api.SetInformationJobObject.argtypes=[w.HANDLE,ctypes.c_int,ctypes.c_void_p,w.DWORD];self.api.SetInformationJobObject.restype=w.BOOL
        self.api.AssignProcessToJobObject.argtypes=[w.HANDLE,w.HANDLE];self.api.AssignProcessToJobObject.restype=w.BOOL
        self.api.CloseHandle.argtypes=[w.HANDLE];self.api.CloseHandle.restype=w.BOOL
        self.handle=self.api.CreateJobObjectW(None,None)
        if not self.handle:raise ctypes.WinError(ctypes.get_last_error())
        try:
            limits=Extended();limits.BasicLimitInformation.LimitFlags=0x2000
            if not self.api.SetInformationJobObject(self.handle,9,ctypes.byref(limits),ctypes.sizeof(limits)):raise ctypes.WinError(ctypes.get_last_error())
            # Popen's own Windows process handle, never an unrelated PID.
            if not self.api.AssignProcessToJobObject(self.handle,int(process._handle)):raise ctypes.WinError(ctypes.get_last_error())
        except BaseException:
            self.close();raise
    def close(self):
        with self.lock:
            if self.handle is not None:
                handle=self.handle
                if not self.api.CloseHandle(handle):raise ctypes.WinError(ctypes.get_last_error())
                self.handle=None

def worker_job(process):
    return WorkerJob(process) if os.name=='nt' else None
