"""Optional adapter using the established static-gallery API and isolated worker."""
import hashlib,os,subprocess,sys
from .native_tail import NativeTailAdapter
from .robust import RobustAdapter
class NativeClosedTailAdapter(NativeTailAdapter):
 def __init__(self,*args,**kwargs):
  from vreid.native_closed_tail import configure
  configure()
  super().__init__(*args,**kwargs)
 def _request(self,message):
  with self._lock:
   if self._worker is None:
    self._registered_gallery=None;self._stderr=(self.data_dir/'robust-worker.log').open('a',encoding='utf-8');self._worker=subprocess.Popen([sys.executable,'-X','utf8','-B','-m','service.native_closed_tail_worker',str(self.cache_dir)],cwd=self.root,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self._stderr,text=True,encoding='utf-8',bufsize=1,env={**os.environ,'VREID_RERANK_PROFILE':self.rerank_profile,'VREID_REFUSAL_PROFILE':self.refusal_profile},creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
   return RobustAdapter._request(self,message)
