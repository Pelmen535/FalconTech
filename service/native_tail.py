"""Optional adapter using the established static-gallery API and isolated worker."""
import hashlib,os,subprocess,sys
from .vric_road import VricRoadAdapter
from .robust import RobustAdapter
class NativeTailAdapter(VricRoadAdapter):
 def __init__(self,*args,**kwargs):
  super().__init__(*args,**kwargs)
  from vreid.native_tail import verified_selection,MANIFEST,PROFILE,digest
  self.tail_spec=verified_selection(self.root);self.tail_selection_sha=digest(self.root/MANIFEST);self.recipe_sha256=hashlib.sha256((self.recipe_sha256+PROFILE+self.tail_selection_sha).encode()).hexdigest()
 def metadata(self):
  from vreid.native_tail import PROFILE
  r=super().metadata();r.update(tail_profile=PROFILE,tail_selection_SHA256=self.tail_selection_sha,confidence_unchanged=True);return r
 def _request(self,message):
  with self._lock:
   if self._worker is None:
    self._registered_gallery=None;self._stderr=(self.data_dir/'robust-worker.log').open('a',encoding='utf-8');self._worker=subprocess.Popen([sys.executable,'-X','utf8','-B','-m','service.native_tail_worker',str(self.cache_dir)],cwd=self.root,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self._stderr,text=True,encoding='utf-8',bufsize=1,env={**os.environ,'VREID_RERANK_PROFILE':self.rerank_profile,'VREID_REFUSAL_PROFILE':self.refusal_profile},creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
   return RobustAdapter._request(self,message)
