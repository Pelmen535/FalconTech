"""Explicit complementary quality/local service; all three passes are real."""
import os,hashlib,subprocess,sys
from .local_recall import LocalRecallAdapter
from .robust import RobustAdapter
class HybridLocalAdapter(LocalRecallAdapter):
 def __init__(self,core_dir,*args,**kwargs):
  from vreid.hybrid_local_recall import verified_selection
  from vreid.recall_rescue import digest
  self.hybrid_spec=verified_selection(core_dir);self.hybrid_sha=digest(os.environ['VREID_HYBRID_LOCAL_SELECTION']);super().__init__(core_dir,*args,**kwargs);self.recipe_sha256=hashlib.sha256((self.recipe_sha256+self.hybrid_sha).encode()).hexdigest()
 def metadata(self):
  r=super().metadata();r.update(rescue_profile='quality-local-recall-v1',composite_profile='quality-local-tail-v1',hybrid_selection_SHA256=self.hybrid_sha,quality_selection_SHA256=self.hybrid_spec['quality_selection_SHA256'],all_auxiliary_image_computation_in_formation=True);return r
 def _request(self,message):
  with self._lock:
   if self._worker is None:
    self._registered_gallery=None;self._stderr=(self.data_dir/'robust-worker.log').open('a',encoding='utf-8');self._worker=subprocess.Popen([sys.executable,'-X','utf8','-B','-m','service.hybrid_local_recall_worker',str(self.cache_dir)],cwd=self.root,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self._stderr,text=True,encoding='utf-8',bufsize=1,env={**os.environ,'VREID_RERANK_PROFILE':self.rerank_profile,'VREID_REFUSAL_PROFILE':self.refusal_profile},creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
   return RobustAdapter._request(self,message)
