"""Optional adapter using the established static-gallery API and isolated worker."""
import hashlib,os,subprocess,sys
from .vric_road import VricRoadAdapter
from .robust import RobustAdapter
from .quality_recall import QualityRecallAdapter
class QualityTailAdapter(QualityRecallAdapter):
 def __init__(self,core_dir,data_dir,device='cuda'):
  from vreid.quality_tail import verified_selection
  from vreid.recall_rescue import digest
  from pathlib import Path
  self.composite_spec=verified_selection(core_dir)
  super().__init__(core_dir,data_dir,device)
  self.composite_selection_sha=digest(Path(os.environ['VREID_QUALITY_TAIL_SELECTION']));self.recipe_sha256=hashlib.sha256((self.recipe_sha256+'quality-recall-tail-v1'+self.composite_selection_sha).encode()).hexdigest()
 def metadata(self):
  r=super().metadata();r.update(composite_profile='quality-recall-tail-v1',composite_selection_SHA256=self.composite_selection_sha,tail_selection_SHA256=self.composite_spec['tail_selection_SHA256'],quality_before_tail=True);return r
 def _request(self,message):
  with self._lock:
   if self._worker is None:
    self._registered_gallery=None;self._stderr=(self.data_dir/'robust-worker.log').open('a',encoding='utf-8');self._worker=subprocess.Popen([sys.executable,'-X','utf8','-B','-m','service.quality_tail_worker',str(self.cache_dir)],cwd=self.root,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self._stderr,text=True,encoding='utf-8',bufsize=1,env={**os.environ,'VREID_RERANK_PROFILE':self.rerank_profile,'VREID_REFUSAL_PROFILE':self.refusal_profile},creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
   return RobustAdapter._request(self,message)
