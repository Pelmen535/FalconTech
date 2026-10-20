"""Optional adapter using the established static-gallery API and isolated worker."""
import hashlib,os,subprocess,sys
from .vric_road import VricRoadAdapter
from .robust import RobustAdapter
class RecallRescueAdapter(VricRoadAdapter):
 def __init__(self,*args,**kwargs):
  super().__init__(*args,**kwargs)
  from vreid.recall_rescue import load_state
  from vreid.recall_rescue import digest
  from pathlib import Path
  self.rescue_spec=load_state(self.root).spec;self.rescue_selection_sha=digest(Path(os.environ['VREID_RECALL_RESCUE_SELECTION']));self.recipe_sha256=hashlib.sha256((self.recipe_sha256+'native-recall-rescue-v1'+self.rescue_selection_sha).encode()).hexdigest()
 def metadata(self):
  r=super().metadata();r.update(rescue_profile='native-recall-rescue-v1',rescue_selection_SHA256=self.rescue_selection_sha,original_accepted_candidates_confidence_preserved=True);return r
 def _request(self,message):
  with self._lock:
   if self._worker is None:
    self._registered_gallery=None;self._stderr=(self.data_dir/'robust-worker.log').open('a',encoding='utf-8');self._worker=subprocess.Popen([sys.executable,'-X','utf8','-B','-m','service.recall_rescue_worker',str(self.cache_dir)],cwd=self.root,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self._stderr,text=True,encoding='utf-8',bufsize=1,env={**os.environ,'VREID_RERANK_PROFILE':self.rerank_profile,'VREID_REFUSAL_PROFILE':self.refusal_profile},creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
   return RobustAdapter._request(self,message)
