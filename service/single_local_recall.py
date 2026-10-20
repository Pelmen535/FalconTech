"""One-pass DINO global/local backend with separately identified raw features."""
import os,hashlib,subprocess,sys
from .local_recall import LocalRecallAdapter
from .robust import RobustAdapter
class SingleLocalAdapter(LocalRecallAdapter):
 def __init__(self,core_dir,*args,**kwargs):
  from vreid.single_local_recall import verified_selection
  from vreid.recall_rescue import digest
  self.single_spec=verified_selection(core_dir);self.single_sha=digest(os.environ['VREID_SINGLE_LOCAL_SELECTION']);super().__init__(core_dir,*args,**kwargs);self.recipe_sha256=hashlib.sha256((self.recipe_sha256+self.single_sha).encode()).hexdigest();self.embedding_sha256=hashlib.sha256((self.embedding_sha256+':distinct-global-local-singlepass').encode()).hexdigest();self.cache_dir=self.data_dir/'raw-features'/self.embedding_sha256;self.cache_dir.mkdir(parents=True,exist_ok=True)
 def metadata(self):
  r=super().metadata();r.update(rescue_profile='single-native-local-recall-v1',composite_profile='single-local-tail-v1',single_selection_SHA256=self.single_sha,native_encoder_passes=2,original_accepted_candidates_confidence_preserved=False,native_base_accepted_confidence_preserved=True,original_global_engine_unchanged=False,quality_before_tail=False,local_before_tail=True);return r
 def _request(self,message):
  with self._lock:
   if self._worker is None:
    self._registered_gallery=None;self._stderr=(self.data_dir/'robust-worker.log').open('a',encoding='utf-8');self._worker=subprocess.Popen([sys.executable,'-X','utf8','-B','-m','service.single_local_recall_worker',str(self.cache_dir)],cwd=self.root,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self._stderr,text=True,encoding='utf-8',bufsize=1,env={**os.environ,'VREID_RERANK_PROFILE':self.rerank_profile,'VREID_REFUSAL_PROFILE':self.refusal_profile},creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
   return RobustAdapter._request(self,message)
