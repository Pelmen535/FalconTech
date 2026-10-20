"""Explicit three-encoder-pass experimental service, separate raw gallery identity."""
import hashlib,os,subprocess,sys
from .vric_road import VricRoadAdapter
from .robust import RobustAdapter
class LocalRecallAdapter(VricRoadAdapter):
 def __init__(self,*args,**kwargs):
  if os.name!='nt':raise ValueError('Local patch profile is not yet qualified for Linux')
  super().__init__(*args,**kwargs)
  from vreid.local_recall import verified_selection
  from vreid.recall_rescue import digest
  self.local_spec=verified_selection(self.root);self.local_sha=digest(os.environ['VREID_LOCAL_RECALL_SELECTION']);engine_sha=self.local_spec['files'][self.local_spec['local_engine']];self.model_sha256=hashlib.sha256((self.model_sha256+engine_sha).encode()).hexdigest();self.recipe_sha256=hashlib.sha256((self.recipe_sha256+self.local_sha).encode()).hexdigest();self.embedding_sha256=hashlib.sha256((self.embedding_sha256+engine_sha+':actual-local36-F16:v1').encode()).hexdigest();self.cache_dir=self.data_dir/'raw-features'/self.embedding_sha256;self.cache_dir.mkdir(parents=True,exist_ok=True)
 def metadata(self):
  r=super().metadata();r.update(rescue_profile='native-local-recall-v1',rescue_selection_SHA256=self.local_sha,composite_profile='local-recall-tail-v1',tail_selection_SHA256=self.local_spec['tail_selection_SHA256'],quality_before_tail=True,original_accepted_candidates_confidence_preserved=True,native_encoder_passes=3,local_descriptor_shape=[36,1024],extra_local_engine_SHA256=self.local_spec['files'][self.local_spec['local_engine']],Linux_A5000_verified=False);return r
 def _request(self,message):
  with self._lock:
   if self._worker is None:
    self._registered_gallery=None;self._stderr=(self.data_dir/'robust-worker.log').open('a',encoding='utf-8');self._worker=subprocess.Popen([sys.executable,'-X','utf8','-B','-m','service.local_recall_worker',str(self.cache_dir)],cwd=self.root,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self._stderr,text=True,encoding='utf-8',bufsize=1,env={**os.environ,'VREID_RERANK_PROFILE':self.rerank_profile,'VREID_REFUSAL_PROFILE':self.refusal_profile},creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
   return RobustAdapter._request(self,message)
