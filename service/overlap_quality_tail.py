"""Explicit scheduling-only service variant; model and safety policy are pinned."""
from pathlib import Path
import hashlib,json,os,subprocess,sys
from .quality_tail import QualityTailAdapter
from .robust import RobustAdapter

def verified_selection(root):
 from vreid.recall_rescue import digest
 root=Path(root).resolve();path=Path(os.environ.get('VREID_OVERLAP_QUALITY_SELECTION',''))
 if not path.is_absolute():raise ValueError('Explicit absolute overlap scheduling selection required')
 s=json.loads(path.read_text(encoding='utf-8'))
 if s.get('profile')!='quality-tail-overlap-v1' or s.get('status')!='FROZEN_FOR_VALIDATION':raise ValueError('Unqualified overlap scheduling profile')
 if digest(s['model_selection'])!=s['model_selection_SHA256']:raise ValueError('Frozen model selection changed')
 for f,want in s['files'].items():
  if digest(root/f)!=want:raise ValueError('Overlap scheduling source changed: '+f)
 os.environ['VREID_QUALITY_TAIL_SELECTION']=s['model_selection'];return s

class OverlapQualityTailAdapter(QualityTailAdapter):
 def __init__(self,core_dir,*args,**kwargs):
  verified_selection(core_dir);super().__init__(core_dir,*args,**kwargs)
  from vreid.recall_rescue import digest
  self.overlap_sha=digest(os.environ['VREID_OVERLAP_QUALITY_SELECTION']);self.recipe_sha256=hashlib.sha256((self.recipe_sha256+self.overlap_sha).encode()).hexdigest()
 def metadata(self):
  r=super().metadata();r.update(extraction_schedule='quality30_overlap_encoding',scheduling_selection_SHA256=self.overlap_sha,model_and_resource_thresholds_unchanged=True);return r
 def _request(self,message):
  with self._lock:
   if self._worker is None:
    self._registered_gallery=None;self._stderr=(self.data_dir/'robust-worker.log').open('a',encoding='utf-8');self._worker=subprocess.Popen([sys.executable,'-X','utf8','-B','-m','service.overlap_quality_tail_worker',str(self.cache_dir)],cwd=self.root,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self._stderr,text=True,encoding='utf-8',bufsize=1,env={**os.environ,'VREID_RERANK_PROFILE':self.rerank_profile,'VREID_REFUSAL_PROFILE':self.refusal_profile},creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
   return RobustAdapter._request(self,message)
