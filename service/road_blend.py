"""Optional global pair profile with the existing static-gallery RPC contract."""
import hashlib,json,os,subprocess,sys
from .robust import RobustAdapter

class RoadBlendAdapter(RobustAdapter):
    def __init__(self,*args,**kwargs):
        if os.environ.get('VREID_REFUSAL_PROFILE')!='expanded-source-v1' or os.environ.get('VREID_RERANK_PROFILE')!='current-query-v1':
            raise ValueError('Road trained adapter requires the preserved expanded/current-query base runtime')
        super().__init__(*args,**kwargs)
        from vreid.refusal_policy_road_blend import verified_selection,PROFILE
        spec=verified_selection(self.root)
        self.recipe_sha256=hashlib.sha256((self.recipe_sha256+PROFILE+json.dumps({key:spec[key] for key in ['files','threshold','alpha','remove_columns']},sort_keys=True)).encode()).hexdigest()

    def metadata(self):
        result=super().metadata();result.update(refusal_profile='road-blend-v1',base_refusal_profile='expanded-source-v1',gallery_style_routing=False)
        return result

    def _request(self,message):
        with self._lock:
            if self._worker is None:
                self._registered_gallery=None
                self._stderr=(self.data_dir/'robust-worker.log').open('a',encoding='utf-8')
                self._worker=subprocess.Popen([sys.executable,'-X','utf8','-B','-m','service.road_blend_worker',str(self.cache_dir)],
                    cwd=self.root,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self._stderr,text=True,encoding='utf-8',bufsize=1,
                    env={**os.environ,'VREID_RERANK_PROFILE':self.rerank_profile,'VREID_REFUSAL_PROFILE':self.refusal_profile},
                    creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            return super()._request(message)
