"""Linux-only transport wrapper; frozen model, refusal and gallery math preserved."""
import os,subprocess,sys
from .vric_road import VricRoadAdapter

class VricRoadLinuxAdapter(VricRoadAdapter):
    def __init__(self,*args,**kwargs):
        if sys.platform!='linux':raise ValueError('Linux adapter requires Linux')
        super().__init__(*args,**kwargs)

    def metadata(self):
        return {**super().metadata(),'worker_transport':'isolated-native-stdout-v1'}

    def _request(self,message):
        with self._lock:
            if self._worker is None:
                self._registered_gallery=None
                self._stderr=(self.data_dir/'robust-worker.log').open('a',encoding='utf-8')
                self._worker=subprocess.Popen([sys.executable,'-X','utf8','-B','-m','service.vric_road_linux_worker',str(self.cache_dir)],
                    cwd=self.root,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self._stderr,
                    text=True,encoding='utf-8',bufsize=1,
                    env={**os.environ,'VREID_RERANK_PROFILE':self.rerank_profile,'VREID_REFUSAL_PROFILE':self.refusal_profile})
            return super()._request(message)
