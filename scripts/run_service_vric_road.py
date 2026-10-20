"""Explicit separate service profile for validation of the global pair rule."""
from pathlib import Path
import os,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]

if __name__=='__main__':
    os.environ['VREID_RERANK_PROFILE']='current-query-v1';os.environ['VREID_REFUSAL_PROFILE']='expanded-source-v1'
    from service import app
    from service.vric_road import VricRoadAdapter
    original=app.create_app
    def create_vric_road_app(settings=None,core=None):
        if settings.backend!='robust' or settings.device!='cuda':raise ValueError('Road trained service requires --backend robust --device cuda')
        return original(settings,core=core if core is not None else VricRoadAdapter(settings.core_dir,settings.data_dir,settings.device))
    app.create_app=create_vric_road_app
    from run_service import main
    sys.argv=[sys.argv[0],'--backend','robust','--device','cuda','--data-dir',str(ROOT/'var/service-vric-road'),'--threads','2',*sys.argv[1:]]
    raise SystemExit(main())
