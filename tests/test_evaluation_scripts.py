from pathlib import Path
import os
import sys
import json
import subprocess
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from eval_common import counts,emitted
from vreid.paired import paired_drop


def utf8_env():
    """Скрипты печатают по-русски. На Windows у дочернего python stdout при перенаправлении
    берёт кодировку консоли (cp1251), и текст перестаёт быть UTF-8 ещё до того, как тест его
    прочитает. В Linux-контейнере жюри этого нет — значит, чинить надо окружение теста,
    а не текст скриптов."""
    env = dict(os.environ)
    env['PYTHONUTF8'] = '1'
    env['PYTHONIOENCODING'] = 'utf-8'
    return env

def test_actual_file_refusal_uses_selected_candidate(tmp_path):
    q={'keys':np.array(['q']),'vids':np.array([1])}
    g={'keys':np.array(['a','b']),'vids':np.array([1,2])}
    # Reranker promotes b (.6) above a (.9), so threshold .8 must refuse.
    accept,correct=emitted(q,g,np.array([[.9,.6]]),np.array([[0.,1.]]),np.array([.9]),.8,tmp_path/'c.csv')
    assert not accept[0] and not correct[0]

def test_counts_follow_written_query_outcomes():
    result=counts(np.array([1,1,0,0],bool),np.array([1,1,1,0],bool),np.array([1,0,0,0],bool))
    assert (result['tp'],result['fp'],result['fn'],result['tn'])==(1,1,1,1)
    assert result['f1']==.5 and result['tnr']==1

def test_paired_bootstrap_zero_effect_and_identity_clusters():
    r=paired_drop([1,.5,0,.5],[1,.5,0,.5],[1,1,2,2],repeats=100)
    assert r['mean_drop']==0 and r['ci95']==[0,0] and r['n_identities']==2

def test_gallery_without_distractors_is_not_estimable(tmp_path):
    e=np.eye(12,dtype=np.float32);common=dict(emb=e,vids=np.arange(12),cams=np.zeros(12,int))
    np.savez(tmp_path/'q.npz',**common,keys=np.array([f'q{i}' for i in range(12)]))
    np.savez(tmp_path/'g.npz',**{**common,'cams':np.ones(12,int)},keys=np.array([f'g{i}' for i in range(12)]))
    (tmp_path/'recipe.json').write_text(json.dumps(dict(threshold=.55)))
    out=tmp_path/'report.json';script=Path(__file__).resolve().parents[1]/'scripts/gallery_size_effect.py'
    result=subprocess.run([sys.executable,str(script),'--query-npz',str(tmp_path/'q.npz'),'--gallery-npz',str(tmp_path/'g.npz'),
        '--recipe',str(tmp_path/'recipe.json'),'--sizes','10','12','--out',str(out)],capture_output=True)
    assert result.returncode==0,result.stderr
    assert json.loads(out.read_text())['status']=='not_estimable'


def test_metrics_resolve_ties_like_submission():
    from vreid.metrics import evaluate
    sims=np.ones((1,20),np.float32);gv=np.arange(20)+10;gv[1]=1
    result=evaluate(sims,np.array([1]),np.array([0]),gv,np.ones(20,int),cutoff=10)
    assert result['mAP']==.5


def test_scorecard_reads_repeated_gallery_measurements(tmp_path):
    recipe=tmp_path/'release';recipe.mkdir();(recipe/'recipe.json').write_text('{}')
    report=dict(design='fixed_queries_all_positives_distractors_only',rows=[
        dict(seed=0,gallery=20,mAP10=.8),dict(seed=0,gallery=30,mAP10=.7),
        dict(seed=1,gallery=20,mAP10=.6),dict(seed=1,gallery=30,mAP10=.5)])
    (tmp_path/'gallery_size_effect.json').write_text(json.dumps(report))
    script=Path(__file__).resolve().parents[1]/'scripts/scorecard.py'
    result=subprocess.run([sys.executable,str(script),'--release',str(recipe),'--results',str(tmp_path)],
        capture_output=True,text=True,encoding='utf-8',env=utf8_env())
    assert result.returncode==0,result.stderr
    assert '70.0' in result.stdout and '60.0' in result.stdout and '-10.0' in result.stdout


@pytest.mark.parametrize('flags',[['--dba','1'],['--refuse-rate','.2'],['--confidence','auto']])
def test_export_rejects_forbidden_recipe_before_loading_weights(flags):
    script=Path(__file__).resolve().parents[1]/'scripts/export_release.py'
    result=subprocess.run([sys.executable,str(script),'--weights','missing.pt','--val','missing.json',*flags],
        capture_output=True,text=True,encoding='utf-8',env=utf8_env())
    assert result.returncode==2
    assert 'Competition export' in result.stderr
