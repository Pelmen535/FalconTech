"""Replay corrected ranking over recorded embeddings; no accuracy claim without labels."""
import argparse,csv,json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from vreid.predict import load_recipe,rerank_chunk_size,sha256
from vreid.rerank import frame_block_mask,k_reciprocal_chunked
from vreid.submit import write_submission,write_candidates
def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('submission','data','release','out'):p.add_argument('--'+k,required=True)
    a=p.parse_args();source=Path(a.submission);out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    def read(p):
        with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
    qk=[r['image_id'] for r in read(Path(a.data)/'test_query.csv')];gk=[r['image_id'] for r in read(Path(a.data)/'test_gallery.csv')]
    e=np.load(source/'embeddings.npy',allow_pickle=False);q,g=e[:len(qk)],e[len(qk):]
    if len(g)!=len(gk):raise ValueError('Embedding row count mismatch')
    r=load_recipe(Path(a.release));raw=np.stack([g@row for row in q]);blocked=frame_block_mask(qk,gk);raw[blocked]=-np.inf
    if r['kr']:
        rank=-k_reciprocal_chunked(q,g,r['k1'],r['k2'],.3,q_keys=qk,g_keys=gk,chunk=rerank_chunk_size(len(q),len(g),12))
        rank[blocked]=-np.inf
    else:rank=raw.copy()
    selected=np.argsort(-rank,axis=1,kind='stable')[:,0];conf=raw[np.arange(len(raw)),selected]
    write_submission(qk,gk,rank,out/'submission.csv',exclude_self=False)
    write_candidates(qk,gk,rank,conf,r['threshold'],out/'candidates.csv',exclude_self=False,conf_sims=raw)
    old,new=read(source/'submission.csv'),read(out/'submission.csv')
    if [x['query_id'] for x in old]!=qk:raise ValueError('Input/output order mismatch')
    oc={r['query_id']:r for r in read(source/'candidates.csv')};nc={r['query_id']:r for r in read(out/'candidates.csv')}
    info=dict(n_query=len(qk),n_gallery=len(gk),embeddings_sha256=sha256(source/'embeddings.npy'),recipe_sha256=sha256(Path(a.release)/'recipe.json'),
        top1_changes=sum(x['gallery_id_1']!=y['gallery_id_1'] for x,y in zip(old,new)),
        changed_rank_cells=sum(x[f'gallery_id_{i}']!=y[f'gallery_id_{i}'] for x,y in zip(old,new) for i in range(1,11)),
        acceptance_changes=len(set(oc)^set(nc)),old_accepted=len(oc),new_accepted=len(nc),
        candidate_id_changes=sum(oc[k]['gallery_id']!=nc[k]['gallery_id'] for k in set(oc)&set(nc)),
        scope='Cached ranking regression only; no new extraction, accuracy or GPU performance measurement.')
    (out/'comparison.json').write_text(json.dumps(info,indent=2),'utf-8');print(json.dumps(info))
if __name__=='__main__':main()
