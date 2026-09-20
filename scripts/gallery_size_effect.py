"""Hold queries and all their positive frames fixed; add only distractor frames."""
import argparse,json
from pathlib import Path
import numpy as np
from eval_common import load_pair,score,counts,save
from vreid.metrics import evaluate
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',default='runs/hack/ft_soup_b336_val')
    p.add_argument('--query-npz');p.add_argument('--gallery-npz');p.add_argument('--recipe',required=True)
    p.add_argument('--query-ids-json',help='Predeclared fixed query cohort; optional, default all')
    p.add_argument('--sizes',type=int,nargs='+',required=True);p.add_argument('--repeats',type=int,default=5)
    p.add_argument('--seed',type=int,default=0);p.add_argument('--out',default='results/gallery_size_effect_fixed.json')
    a=p.parse_args();q,g,r,meta=load_pair(a)
    if a.query_ids_json:
        keys=json.loads(Path(a.query_ids_json).read_text('utf-8'))
        if len(keys)!=len(set(keys)) or not set(keys)<=set(q['keys']):raise ValueError('Invalid query cohort')
        keep=np.isin(q['keys'],keys);q={k:v[keep] for k,v in q.items()}
    anchors=np.flatnonzero(np.isin(g['vids'],q['vids']));negative=np.flatnonzero(~np.isin(g['vids'],q['vids']))
    report=dict(**meta,design='fixed_queries_all_positives_distractors_only',threshold=r['threshold'],
        fixed_query_ids=list(map(str,q['keys'])),positive_rows_retained=len(anchors),available_distractors=len(negative),rows=[])
    if not len(negative):
        report.update(status='not_estimable',reason='All gallery rows are positives for the fixed query cohort; no distractor pool. Do not invent a size effect.')
        save(a.out,report);return
    if any(n<max(10,len(anchors)) or n>len(g['emb']) for n in a.sizes):raise ValueError('Requested size cannot retain all positives or exceeds available gallery')
    for rep in range(a.repeats):
        order=np.random.default_rng(a.seed+rep).permutation(negative)
        for size in a.sizes:
            idx=np.sort(np.concatenate([anchors,order[:size-len(anchors)]]));gg={k:v[idx] for k,v in g.items()}
            raw,rank,conf,has,correct=score(q,gg,r)
            m=evaluate(rank,q['vids'],q['cams'],gg['vids'],gg['cams'],cutoff=10)
            report['rows'].append(dict(seed=a.seed+rep,gallery=size,mAP10=m['mAP'],rank1=m['rank1'],
                refusal=counts(conf>=r['threshold'],has,correct),n_query=len(q['emb'])))
    report['status']='measured_fixed_cohort';save(a.out,report)
if __name__=='__main__':main()

