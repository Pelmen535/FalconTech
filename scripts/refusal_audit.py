"""Calibrate only on A; check frozen recipe on emitted CSV. Prior validation is not a new untouched holdout."""
import argparse
from pathlib import Path
import numpy as np
from eval_common import load_pair,score,counts,emitted,save
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',default='runs/hack/ft_soup_b336_val')
    p.add_argument('--query-npz');p.add_argument('--gallery-npz');p.add_argument('--recipe',required=True)
    p.add_argument('--seed',type=int,default=0);p.add_argument('--out',default='results/refusal_audit_fixed.json')
    a=p.parse_args();q,g,r,meta=load_pair(a);raw,rank,conf,has,correct=score(q,g,r)
    ids=np.unique(q['vids']);np.random.default_rng(a.seed).shuffle(ids)
    if len(ids)<2:raise ValueError('At least two query identities required')
    A=np.isin(q['vids'],ids[:len(ids)//2]);B=~A
    if not np.any(has[A]) or np.all(has[A]):raise ValueError('Calibration A needs matched and no-match queries')
    grid=np.append(np.unique(conf[A]),np.nextafter(np.max(conf[A]),np.float32(np.inf)))
    best=max((dict(threshold=float(t),**counts(conf[A]>=t,has[A],correct[A])) for t in grid),key=lambda x:(x['score'],x['tnr'],x['threshold']))
    out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True)
    results={}
    for name,t in [('frozen_recipe',r['threshold']),('selected_on_A',best['threshold'])]:
        accepted,actual=emitted(q,g,raw,rank,conf,t,out.parent/(out.stem+'_'+name+'_candidates.csv'))
        results[name]=dict(threshold=t,A=counts(accepted[A],has[A],actual[A]),B=counts(accepted[B],has[B],actual[B]),all=counts(accepted,has,actual))
    save(out,dict(**meta,seed=a.seed,n_query=len(conf),calibration_vehicle_ids=list(map(int,ids[:len(ids)//2])),
        evaluation_vehicle_ids=list(map(int,ids[len(ids)//2:])),results=results,threshold_grid_source='A only',
        history='If these validation identities informed earlier model/threshold decisions, B is a post-selection check, not untouched holdout.',
        release_recipe_modified=False,official_scorer='not_run'))
if __name__=='__main__':main()

