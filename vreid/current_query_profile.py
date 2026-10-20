"""CAL-frozen current-query reranking; existing encoders and refusal stay fixed."""
import numpy as np
from .current_query_graph import CurrentQueryGraph
from .rerank import frame_block_mask

PROFILE='current-query-v1'

def route(q,g,cfg):
    # Existing runtime establishes these imports and installs its cached backend.
    import fusion_evaluate as fusion
    import evaluate as single
    from fast_rerank import GalleryGraph
    if cfg['fusion']!=.5:raise ValueError('Profile requires the frozen equal-weight ensemble')
    r0=single.route(q,g,{})
    r1=single.route(q,g,{'cnn':1.,'cnn_head':'specialist-cnn.pt'})
    raw=.5*r0['raw']+.5*r1['raw'];base=.5*r0['rank']+.5*r1['rank']
    qjoint=(np.concatenate([r0['query'],r1['query']],axis=1)/np.sqrt(2.)).astype(np.float32)
    gjoint=(np.concatenate([r0['gallery'],r1['gallery']],axis=1)/np.sqrt(2.)).astype(np.float32)
    gm=frame_block_mask(g['keys'],g['keys']);qm=frame_block_mask(q['keys'],g['keys'])
    new=-CurrentQueryGraph(gjoint,GalleryGraph,g_mask=gm).distances(qjoint,qm)
    rank=base.copy();valid=np.isfinite(raw);rank[valid]=.75*base[valid]+.25*new[valid];rank[~valid]=-np.inf
    r={**r1,'raw':raw,'rank':rank};order=np.argsort(-rank,axis=1,kind='stable');top=order[:,0];r['top']=top;ii=np.arange(len(top));conf=raw[ii,top]
    top5=np.sort(raw,axis=1)[:,-5:];other=raw.copy();other[ii,top]=-np.inf
    gg=.5*(r0['gallery']@r0['gallery'].T)+.5*(r1['gallery']@r1['gallery'].T);np.fill_diagonal(gg,-np.inf);density=np.partition(gg,-5,axis=1)[:,-5:].mean(1)[top]
    r['X']=np.column_stack([conf,conf-np.max(other,1),top5[:,-1]-top5[:,-2],top5.mean(1),top5.std(1),rank[ii,top]-rank[ii,order[:,1]],density,conf-density]).astype(np.float64)
    bc,br=fusion.bc,fusion.br
    oq=bc.transform(q,bc.OLDZ['transform'],bc.OLDZ['mean']);og=bc.transform(g,bc.OLDZ['transform'],bc.OLDZ['mean']);aux=br.rank_infer(oq,og)
    sim=aux['raw'][ii,top];best=np.max(aux['raw'],axis=1);ag=og['emb']@og['emb'].T;np.fill_diagonal(ag,-np.inf);adensity=np.partition(ag,-5,axis=1)[:,-5:].mean(1)[top]
    r['X20']=np.column_stack([r['X'],aux['X'],top==aux['top'],sim,sim-best,sim-adensity])
    r['_query_features']={k:q[k] for k in ['emb','specialist']};r['_gallery_features']={k:g[k] for k in ['emb','specialist']}
    return r
