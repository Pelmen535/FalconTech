"""Experimental exact factorization of the existing isolated-query reranker."""
import numpy as np
from vreid.rerank import _l2n, _FAR, frame_block_mask

class GalleryGraph:
    def __init__(self,g,k1=3,k2=2,lam=.2,g_mask=None):
        self.g=_l2n(g.astype(np.float32));self.lam=lam
        G=len(g);self.k1=max(1,min(k1,G-1));self.k2=max(1,min(k2,self.k1))
        orig=2.-2.*(self.g@self.g.T)
        if g_mask is not None:orig[g_mask]=_FAR
        np.fill_diagonal(orig,0.)
        if np.any((orig<_FAR).sum(axis=1)<self.k1+1):raise ValueError('Sparse eligible gallery; use reference')
        rank=np.argsort(orig,axis=1,kind='stable')
        def reciprocal(i,k):
            fwd=rank[i,:k+1];back=rank[fwd,:k+1]
            return fwd[(back==i).any(axis=1)]
        V=np.zeros((G,G),np.float32)
        for i in range(G):
            R=reciprocal(i,self.k1);expanded=set(R.tolist())
            for j in R:
                Rj=reciprocal(j,max(1,self.k1//2))
                if len(np.intersect1d(R,Rj,assume_unique=True))>2/3*len(Rj):expanded.update(Rj.tolist())
            ids=np.array(sorted(expanded),dtype=np.int64);w=np.exp(-orig[i,ids]);V[i,ids]=w/w.sum()
        self.before=V
        self.after=np.stack([V[rank[i,:self.k2]].mean(axis=0) for i in range(G)]) if self.k2>1 else V
        self.inv=[np.nonzero(self.after[:,j])[0] for j in range(G)]

    def distances(self,q,q_mask=None):
        q=_l2n(q.astype(np.float32));G=len(self.g);out=np.empty((len(q),G),np.float32)
        for i,row in enumerate(q):
            orig=2.-2.*(self.g@row)
            if q_mask is not None:orig[q_mask[i]]=_FAR
            if (orig<_FAR).sum()<self.k2-1:raise ValueError('Sparse eligible query; use reference')
            ids=np.argsort(np.r_[np.float32(0),orig],kind='stable')[:self.k2]-1
            v=np.stack([np.zeros(G,np.float32) if j<0 else self.before[j] for j in ids]).mean(axis=0)
            tmin=np.zeros(G,np.float32)
            for j in np.nonzero(v)[0]:
                neighbors=self.inv[j]
                tmin[neighbors]+=np.minimum(v[j],self.after[neighbors,j])
            out[i]=(1.-tmin/(2.-tmin))*(1-self.lam)+orig*self.lam
        return out
