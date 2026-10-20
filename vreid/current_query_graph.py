"""Full gallery reciprocity for one current query; no other query is visible.

Reuse gallery-only reciprocal rows and update only rows affected by insertion.
This is a candidate algorithm, not a replacement for the frozen runtime.
"""
import numpy as np
from vreid.rerank import _l2n, _FAR


class CurrentQueryGraph:
    def __init__(self, gallery, graph_class, g_mask=None, k1=3, k2=2, lam=.2):
        self.base=graph_class(gallery,k1=k1,k2=k2,lam=lam,g_mask=g_mask)
        self.g=self.base.g;self.k1=self.base.k1;self.k2=self.base.k2;self.lam=lam
        d=2.-2.*(self.g@self.g.T)
        if g_mask is not None:d[g_mask]=_FAR
        np.fill_diagonal(d,0.)
        self.orig=d
        self.rank=np.argsort(d,axis=1,kind='stable')[:,:self.k1+1]
        self.ds=np.take_along_axis(d,self.rank,axis=1)
        self.oldrank=self.rank+1
        self.oldr=self._reciprocal(np.r_[np.zeros((1,self.k1+1),np.int64),self.oldrank],self.k1)[1:]
        self.oldhalf=self._reciprocal(np.r_[np.zeros((1,self.k1+1),np.int64),self.oldrank],max(1,self.k1//2))[1:]

    @staticmethod
    def _reciprocal(rank,k):
        f=rank[:,:k+1];b=rank[f,:k+1]
        ok=(b==np.arange(len(rank))[:,None,None]).any(axis=2)
        return np.where(ok,f,-1)

    def distances(self, query, q_mask=None):
        query=_l2n(np.asarray(query,dtype=np.float32));G=len(self.g);out=[]
        for i,q in enumerate(query):
            d=2.-2.*(self.g@q)
            if q_mask is not None:d[q_mask[i]]=_FAR
            # Query is index zero, so stable ties put it before gallery indices.
            candidates=np.c_[np.zeros(G,dtype=np.int64),self.oldrank]
            canddist=np.c_[d,self.ds]
            chosen=np.argsort(canddist,axis=1,kind='stable')[:,:self.k1+1]
            grank=np.take_along_axis(candidates,chosen,axis=1)
            qr=np.argsort(np.r_[np.float32(0),d],kind='stable')[:self.k1+1]
            rank=np.r_[qr[None,:],grank]
            R=self._reciprocal(rank,self.k1);H=self._reciprocal(rank,max(1,self.k1//2))
            # Compare sets, because insertion may shift an unchanged reciprocal set.
            changedR=np.any(np.sort(R[1:],axis=1)!=np.sort(self.oldr,axis=1),axis=1)
            changedH=np.r_[True,np.any(np.sort(H[1:],axis=1)!=np.sort(self.oldhalf,axis=1),axis=1)]
            affected=np.r_[True,changedR] | np.any(np.where(R>=0,changedH[np.maximum(R,0)],False),axis=1)
            before={}
            for row in np.flatnonzero(affected):
                rec=R[row];rec=rec[rec>=0];expanded=set(rec.tolist())
                for j in rec:
                    rh=H[j];rh=rh[rh>=0]
                    if len(np.intersect1d(rec,rh,assume_unique=True))>2/3*len(rh):expanded.update(rh.tolist())
                ids=np.array(sorted(expanded),dtype=np.int64)
                if row==0:dist=np.r_[np.float32(0),d][ids]
                else:
                    dist=np.array([d[row-1] if j==0 else self.orig[row-1,j-1] for j in ids],np.float32)
                w=np.exp(-dist);v=np.zeros(G+1,np.float32);v[ids]=w/w.sum();before[row]=v
            def vbefore(row):
                if row in before:return before[row]
                return np.r_[np.float32(0),self.base.before[row-1]]
            expanded_changed=np.r_[True,np.any(rank[1:,:self.k2]!=self.oldrank[:,:self.k2],axis=1)] | np.any(affected[rank[:,:self.k2]],axis=1)
            qv=np.stack([vbefore(j) for j in rank[0,:self.k2]]).mean(axis=0)
            nz=np.flatnonzero(qv);gnz=nz[nz>0]
            tmin=np.minimum(self.base.after[:,gnz-1],qv[gnz][None,:]).sum(axis=1)
            for row in np.flatnonzero(expanded_changed[1:])+1:
                v=np.stack([vbefore(j) for j in rank[row,:self.k2]]).mean(axis=0)
                tmin[row-1]=np.minimum(v[nz],qv[nz]).sum()
            out.append((1.-tmin/(2.-tmin))*(1-self.lam)+d*self.lam)
        return np.asarray(out,dtype=np.float32)

