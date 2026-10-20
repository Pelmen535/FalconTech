"""Exact gallery graph storage and candidate-only Jaccard evaluation.

The full fixed gallery still determines reciprocity, query neighbours and masks.
Only final candidate distances are restricted in ANN mode. No labels or fitting.
"""
import numpy as np
from scipy import sparse
from vreid.current_query_graph import CurrentQueryGraph
from vreid.rerank import _l2n,_FAR


class SparseRows:
    def __init__(self,array):
        self.rows=sparse.csr_matrix(array)
        self.columns=self.rows.tocsc()
        self.shape=array.shape;self.dtype=array.dtype
    @property
    def nbytes(self):
        return sum(x.nbytes for m in (self.rows,self.columns) for x in (m.data,m.indices,m.indptr))
    def take(self,rows,columns):
        return np.asfortranarray(self.columns[:,columns][rows,:].toarray())
    def __getitem__(self,index):
        if isinstance(index,tuple):
            rows,cols=index
            result=(self.columns if isinstance(rows,slice) else self.rows)[index]
            if sparse.issparse(result):result=result.toarray()
            result=np.asarray(result,dtype=self.dtype)
            if np.isscalar(rows) or np.isscalar(cols):return result.reshape(-1) if result.size>1 else self.dtype.type(result.item())
            # Match NumPy's original advanced column indexing layout/reduction.
            if isinstance(rows,slice) and isinstance(cols,np.ndarray):return np.asfortranarray(result)
            return result
        result=self.rows[index].toarray()
        return result[0] if np.isscalar(index) else result


def gallery_graph_class(base):
    class OctoberGalleryGraph(base):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs)
            self.before=SparseRows(self.before);self.after=SparseRows(self.after)
        def distances(self,q,q_mask=None,pools=None):
            if pools is None:return super().distances(q,q_mask)
            q=_l2n(q.astype(np.float32));G=len(self.g);out=np.full((len(q),G),np.inf,np.float32)
            for i,row in enumerate(q):
                pool=np.asarray(pools[i],np.int64)
                orig=2.-2.*(self.g@row)
                if q_mask is not None:orig[q_mask[i]]=_FAR
                if (orig<_FAR).sum()<self.k2-1:raise ValueError('Sparse eligible query')
                ids=np.argsort(np.r_[np.float32(0),orig],kind='stable')[:self.k2]-1
                v=np.stack([np.zeros(G,np.float32) if j<0 else self.before[j] for j in ids]).mean(axis=0)
                tmin=np.zeros(len(pool),np.float32)
                for j in np.nonzero(v)[0]:
                    # Evaluate overlap only for the shortlisted observations.
                    values=np.asarray(self.after[pool,j],np.float32).reshape(-1)
                    tmin+=np.minimum(v[j],values)
                out[i,pool]=(1.-tmin/(2.-tmin))*(1-self.lam)+orig[pool]*self.lam
            return out
    return OctoberGalleryGraph


class OctoberCurrentQueryGraph(CurrentQueryGraph):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        # A narrow NumPy view otherwise retains an entire GxG int64 argsort.
        self.rank=self.rank.copy()
    def distances(self,query,q_mask=None,pools=None):
        if pools is None:return super().distances(query,q_mask)
        query=_l2n(np.asarray(query,dtype=np.float32));G=len(self.g);out=[]
        for i,q in enumerate(query):
            pool=np.asarray(pools[i],np.int64)
            d=2.-2.*(self.g@q)
            if q_mask is not None:d[q_mask[i]]=_FAR
            candidates=np.c_[np.zeros(G,dtype=np.int64),self.oldrank]
            canddist=np.c_[d,self.ds]
            chosen=np.argsort(canddist,axis=1,kind='stable')[:,:self.k1+1]
            grank=np.take_along_axis(candidates,chosen,axis=1)
            qr=np.argsort(np.r_[np.float32(0),d],kind='stable')[:self.k1+1]
            rank=np.r_[qr[None,:],grank]
            R=self._reciprocal(rank,self.k1);H=self._reciprocal(rank,max(1,self.k1//2))
            changedR=np.any(np.sort(R[1:],axis=1)!=np.sort(self.oldr,axis=1),axis=1)
            changedH=np.r_[True,np.any(np.sort(H[1:],axis=1)!=np.sort(self.oldhalf,axis=1),axis=1)]
            affected=np.r_[True,changedR]|np.any(np.where(R>=0,changedH[np.maximum(R,0)],False),axis=1)
            before={}
            def vbefore(row):
                if row not in before and affected[row]:
                    rec=R[row];rec=rec[rec>=0];expanded=set(rec.tolist())
                    for j in rec:
                        rh=H[j];rh=rh[rh>=0]
                        if len(np.intersect1d(rec,rh,assume_unique=True))>2/3*len(rh):expanded.update(rh.tolist())
                    ids=np.array(sorted(expanded),dtype=np.int64)
                    if row==0:dist=np.r_[np.float32(0),d][ids]
                    else:dist=np.array([d[row-1] if j==0 else self.orig[row-1,j-1] for j in ids],np.float32)
                    w=np.exp(-dist);v=np.zeros(G+1,np.float32);v[ids]=w/w.sum();before[row]=v
                return before[row] if row in before else np.r_[np.float32(0),self.base.before[row-1]]
            changed=np.r_[True,np.any(rank[1:,:self.k2]!=self.oldrank[:,:self.k2],axis=1)]|np.any(affected[rank[:,:self.k2]],axis=1)
            qv=np.stack([vbefore(j) for j in rank[0,:self.k2]]).mean(axis=0)
            nz=np.flatnonzero(qv);gnz=nz[nz>0]
            # Only these candidates need their final expanded reciprocal vector.
            base_values=self.base.after.take(pool,gnz-1)
            tmin=np.minimum(base_values,qv[gnz][None,:]).sum(axis=1)
            for offset,row in enumerate(pool+1):
                if changed[row]:
                    v=np.stack([vbefore(j) for j in rank[row,:self.k2]]).mean(axis=0)
                    tmin[offset]=np.minimum(v[nz],qv[nz]).sum()
            value=np.full(G,np.inf,np.float32)
            value[pool]=(1.-tmin/(2.-tmin))*(1-self.lam)+d[pool]*self.lam;out.append(value)
        return np.asarray(out,np.float32)
