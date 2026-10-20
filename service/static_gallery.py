"""Bounded gallery-only memoization of the existing numerical implementation.

No query embeddings, query transforms or query ranking results are retained.
Used only inside the isolated expanded-profile worker, which serializes RPCs.
"""
from collections import OrderedDict
import hashlib
import json
import numpy as np


def array_token(value):
    a = np.asarray(value)
    return a.shape, a.dtype.str, hashlib.sha256(a.tobytes(order='C')).digest()


def gallery_signature(vectors, ids):
    a = np.asarray(vectors, dtype='<f4')
    if (a.ndim!=2 or a.shape!=(len(ids),512) or len(ids)<10 or not np.isfinite(a).all()
            or any(not isinstance(key,str) or not key.strip() for key in ids)
            or len(set(ids))!=len(ids)):
        raise ValueError('Need at least 10 unique IDs and finite 512D gallery projections')
    digest = hashlib.sha256(a.tobytes())
    digest.update(json.dumps(list(ids),ensure_ascii=False,separators=(',',':')).encode('utf-8'))
    return digest.hexdigest()


class RegisteredGallery:
    """One immutable gallery snapshot; raw-file changes invalidate numerical caches."""
    def __init__(self, cache_dir, token, pack, on_change):
        self.cache_dir, self.token, self.pack, self.on_change = cache_dir, token, pack, on_change
        self.state = None
        self.registrations = self.refreshes = self.hits = 0

    @staticmethod
    def stat_files(paths):
        try:
            return tuple((p.stat().st_size,p.stat().st_mtime_ns) for p in paths)
        except FileNotFoundError as exc:
            raise ValueError('Raw gallery features missing; rebuild this gallery') from exc

    def register(self, vectors, ids, expected):
        signature = gallery_signature(vectors,ids)
        if signature!=expected:
            raise ValueError('Gallery registration signature mismatch')
        a = np.asarray(vectors,dtype='<f4').copy()
        keys = list(ids)
        paths = [self.cache_dir/(self.token(v)+'.npz') for v in a]
        stats = self.stat_files(paths)
        packed = self.pack(a,keys)
        self.on_change()
        self.state = dict(signature=signature,vectors=a,ids=keys,paths=paths,stats=stats,packed=packed)
        self.registrations += 1
        return signature

    def get(self, signature):
        state = self.state
        if state is None or signature!=state['signature']:
            raise ValueError('Gallery snapshot unavailable; register current gallery')
        stats = self.stat_files(state['paths'])
        if stats!=state['stats']:
            packed = self.pack(state['vectors'],state['ids'])
            self.on_change()
            state.update(packed=packed,stats=stats)
            self.refreshes += 1
        else:
            self.hits += 1
        return state['packed'],state['ids']


class StaticGalleryMemo:
    def __init__(self):
        import evaluate as single
        import fusion_evaluate as fusion
        import fast_rerank
        from vreid import current_query_profile
        self.single, self.profile = single, current_query_profile
        self.original_route = current_query_profile.route
        self.gallery = None
        self.transforms = {}
        self.adaptations = OrderedDict()
        self.graphs = OrderedDict()
        self.current_graphs = OrderedDict()
        self.gallery_mask = None
        self.hits = {'transform':0, 'adapt':0, 'graph':0, 'current_graph':0, 'gallery_mask':0}
        self.restores = []

        def replace(obj, name, value):
            self.restores.append((obj, name, getattr(obj, name)))
            setattr(obj, name, value)

        transform = single.bc.transform

        def cached_transform(rows, matrix, mean):
            if rows is not self.gallery:
                return transform(rows, matrix, mean)
            key = id(matrix), id(mean)
            if key in self.transforms:
                self.hits['transform'] += 1
                # Keep input arrays alive as well: object IDs cannot be reused.
                return dict(self.transforms[key][2])
            value = transform(rows, matrix, mean)
            self.transforms[key] = matrix, mean, value
            # Routing replaces mapped['emb'] for the CNN branch. Share arrays,
            # never the mutable result dictionary between branch evaluations.
            return dict(value)

        replace(single.bc, 'transform', cached_transform)
        assert single.bc is fusion.bc
        mask_module = single.br.rank_infer.__globals__['s']
        original_mask = mask_module.frame_block_mask

        def cached_mask(left, right):
            if self.gallery is not None and left is right and left is self.gallery['keys']:
                if self.gallery_mask is None:
                    self.gallery_mask = original_mask(left,right)
                    self.gallery_mask.setflags(write=False)
                else:
                    self.hits['gallery_mask'] += 1
                return self.gallery_mask
            return original_mask(left,right)

        replace(mask_module, 'frame_block_mask', cached_mask)
        replace(current_query_profile, 'frame_block_mask', cached_mask)
        original_adapt = single.br.adapt_gallery

        def cached_adapt(q, g, cfg):
            if cfg['center']==0 and cfg['power']==0:
                return original_adapt(q, g, cfg)
            from search import norm
            key = array_token(g['emb']), float(cfg['center']), float(cfg['power'])
            if key in self.adaptations:
                center, matrix, gallery = self.adaptations[key]
                self.hits['adapt'] += 1
            else:
                # Identical order/dtypes/operations to model.adapt_gallery.
                center = g['emb'].mean(0)*cfg['center']
                ge = g['emb'].astype(np.float64)-center
                matrix = None
                if cfg['power']:
                    eig, u = np.linalg.eigh(ge.T@ge/len(ge))
                    scale = np.maximum(eig,.1*eig.mean())**(-cfg['power'])
                    scale /= np.median(scale)
                    matrix = (u*scale[None,:])@u.T
                    ge = ge@matrix
                gallery = norm(ge)
                self.adaptations[key] = center, matrix, gallery
                while len(self.adaptations)>4:
                    self.adaptations.popitem(last=False)
            qe = q['emb'].astype(np.float64)-center
            if matrix is not None:
                qe = qe@matrix
            return {**q,'emb':norm(qe)}, {**g,'emb':gallery}

        replace(single.br, 'adapt_gallery', cached_adapt)
        graph_class = fast_rerank.GalleryGraph

        def cached_graph(g, k1=3, k2=2, lam=.2, g_mask=None):
            key = array_token(g), k1, k2, lam, None if g_mask is None else array_token(g_mask)
            if key in self.graphs:
                self.hits['graph'] += 1
                self.graphs.move_to_end(key)
                return self.graphs[key]
            value = graph_class(g,k1=k1,k2=k2,lam=lam,g_mask=g_mask)
            self.graphs[key] = value
            while len(self.graphs)>8:
                self.graphs.popitem(last=False)
            return value

        replace(fast_rerank, 'GalleryGraph', cached_graph)
        infer_globals = single.br.rank_infer.__globals__
        self.infer_globals, self.old_infer_graph = infer_globals, infer_globals['GalleryGraph']
        assert self.old_infer_graph is graph_class
        infer_globals['GalleryGraph'] = cached_graph
        current_class = current_query_profile.CurrentQueryGraph

        def cached_current(g, graph_class, g_mask=None, k1=3, k2=2, lam=.2):
            key = array_token(g), k1, k2, lam, None if g_mask is None else array_token(g_mask)
            if key in self.current_graphs:
                self.hits['current_graph'] += 1
                return self.current_graphs[key]
            value = current_class(g,graph_class,g_mask=g_mask,k1=k1,k2=k2,lam=lam)
            self.current_graphs[key] = value
            while len(self.current_graphs)>2:
                self.current_graphs.popitem(last=False)
            return value

        replace(current_query_profile, 'CurrentQueryGraph', cached_current)

    def clear(self):
        self.gallery = None
        self.transforms.clear()
        self.adaptations.clear()
        self.graphs.clear()
        self.current_graphs.clear()
        self.gallery_mask = None

    def route(self, q, g, config):
        if self.gallery is not g:
            self.clear()
            self.gallery = g
        return self.original_route(q,g,config)

    def close(self):
        self.clear()
        self.infer_globals['GalleryGraph'] = self.old_infer_graph
        for obj, name, previous in reversed(self.restores):
            setattr(obj, name, previous)
