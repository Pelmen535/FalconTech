"""Explicit October A math: immutable gallery context, unchanged learned parameters.

No class/module replacement, query collection, labels, or inference-time fitting.
The numeric operation order matches the frozen current-query-v1 implementation.
"""
from pathlib import Path
import hashlib
import importlib.util
import json
import threading

import joblib
import numpy as np

from .october_graph import gallery_graph_class, OctoberCurrentQueryGraph as CurrentQueryGraph
from vreid.rerank import frame_block_mask


def normalize(value):
    return (value / np.maximum(np.linalg.norm(value, axis=1, keepdims=True), 1e-12)).astype(np.float32)


def transform(pack, geometry):
    x = (pack['emb'].astype(np.float64) - geometry['mean']) @ geometry['transform']
    return {**pack, 'emb': (x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)).astype(np.float32)}


def load_arrays(path):
    with np.load(path, allow_pickle=False) as source:
        result = {name: source[name].copy(order="K") for name in source.files}
    for array in result.values():
        array.setflags(write=False)
    return result


class OctoberMath:
    def __init__(self, root, *, context_identity=None):
        self.root = Path(root).resolve()
        self._gallery_lock=threading.RLock()
        if context_identity is None:
            paths=[self.root/'release_current/runtime-manifest.json',self.root/'var/october-v14-compatibility-20261019/runtime-manifest.json']
            selected=next((p for p in paths if p.is_file()),None)
            spec=json.loads(selected.read_text(encoding='utf-8')) if selected else {}
            context_identity={'identity':spec.get('identity'),'artifacts':spec.get('files'),'root':str(self.root)}
        self.ann_context=hashlib.sha256(json.dumps(context_identity,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        self.ann_snapshot=None
        self.source = load_arrays(self.root / 'var/quality-margin-20261007/extra_views-transform.npz')
        self.auxiliary = load_arrays(self.root / 'var/competitive-margin-20261007/pca512-transform.npz')
        self.night = load_arrays(self.root / 'var/nive-night-transfer-20261009/night_soft-transform.npz')
        self.domain = joblib.load(self.root / 'var/nive-night-transfer-20261009/domain.joblib')
        self.source_head = joblib.load(self.root / 'var/quality-margin-20261007/extra-refusal-multinomial_hgb20.joblib')
        self.night_head = joblib.load(self.root / 'var/nive-night-transfer-20261009/night-head.joblib')
        self.base = json.loads((self.root / 'var/quality-margin-20261007/candidate-extra.json').read_text(encoding='utf-8'))
        self.config = json.loads((self.root / 'var/three-directions-20261009/fusion-selection.json').read_text(encoding='utf-8'))['config']
        if self.config['fusion'] != .5 or self.config['recipe']['kind'] != 'retune_old':
            raise ValueError('Unsupported October math recipe')
        self.boundary = self.base['threshold']
        graph_path = self.root / 'var/competitive-margin-20261007/fast_rerank.py'
        spec = importlib.util.spec_from_file_location('_falcon_october_gallery_graph', graph_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.graph_class = gallery_graph_class(module.GalleryGraph)
        import torch
        from torch import nn
        class Residual(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = nn.Sequential(nn.Linear(2048, 256), nn.LayerNorm(256), nn.GELU(), nn.Linear(256, 512))
            def forward(self, value, base):
                return torch.nn.functional.normalize(base + self.net(value), dim=1)
        checkpoint = torch.load(self.root / 'var/three-directions-20261009/specialist-cnn.pt', map_location='cpu', weights_only=True, mmap=True)
        self.cnn_head = Residual()
        self.cnn_head.load_state_dict(checkpoint['head'])
        self.cnn_head.eval()
        self.torch = torch
        from .ann import ShortlistIndex
        self.ann = ShortlistIndex()
        self.generation = None
        self.gallery = None
        self.registrations = 0
        self.hits = 0
        from vreid.refusal_policy_vric_road import VricRoadRefusalPolicy
        from vreid.shared_refusal import SharedVricRoadPolicy
        self.refusal = SharedVricRoadPolicy(VricRoadRefusalPolicy.load(self.root), rescue_guard=True)

    def teacher(self, raw):
        result = transform({'emb': raw}, self.source)['emb']
        mask = self.domain.predict_proba(result)[:, 1] >= .95
        if mask.any():
            result[mask] = transform({'emb': raw[mask]}, self.night)['emb']
        return result

    @staticmethod
    def fingerprint(gallery):
        h = hashlib.sha256()
        for key in ('keys', 'emb', 'specialist'):
            value = np.ascontiguousarray(gallery[key])
            h.update(key.encode())
            h.update(str(value.dtype).encode())
            h.update(str(value.shape).encode())
            h.update(value.tobytes())
        return h.hexdigest()

    def mapped(self, pack, cnn=False):
        base = transform(pack, self.source)
        night = transform(pack, self.night)
        mask = self.domain.predict_proba(base['emb'])[:, 1] >= .95
        value = np.where(mask[:, None], night['emb'], base['emb'])
        if cnn:
            with self.torch.inference_mode():
                head = self.cnn_head(self.torch.tensor(np.asarray(pack['specialist'], dtype=np.float32)),
                                     self.torch.tensor(value.copy())).numpy()
            delta = np.zeros_like(value)
            delta += 1.0 * (head - value)
            value = normalize(value + delta)
        return {**base, 'emb': value}, mask

    def adaptation(self, gallery):
        cfg = self.base['gallery_adaptation']
        if cfg['center'] == 0 and cfg['power'] == 0:
            return None, None, gallery
        center = gallery['emb'].mean(0) * cfg['center']
        value = gallery['emb'].astype(np.float64) - center
        matrix = None
        if cfg['power']:
            eigenvalues, vectors = np.linalg.eigh(value.T @ value / len(value))
            scale = np.maximum(eigenvalues, .1 * eigenvalues.mean()) ** (-cfg['power'])
            scale /= np.median(scale)
            matrix = (vectors * scale[None, :]) @ vectors.T
            value = value @ matrix
        return center, matrix, {**gallery, 'emb': normalize(value)}

    @staticmethod
    def adapt_query(query, state):
        center, matrix, _ = state
        if center is None:
            return query
        value = query['emb'].astype(np.float64) - center
        if matrix is not None:
            value = value @ matrix
        return {**query, 'emb': normalize(value)}

    def register(self,gallery,*,observation_generation=None):
        with self._gallery_lock:
            keys=np.asarray(gallery['keys'])
            if keys.ndim!=1 or len(keys)<10 or len(set(keys.tolist()))!=len(keys):
                raise ValueError('Need at least ten unique gallery observation IDs')
            for field in ('emb','specialist'):
                values=np.asarray(gallery[field])
                if values.ndim!=2 or len(values)!=len(keys) or not values.shape[1] or not np.isfinite(values).all():
                    raise ValueError('Finite gallery descriptors required before publication')
            # Static k1=3 requires three other-frame observations per row.
            # Reject unsupported geometry before building or publishing graphs;
            # never silently change k1, frame masking, or ranking.
            blocked=frame_block_mask(keys,keys)
            if np.any((~blocked).sum(axis=1)<3):
                raise ValueError('Unsupported gallery geometry: each observation requires at least three observations from other frames')
            generation=self.fingerprint(gallery)
            if generation==self.generation:
                self.hits+=1
                if observation_generation is not None and observation_generation!=self.ann_snapshot.generation:
                    self.ann_snapshot=self.ann_snapshot.rebind(observation_generation);self.ann.clear()
                return generation
            # Build all graphs/densities locally. Exceptions never replace the
            # last valid gallery or clear its usable index.
            shadow=object.__new__(type(self));shadow.__dict__=dict(self.__dict__)
            value=shadow._register_build(gallery,generation,observation_generation)
            self.ann.clear();self.__dict__.update(shadow.__dict__)
            return value

    def _register_build(self,gallery,generation,observation_generation):
        if generation == self.generation:
            self.hits += 1
            return generation
        if len(gallery['keys']) < 10 or len(set(gallery['keys'].tolist())) != len(gallery['keys']):
            raise ValueError('Need at least ten unique gallery observation IDs')
        detached = {}
        for key, value in gallery.items():
            detached[key] = np.array(value, copy=True)
            detached[key].setflags(write=False)
        self.gallery = detached
        self.block = frame_block_mask(detached['keys'], detached['keys'])
        self.branches = []
        self.gallery_night = None
        for cnn in (False, True):
            mapped, night = self.mapped(detached, cnn)
            state = self.adaptation(mapped)
            adapted = state[2]
            graph = self.graph_class(adapted['emb'], g_mask=self.block)
            self.branches.append((state, graph))
            self.gallery_night = night
        self.aux_gallery = transform(detached, self.auxiliary)
        self.aux_graph = self.graph_class(self.aux_gallery['emb'], g_mask=self.block)
        g0, g1 = [state[2]['emb'] for state, _ in self.branches]
        joint = (np.concatenate([g0, g1], axis=1) / np.sqrt(2.)).astype(np.float32)
        from .ann import PublishedGallery
        self.ann_snapshot=PublishedGallery(detached['keys'].tolist(),joint,
            generation=observation_generation or generation,identity=self.ann_context)
        joint=self.ann_snapshot.vectors
        self.joint_gallery = joint
        self.current_graph = CurrentQueryGraph(joint, self.graph_class, g_mask=self.block)
        # All densities depend on this immutable gallery, never on a query.
        # Cache row means rather than retaining additional quadratic matrices.
        self.branch_density = [self.density(state[2]['emb']) for state, _ in self.branches]
        self.aux_density = self.density(self.aux_gallery['emb'])
        density_matrix = .5 * (g0 @ g0.T) + .5 * (g1 @ g1.T)
        np.fill_diagonal(density_matrix, -np.inf)
        self.joint_density = np.partition(density_matrix, -5, axis=1)[:, -5:].mean(1)
        self.generation = generation
        self.registrations += 1
        return generation

    @staticmethod
    def density(gallery):
        matrix = gallery @ gallery.T
        np.fill_diagonal(matrix, -np.inf)
        return np.partition(matrix, -5, axis=1)[:, -5:].mean(1)

    @staticmethod
    def features(raw, rank, gallery, top=None, density_matrix=None, density_values=None):
        order = np.argsort(-rank, axis=1, kind='stable')
        top = order[:, 0] if top is None else top
        ii = np.arange(len(top))
        confidence = raw[ii, top]
        top5 = np.sort(raw, axis=1)[:, -5:]
        other = raw.copy()
        other[ii, top] = -np.inf
        if density_values is None:
            if density_matrix is None:
                density_matrix = gallery @ gallery.T
                np.fill_diagonal(density_matrix, -np.inf)
            density_values = np.partition(density_matrix, -5, axis=1)[:, -5:].mean(1)
        density = density_values[top]
        features = np.column_stack([confidence, confidence - np.max(other, 1), top5[:, -1] - top5[:, -2],
                                    top5.mean(1), top5.std(1), rank[ii, top] - rank[ii, order[:, 1]],
                                    density, confidence - density]).astype(np.float64)
        return top, features

    def infer(self, query, gallery, graph, block, density=None, pools=None):
        raw = np.stack([gallery['emb'] @ row for row in query['emb']])
        raw[block] = -np.inf
        rank = -graph.distances(query['emb'], block, pools=pools)
        rank[block] = -np.inf
        top, features = self.features(raw, rank, gallery['emb'], density_values=density)
        return {'raw': raw, 'rank': rank, 'top': top, 'X': features}

    def candidate_pools(self,joint,block,size):
        info={'search_mode':'exact','shortlist_seconds':None,'notes':[]}
        if size is None:return None,info
        from .ann import AnnUnavailable
        selected=[];seconds=0.
        try:
            for i,row in enumerate(joint):
                found,elapsed=self.ann.shortlist_published(self.ann_snapshot,row,
                                               int(size)+int(block[i].sum()))
                found=found[~block[i,found]]
                if len(found)<10:raise AnnUnavailable('fewer than ten eligible observations')
                selected.append(found);seconds+=elapsed
        except AnnUnavailable as reason:
            info['notes'].append('Exact fallback: '+str(reason));return None,info
        info.update(search_mode='ann',shortlist_seconds=seconds,candidate_pool_sizes=[len(p) for p in selected])
        info['notes'].append('HNSW shortlist precedes candidate-only query-insertion reciprocity. Full-gallery static graph distances, transforms, neighbours, raw similarities and confidence statistics are retained.')
        return selected,info

    def route(self,query,gallery=None,ann_size=None):
        with self._gallery_lock:
            return self._route(query,gallery,ann_size)

    def _route(self, query, gallery=None, ann_size=None):
        if gallery is not None:
            self.register(gallery)
        if self.gallery is None:
            raise ValueError('Gallery is not registered')
        gallery = self.gallery
        block = frame_block_mask(query['keys'], gallery['keys'])
        if np.any((~block).sum(axis=1)<10):
            raise ValueError('Query requires at least ten eligible gallery observations from other frames')
        prepared=[]
        for cnn,(state,graph) in zip((False,True),self.branches):
            mapped,night=self.mapped(query,cnn)
            prepared.append((self.adapt_query(mapped,state),night))
        joint=(np.concatenate([p[0]['emb'] for p in prepared],axis=1)/np.sqrt(2.)).astype(np.float32)
        pools,search_info=self.candidate_pools(joint,block,ann_size)
        branches=[]
        for (adapted,night),(state,graph),density in zip(prepared,self.branches,self.branch_density):
            # Preserve the inexpensive static-graph distances as full-gallery
            # context. ANN restricts the costly query-insertion graph below.
            result=self.infer(adapted,state[2],graph,block,density)
            result.update(query=adapted['emb'],gallery=state[2]['emb'],query_night=night,
                          gallery_night=self.gallery_night)
            branches.append(result)
        r0, r1 = branches
        raw = .5 * r0['raw'] + .5 * r1['raw']
        base = .5 * r0['rank'] + .5 * r1['rank']
        joint = (np.concatenate([r0['query'], r1['query']], axis=1) / np.sqrt(2.)).astype(np.float32)
        new = -self.current_graph.distances(joint, block, pools=pools)
        rank = base.copy()
        valid = np.isfinite(raw)
        rank[valid] = .75 * base[valid] + .25 * new[valid]
        rank[~valid] = -np.inf
        if pools is not None:
            # Jaccard distances are nonnegative. Their omitted terms therefore
            # give a conservative lower bound on each final distance. Expand
            # the HNSW pool whenever an omitted row could enter the top ten.
            # Full-gallery statistics above are unchanged, and ties are kept.
            lower=-.75*base
            from vreid.rerank import _l2n
            qn=_l2n(joint.astype(np.float32))
            lower+=.25*self.current_graph.lam*np.stack([2.-2.*(self.current_graph.g@row) for row in qn])
            expanded=[]
            for i,pool in enumerate(pools):
                cutoff=np.partition(-rank[i,pool],9)[9]
                possible=np.flatnonzero((lower[i]-1e-6<=cutoff)&~block[i])
                expanded.append(np.union1d(pool,possible))
            if any(len(a)>len(b) for a,b in zip(expanded,pools)):
                new=-self.current_graph.distances(joint,block,pools=expanded)
                rank[valid]=.75*base[valid]+.25*new[valid]
                rank[~valid]=-np.inf
                info=search_info
                info.update(search_mode='ann',shortlist_seconds=search_info['shortlist_seconds'],
                            initial_candidate_pool_sizes=[len(p) for p in pools],
                            candidate_pool_sizes=[len(p) for p in expanded],
                            exact_completion=all(len(p)==int((~mask).sum()) for p,mask in zip(expanded,block)))
                info['notes']=search_info['notes']+['Static-graph distances remain full context; HNSW restricts query-insertion reciprocity. Conservative distance-bound completion preserves exact top ten; pool may expand to the full eligible gallery. No speed advantage is implied.']
                if info['exact_completion']:
                    info['search_mode']='exact'
                    info['notes'].append('Exact fallback: the conservative certificate requires every eligible observation.')
        top, features = self.features(raw, rank, r1['gallery'], density_values=self.joint_density)
        result = {**r1, 'raw': raw, 'rank': rank, 'top': top, 'X': features, 'search_info':search_info}
        auxiliary_query = transform(query, self.auxiliary)
        auxiliary = self.infer(auxiliary_query, self.aux_gallery, self.aux_graph, block, self.aux_density)
        ii = np.arange(len(top))
        similarity = auxiliary['raw'][ii, top]
        best = np.max(auxiliary['raw'], axis=1)
        density = self.aux_density[top]
        result['X20'] = np.column_stack([features, auxiliary['X'], top == auxiliary['top'], similarity,
                                        similarity - best, similarity - density])
        result.update(_query_features={k: query[k] for k in ('emb', 'specialist')},
                      _gallery_features={k: gallery[k] for k in ('emb', 'specialist')},
                      _gallery_keys=gallery['keys'])
        return result

    def confidence(self, result, *, public_only=False):
        def odds(head,features):
            probability = head.predict_proba(features)
            if list(head.classes_) != [0, 1, 2]:
                raise ValueError('Unexpected legacy confidence classes')
            return probability[:, 2] / np.maximum(probability[:, 0] + probability[:, 2], 1e-15)
        def logit(value):
            value = np.clip(value, 1e-12, 1-1e-12)
            return np.log(value / (1-value))
        def adjusted(probability, cut):
            value = 1 / (1 + np.exp(-np.clip(logit(probability)-logit(cut)+logit(self.boundary), -50, 50)))
            return np.where(probability >= cut, np.maximum(value, self.boundary),
                            np.minimum(value, np.nextafter(self.boundary, -np.inf)))
        recipe = self.config['recipe']
        # Validate both pinned heads, then evaluate only each query's fixed route.
        if list(self.night_head.classes_) != [0,1,2] or list(self.source_head.classes_) != [0,1,2]:
            raise ValueError('Unexpected legacy confidence classes')
        night=np.asarray(result['query_night'],dtype=bool)
        legacy=np.empty(len(night),dtype=float)
        absent=np.empty(len(night),dtype=float)
        for mask,head,cut in [(night,self.night_head,recipe['night_threshold']),
                              (~night,self.source_head,recipe['source_threshold'])]:
            if mask.any():
                probability=head.predict_proba(result['X20'][mask])
                absent[mask]=probability[:,0]
                ratio=probability[:,2]/np.maximum(probability[:,0]+probability[:,2],1e-15)
                legacy[mask]=adjusted(ratio,cut)
        if public_only and len(legacy)==1 and legacy[0]>=self.boundary:
            from vreid.shared_refusal import SharedVricRoadPolicy
            from vreid.refusal_policy_vric_road import VricRoadRefusalPolicy
            # This exact frozen chain only supplements declines. None of its
            # later heads may revoke or modify an already accepted legacy score.
            # Full diagnostics still use confidence(..., public_only=False).
            if type(self.refusal) is SharedVricRoadPolicy and type(self.refusal.original) is VricRoadRefusalPolicy:
                return {'confidence':legacy,'accepted':np.ones(1,dtype=bool)}
        return self.refusal.predict({**result,'legacy_absent_probability':absent}, legacy, self.boundary)

    def confidence_public(self,result):
        return self.confidence(result,public_only=True)
