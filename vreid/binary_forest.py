"""Vectorized inference for a finite-input sklearn binary ExtraTrees forest."""
import numpy as np


class BinaryForest:
    def __init__(self, model):
        if list(model.classes_) != [0, 1] or not model.estimators_:
            raise ValueError('Expected fitted binary forest')
        trees = [est.tree_ for est in model.estimators_]
        offsets = np.cumsum([0] + [tree.node_count for tree in trees[:-1]])
        self.roots = offsets.astype(np.int64)
        self.left = np.concatenate([np.where(t.children_left >= 0, t.children_left+off, -1)
                                    for t, off in zip(trees, offsets)])
        self.right = np.concatenate([np.where(t.children_right >= 0, t.children_right+off, -1)
                                     for t, off in zip(trees, offsets)])
        self.feature = np.concatenate([t.feature for t in trees])
        self.threshold = np.concatenate([t.threshold for t in trees])
        values = np.concatenate([t.value[:, 0, :] for t in trees])
        if not np.allclose(values.sum(axis=1), 1., rtol=0, atol=1e-12):
            raise ValueError('Forest artifact does not contain normalized probabilities')
        # Current sklearn returns stored probabilities without another normalization.
        self.positive = values[:, 1]
        self.depth = max(t.max_depth for t in trees)
        self.columns = model.n_features_in_

    def predict_positive(self, features):
        # sklearn's tree predictor converts its input to float32 before splitting.
        X = np.asarray(features, dtype=np.float32)
        if X.ndim != 2 or X.shape[1] != self.columns or not np.isfinite(X).all():
            raise ValueError('Invalid forest features')
        nodes = np.broadcast_to(self.roots, (len(X), len(self.roots))).copy()
        rows = np.arange(len(X))[:, None]
        for _ in range(self.depth):
            field = self.feature[nodes]
            active = field >= 0
            chosen = np.where(X[rows, np.maximum(field, 0)] <= self.threshold[nodes],
                              self.left[nodes], self.right[nodes])
            nodes = np.where(active, chosen, nodes)
        if np.any(self.feature[nodes] >= 0):
            raise ValueError('Forest traversal did not reach leaves')
        # cumsum preserves serial estimator addition order, matching sklearn n_jobs=1.
        return np.cumsum(self.positive[nodes], axis=1)[:, -1]/len(self.roots)
