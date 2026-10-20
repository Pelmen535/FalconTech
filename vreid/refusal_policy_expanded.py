"""Expanded-source verifier, retaining current ranking and all prior accepts."""
from pathlib import Path
import json
import joblib
import numpy as np
from .raw_refusal import raw_features
from .refusal_policy_contrast import alternative
from .refusal_policy_contrast_learned import LearnedContrastRefusalPolicy, contrast_features
from .refusal_policy_bagged import BaggedRefusalPolicy, digest

PROFILE = 'expanded-source-v1'


def verified_selection(root):
    root = Path(root)
    path = root/'var/systematic-ranking-20261017/expanded-runtime-selection.json'
    selected = json.loads(path.read_text(encoding='utf-8'))
    if (selected['profile'] != PROFILE or selected['rerank_profile'] != 'current-query-v1'
            or selected['scope'] != 'all' or selected['columns'] != 93
            or selected['feature_dtype'] != 'float32'):
        raise ValueError('Expanded refusal identity mismatch')
    if selected['status'] not in ('FRESH_ID_CONFIRMED_PENDING_NATIVE', 'READY_LOCAL_WINDOWS_WITH_TRADEOFF'):
        raise ValueError('Expanded refusal is not qualified for native validation')
    for name, expected in selected['files'].items():
        if digest(root/name) != expected:
            raise ValueError('Expanded refusal checksum mismatch: '+name)
    if not 0 < selected['threshold'] < 1:
        raise ValueError('Invalid expanded refusal threshold')
    return selected


class ExpandedRefusalPolicy:
    def __init__(self, baseline, model, threshold):
        if list(model.classes_) != [0, 1] or model.n_features_in_ != 93 or not 0 < threshold < 1:
            raise ValueError('Invalid expanded verifier')
        self.baseline = baseline
        self.model = model
        self.threshold = float(threshold)

    @classmethod
    def load(cls, root):
        selected = verified_selection(root)
        return cls(LearnedContrastRefusalPolicy.load(root), joblib.load(Path(root)/selected['head']), selected['threshold'])

    def previous_features(self, result, features, legacy_confidence, boundary):
        """Same prior algorithm with shared primary raw93; verify against old .predict."""
        base = self.baseline
        night = np.asarray(result['query_night'], bool)
        previous = base.baseline.predict_features(features, legacy_confidence, night, boundary)
        rows = np.flatnonzero(~previous['accepted'] & ~night)
        if not len(rows):
            return previous
        selected = base.head_scores(features[rows])
        known = np.column_stack([features[rows], selected])
        possible = base.partial.upper_probability(known) >= base.threshold-1e-12
        rows = rows[possible]
        selected = selected[possible]
        if not len(rows):
            return previous
        sub = {k: np.asarray(result[k])[rows] for k in ('raw', 'rank', 'query', 'query_night', 'top')}
        sub.update(gallery=result['gallery'],
                   _query_features={k: v[rows] for k, v in result['_query_features'].items()},
                   _gallery_features=result['_gallery_features'])
        indices, valid = alternative(sub, result['_gallery_keys'], width=5)
        sub['top'] = indices
        other = base.head_scores(raw_features(sub))
        X = contrast_features(features[rows], selected, other)
        score = np.zeros(len(features), dtype=float)
        score[rows] = np.where(valid, base.model.predict_proba(X)[:, 1], 0.)
        return base.supplement(previous, score, night, boundary)

    def predict(self, result, legacy_confidence, boundary):
        features = raw_features(result)
        previous = self.previous_features(result, features, legacy_confidence, boundary)
        rows = np.flatnonzero(~previous['accepted'])
        if not len(rows):
            return previous
        score = np.zeros(len(features), dtype=float)
        score[rows] = self.model.predict_proba(features[rows].astype(np.float32))[:, 1]
        # The new CAL-frozen rule covers both source/night routes.
        got = BaggedRefusalPolicy.supplement(self, previous, score, np.zeros(len(features), bool), boundary)
        return {**got, 'expanded_score': score}
