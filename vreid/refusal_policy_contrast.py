"""CAL-frozen relative candidate evidence, supplementing domain-source-v1."""
from pathlib import Path
import json
import numpy as np
from .raw_refusal import raw_features
from .refusal_policy_domain import DomainRefusalPolicy
from .refusal_policy_bagged import BaggedRefusalPolicy, digest

PROFILE = 'contrast-source-v1'


def verified_selection(root):
    root = Path(root)
    selected = json.loads((root/'var/confidence-contrast-20261017/runtime-selection.json').read_text(encoding='utf-8'))
    if (selected['profile'] != PROFILE or selected['rerank_profile'] != 'current-query-v1'
            or selected['scope'] != 'source' or selected['config'] != {
                'head': 'rawHGB15', 'cluster': 5, 'family': 'odds', 'selected_min': .1,
                'scope': 'source', 'threshold': selected['threshold']}):
        raise ValueError('Contrast refusal identity mismatch')
    if selected['status'] not in ('Q_DOMINANT_PENDING_NATIVE', 'READY_LOCAL_WINDOWS'):
        raise ValueError('Contrast refusal has not passed quality validation')
    for name, expected in selected['files'].items():
        if digest(root/name) != expected:
            raise ValueError('Contrast refusal checksum mismatch: '+name)
    if not 0 < selected['threshold'] < 1:
        raise ValueError('Invalid contrast threshold')
    return selected


def alternative(result, gallery_keys, width=5):
    """Strongest ranked alternative outside anchor's static CNN neighborhood.

    Stable gallery-key ties, no labels or other queries. Invalid alternatives
    remain explicitly masked, including a gallery smaller than the exclusion.
    """
    raw = np.asarray(result['raw'])
    rank = np.asarray(result['rank'])
    top = np.asarray(result['top'], dtype=np.int64)
    gallery = np.asarray(result['gallery'])
    keys = np.asarray(gallery_keys)
    if (raw.ndim != 2 or raw.shape != rank.shape or len(keys) != raw.shape[1]
            or len(set(keys.tolist())) != len(keys) or not len(keys)):
        raise ValueError('Invalid candidate geometry/keys')
    canonical = np.argsort(keys, kind='stable')
    neighbor = gallery[top] @ gallery.T
    neighbor[np.arange(len(top)), top] = np.inf
    order = canonical[np.argsort(-neighbor[:, canonical], axis=1, kind='stable')]
    allowed = np.isfinite(raw).copy()
    np.put_along_axis(allowed, order[:, :min(width, len(keys))], False, axis=1)
    scores = np.where(allowed, rank, -np.inf)
    valid = np.isfinite(scores).any(axis=1)
    best = canonical[np.argmax(scores[:, canonical], axis=1)]
    return np.where(valid, best, top), valid


def relative_odds(selected, other):
    selected = np.clip(selected, 1e-12, 1-1e-12)
    other = np.clip(other, 1e-12, 1-1e-12)
    delta = np.log(selected/(1-selected))-np.log(other/(1-other))
    return np.clip(1/(1+np.exp(-np.clip(delta, -50, 50))), 1e-12, 1-1e-12)


class ContrastRefusalPolicy:
    def __init__(self, baseline, threshold):
        if not 0 < threshold < 1:
            raise ValueError('Invalid contrast threshold')
        self.baseline = baseline
        self.threshold = float(threshold)

    @classmethod
    def load(cls, root):
        selected = verified_selection(root)
        return cls(DomainRefusalPolicy.load(root), selected['threshold'])

    def supplement(self, previous, selected_score, alternative_score, valid, night, boundary):
        score = relative_odds(selected_score, alternative_score)
        eligible = np.asarray(valid, bool) & (np.asarray(selected_score) >= .1)
        # Reuse the exact confidence mapping of previous profiles, without
        # claiming that this relative evidence is a calibrated probability.
        return BaggedRefusalPolicy.supplement(
            self, previous, np.where(eligible, score, 0.), night, boundary)

    def predict(self, result, legacy_confidence, boundary):
        previous = self.baseline.predict(result, legacy_confidence, boundary)
        selected = previous['raw_score']
        night = np.asarray(result['query_night'], bool)
        rows = np.flatnonzero(~previous['accepted'] & ~night & (selected >= .1))
        if not len(rows):
            return previous
        sub = {k: np.asarray(result[k])[rows] for k in ('raw', 'rank', 'query', 'query_night', 'top')}
        sub.update(gallery=result['gallery'],
                   _query_features={k: v[rows] for k, v in result['_query_features'].items()},
                   _gallery_features=result['_gallery_features'])
        indices, valid = alternative(sub, result['_gallery_keys'])
        sub['top'] = indices
        other = self.baseline.baseline.baseline.model.predict_proba(raw_features(sub))[:, 1]
        scores = np.zeros(len(selected), dtype=float)
        scores[rows] = other
        eligible = np.zeros(len(selected), dtype=bool)
        eligible[rows] = valid
        return self.supplement(previous, selected, scores, eligible, night, boundary)
