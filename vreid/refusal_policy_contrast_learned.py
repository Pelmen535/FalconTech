"""FIT-trained, CAL-frozen verifier of selected and alternative identity evidence."""
from pathlib import Path
import json
import joblib
import numpy as np
from .raw_refusal import raw_features
from .refusal_policy_domain import DomainRefusalPolicy
from .refusal_policy_bagged import BaggedRefusalPolicy, digest
from .refusal_policy_contrast import alternative, relative_odds
from .partial_hgb import PartialHGB

PROFILE = 'contrast-learned-v1'


def verified_selection(root):
    root = Path(root)
    selected = json.loads((root/'var/confidence-contrast-20261017/learned/runtime-selection.json').read_text(encoding='utf-8'))
    if (selected['profile'] != PROFILE or selected['rerank_profile'] != 'current-query-v1'
            or selected['scope'] != 'source' or selected['columns'] != 102):
        raise ValueError('Learned contrast identity mismatch')
    if selected['status'] not in ('Q_DOMINANT_PENDING_NATIVE', 'READY_LOCAL_WINDOWS'):
        raise ValueError('Learned contrast has not passed quality validation')
    for name, expected in selected['files'].items():
        if digest(root/name) != expected:
            raise ValueError('Learned contrast checksum mismatch: '+name)
    if not 0 < selected['threshold'] < 1:
        raise ValueError('Invalid learned contrast threshold')
    return selected


def contrast_features(selected_features, selected_scores, alternative_scores):
    result = np.column_stack([selected_features, selected_scores, alternative_scores,
                              relative_odds(selected_scores, alternative_scores)])
    if result.shape != (len(selected_features), 102) or not np.isfinite(result).all():
        raise ValueError('Invalid learned contrast features')
    return result


class LearnedContrastRefusalPolicy:
    def __init__(self, baseline, model, threshold):
        if list(model.classes_) != [0, 1] or model.n_features_in_ != 102 or not 0 < threshold < 1:
            raise ValueError('Invalid learned contrast model')
        self.baseline = baseline
        self.model = model
        self.threshold = float(threshold)
        self.partial = PartialHGB(model)

    @classmethod
    def load(cls, root):
        selected = verified_selection(root)
        return cls(DomainRefusalPolicy.load(root), joblib.load(Path(root)/selected['head']), selected['threshold'])

    def head_scores(self, features):
        return np.column_stack([self.baseline.baseline.baseline.model.predict_proba(features)[:, 1],
                                self.baseline.baseline.score_features(features),
                                self.baseline.score_features(features)])

    def supplement(self, previous, score, night, boundary):
        return BaggedRefusalPolicy.supplement(self, previous, score, night, boundary)

    def predict(self, result, legacy_confidence, boundary):
        features = raw_features(result)
        night = np.asarray(result['query_night'], bool)
        previous = self.baseline.predict_features(features, legacy_confidence, night, boundary)
        rows = np.flatnonzero(~previous['accepted'] & ~night)
        if not len(rows):
            return previous
        selected = self.head_scores(features[rows])
        known = np.column_stack([features[rows], selected])
        # A margin covers probability rounding at the threshold. Rows that
        # cannot possibly pass keep their original confidence without raw93
        # extraction for an alternate anchor.
        possible = self.partial.upper_probability(known) >= self.threshold-1e-12
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
        other = self.head_scores(raw_features(sub))
        # Preserve the original selected descriptors and every prior confidence.
        X = contrast_features(features[rows], selected, other)
        score = np.zeros(len(features), dtype=float)
        score[rows] = np.where(valid, self.model.predict_proba(X)[:, 1], 0.)
        return self.supplement(previous, score, night, boundary)
