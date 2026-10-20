"""Optional FIT-trained bagged supplement; preserve all prior decisions."""
from pathlib import Path
import hashlib
import json
import joblib
import numpy as np
from .raw_refusal import raw_features
from .refusal_policy_raw import GeneralRefusalPolicy
from .binary_forest import BinaryForest

PROFILE = 'bagged-source-v1'


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def verified_selection(root):
    root = Path(root)
    path = root/'var/confidence-repair-20261017/runtime-selection.json'
    selected = json.loads(path.read_text(encoding='utf-8'))
    if selected['profile'] != PROFILE or selected['rerank_profile'] != 'current-query-v1':
        raise ValueError('Bagged profile identity mismatch')
    if selected['status'] not in ('PASS_FIXED_FEATURE_PENDING_NATIVE', 'READY_LOCAL_WINDOWS'):
        raise ValueError('Bagged profile has not passed validation')
    for name, expected in selected['files'].items():
        if digest(root/name) != expected:
            raise ValueError('Bagged profile checksum mismatch: '+name)
    if not 0 < selected['threshold'] < 1:
        raise ValueError('Invalid bagged threshold')
    return selected


class BaggedRefusalPolicy:
    def __init__(self, baseline, model, threshold):
        if list(model.classes_) != [0, 1] or not 0 < threshold < 1:
            raise ValueError('Invalid bagged model')
        self.baseline = baseline
        self.model = model
        self.forest = BinaryForest(model)
        self.threshold = float(threshold)

    @classmethod
    def load(cls, root):
        root = Path(root)
        selected = verified_selection(root)
        return cls(GeneralRefusalPolicy.load(root/'var/general-refusal-final-20261017/selection.json'),
                   joblib.load(root/selected['head']), selected['threshold'])

    def predict(self, result, legacy_confidence, boundary):
        features = raw_features(result)
        old = self.baseline_from_features(features, legacy_confidence, result['query_night'], boundary)
        score = self.score_features(features)
        return self.supplement(old, score, result['query_night'], boundary)

    def baseline_from_features(self, features, legacy_confidence, query_night, boundary):
        """Same frozen raw-policy math, sharing its expensive features with the supplement.

        Exact parity against the unchanged original is required by real-cache acceptance.
        """
        legacy = np.asarray(legacy_confidence, dtype=float)
        night = np.asarray(query_night, dtype=bool)
        if legacy.shape != (len(features),) or night.shape != legacy.shape or not np.isfinite(legacy).all() or not 0 < boundary < 1:
            raise ValueError('Invalid legacy confidence')
        score = self.baseline.model.predict_proba(features)[:, 1]
        if not np.isfinite(score).all() or np.any((score < 0) | (score > 1)):
            raise ValueError('Invalid local score')
        accepted = (score >= self.baseline.threshold) & ~night
        def logit(p):
            p = np.clip(p, 1e-12, 1-1e-12)
            return np.log(p/(1-p))
        shifted = 1/(1+np.exp(-np.clip(logit(score)-logit(self.baseline.threshold)+logit(boundary), -50, 50)))
        shifted = np.where(accepted, np.maximum(shifted, boundary),
                           np.minimum(shifted, np.nextafter(boundary, -np.inf)))
        confidence = np.where(accepted & (legacy < boundary), shifted, legacy)
        final = (legacy >= boundary) | accepted
        if not np.array_equal(final, confidence >= boundary):
            raise ValueError('Frozen baseline confidence disagreement')
        return {'raw_score':score, 'confidence':confidence, 'accepted':final}

    def score_features(self, features):
        return self.forest.predict_positive(features)

    def supplement(self, old, score, query_night, boundary):
        score = np.asarray(score, dtype=float)
        night = np.asarray(query_night, dtype=bool)
        if score.shape != old['confidence'].shape or night.shape != score.shape or not 0 < boundary < 1:
            raise ValueError('Invalid bagged supplement inputs')
        if not np.isfinite(score).all() or np.any((score < 0) | (score > 1)):
            raise ValueError('Invalid bagged score')
        additional = (score >= self.threshold) & ~night
        additional &= ~old['accepted']
        def logit(p):
            p = np.clip(p, 1e-12, 1-1e-12)
            return np.log(p/(1-p))
        shifted = 1/(1+np.exp(-np.clip(logit(score)-logit(self.threshold)+logit(boundary), -50, 50)))
        confidence = np.where(additional, np.maximum(shifted, boundary), old['confidence'])
        accepted = old['accepted'] | additional
        if not np.array_equal(accepted, confidence >= boundary):
            raise ValueError('Bagged confidence/acceptance disagreement')
        return {**old, 'bagged_score': score, 'confidence': confidence, 'accepted': accepted}
