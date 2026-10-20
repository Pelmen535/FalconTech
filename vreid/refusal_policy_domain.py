"""Organizer-FIT-trained source supplement with an explicit local TNR tradeoff."""
from pathlib import Path
import json
import joblib
from .raw_refusal import raw_features
from .refusal_policy_bagged import BaggedRefusalPolicy, digest

PROFILE = 'domain-source-v1'


def verified_selection(root):
    root = Path(root)
    selected = json.loads((root/'var/confidence-domain-20261017/runtime-selection.json').read_text(encoding='utf-8'))
    if selected['profile'] != PROFILE or selected['rerank_profile'] != 'current-query-v1' or selected['scope'] != 'source':
        raise ValueError('Domain refusal profile identity mismatch')
    if selected['status'] not in ('Q_DOMINANT_PENDING_NATIVE_WITH_TNR_TRADEOFF',
                                  'READY_LOCAL_WINDOWS_Q_GAIN_WITH_TNR_TRADEOFF'):
        raise ValueError('Domain refusal has not passed its quality review')
    for name, expected in selected['files'].items():
        if digest(root/name) != expected:
            raise ValueError('Domain refusal checksum mismatch: '+name)
    if not 0 < selected['threshold'] < 1:
        raise ValueError('Invalid domain refusal threshold')
    return selected


class DomainRefusalPolicy(BaggedRefusalPolicy):
    @classmethod
    def load(cls, root):
        root = Path(root)
        selected = verified_selection(root)
        return cls(BaggedRefusalPolicy.load(root), joblib.load(root/selected['head']), selected['threshold'])

    def predict(self, result, legacy_confidence, boundary):
        features = raw_features(result)
        return self.predict_features(features, legacy_confidence, result['query_night'], boundary)

    def predict_features(self, features, legacy_confidence, query_night, boundary):
        raw = self.baseline.baseline_from_features(features, legacy_confidence, query_night, boundary)
        previous = self.baseline.supplement(raw, self.baseline.score_features(features), query_night, boundary)
        return self.supplement(previous, self.score_features(features), query_night, boundary)
