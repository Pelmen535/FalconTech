"""Fixed conservative check for contradictory final recovery evidence.

Earlier accepted decisions are preserved. Raw similarities are gallery-adapted
cosines, never compared to the public calibrated confidence threshold.
"""
import numpy as np

POLICY = {'name': 'source-contradictory-road-rescue-v2', 'absent_ge': 0.8,
          'road_lt': 0.9, 'raw_cosine_lt': 0.45}

def contradictory_rescue(result, previous, current, road_score):
    absent=np.asarray(result['legacy_absent_probability'],dtype=float)
    top=np.asarray(result['top'],dtype=int)
    raw=np.asarray(result['raw'])[np.arange(len(top)),top]
    return (np.asarray(current['accepted'],dtype=bool)
            & ~np.asarray(previous['accepted'],dtype=bool)
            & ~np.asarray(result['query_night'],dtype=bool)
            & (absent >= POLICY['absent_ge'])
            & (np.asarray(road_score) < POLICY['road_lt'])
            & (raw < POLICY['raw_cosine_lt']))
