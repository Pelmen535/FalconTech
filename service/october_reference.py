"""Versioned raw-pair references, with no transport or storage projection."""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import os
import tempfile
import re
import numpy as np
from .october_features import RawFeatureStore

SCHEMA = 'raw-pair-le-v2'
_HEX_REF = re.compile(r'[0-9a-f]{64}')


def validate_ref(value):
    if not isinstance(value, str) or len(value) != 64 or _HEX_REF.fullmatch(value) is None:
        raise ValueError('Invalid feature_ref')
    return value


@dataclass(frozen=True)
class FeatureReference:
    feature_ref: str
    namespace: str
    worker_generation: object = None

    def __post_init__(self):
        validate_ref(self.feature_ref)
        validate_ref(self.namespace)


@dataclass(frozen=True)
class GalleryReferences:
    feature_refs: tuple
    namespace: str
    content_revisions: tuple = ()

    def __post_init__(self):
        validate_ref(self.namespace)
        object.__setattr__(self, 'feature_refs', tuple(validate_ref(r) for r in self.feature_refs))
        if self.content_revisions:
            if len(self.content_revisions)!=len(self.feature_refs):raise ValueError('Revision count mismatch')
            object.__setattr__(self,'content_revisions',tuple(validate_ref(r) for r in self.content_revisions))

    def __len__(self):
        return len(self.feature_refs)


def reference_signature(ids, refs, namespace, revisions=None):
    ids = list(ids)
    refs = tuple(refs)
    if len(ids) != len(refs) or len(set(ids)) != len(ids):
        raise ValueError('Observation IDs/references mismatch')
    for ref in refs:
        validate_ref(ref)
    revisions=tuple(revisions or (None,)*len(ids))
    if len(revisions)!=len(ids):raise ValueError('Observation revision count mismatch')
    for revision in revisions:
        if revision is not None:validate_ref(revision)
    payload = {'schema': SCHEMA, 'namespace': validate_ref(namespace), 'observations': list(zip(ids, refs, revisions))}
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()


class RawReferenceStore(RawFeatureStore):
    """Explicit v2 namespace: both raw branches, canonical byte order and schema."""
    @staticmethod
    def arrays(raw, cnn, base=None):
        if base is not None:
            raise ValueError('v2 raw references do not contain service projections')
        result = {'emb': np.ascontiguousarray(raw, dtype='<f4'),
                  'specialist': np.ascontiguousarray(cnn, dtype='<f2')}
        for name, value in result.items():
            if value.shape != (2048,) or not np.isfinite(value).all():
                raise ValueError('Invalid raw pair: ' + name)
        return result

    def reference(self, values):
        h = hashlib.sha256(json.dumps({'schema': SCHEMA, 'extractor_namespace': self.identity}, sort_keys=True).encode())
        for name, dtype in [('emb', '<f4'), ('specialist', '<f2')]:
            value = np.ascontiguousarray(values[name], dtype=dtype)
            header = json.dumps({'name': name, 'dtype': dtype, 'shape': list(value.shape)}, sort_keys=True).encode()
            h.update(len(header).to_bytes(4, 'little'));h.update(header);h.update(value.tobytes())
        return h.hexdigest()

    def put(self, raw, cnn, base=None, *, persist=False, explicit_ref=True):
        values = self.arrays(raw, cnn, base)
        ref = self.reference(values)
        if not persist:
            self.recent[ref] = {k: v.copy() for k, v in values.items()}
            self.recent.move_to_end(ref)
            while len(self.recent) > 64:
                self.recent.popitem(last=False)
            return ref
        target = self.objects / (ref + '.npz')
        if target.exists():
            self.get(ref, persisted=True)
            return ref
        with tempfile.NamedTemporaryFile(dir=self.objects, suffix='.writing', delete=False) as stream:
            temporary = Path(stream.name)
            np.savez(stream, **values)
        try:
            try:
                os.link(temporary, target)
            except FileExistsError:
                pass
            self.get(ref, persisted=True)
        finally:
            temporary.unlink()
        return ref

    def get(self, ref, *, persisted=False):
        validate_ref(ref)
        if not persisted and ref in self.recent:
            return self.recent[ref]
        path = self.objects / (ref + '.npz')
        if not path.is_file():
            raise ValueError('Raw feature_ref unavailable; migrate or re-encode')
        with np.load(path, allow_pickle=False) as source:
            if set(source.files) != {'emb', 'specialist'}:
                raise ValueError('Wrong raw pair schema')
            for name, dtype in [('emb', '<f4'), ('specialist', '<f2')]:
                if source[name].dtype.str != dtype:
                    raise ValueError('Wrong raw pair dtype: ' + name)
            values = self.arrays(source['emb'], source['specialist'])
        if self.reference(values) != ref:
            raise ValueError('Raw pair integrity or extractor namespace mismatch')
        return values

    def legacy(self, *args, **kwargs):
        raise ValueError('v2 requires explicit feature_ref; legacy lookup is migration-only')
