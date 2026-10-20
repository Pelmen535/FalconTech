"""Typed content-addressed raw features; observations keep their own IDs.

The old 512D cache token is a compatibility lookup only, never a search key.
Query feature objects stay in memory and are not written to the gallery cache.
"""
from collections import OrderedDict
from pathlib import Path
import hashlib
import json
import os
import tempfile
import numpy as np


class FeatureVector(np.ndarray):
    def __new__(cls, vector, feature_ref, generation=None):
        obj = np.asarray(vector, dtype=np.float32).view(cls)
        obj.feature_ref, obj.worker_generation = feature_ref, generation
        return obj
    def __array_finalize__(self, source):
        self.feature_ref = getattr(source, 'feature_ref', None)
        self.worker_generation = getattr(source, 'worker_generation', None)


class RawFeatureStore:
    def __init__(self, root, identity):
        self.root = Path(root).resolve()
        self.identity = identity
        self.objects = self.root / 'objects'
        self.aliases = self.root / 'legacy-vector-refs'
        self.objects.mkdir(parents=True, exist_ok=True)
        self.aliases.mkdir(exist_ok=True)
        self.inputs = self.root/'gallery-input-refs'
        self.inputs.mkdir(exist_ok=True)
        self.recent = OrderedDict()

    def input_key(self,path,bbox):
        with Path(path).open('rb') as stream:
            image=hashlib.file_digest(stream,'sha256').hexdigest()
        payload={'schema':1,'embedding_identity':self.identity,'image_SHA256':image,
                 'bbox':None if bbox is None else [float(v) for v in bbox]}
        return hashlib.sha256(json.dumps(payload,sort_keys=True,allow_nan=False).encode()).hexdigest()

    def find_input(self,key):
        if not isinstance(key,str) or len(key)!=64 or any(c not in '0123456789abcdef' for c in key):
            raise ValueError('Invalid gallery input key')
        path=self.inputs/(key+'.json')
        if not path.is_file():
            return None
        ref=json.loads(path.read_text(encoding='utf-8'))['feature_ref']
        return ref,self.get(ref,persisted=True)

    def remember_input(self,key,ref):
        if not isinstance(key,str) or len(key)!=64 or any(c not in '0123456789abcdef' for c in key):
            raise ValueError('Invalid gallery input key')
        self.get(ref,persisted=True)
        # Publish only a complete immutable record; first valid extraction wins.
        target=self.inputs/(key+'.json')
        with tempfile.NamedTemporaryFile(dir=self.inputs,suffix='.writing',delete=False) as stream:
            temporary=Path(stream.name)
            stream.write(json.dumps({'feature_ref':ref,'schema':1}).encode('utf-8'))
        try:
            try:
                os.link(temporary,target)
            except FileExistsError:
                pass
        finally:
            temporary.unlink()
        return self.find_input(key)

    @staticmethod
    def arrays(raw, cnn, base):
        values = dict(emb=np.asarray(raw, dtype=np.float32), specialist=np.asarray(cnn, dtype=np.float16),
                      base=np.asarray(base, dtype=np.float32))
        for key, shape in [('emb',(2048,)),('specialist',(2048,)),('base',(512,))]:
            if values[key].shape != shape or not np.isfinite(values[key]).all():
                raise ValueError('Invalid typed raw feature: '+key)
        if abs(float(np.linalg.norm(values['base'].astype(np.float64)))-1) > 1e-3:
            raise ValueError('Stored projection must be normalized')
        return values

    def reference(self, values):
        h = hashlib.sha256(self.identity.encode())
        for key in ('emb','specialist','base'):
            value = np.ascontiguousarray(values[key])
            h.update(key.encode());h.update(str(value.dtype).encode())
            h.update(str(value.shape).encode());h.update(value.tobytes())
        return h.hexdigest()

    @staticmethod
    def token(vector):
        value = np.asarray(vector, dtype='<f4')
        if value.shape != (512,) or not np.isfinite(value).all():
            raise ValueError('Invalid legacy projection vector')
        return hashlib.sha256(value.tobytes()).hexdigest()

    def put(self, raw, cnn, base, *, persist=False, explicit_ref=False):
        values = self.arrays(raw, cnn, base)
        ref = self.reference(values)
        if not persist:
            self.recent[ref] = {k:v.copy() for k,v in values.items()}
            self.recent.move_to_end(ref)
            while len(self.recent)>64:
                self.recent.popitem(last=False)
            return ref
        path = self.objects / (ref+'.npz')
        if path.exists():
            self.get(ref, persisted=True)
        else:
            with tempfile.NamedTemporaryFile(dir=self.objects, suffix='.writing', delete=False) as stream:
                temporary = Path(stream.name)
                np.savez(stream, **values)
            temporary.replace(path)
        alias = self.aliases / (self.token(values['base'])+'.json')
        if alias.exists():
            previous = json.loads(alias.read_text(encoding='utf-8'))['feature_ref']
            if previous != ref:
                if not explicit_ref:
                    raise ValueError('Ambiguous legacy 512D reference; migrate with explicit observation feature_ref')
                # New observations retain their exact raw reference in the DB.
                # A projection-only lookup cannot disambiguate these objects.
                temporary=alias.with_suffix('.writing')
                temporary.write_text(json.dumps({'feature_ref':previous,'ambiguous':True}),encoding='utf-8')
                temporary.replace(alias)
                return ref
        else:
            temporary = alias.with_suffix('.writing')
            temporary.write_text(json.dumps({'feature_ref':ref,'kind':'legacy_projection_lookup'}),encoding='utf-8')
            temporary.replace(alias)
        legacy = self.root/(self.token(values['base'])+'.npz')
        if legacy.exists():
            with np.load(legacy,allow_pickle=False) as source:
                if not all(np.array_equal(source[k],v) for k,v in values.items()):
                    raise ValueError('Legacy raw cache collision; explicit feature_ref migration required')
        else:
            try:
                os.link(path,legacy)
            except OSError:
                import shutil
                shutil.copy2(path,legacy)
        return ref

    def get(self, ref, *, persisted=False):
        if not isinstance(ref,str) or len(ref)!=64 or any(c not in '0123456789abcdef' for c in ref):
            raise ValueError('Invalid feature_ref')
        if not persisted and ref in self.recent:
            return self.recent[ref]
        path = self.objects/(ref+'.npz')
        if not path.is_file():
            raise ValueError('Raw feature_ref not available; re-encode the observation')
        with np.load(path,allow_pickle=False) as source:
            for key,dtype in [('emb',np.dtype('float32')),('specialist',np.dtype('float16')),('base',np.dtype('float32'))]:
                if source[key].dtype!=dtype:
                    raise ValueError('Wrong typed raw feature storage: '+key)
            values = self.arrays(source['emb'],source['specialist'],source['base'])
        if self.reference(values)!=ref:
            raise ValueError('Raw feature object integrity mismatch')
        return values

    def legacy(self, vector, *, persisted=False):
        token = self.token(vector)
        alias = self.aliases/(token+'.json')
        if alias.is_file():
            record=json.loads(alias.read_text(encoding='utf-8'))
            if record.get('ambiguous'):
                raise ValueError('Ambiguous legacy 512D reference; explicit observation feature_ref required')
            ref=record['feature_ref']
            values = self.get(ref,persisted=persisted)
        else:
            path = self.root/(token+'.npz')
            if not path.is_file():
                raise ValueError('Raw gallery features missing; migrate/re-encode gallery with October runtime')
            with np.load(path,allow_pickle=False) as source:
                values = self.arrays(source['emb'],source['specialist'],source['base'])
            if self.token(values['base'])!=token:
                raise ValueError('Legacy raw cache integrity mismatch')
            ref = self.put(values['emb'],values['specialist'],values['base'],persist=True)
        if not np.array_equal(values['base'],np.asarray(vector,dtype=np.float32)):
            raise ValueError('Projection and feature_ref disagree')
        return ref,values


def snapshot_signature(vectors,ids,refs=None):
    from .static_gallery import gallery_signature
    base=gallery_signature(vectors,ids)
    if refs is None:return base
    if len(refs)!=len(ids):raise ValueError('Gallery references and IDs disagree')
    return hashlib.sha256((base+json.dumps(list(refs),separators=(',',':'))).encode()).hexdigest()
