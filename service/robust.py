"""Service boundary for the frozen DINOv3/CNN candidate.

The subprocess owns research-module imports and the GPU. SQLite/pgvector keeps
normalized pre-gallery 512D vectors; a separate immutable cache keeps raw
features required to reproduce the selected refusal rule after a restart.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PureWindowsPath
import subprocess
import sys
import threading

import numpy as np


class ResourceUnavailableError(RuntimeError):
    pass


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


class RobustAdapter:
    dimension = 512
    supports_ann = False
    supports_explain = False
    confidence_scale = 'calibrated_acceptance_score'

    def __init__(self, core_dir, data_dir, device='cuda'):
        self.root = Path(core_dir).resolve()
        self.data_dir = Path(data_dir).resolve()
        if device != 'cuda':
            raise ValueError('The robust backend requires --device cuda')
        self.device = device
        self.rerank_profile = os.environ.get('VREID_RERANK_PROFILE', 'frozen')
        if self.rerank_profile not in ('frozen', 'current-query-v1'):
            raise ValueError('Unknown robust reranking profile')
        self.refusal_profile = os.environ.get('VREID_REFUSAL_PROFILE', 'frozen')
        if self.refusal_profile not in ('frozen', 'bagged-source-v1', 'domain-source-v1', 'contrast-learned-v1', 'expanded-source-v1'):
            raise ValueError('Unknown robust refusal profile')
        if self.refusal_profile != 'frozen' and self.rerank_profile != 'current-query-v1':
            raise ValueError('Supplemental refusal requires current-query-v1 ranking')
        self.selection = self.root / 'var/general-refusal-final-20261017/selection.json'
        decision = json.loads(self.selection.with_name('decision.json').read_text(encoding='utf-8'))
        if decision['status'] != 'READY_LOCAL_WINDOWS_Q_GAIN_WITH_TNR_TRADEOFF':
            raise ValueError('Robust candidate has not passed native acceptance')
        if digest(self.selection) != decision['selection_SHA']:
            raise ValueError('Frozen robust selection changed')
        selected = json.loads(self.selection.read_text(encoding='utf-8'))
        for filename, expected in [
            (self.selection.with_name(selected['winner']['name'] + '.joblib'), selected['winner']['model_SHA']),
            (self.root/'vreid/raw_refusal.py', selected['feature_code_SHA']),
            (self.root/'vreid/refusal_policy_raw.py', selected['policy_code_SHA']),
        ]:
            if digest(filename) != expected:
                raise ValueError(f'Frozen robust artifact changed: {filename.name}')
        manifest = json.loads((self.root/'var/cached-feature-post-20261010/candidate-manifest.json').read_text(encoding='utf-8'))
        for name, expected in manifest['runtime_files'].items():
            if digest(self.root.joinpath(*PureWindowsPath(name).parts)) != expected:
                raise ValueError(f'Frozen base runtime changed: {name}')
        identities = []
        for item in manifest['inference_weights']:
            if sys.platform == 'linux' and item['path'].endswith('.engine'):
                # Windows engines are provenance only. Linux replacements have
                # a separate SHA manifest tied to this frozen source manifest.
                continue
            path = PureWindowsPath(item['path'])
            # The frozen manifest retains provenance; resolve its var/ suffix
            # inside this checkout/package, never by falling back to the source PC.
            path = self.root.joinpath(*path.parts[path.parts.index('var'):])
            if digest(path) != item['SHA256']:
                raise ValueError(f'Inference weight checksum mismatch: {path.name}')
            identities.append(item['SHA256'])
        if sys.platform == 'linux':
            from service.linux_runtime import verified_manifest
            linux = verified_manifest(self.root)
            identities.extend(item['SHA256'] for item in linux['files'])
        elif sys.platform != 'win32':
            raise ValueError('Robust TensorRT backend requires Windows or Linux')
        self.model_sha256 = hashlib.sha256(json.dumps(identities).encode()).hexdigest()
        self.recipe_sha256 = digest(self.selection)
        if self.rerank_profile == 'current-query-v1':
            profile = json.loads((self.root/'var/rerank-repair-20261017/runtime-selection.json').read_text(encoding='utf-8'))
            if profile['profile'] != self.rerank_profile:
                raise ValueError('Reranking profile identity mismatch')
            if profile['status'] not in ('PASS_FIXED_FEATURES_WITH_CFV14_TRADEOFF_PENDING_NATIVE',
                                         'READY_LOCAL_WINDOWS_WITH_CFV14_TRADEOFF'):
                raise ValueError('Reranking profile has not passed its validation gates')
            for filename, expected in profile['files'].items():
                if digest(self.root/filename) != expected:
                    raise ValueError('Reranking profile checksum mismatch: '+filename)
            identity = self.recipe_sha256 + self.rerank_profile + json.dumps(profile['files'], sort_keys=True)
            self.recipe_sha256 = hashlib.sha256(identity.encode()).hexdigest()
        if self.refusal_profile != 'frozen':
            if self.refusal_profile == 'bagged-source-v1':
                from vreid.refusal_policy_bagged import verified_selection
            elif self.refusal_profile == 'contrast-learned-v1':
                from vreid.refusal_policy_contrast_learned import verified_selection
            elif self.refusal_profile == 'expanded-source-v1':
                from vreid.refusal_policy_expanded import verified_selection
            else:
                from vreid.refusal_policy_domain import verified_selection
            selected_refusal = verified_selection(self.root)
            identity = (self.recipe_sha256 + self.refusal_profile
                        + json.dumps(selected_refusal['files'], sort_keys=True)
                        + str(selected_refusal['threshold']))
            self.recipe_sha256 = hashlib.sha256(identity.encode()).hexdigest()
        # Raw-feature cache and projected vectors share the same extraction identity.
        identity = self.model_sha256 + ':padded352:crop.05:raw2048+fp16cnn2048:teacher512:v1'
        self.embedding_sha256 = hashlib.sha256(identity.encode()).hexdigest()
        self.cache_dir = self.data_dir/'raw-features'/self.embedding_sha256
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._worker = None
        self._stderr = None
        self._loaded = False
        self._registered_gallery = None

    def metadata(self):
        return dict(model_sha256=self.model_sha256, recipe_sha256=self.recipe_sha256,
                    embedding_sha256=self.embedding_sha256, dimension=512,
                    threshold=0.7988346717209206, device=self.device,
                    loaded=self._loaded and self._worker is not None and self._worker.poll() is None,
                    confidence_kind=self.confidence_scale,
                    confidence_scale=self.confidence_scale, bbox_format='xywh',
                    backend='robust', rerank=('current_query_reciprocity_joint25'
                    if self.rerank_profile == 'current-query-v1' else 'frozen_dino_cnn_full_gallery'),
                    storage_embedding_kind='pre_gallery_dino_projection',
                    supports_ann=False, supports_explain=False,
                    raw_feature_cache_required=True, refusal_profile=self.refusal_profile)

    def _request(self, message):
        with self._lock:
            if self._worker is None:
                self._registered_gallery = None
                self._stderr = (self.data_dir/'robust-worker.log').open('a', encoding='utf-8')
                self._worker = subprocess.Popen(
                    [sys.executable, '-X', 'utf8', '-B', '-m', 'service.robust_worker',
                     str(self.cache_dir)], cwd=self.root, stdin=subprocess.PIPE,
                    env={**os.environ, 'VREID_RERANK_PROFILE': self.rerank_profile,
                         'VREID_REFUSAL_PROFILE': self.refusal_profile},
                    stdout=subprocess.PIPE, stderr=self._stderr, text=True,
                    encoding='utf-8', bufsize=1,
                    creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            if self._worker.poll() is not None:
                raise RuntimeError('Robust worker exited; inspect robust-worker.log and restart service')
            self._worker.stdin.write(json.dumps(message, allow_nan=False) + '\n')
            self._worker.stdin.flush()
            response = self._worker.stdout.readline()
            if not response:
                raise RuntimeError('Robust worker exited; inspect robust-worker.log')
            payload = json.loads(response)
            if not payload['ok']:
                if payload.get('resource_wait'):
                    self._loaded = False
                    raise ResourceUnavailableError(payload['error'])
                raise RuntimeError(payload['error'])
            return payload['result']

    def encode(self, image_path, bbox_xywh):
        return self._encode([dict(path=str(Path(image_path).resolve()), bbox=list(bbox_xywh))], False)[0]

    def encode_gallery(self, image_path, bbox_xywh):
        return self._encode([dict(path=str(Path(image_path).resolve()), bbox=list(bbox_xywh))], True)[0]

    def encode_gallery_many(self, rows):
        return self._encode(rows, True)

    def _encode(self, rows, persist):
        result = self._request(dict(command='encode', rows=rows, persist=persist))
        self._loaded = True
        return np.asarray(result, dtype=np.float32)

    def rank(self, query_vector, gallery_vectors, gallery_ids, query_id=None):
        if self.refusal_profile == 'expanded-source-v1':
            from .static_gallery import gallery_signature
            vectors = np.asarray(gallery_vectors,dtype=np.float32)
            ids = list(gallery_ids)
            signature = gallery_signature(vectors,ids)
            with self._lock:
                if self._worker is None:
                    self._registered_gallery = None
                if self._registered_gallery!=signature:
                    got = self._request(dict(command='gallery',gallery=vectors.tolist(),ids=ids,signature=signature))
                    if got!=signature:
                        raise RuntimeError('Worker registered a different gallery snapshot')
                    self._registered_gallery = signature
                return self._request(dict(command='rank',query=np.asarray(query_vector).tolist(),
                                          gallery_signature=signature,query_id=query_id))
        return self._request(dict(command='rank', query=np.asarray(query_vector).tolist(),
                                  gallery=np.asarray(gallery_vectors).tolist(),
                                  ids=list(gallery_ids), query_id=query_id))

    def close(self):
        with self._lock:
            if self._worker is not None:
                if self._worker.poll() is None:
                    try:
                        self._request(dict(command='close'))
                    finally:
                        self._worker.wait(timeout=30)
                self._worker.stdin.close()
                self._worker.stdout.close()
                self._worker = None
                self._loaded = False
            if self._stderr is not None:
                self._stderr.close()
                self._stderr = None
