"""One explicit, portable identity for October CLI/API runtimes."""
from pathlib import Path, PureWindowsPath
import hashlib
import json
import math
import sys

PROFILE = 'vric-road-verifier-v1/current-query-v1'
BASELINE_SHA256 = 'a4294b93acd9ddce36c06b03c045aca9667946e879d5ac92a6e61b225d65d220'


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def artifact(root, name):
    if not isinstance(name, str) or not name or PureWindowsPath(name).is_absolute():
        raise ValueError('Runtime artifact must use a relative path')
    path = (Path(root) / name.replace('\\', '/')).resolve()
    if not path.is_relative_to(Path(root).resolve()):
        raise ValueError('Runtime artifact escapes project root')
    return path


def load_manifest(path, root, *, verify=True):
    source = Path(path).resolve()
    spec = json.loads(source.read_text(encoding='utf-8'))
    if spec.get('schema') != 1 or spec.get('profile') != PROFILE or spec.get('baseline_manifest_SHA256') != BASELINE_SHA256:
        raise ValueError('Unsupported October runtime identity')
    platform='windows' if sys.platform=='win32' else sys.platform
    if spec.get('platform','windows')!=platform:
        raise ValueError('Runtime manifest targets a different operating system; use the matching platform package')
    expected_storage = 'feature_ref/raw-pair-le-v2' if spec.get('storage_contract') == 'raw-pair-le-v2' else 'pre_gallery_dino_projection'
    if spec.get('dimension') != 512 or spec.get('storage_embedding_kind') != expected_storage:
        raise ValueError('Invalid October embedding semantics')
    for key in ['model_sha256','recipe_sha256','embedding_sha256']:
        value = spec.get('identity', {}).get(key, '')
        if len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
            raise ValueError('Missing October model fingerprint: '+key)
    deadline = spec.get('operations', {}).get('worker_timeout_seconds', 0)
    if not isinstance(deadline, (int,float)) or not math.isfinite(deadline) or deadline <= 0:
        raise ValueError('Runtime requires a finite positive worker deadline')
    if verify and spec.get('expected_versions'):
        import importlib.metadata
        for name,wanted in spec['expected_versions'].items():
            try:actual=importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError as exc:raise ValueError('Missing pinned runtime dependency: '+name) from exc
            if actual!=wanted:raise ValueError('Runtime dependency mismatch: '+name+' expected '+wanted+', found '+actual)
    if verify:
        for name, record in spec['files'].items():
            target = artifact(root, name)
            if not target.is_file() or target.stat().st_size != record['bytes'] or digest(target) != record['SHA256']:
                raise ValueError('October runtime artifact mismatch: '+name)
    if verify:
        for name, record in spec.get('implementation_files', {}).items():
            target = artifact(root, name)
            if not target.is_file() or target.stat().st_size != record['bytes'] or digest(target) != record['SHA256']:
                raise ValueError('October runtime implementation mismatch: '+name)
    weights = spec.get('weight_files', [])
    if len(weights) != len(set(weights)) or any(n not in spec['files'] for n in weights):
        raise ValueError('Invalid runtime weight inventory')
    total = sum(spec['files'][n]['bytes'] for n in weights)
    if spec.get('weight_budget_bytes') != 2_000_000_000 or total != spec.get('weights_bytes') or total > 2_000_000_000:
        raise ValueError('Runtime weights exceed declared byte budget')
    return spec
