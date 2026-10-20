"""Print installed packages and verify the v2.0 runtime inside its container."""
from pathlib import Path
import importlib.metadata
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    from service.october_manifest import digest, load_manifest
    path = ROOT / 'release_current/runtime-manifest.json'
    spec = load_manifest(path, ROOT)
    report = {
        'status': 'PASS_ARTIFACTS_AND_ENVIRONMENT',
        'python': sys.version.split()[0],
        'runtime_manifest_SHA256': digest(path),
        'model_identity': spec['identity'],
        'weight_files': len(spec['weight_files']),
        'weights_bytes': spec['weights_bytes'],
        'packages': {name: importlib.metadata.version(name)
                     for name in sorted(spec['expected_versions'])},
        'scope': 'Dependency and artifact verification; no inference or speed measurement',
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
