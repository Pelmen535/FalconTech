"""Exercise offline CPU inference and HTTP metadata with synthetic JPEG input."""
from pathlib import Path
import csv
import json
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def main():
    import numpy as np
    from PIL import Image
    with tempfile.TemporaryDirectory(prefix='falcon-container-smoke-') as temporary:
        directory = Path(temporary)
        data = directory / 'input'
        images = data / 'images'
        images.mkdir(parents=True)
        rng = np.random.default_rng(20261020)
        for i in range(16):
            pixels = rng.integers(0, 256, (192, 256, 3), dtype=np.uint8)
            pixels[32:160, 40:216] = np.array([30 + 11*i, 200 - 7*i, 40 + 9*i], dtype=np.uint8)
            Image.fromarray(pixels).save(images / f'g{i:02}.jpg', quality=92)
        for i in range(2):
            with Image.open(images / f'g{i:02}.jpg') as image:
                image.save(images / f'q{i:02}.jpg', quality=90)
        for kind, count, prefix in [('gallery', 16, 'g'), ('query', 2, 'q')]:
            with (data / f'test_{kind}.csv').open('w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['image_id', 'x', 'y', 'w', 'h'])
                writer.writerows((f'{prefix}{i:02}.jpg', 0, 0, 256, 192) for i in range(count))
        output = directory / 'output'
        subprocess.run([
            sys.executable, '-B', str(ROOT / 'scripts/october_entrypoint.py'),
            'predict', '--data', str(data), '--out', str(output), '--device', 'cpu',
            '--batch-size', '1', '--workers', '2',
        ], check=True, timeout=900)
        embedding = np.load(output / 'embeddings.npy', allow_pickle=False)
        assert embedding.shape == (18, 512) and embedding.dtype == np.float32
        assert np.isfinite(embedding).all()
        with (output / 'submission.csv').open(newline='', encoding='utf-8') as f:
            submission = list(csv.reader(f))
        gallery_ids = {f'g{i:02}.jpg' for i in range(16)}
        assert len(submission) == 2
        for i, row in enumerate(submission):
            assert len(row) == 11 and row[0] == f'q{i:02}.jpg'
            assert len(set(row[1:])) == 10 and set(row[1:]) <= gallery_ids
        with (output / 'candidates.csv').open(newline='', encoding='utf-8') as f:
            candidates = list(csv.reader(f))
        assert candidates[0] == ['query_id', 'gallery_id', 'confidence']
        assert (output / 'run_info.json').is_file()
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0))
            port = sock.getsockname()[1]
        with (directory / 'http.log').open('w', encoding='utf-8') as log:
            server = subprocess.Popen([
                sys.executable, '-B', str(ROOT / 'scripts/october_entrypoint.py'),
                'serve', '--device', 'cpu', '--host', '127.0.0.1', '--port', str(port),
                '--data-dir', str(directory / 'http-data'),
            ], stdout=log, stderr=subprocess.STDOUT)
            try:
                base = f'http://127.0.0.1:{port}'
                deadline = time.monotonic() + 120
                while True:
                    if server.poll() is not None:
                        raise RuntimeError((directory / 'http.log').read_text(encoding='utf-8')[-6000:])
                    try:
                        with urllib.request.urlopen(base + '/health', timeout=2) as response:
                            assert response.status == 200
                        break
                    except OSError:
                        if time.monotonic() >= deadline:
                            raise TimeoutError('HTTP startup timeout')
                        time.sleep(1)
                with urllib.request.urlopen(base + '/openapi.json', timeout=5) as response:
                    schema = json.load(response)
                assert schema['info']['version'] == '2.0.0'
                with urllib.request.urlopen(base + '/v1/model', timeout=5) as response:
                    model = json.load(response)
                assert model['backend'] == 'october'
                assert model['confidence_scale'] == 'calibrated_acceptance_score'
                with urllib.request.urlopen(base + '/docs', timeout=5) as response:
                    assert response.status == 200
            finally:
                server.terminate()
                try:
                    server.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    server.kill()
                    server.wait(timeout=10)
        print(json.dumps({
            'status': 'PASS_OFFLINE_CPU_CLI_AND_HTTP_METADATA',
            'query': 2, 'gallery': 16, 'export_shape': list(embedding.shape),
            'api_version': schema['info']['version'],
            'scope': 'Synthetic functional smoke; not GPU, quality or performance qualification',
        }))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
