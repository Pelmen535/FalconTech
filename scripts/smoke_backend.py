"""Check real-model HTTP import/search with the small bundled demo, not accuracy."""
import argparse
import json
import os
from pathlib import Path
import time

import httpx


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:8765')
    parser.add_argument('--demo-dir', type=Path, default=root / 'demo')
    parser.add_argument('--output', type=Path, default=root / 'var/smoke_result.json')
    parser.add_argument('--timeout', type=float, default=600)
    args = parser.parse_args()
    demo = args.demo_dir.resolve()
    query = json.loads((demo / 'query.json').read_text(encoding='utf-8'))
    expected = json.loads((demo / 'reference.json').read_text(encoding='utf-8'))
    headers = {'X-API-Key': os.environ['VREID_API_KEY']} if os.environ.get('VREID_API_KEY') else {}
    started = time.perf_counter()
    with httpx.Client(base_url=args.url.rstrip('/'), headers=headers,
                      timeout=args.timeout, trust_env=False) as client:
        def get(path):
            response = client.get(path)
            response.raise_for_status()
            return response.json()

        health = get('/health')
        imported = 0
        if health['gallery_count'] == 0:
            with (demo / 'gallery.zip').open('rb') as stream:
                response = client.post('/v1/gallery/import', files={
                    'archive': ('gallery.zip', stream, 'application/zip')})
            response.raise_for_status()
            imported = response.json()['added']
            require(imported == 20, 'Expected 20 imported demo images')
        else:
            print('Existing gallery preserved; checking whether it is exactly the demo gallery.', flush=True)
        gallery = get('/v1/gallery?limit=100')
        require(gallery['count'] == 20 and
                {item['image_id'] for item in gallery['items']} == set(expected['gallery_ids']),
                'This is not the demo gallery. Start a separate instance with --data-dir var/demo-check. No data was deleted.')
        image_path = demo / query['image']
        with image_path.open('rb') as stream:
            response = client.post('/v1/search',
                data={'bbox': json.dumps(query['bbox']), 'query_id': query['image_id']},
                files={'image': (image_path.name, stream, 'image/jpeg')})
        response.raise_for_status()
        answer = response.json()
        ranks = answer['ranking']
        require(len(ranks) == 10 and len({item['image_id'] for item in ranks}) == 10,
                'Search must return 10 different candidates')
        require([item['rank'] for item in ranks] == list(range(1, 11)), 'Invalid rank order')
        require(answer['confidence_scale'] == 'raw_cosine', 'Unexpected confidence scale')
        require(answer['accepted'] == (answer['candidate'] is not None), 'Invalid refusal response')
        health = get('/health')
        same_release = all(health[key] == expected[key] for key in ('model_sha256', 'recipe_sha256'))
        exact_top10 = [item['image_id'] for item in ranks] == expected['ranking_ids']
        if same_release:
            require(ranks[0]['image_id'] == expected['ranking_ids'][0], '19b top-1 differs from the recorded demo')
            require(answer['accepted'] == expected['accepted'], '19b refusal differs from the recorded demo')
        thumbnail = client.get(ranks[0]['thumbnail_url'])
        thumbnail.raise_for_status()
        require(thumbnail.headers.get('content-type', '').startswith('image/'), 'Thumbnail is missing')
        report = dict(passed=True, scope='HTTP execution smoke, not accuracy',
                      imported=imported, gallery_count=20, ranking_count=10,
                      accepted=answer['accepted'], top1=ranks[0]['image_id'],
                      device=health['device'], dimension=health['dimension'],
                      model_sha256=health['model_sha256'], recipe_sha256=health['recipe_sha256'],
                      same_reference_release=same_release, exact_reference_top10=exact_top10,
                      seconds=round(time.perf_counter() - started, 2))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
        print(json.dumps(report, ensure_ascii=False, indent=2))
        if not same_release:
            print('Different model/recipe: only HTTP execution was checked, not equivalence to 19b.')
        elif not exact_top10:
            print('Top-1/refusal match; lower candidate order differs from the CPU reference.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
