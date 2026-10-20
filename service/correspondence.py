"""Experimental DINO patch correspondences, isolated from retrieval and CUDA."""
from pathlib import Path
import base64
import ctypes
import json
import math
import os
import subprocess
import sys
import tempfile
import numpy as np

WARNING = ('Экспериментальный метод может ошибаться: линии соединяют визуально похожие '
           'участки, в том числе фон и разные детали. Это не доказательство идентичности '
           'автомобиля и не объяснение решения об отказе.')


def match_patches(query, candidate, *, grid=24, count=8):
    """Mutual cosine neighbours with ambiguity and spatial-diversity filters."""
    arrays = [np.asarray(x, dtype=np.float32) for x in (query, candidate)]
    if any(x.ndim != 2 or x.shape[0] != grid * grid or not np.isfinite(x).all() for x in arrays):
        raise ValueError('Invalid patch features')
    if arrays[0].shape != arrays[1].shape or arrays[0].shape[1] == 0:
        raise ValueError('Incompatible patch features')
    a, b = [x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12) for x in arrays]
    sim = np.clip(a @ b.T, -1, 1)
    forward, reverse = sim.argmax(axis=1), sim.argmax(axis=0)
    proposals = []
    for i, j in enumerate(forward):
        if reverse[j] != i:
            continue
        row = np.partition(sim[i], -2)[-2:]
        column = np.partition(sim[:, j], -2)[-2:]
        margin = float(min(row[-1] - row[-2], column[-1] - column[-2]))
        cosine = float(sim[i, j])
        if cosine >= .65 and margin >= .02:
            proposals.append({'query_index': int(i), 'candidate_index': int(j),
                              'cosine': cosine, 'margin': margin})
    proposals.sort(key=lambda p: (-p['margin'], p['query_index'], p['candidate_index']))
    selected = []
    for pair in proposals:
        if any(any(math.hypot(pair[k] // grid - old[k] // grid,
                             pair[k] % grid - old[k] % grid) < 3
                       for k in ('query_index', 'candidate_index')) for old in selected):
            continue
        selected.append(pair)
        if len(selected) >= count:
            break
    return selected


def memory_headroom():
    if sys.platform == 'win32':
        class Memory(ctypes.Structure):
            _fields_ = [('length', ctypes.c_ulong), ('load', ctypes.c_ulong)] + [
                (key, ctypes.c_ulonglong) for key in ('total', 'available', 'commit_total',
                    'commit_available', 'virtual_total', 'virtual_available', 'extended')]
        value = Memory(); value.length = ctypes.sizeof(value)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(value)):
            raise RuntimeError('Memory telemetry unavailable')
        return min(value.available, value.commit_available) / 2**30
    from .linux_resources import memory_headroom_gib
    return memory_headroom_gib()


def run_isolated(root, spec, query_path, query_bbox, candidate_path, candidate_bbox):
    """One bounded CPU process per explicit request; no resident extra model."""
    from .robust import ResourceUnavailableError
    policy = spec['correspondence_policy']
    from .disk_reserve import WriteDiskReserve, DiskReserveError
    try:
        WriteDiskReserve({'temporary':tempfile.gettempdir()},
                         spec['operations']['start_C_disk_free_GiB']).require()
    except DiskReserveError as exc:
        raise ResourceUnavailableError(str(exc)) from exc
    if memory_headroom() < policy['minimum_headroom_GiB']:
        raise ResourceUnavailableError('Для экспериментальной карты недостаточно свободной RAM/commit. Повторите позже.')
    with tempfile.TemporaryDirectory(prefix='falcon-parts-') as temporary:
        directory = Path(temporary)
        config = dict(root=str(root), checkpoint=spec['cpu_environment']['checkpoint'],
                      checkpoint_pin=spec['files'][spec['cpu_environment']['checkpoint']],
                      query_path=str(query_path), query_bbox=query_bbox,
                      candidate_path=str(candidate_path), candidate_bbox=candidate_bbox)
        source, output = directory / 'input.json', directory / 'output.json'
        source.write_text(json.dumps(config), encoding='utf-8')
        env = {**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'OMP_NUM_THREADS': '2',
               'MKL_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '2', 'PYTHONUTF8': '1'}
        try:
            result = subprocess.run([sys.executable, '-u', '-X', 'utf8', '-B', '-m',
                'service.correspondence', str(source), str(output)], cwd=root, env=env,
                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                timeout=policy['timeout_seconds'], creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        except subprocess.TimeoutExpired as exc:
            raise ResourceUnavailableError('Время построения экспериментальной карты истекло. Повторите позже.') from exc
        if result.returncode:
            raise ResourceUnavailableError('Не удалось построить экспериментальную карту. Основной поиск доступен; повторите позже.')
        return json.loads(output.read_text(encoding='utf-8'))


def draw_pairs(images, pairs, grid=24):
    from io import BytesIO
    from PIL import Image, ImageDraw
    colors = ['#ef4444', '#22c55e', '#3b82f6', '#facc15', '#a855f7', '#06b6d4', '#fb923c', '#ec4899']
    shown = []
    for image in images:
        image = image.copy(); image.thumbnail((580, 480), Image.Resampling.LANCZOS); shown.append(image)
    canvas = Image.new('RGB', (1200, max(im.height for im in shown) + 40), '#17212b')
    offsets = [(int((580-im.width)/2)+side*620, 30) for side,im in enumerate(shown)]
    for im, offset in zip(shown, offsets):
        canvas.paste(im, offset)
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 7), 'QUERY', fill='white'); draw.text((630, 7), 'CANDIDATE', fill='white')
    points = []
    for number, pair in enumerate(pairs):
        ends = []
        for side, key in enumerate(('query_index', 'candidate_index')):
            index = pair[key]; image = shown[side]; x0,y0 = offsets[side]
            ends.append((x0+(index % grid+.5)*image.width/grid, y0+(index//grid+.5)*image.height/grid))
        draw.line(ends, fill=colors[number], width=2); points.append(ends)
    for number, ends in enumerate(points):
        for x,y in ends:
            draw.ellipse((x-11,y-11,x+11,y+11), fill=colors[number], outline='white', width=2)
            draw.text((x-3,y-5), str(number+1), fill='black')
    stream = BytesIO(); canvas.save(stream, format='PNG')
    return stream.getvalue()


def worker(source, output):
    import time
    from PIL import Image
    from .october_manifest import artifact, digest
    from .october_cpu import load_dino
    import torch
    torch.set_num_threads(2)
    started = time.perf_counter()
    config = json.loads(Path(source).read_text(encoding='utf-8'))
    checkpoint = artifact(config['root'], config['checkpoint'])
    pin = config['checkpoint_pin']
    if checkpoint.stat().st_size != pin['bytes'] or digest(checkpoint) != pin['SHA256']:
        raise ValueError('Correspondence checkpoint checksum mismatch')
    model, metadata = load_dino(checkpoint)
    images, features = [], []
    for side in ('query', 'candidate'):
        with Image.open(config[side+'_path']) as source_image:
            image = source_image.convert('RGB')
        box = config[side+'_bbox']
        if box is not None:
            x,y,w,h = box
            image = image.crop((x,y,x+w,y+h))
        images.append(image)
        x = np.asarray(image.resize((384,384), Image.Resampling.BICUBIC), dtype=np.float32) / np.float32(255)
        x = ((x-np.asarray(metadata['mean'], np.float32))/np.asarray(metadata['std'], np.float32)).transpose(2,0,1).copy()
        with torch.inference_mode():
            tokens = model.backbone.forward_features(torch.from_numpy(x)[None])
        patches = tokens[0, int(model.backbone.num_prefix_tokens):].float().numpy().copy()
        if patches.shape != (576,1024):
            raise ValueError('Unexpected patch token grid')
        features.append(patches)
    pairs = match_patches(*features)
    png = draw_pairs(images, pairs)
    response = dict(method='experimental_dinov3_mutual_patch_similarity_v1', experimental=True,
        warning=WARNING, device='cpu', grid=[24,24], pairs=pairs, pair_count=len(pairs),
        status='matches_found' if pairs else 'no_unambiguous_matches',
        image_sizes=[list(im.size) for im in images],
        correspondences_png_base64=base64.b64encode(png).decode('ascii'),
        seconds=round(time.perf_counter()-started,4))
    Path(output).write_text(json.dumps(response, allow_nan=False), encoding='utf-8')


if __name__ == '__main__':
    worker(*sys.argv[1:])
