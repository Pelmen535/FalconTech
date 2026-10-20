"""Isolated persistent worker; all model math comes from the frozen CLI modules."""
from pathlib import Path
from collections import OrderedDict
import hashlib
import json
import os
import sys
import tempfile
import time
import subprocess
import itertools
import traceback

for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[name] = '2'


def main():
    control = sys.stdout
    sys.stdout = sys.stderr  # Third-party diagnostics must not corrupt the RPC stream.
    root = Path(__file__).resolve().parents[1]
    cache = Path(sys.argv[1]).resolve()
    audited_paths = set()
    forbidden = os.environ.get('VREID_FORBID_SOURCE_ROOT')
    if forbidden:
        forbidden = Path(forbidden).resolve()

        def audit(event, args):
            if event != 'open' or not args or not isinstance(args[0], (str, bytes, os.PathLike)):
                return
            path = Path(os.fsdecode(args[0])).resolve()
            if path.is_relative_to(forbidden):
                if not path.is_relative_to(root) and not path.is_relative_to(cache.parents[1]):
                    raise RuntimeError('Portable worker escaped package: '+str(path))
                audited_paths.add(str(path))

        sys.addaudithook(audit)
    sys.path.insert(0, str(root/'var/teacher-projection-20261011'))
    if sys.platform == 'linux':
        from service import linux_resources as resources
    else:
        import resources
    resources.restrict_current_process()

    def memory_headroom():
        if sys.platform == 'linux':
            return resources.memory_headroom_gib()
        mem=resources.Mem();mem.length=resources.ctypes.sizeof(mem)
        assert resources.ctypes.windll.kernel32.GlobalMemoryStatusEx(resources.ctypes.byref(mem))
        return mem.commit_available/2**30
    # Match the verified CLI reserve boundary: test the complete 7GiB budget
    # BEFORE importing the model stack, not again after it already consumed RAM.
    pending = None
    while pending is None:
        line = sys.stdin.readline()
        if not line:
            return
        message = json.loads(line)
        if message['command'] == 'close':
            control.write(json.dumps(dict(ok=True, result=True))+'\n');control.flush()
            return
        resources.snapshot();time.sleep(.2);snap=resources.snapshot()
        reasons=[r for r in resources.start_reasons(snap,'gpu') if r!='GPU free below9GiB or unknown']
        if snap.get('GPU_free_GiB',0)<7 or memory_headroom()<7 or snap['RAM_free_GiB']<4.35:
            reasons.append('Need measured pre-import start reserve: VRAM7GiB/Linux memory7GiB' if sys.platform == 'linux'
                           else 'Need measured pre-import start reserve: VRAM7GiB/commit7GiB/RAM4.35GiB')
        if reasons:
            control.write(json.dumps(dict(ok=False,resource_wait=True,error='; '.join(reasons)))+'\n');control.flush()
        else:
            pending=line
    sys.path.insert(0, str(root/'var/cached-feature-post-20261010'))
    import context
    from cached_post import install, teacher_cached
    install()
    if sys.platform == 'linux':
        from service.linux_runtime import joint_features_class
        JointFeatures = joint_features_class()
    else:
        from joint_runtime import JointFeatures
    if sys.platform != 'linux' and os.environ.get('VREID_WORKER_PYTHON'):
        import joint_runtime
        required = ['VREID_TRT_SITE', 'VREID_CUDA_DLL_DIR', 'VREID_NORMALIZER']
        if any(not os.environ.get(name) for name in required):
            raise ValueError('Portable worker requires TRT site, CUDA DLL directory and normalizer')
        joint_runtime.NORMALIZER = Path(os.environ['VREID_NORMALIZER']).resolve()
        original_popen = subprocess.Popen

        def relocated_popen(args, *extra, **kw):
            if isinstance(args, (list, tuple)) and len(args) >= 6 and Path(args[4]).name == 'joint_worker.py':
                args = [os.environ['VREID_WORKER_PYTHON'], '-X', 'utf8', '-B',
                        str(root/'service/trt_entry.py'), *args[5:]]
            return original_popen(args, *extra, **kw)

        subprocess.Popen = relocated_popen
    from fusion_evaluate import route
    profile = os.environ.get('VREID_RERANK_PROFILE', 'frozen')
    if profile == 'current-query-v1':
        from vreid.current_query_profile import route
    elif profile != 'frozen':
        raise ValueError('Unknown robust reranking profile')
    import numpy as np
    from vreid.refusal_policy_raw import GeneralRefusalPolicy
    policy = GeneralRefusalPolicy.load(root/'var/general-refusal-final-20261017/selection.json')
    refusal_profile = os.environ.get('VREID_REFUSAL_PROFILE', 'frozen')
    if refusal_profile == 'bagged-source-v1':
        if profile != 'current-query-v1':
            raise ValueError('Bagged refusal requires current-query-v1 ranking')
        from vreid.refusal_policy_bagged import BaggedRefusalPolicy
        policy = BaggedRefusalPolicy.load(root)
    elif refusal_profile == 'domain-source-v1':
        if profile != 'current-query-v1':
            raise ValueError('Domain refusal requires current-query-v1 ranking')
        from vreid.refusal_policy_domain import DomainRefusalPolicy
        policy = DomainRefusalPolicy.load(root)
    elif refusal_profile == 'contrast-learned-v1':
        if profile != 'current-query-v1':
            raise ValueError('Contrast refusal requires current-query-v1 ranking')
        from vreid.refusal_policy_contrast_learned import LearnedContrastRefusalPolicy
        policy = LearnedContrastRefusalPolicy.load(root)
    elif refusal_profile == 'expanded-source-v1':
        if profile != 'current-query-v1':
            raise ValueError('Expanded refusal requires current-query-v1 ranking')
        from vreid.refusal_policy_expanded import ExpandedRefusalPolicy
        policy = ExpandedRefusalPolicy.load(root)
    elif refusal_profile != 'frozen':
        raise ValueError('Unknown robust refusal profile')
    config = json.loads((root/'var/three-directions-20261009/fusion-selection.json').read_text())['config']
    gallery_memo = None
    registered_gallery = None
    if refusal_profile == 'expanded-source-v1':
        from service.static_gallery import StaticGalleryMemo, RegisteredGallery
        gallery_memo = StaticGalleryMemo()
        route = gallery_memo.route
    recent = OrderedDict()
    runtime = None
    initial_start_reserved = True

    class LiveReserveError(RuntimeError):
        pass

    def check_live():
        snap = resources.snapshot()
        if min(snap['RAM_free_GiB'], memory_headroom()) < 1.5 or snap.get('GPU_free_GiB', 0) < 3:
            raise LiveReserveError('Insufficient live RAM/VRAM reserve; models unloaded' if sys.platform == 'linux'
                                   else 'Insufficient live RAM/commit/VRAM reserve; models unloaded')

    def token(vector):
        a = np.asarray(vector, dtype='<f4')
        if a.shape != (512,) or not np.isfinite(a).all():
            raise ValueError('Invalid stored projection')
        return hashlib.sha256(a.tobytes()).hexdigest()

    def lookup(vector, persisted=False):
        key = token(vector)
        if not persisted and key in recent:
            return recent[key]
        path = cache/(key+'.npz')
        if not path.is_file():
            raise ValueError('Raw gallery features missing; rebuild this gallery with the robust backend')
        with np.load(path, allow_pickle=False) as z:
            raw, cnn, base = z['emb'].copy(), z['specialist'].copy(), z['base'].copy()
        if token(base) != key or raw.shape != (2048,) or cnn.shape != (2048,) or not np.isfinite(raw).all() or not np.isfinite(cnn).all():
            raise ValueError('Corrupt raw-feature cache; rebuild gallery')
        return raw, cnn

    def pack(vectors, keys, persisted=False):
        pairs = [lookup(v,persisted=persisted) for v in vectors]
        return JointFeatures.pack(None, [dict(key=key) for key in keys],
                                  dict(emb=np.stack([p[0] for p in pairs]),
                                       specialist=np.stack([p[1] for p in pairs])))

    if gallery_memo is not None:
        registered_gallery = RegisteredGallery(cache,token,
            lambda vectors,ids:pack(vectors,ids,persisted=True),gallery_memo.clear)

    def encode(message):
        nonlocal runtime, initial_start_reserved
        if runtime is None:
            if not initial_start_reserved:
                resources.snapshot();time.sleep(.2);snap=resources.snapshot()
                reasons=[r for r in resources.start_reasons(snap,'gpu') if r!='GPU free below9GiB or unknown']
                if snap.get('GPU_free_GiB',0)<7 or memory_headroom()<7 or snap['RAM_free_GiB']<4.35:
                    reasons.append('Insufficient measured restart reserve')
                if reasons:return None,'; '.join(reasons)
            initial_start_reserved = False
            logs = cache/'logs'; logs.mkdir(exist_ok=True)
            attempt = Path(tempfile.mkdtemp(prefix='worker-', dir=logs))
            runtime = JointFeatures(attempt, 16)
            original_embed = runtime.embed

            def guarded_embed(batch):
                check_live()
                return original_embed(batch)

            runtime.embed = guarded_embed
        rows = [{**r, 'key': str(i)} for i, r in enumerate(message['rows'])]
        try:
            features, timing = runtime.extract(rows)
        except LiveReserveError as exc:
            runtime.close(); runtime = None
            return None, str(exc)
        base = teacher_cached(features['emb'])
        for i, vector in enumerate(base):
            key = token(vector)
            pair = features['emb'][i].copy(), features['specialist'][i].copy()
            path = cache/(key+'.npz')
            if message['persist']:
                if path.exists():
                    old = lookup(vector,persisted=refusal_profile=='expanded-source-v1')
                    if not all(np.array_equal(a, b) for a, b in zip(old, pair)):
                        raise ValueError('Projection cache collision or extraction changed')
                else:
                    with tempfile.NamedTemporaryFile(dir=cache, suffix='.writing', delete=False) as f:
                        tmp = Path(f.name)
                        np.savez(f, emb=pair[0], specialist=pair[1], base=vector)
                    tmp.replace(path)
            else:
                recent[key] = pair
                recent.move_to_end(key)
                while len(recent) > 64:
                    recent.popitem(last=False)
        return base.tolist(), None

    def rank(message):
        if 'gallery_signature' in message:
            if registered_gallery is None:
                raise ValueError('Gallery registration requires expanded-source-v1')
            g,ids = registered_gallery.get(message['gallery_signature'])
        else:
            ids = message['ids']
            g = None
        if len(ids) < 10 or len(set(ids)) != len(ids):
            raise ValueError('Need at least 10 unique gallery IDs')
        query_id = message.get('query_id')
        if query_id is None:
            query_id = '__service_query__'
            while query_id.split('#')[0] in {k.split('#')[0] for k in ids}:
                query_id += '_'
        q = pack([message['query']], [query_id])
        if g is None:
            g = pack(message['gallery'],ids,persisted=refusal_profile=='expanded-source-v1')
        result = route(q, g, config)
        result['_query_features'] = {k: q[k] for k in ['emb', 'specialist']}
        result['_gallery_features'] = {k: g[k] for k in ['emb', 'specialist']}
        if refusal_profile in ('contrast-learned-v1', 'expanded-source-v1'):
            result['_gallery_keys'] = g['keys']
        boundary = context.bc.BASE['threshold']
        confidence = float(policy.predict(result, context.confidence(result, config['recipe']), boundary)['confidence'][0])
        order = np.argsort(-result['rank'][0], kind='stable')[:10]
        if not np.isfinite(result['rank'][0, order]).all():
            raise ValueError('Need 10 eligible gallery observations outside the query frame')
        ranking = [dict(image_id=ids[j], rank=i+1, cosine=float(result['raw'][0, j]),
                        ranking_score=float(result['rank'][0, j])) for i, j in enumerate(order)]
        accepted = confidence >= boundary
        return dict(ranking=ranking, accepted=accepted,
                    candidate={**ranking[0], 'confidence': confidence} if accepted else None,
                    confidence=confidence, threshold=boundary, stage='robust_full_gallery',
                    supports_explain=False)

    try:
        for line in itertools.chain([pending], sys.stdin):
            closing = False
            try:
                message = json.loads(line)
                if message['command'] == 'close':
                    closing = True
                    if runtime is not None:
                        runtime.close(); runtime = None
                    value, wait = True, None
                elif message['command'] == 'encode':
                    value, wait = encode(message)
                elif message['command'] == 'rank':
                    value, wait = rank(message), None
                elif message['command'] == 'gallery':
                    if registered_gallery is None:
                        raise ValueError('Gallery registration requires expanded-source-v1')
                    value, wait = registered_gallery.register(message['gallery'],message['ids'],message['signature']),None
                else:
                    raise ValueError('Unknown worker command')
                response = dict(ok=not bool(wait), result=value, resource_wait=bool(wait), error=wait)
            except Exception as exc:
                traceback.print_exc(file=sys.stderr)
                response = dict(ok=False, error=str(exc), resource_wait=False)
            control.write(json.dumps(response, allow_nan=False)+'\n'); control.flush()
            if closing:
                break
    finally:
        if runtime is not None:
            runtime.close()
        if gallery_memo is not None:
            print(json.dumps(dict(event='gallery.cache',registrations=registered_gallery.registrations,
                raw_file_refreshes=registered_gallery.refreshes,snapshot_hits=registered_gallery.hits,
                numerical_hits=gallery_memo.hits)),file=sys.stderr)
            gallery_memo.close()
        if forbidden:
            (cache/'package-read-audit.json').write_text(json.dumps(dict(
                status='PASS_NO_SOURCE_ESCAPE', root=str(root), opened=sorted(audited_paths)),
                ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
