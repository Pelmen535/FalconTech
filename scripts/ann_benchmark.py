"""Масштабируемость поиска: точный перебор против ANN на галерее до 10^6 (ТЗ §10).

Что меряется и почему именно так.

Поиск по галерее НЕ входит в замер производительности жюри (ответ 31: латентность — это
чтение, декодирование, кроп, препроцессинг, forward и нормализация; поиск и ре-ранжирование
исключены). Поэтому здесь не борьба за балл, а ответ на вопрос «что будет, когда галерея
станет городской»: 10^6 наблюдений вместо 750.

Откуда берутся вектора. Белый шум размерности 1536 для такой проверки не годится: у него
нет ковариационной структуры настоящих эмбеддингов, PCA на нём не работает, и любой ANN
выглядит плохо просто потому, что задача выродилась. Поэтому база СИНТЕЗИРУЕТСЯ ИЗ НАШИХ
НАСТОЯЩИХ ВЕКТОРОВ: по ним считаются среднее и главные компоненты, и миллион точек
разыгрывается в том же базисе с теми же дисперсиями по компонентам. Спектр ковариации
совпадает с реальным, значит и выводы про PCA переносимы.

Структура поиска задаётся явно: сначала разыгрываются «личности», затем каждая попадает в
базу как одно наблюдение с малым шумом, а запрос — как ещё одно наблюдение той же личности.
У каждого запроса есть подсаженный правильный ответ, и его можно проверять, а не только
сравнивать списки между собой.

Три режима на одних и тех же данных:

  exact    numpy-перебор по всей базе в float32. Эталон: recall@10 = 1 по определению.
           Память = N*D*4 байта, для 10^6 x 1536 это 6.1 ГиБ.
  ivfpq    IVF + произведение квантователей: 192 байта на вектор вместо 6144,
           то есть 10^6 наблюдений помещаются в 192 МиБ.
  pca_hnsw PCA до 256 измерений + граф HNSW (faiss). Память в шесть раз меньше,
           поиск сублинейный. Потеря качества измеряется, а не декларируется.

    python scripts/ann_benchmark.py --sizes 10000 100000 1000000 --out results/ann_scalability.json

Числа из этого файла показывает вкладка «Масштабируемость» в демо-интерфейсе.
"""
from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np

CHUNK = 4096


# --------------------------------------------------------------------------- данные
def load_real(paths: list[Path]) -> np.ndarray:
    """Собрать наши настоящие эмбеддинги из .npy сдачи и .npz кешей валидации."""
    blocks = []
    for path in paths:
        if not path.is_file():
            continue
        if path.suffix == ".npz":
            with np.load(path, allow_pickle=True) as data:
                if "emb" in data:
                    blocks.append(np.asarray(data["emb"], dtype=np.float32))
        else:
            blocks.append(np.asarray(np.load(path, allow_pickle=False), dtype=np.float32))
    if not blocks:
        raise SystemExit("не найдено ни одного файла с настоящими эмбеддингами — "
                         "укажи --real явно (embeddings.npy или *_track.npz)")
    dims = {block.shape[1] for block in blocks}
    if len(dims) != 1:
        raise SystemExit(f"разные размерности в источниках: {sorted(dims)}")
    matrix = np.concatenate(blocks, axis=0)
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-12
    return matrix


class RealisticSampler:
    """Гауссова модель в базисе главных компонент настоящих эмбеддингов.

    Второй момент распределения совпадает с реальным по построению: мы берём те же
    собственные векторы и те же собственные значения. Дальше всё разыгрывается блоками,
    чтобы миллион точек не требовал второй копии в памяти.
    """

    def __init__(self, real: np.ndarray, seed: int):
        self.dim = int(real.shape[1])
        self.mean = real.mean(axis=0, dtype=np.float32)
        centered = real - self.mean
        _, singular, right = np.linalg.svd(centered, full_matrices=False)
        keep = min(centered.shape[0] - 1, self.dim)
        self.basis = np.ascontiguousarray(right[:keep].T.astype(np.float32))      # [D, K]
        self.sigma = (singular[:keep] / np.sqrt(max(1, centered.shape[0] - 1))).astype(np.float32)
        self.rng = np.random.default_rng(seed)

    def identities(self, count: int) -> np.ndarray:
        out = np.empty((count, self.dim), dtype=np.float32)
        for start in range(0, count, CHUNK):
            stop = min(start + CHUNK, count)
            coefficients = self.rng.normal(size=(stop - start, self.sigma.size)).astype(np.float32)
            coefficients *= self.sigma
            block = coefficients @ self.basis.T + self.mean
            block /= np.linalg.norm(block, axis=1, keepdims=True) + 1e-12
            out[start:stop] = block
        return out

    def observation(self, identities: np.ndarray, noise: float) -> np.ndarray:
        """Ещё одно наблюдение тех же личностей: шум в том же базисе, не изотропный."""
        out = np.empty_like(identities)
        for start in range(0, identities.shape[0], CHUNK):
            stop = min(start + CHUNK, identities.shape[0])
            coefficients = self.rng.normal(size=(stop - start, self.sigma.size)).astype(np.float32)
            coefficients *= self.sigma * noise
            block = identities[start:stop] + coefficients @ self.basis.T
            block /= np.linalg.norm(block, axis=1, keepdims=True) + 1e-12
            out[start:stop] = block
        return out


# --------------------------------------------------------------------------- метрики
def exact_topk(base: np.ndarray, queries: np.ndarray, k: int) -> tuple[np.ndarray, float]:
    """Перебор блоками запросов: без второй копии базы и без матрицы Q x N целиком."""
    out = np.empty((queries.shape[0], k), dtype=np.int64)
    started = time.perf_counter()
    for start in range(0, queries.shape[0], 64):
        block = queries[start:start + 64]
        sims = block @ base.T
        part = np.argpartition(-sims, k - 1, axis=1)[:, :k]
        rows = np.arange(block.shape[0])[:, None]
        order = np.argsort(-sims[rows, part], axis=1)
        out[start:start + 64] = part[rows, order]
    return out, time.perf_counter() - started


def agreement(found: np.ndarray, truth: np.ndarray) -> float:
    """Доля позиций точного топ-k, которые ANN тоже вернул."""
    hits = sum(len(set(a.tolist()) & set(b.tolist())) for a, b in zip(found, truth))
    return hits / (truth.shape[0] * truth.shape[1])


def planted(found: np.ndarray, answers: np.ndarray, cutoff: int) -> float:
    """Доля запросов, у которых подсаженный правильный ответ попал в первые cutoff мест."""
    return float(np.mean([answers[i] in set(found[i, :cutoff].tolist())
                          for i in range(found.shape[0])]))


def scores(found: np.ndarray, truth: np.ndarray, answers: np.ndarray, k: int) -> dict:
    return {"recall_at_k": round(agreement(found, truth), 4),
            "true_match_at_1": round(planted(found, answers, 1), 4),
            "true_match_at_k": round(planted(found, answers, k), 4)}


def timed_search(index, queries: np.ndarray, k: int, repeats: int = 3) -> tuple[np.ndarray, float]:
    best, found = float("inf"), None
    for _ in range(repeats):
        started = time.perf_counter()
        _, ids = index.search(queries, k)
        best = min(best, time.perf_counter() - started)
        found = ids
    return found, best


# --------------------------------------------------------------------------- проекция
def fit_pca(base: np.ndarray, dim: int, sample: int, seed: int) -> tuple[np.ndarray, np.ndarray, float]:
    rng = np.random.default_rng(seed)
    take = base[rng.choice(base.shape[0], size=min(sample, base.shape[0]), replace=False)]
    mean = take.mean(axis=0, dtype=np.float32)
    _, singular, right = np.linalg.svd(take - mean, full_matrices=False)
    energy = float((singular[:dim] ** 2).sum() / (singular ** 2).sum())
    return mean, np.ascontiguousarray(right[:dim].T.astype(np.float32)), energy


def project(matrix: np.ndarray, mean: np.ndarray, basis: np.ndarray) -> np.ndarray:
    out = np.empty((matrix.shape[0], basis.shape[1]), dtype=np.float32)
    for start in range(0, matrix.shape[0], CHUNK):
        block = (matrix[start:start + CHUNK] - mean) @ basis
        block /= np.linalg.norm(block, axis=1, keepdims=True) + 1e-12
        out[start:start + CHUNK] = block
    return out


# --------------------------------------------------------------------------- прогон
def measure(sampler: RealisticSampler, n: int, k: int, n_queries: int, seed: int,
            pca_dim: int, noise: float, faiss_flat_limit: int) -> dict:
    import faiss

    dim = sampler.dim
    picked = np.sort(np.random.default_rng(seed + 7).choice(n, size=n_queries, replace=False))
    # База строится блоками: держать одновременно матрицу личностей и матрицу наблюдений
    # для миллиона векторов 1536 значило бы 12 ГиБ. Личности нужны только для выбранных
    # запросов, их и сохраняем.
    base = np.empty((n, dim), dtype=np.float32)
    kept = np.empty((n_queries, dim), dtype=np.float32)
    taken = 0
    for start in range(0, n, CHUNK):
        stop = min(start + CHUNK, n)
        identities = sampler.identities(stop - start)
        base[start:stop] = sampler.observation(identities, noise)
        inside = picked[(picked >= start) & (picked < stop)] - start
        if inside.size:
            kept[taken:taken + inside.size] = identities[inside]
            taken += inside.size
        del identities
    assert taken == n_queries
    queries = sampler.observation(kept, noise)
    del kept
    answers = picked.astype(np.int64)

    row = {"n": n, "dim": dim, "k": k, "n_queries": n_queries,
           "observation_noise": noise, "base_bytes": int(base.nbytes), "modes": {}}

    truth, exact_seconds = exact_topk(base, queries, k)
    row["modes"]["exact"] = {
        "description": "полный перебор float32 по всей базе",
        "bytes_per_vector": dim * 4, "index_bytes": int(base.nbytes),
        "build_seconds": 0.0,
        "ms_per_query": round(1000 * exact_seconds / n_queries, 3),
        "qps": round(n_queries / exact_seconds, 1),
        **scores(truth, truth, answers, k),
    }

    if n <= faiss_flat_limit:
        flat = faiss.IndexFlatIP(dim)
        started = time.perf_counter()
        flat.add(base)
        build = time.perf_counter() - started
        found, seconds = timed_search(flat, queries, k)
        row["modes"]["faiss_flat"] = {
            "description": "faiss IndexFlatIP, тот же точный ответ",
            "bytes_per_vector": dim * 4, "index_bytes": int(base.nbytes),
            "build_seconds": round(build, 2),
            "ms_per_query": round(1000 * seconds / n_queries, 3),
            "qps": round(n_queries / seconds, 1),
            **scores(found, truth, answers, k),
        }
        del flat

    sub_vectors = 192
    lists = int(max(16, min(4 * np.sqrt(n), n // 39)))
    quantizer = faiss.IndexFlatIP(dim)
    ivfpq = faiss.IndexIVFPQ(quantizer, dim, lists, sub_vectors, 8, faiss.METRIC_INNER_PRODUCT)
    started = time.perf_counter()
    ivfpq.train(base[: min(n, max(lists * 39, 50_000))])
    ivfpq.add(base)
    build = time.perf_counter() - started
    ivfpq.nprobe = int(max(8, lists // 16))
    found, seconds = timed_search(ivfpq, queries, k)
    row["modes"]["ivfpq"] = {
        "description": f"faiss IVF{lists},PQ{sub_vectors}x8, nprobe={ivfpq.nprobe}",
        "bytes_per_vector": sub_vectors, "index_bytes": int(sub_vectors * n),
        "build_seconds": round(build, 2),
        "ms_per_query": round(1000 * seconds / n_queries, 3),
        "qps": round(n_queries / seconds, 1),
        **scores(found, truth, answers, k),
    }
    del ivfpq, quantizer

    mean, basis, energy = fit_pca(base, pca_dim, 50_000, seed + 2)
    reduced, reduced_queries = project(base, mean, basis), project(queries, mean, basis)
    del base
    hnsw = faiss.IndexHNSWFlat(pca_dim, 32, faiss.METRIC_INNER_PRODUCT)
    hnsw.hnsw.efConstruction = 80
    started = time.perf_counter()
    hnsw.add(reduced)
    build = time.perf_counter() - started
    hnsw.hnsw.efSearch = 128
    found, seconds = timed_search(hnsw, reduced_queries, k)
    row["modes"]["pca_hnsw"] = {
        "description": f"PCA-{pca_dim} ({energy:.1%} дисперсии) + HNSW32, efSearch={hnsw.hnsw.efSearch}",
        "bytes_per_vector": pca_dim * 4, "index_bytes": int(reduced.nbytes),
        "build_seconds": round(build, 2),
        "ms_per_query": round(1000 * seconds / n_queries, 3),
        "qps": round(n_queries / seconds, 1),
        **scores(found, truth, answers, k),
    }
    del hnsw, reduced, reduced_queries
    return row


def main() -> int:
    import faiss

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sizes", type=int, nargs="+", default=[10_000, 100_000, 1_000_000])
    parser.add_argument("--real", type=Path, nargs="+", default=[
        Path("submission/embeddings.npy"),
        Path("runs/hack/ft_soup_b336_fit/val_gallery_track.npz"),
        Path("runs/hack/ft_soup_b336_fit/val_query_track.npz")])
    parser.add_argument("--pca-dim", type=int, default=256)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--queries", type=int, default=256)
    parser.add_argument("--noise", type=float, default=0.6,
                        help="разброс второго наблюдения той же личности в долях реального")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--faiss-flat-limit", type=int, default=200_000,
                        help="выше этого размера вторая точная копия базы не помещается в память")
    parser.add_argument("--out", type=Path, default=Path("results/ann_scalability.json"))
    args = parser.parse_args()

    real = load_real(list(args.real))
    sampler = RealisticSampler(real, args.seed)
    print(f"[ann] модель распределения построена по {real.shape[0]} настоящим векторам, "
          f"D={real.shape[1]}", flush=True)

    report = {
        "purpose": "масштабируемость поиска по галерее; в замер производительности жюри не входит",
        "vectors": f"синтез по ковариации {real.shape[0]} настоящих эмбеддингов релиза",
        "sources": [str(p) for p in args.real if Path(p).is_file()],
        "machine": {"platform": platform.platform(), "faiss": faiss.__version__,
                    "threads": int(faiss.omp_get_max_threads())},
        "metrics": {
            "recall_at_k": "доля позиций точного топ-k, которые вернул ANN",
            "true_match_at_1": "доля запросов, у которых подсаженный ответ оказался первым",
            "true_match_at_k": "то же, но в пределах топ-k",
        },
        "rows": [],
    }
    faiss.omp_set_num_threads(min(16, faiss.omp_get_max_threads()))
    for n in args.sizes:
        print(f"[ann] N={n} ...", flush=True)
        row = measure(sampler, n, args.k, args.queries, args.seed, args.pca_dim,
                      args.noise, args.faiss_flat_limit)
        report["rows"].append(row)
        for name, mode in row["modes"].items():
            print(f"[ann]   {name:11s} {mode['ms_per_query']:8.3f} мс/запрос  "
                  f"recall@{args.k}={mode['recall_at_k']:.3f}  "
                  f"top1={mode['true_match_at_1']:.3f}  "
                  f"{mode['index_bytes'] / 1024 ** 2:8.0f} МиБ", flush=True)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ann] → {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
