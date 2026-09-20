"""Режим отказа (ТЗ §3, §9): вернуть кандидатов с уверенностью или пустой ответ.

Оценивается: F1 при пороге, выбранном командой; TNR (доля корректных отказов на запросах
без пары в галерее); mINP; AUC по кривой precision-recall.

Определения (для запроса q с топ-1 кандидатом c и уверенностью s):
  «принят» — s ≥ порога и возвращён c;
  TP — принят и c той же машины;  FP — принят, но c другой машины (или пары в галерее нет);
  FN — отказ, хотя пара в галерее есть;  TN — отказ, и пары в галерее нет.
  F1 = 2TP / (2TP + FP + FN);  TNR = TN / (TN + FP_на_запросах_без_пары).

Уверенность s можно задать по-разному — сравниваем несколько:
  top1      — сходство с топ-1
  margin    — top1 − top2  (насколько лидер оторвался)
  bits      — −log2(N_кандидатов_в_радиусе / N_галереи) при откалиброванном радиусе
  combo     — top1 + α·margin (α подбирается на val)
"""
from __future__ import annotations

import numpy as np


def rank_gallery(sims: np.ndarray, q_cams: np.ndarray, g_cams: np.ndarray,
                 cross_camera_only: bool = True):
    """sims [Q,G] → отсортированные индексы галереи per query с учётом кросс-камерного правила."""
    s = sims.copy()
    if cross_camera_only:
        s[q_cams[:, None] == g_cams[None, :]] = -np.inf
    # стабильная сортировка: на ничьей порядок задаёт порядок строк test_gallery.csv (ответ 16)
    return np.argsort(-s, axis=1, kind="stable"), s


def confidence_scores(sims_masked: np.ndarray, order: np.ndarray, tau: float | None = None) -> dict[str, np.ndarray]:
    Q = sims_masked.shape[0]
    top1 = sims_masked[np.arange(Q), order[:, 0]]
    top2 = sims_masked[np.arange(Q), order[:, 1]] if order.shape[1] > 1 else np.full(Q, -1.0)
    top2 = np.where(np.isfinite(top2), top2, -1.0)
    out = {"top1": top1, "margin": top1 - top2}
    if tau is not None:
        n_g = np.isfinite(sims_masked).sum(axis=1)
        n_cand = np.maximum((sims_masked >= tau).sum(axis=1), 1)
        out["bits"] = -np.log2(n_cand / np.maximum(n_g, 1))
    return out


def refusal_curve(conf: np.ndarray, top1_correct: np.ndarray, has_match: np.ndarray,
                  n_thresholds: int = 200) -> list[dict]:
    """Порог → precision, recall, F1, TNR, acceptance rate."""
    ths = np.quantile(conf, np.linspace(0, 1, n_thresholds))
    rows = []
    for t in np.unique(ths):
        accept = conf >= t
        tp = int((accept & has_match & top1_correct).sum())
        fp = int((accept & ~(has_match & top1_correct)).sum())
        fn = int((~accept & has_match).sum())
        tn = int((~accept & ~has_match).sum())
        neg = int((~has_match).sum())
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-12)
        rows.append({"threshold": float(t), "precision": prec, "recall": rec, "f1": f1,
                     "tnr": (tn / neg) if neg else float("nan"), "accept_rate": float(accept.mean())})
    return rows


def choose_threshold(curve: list[dict], min_tnr: float | None = None,
                     w_f1: float = 0.7, w_tnr: float = 0.3) -> dict:
    """Максимум балла жюри 0.7·F1 + 0.3·TNR (ответ организаторов Q-14/Q-15 от 16.09);
    при заданном min_tnr — только среди порогов с TNR ≥ min_tnr."""
    cand = [r for r in curve if min_tnr is None or (not np.isnan(r["tnr"]) and r["tnr"] >= min_tnr)]
    if not cand:
        cand = curve
    score = lambda r: w_f1 * r["f1"] + w_tnr * (0.0 if np.isnan(r["tnr"]) else r["tnr"])
    best = max(cand, key=score)
    return {**best, "jury_score": score(best)}


def pr_auc(conf: np.ndarray, positive: np.ndarray) -> float:
    """Площадь под precision-recall для бинарной задачи «топ-1 верен и пара есть»."""
    order = np.argsort(-conf)
    pos = positive[order].astype(np.float64)
    tp = np.cumsum(pos)
    n_pos = pos.sum()
    if n_pos == 0:
        return float("nan")
    prec = tp / np.arange(1, len(pos) + 1)
    rec = tp / n_pos
    # интеграл по recall (step-wise)
    rec_prev = np.concatenate([[0.0], rec[:-1]])
    return float(np.sum((rec - rec_prev) * prec))


def minp(order: np.ndarray, sims_masked: np.ndarray, q_vids: np.ndarray, g_vids: np.ndarray,
         has_match: np.ndarray) -> float:
    """mean Inverse Negative Penalty (Ye et al. 2021): для каждого запроса
    NP = (позиция последнего истинного совпадения − число совпадений) / позиция последнего;
    INP = 1 − NP. Считается только по запросам с парой."""
    vals = []
    for i in range(order.shape[0]):
        if not has_match[i]:
            continue
        valid = np.isfinite(sims_masked[i, order[i]])
        ranked = g_vids[order[i]][valid]
        m = np.nonzero(ranked == q_vids[i])[0]
        if len(m) == 0:
            continue
        last = m[-1] + 1
        vals.append(1.0 - (last - len(m)) / last)
    return float(np.mean(vals)) if vals else float("nan")


def evaluate_refusal(sims: np.ndarray, q_vids: np.ndarray, q_cams: np.ndarray,
                     g_vids: np.ndarray, g_cams: np.ndarray, has_match: np.ndarray,
                     tau: float | None = None, min_tnr: float | None = None,
                     cross_camera_only: bool = True) -> dict:
    order, sm = rank_gallery(sims, q_cams, g_cams, cross_camera_only)
    top1_correct = g_vids[order[:, 0]] == q_vids
    scores = confidence_scores(sm, order, tau)
    if "bits" in scores:
        scores["top1+bits"] = scores["top1"] + 0.02 * scores["bits"]
    res = {"n_query": int(len(q_vids)), "n_no_match": int((~has_match).sum()),
           "mINP": minp(order, sm, q_vids, g_vids, has_match), "by_confidence": {}}
    positive = has_match & top1_correct
    best_name, best = None, None
    for name, conf in scores.items():
        curve = refusal_curve(conf, top1_correct, has_match)
        pick = choose_threshold(curve, min_tnr)
        entry = {"chosen": pick, "pr_auc": pr_auc(conf, positive), "curve": curve}
        res["by_confidence"][name] = entry
        if best is None or pick.get("jury_score", pick["f1"]) > best.get("jury_score", best["f1"]):
            best_name, best = name, pick
    res["best_confidence"] = best_name
    res["best"] = best
    return res


def format_refusal(res: dict) -> str:
    lines = [f"режим отказа: запросов {res['n_query']}, без пары {res['n_no_match']}, mINP={res['mINP'] * 100:.1f}%"]
    for name, e in res["by_confidence"].items():
        c = e["chosen"]
        lines.append(f"  {name:<10} порог={c['threshold']:.3f}  балл={c.get('jury_score', 0) * 100:5.1f}  "
                     f"F1={c['f1'] * 100:5.1f}%  P={c['precision'] * 100:5.1f}%  "
                     f"R={c['recall'] * 100:5.1f}%  TNR={c['tnr'] * 100:5.1f}%  accept={c['accept_rate'] * 100:4.0f}%")
    lines.append(f"  → лучшая уверенность: {res['best_confidence']}")
    return "\n".join(lines)
