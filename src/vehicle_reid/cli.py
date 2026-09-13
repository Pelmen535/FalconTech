from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import audit_records, load_manifest
from .demo_data import make_demo
from .features import ColorGridEmbedder, OnnxEmbedder
from .retrieval import evaluate, rank_candidates


def _embedder(args: argparse.Namespace):
    if args.backend == "color-grid":
        return ColorGridEmbedder()
    if not args.model or not args.model_spec:
        raise ValueError("ONNX backend requires --model and --model-spec")
    return OnnxEmbedder(args.model, args.model_spec)


def _add_backend(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--backend", choices=("color-grid", "onnx"), default="color-grid")
    parser.add_argument("--model", help="Path to ONNX embedding model")
    parser.add_argument("--model-spec", help="JSON preprocessing specification")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Vehicle Re-ID baseline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    demo = subparsers.add_parser("make-demo", help="create synthetic smoke-test images")
    demo.add_argument("--out", required=True)

    audit = subparsers.add_parser("audit", help="validate manifest and leakage constraints")
    audit.add_argument("--manifest", required=True)

    evaluation = subparsers.add_parser("evaluate", help="evaluate cross-camera retrieval")
    evaluation.add_argument("--manifest", required=True)
    evaluation.add_argument("--out", required=True)
    _add_backend(evaluation)

    query = subparsers.add_parser("query", help="rank gallery for one manifest query")
    query.add_argument("--manifest", required=True)
    query.add_argument("--query-index", type=int, default=0)
    query.add_argument("--top-k", type=int, default=5)
    _add_backend(query)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "make-demo":
        manifest = make_demo(args.out)
        print(json.dumps({"manifest": str(manifest)}, ensure_ascii=False, indent=2))
        return 0

    records = load_manifest(args.manifest)
    report = audit_records(records)
    if args.command == "audit":
        print(json.dumps(report.as_dict(), ensure_ascii=False, indent=2))
        return 0

    queries = [record for record in records if record.split == "query"]
    gallery = [record for record in records if record.split == "gallery"]
    embedder = _embedder(args)
    query_features = embedder.embed(queries)
    gallery_features = embedder.embed(gallery)
    if args.command == "evaluate":
        metrics, rankings = evaluate(queries, query_features, gallery, gallery_features)
        result = {"audit": report.as_dict(), "backend": args.backend, "metrics": metrics, "rankings": rankings}
        output = Path(args.out).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"output": str(output), "metrics": metrics}, ensure_ascii=False, indent=2))
        return 0


    if not queries:
        raise ValueError("manifest contains no query rows")
    if not 0 <= args.query_index < len(queries):
        raise IndexError(f"query-index must be between 0 and {len(queries) - 1}")
    ranked = rank_candidates(queries[args.query_index], query_features[args.query_index], gallery, gallery_features)
    payload = [
        {
            "image_path": str(candidate.record.image_path),
            "camera_id": candidate.record.camera_id,
            "score": round(candidate.score, 6),
        }
        for candidate in ranked[: args.top_k]
    ]
    print(json.dumps({"query": str(queries[args.query_index].image_path), "candidates": payload}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
