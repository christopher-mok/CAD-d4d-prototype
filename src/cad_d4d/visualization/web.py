"""Build the interactive viewer: run replay (three.js) + benchmark dashboard (Plotly).

    python -m cad_d4d.visualization.web --recording run.json [run2.json ...] \
        --benchmark experiments/benchmark_out/results.jsonl --tag v1 -o viewer.html

The output is a single self-contained HTML file (data embedded); it loads
three.js and Plotly from cdn.jsdelivr.net, so it needs internet access once.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..benchmark.analysis import annotate

TEMPLATE = Path(__file__).with_name("viewer_template.html")
BENCH_FIELDS = ("target", "tag", "split", "family", "method", "kind", "seed", "fit", "n_cp", "n_faces", "refinements",
                "simplifications", "runtime_s", "efficiency", "cp_saving", "sdf_rms", "cov_rms")


def benchmark_rows(path: Path) -> list[dict]:
    rows = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
    return [{k: r.get(k) for k in BENCH_FIELDS} for r in annotate(rows)]


def build_html(recordings=(), bench_rows=None, title: str = "D4D Run Viewer", default_tag: str | None = None) -> str:
    data = {"title": title, "recordings": list(recordings),
            "benchmark": {"rows": bench_rows or [], "default_tag": default_tag}}
    payload = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    return TEMPLATE.read_text(encoding="utf-8").replace("__D4D_DATA__", payload)


def write_html(path, recordings=(), bench_rows=None, **kw) -> Path:
    path = Path(path)
    path.write_text(build_html(recordings, bench_rows, **kw), encoding="utf-8")
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--recording", nargs="*", default=[], help="Recorder JSON files (run replay tab)")
    ap.add_argument("--benchmark", default=None, help="benchmark results.jsonl (dashboard tab)")
    ap.add_argument("--tag", default=None, help="benchmark tag shown first")
    ap.add_argument("--title", default="D4D Run Viewer")
    ap.add_argument("-o", "--out", default="viewer.html")
    args = ap.parse_args()
    recs = [json.loads(Path(p).read_text()) for p in args.recording]
    rows = benchmark_rows(Path(args.benchmark)) if args.benchmark else None
    out = write_html(args.out, recs, rows, title=args.title, default_tag=args.tag)
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
