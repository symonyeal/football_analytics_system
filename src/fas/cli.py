"""Command-line entry point for fas.

    fas demo            run the local-data or synthetic end-to-end pipeline
    fas product-build   materialize the full product artifact set
    fas ui              launch the Streamlit analytics workspace
    fas report          build a static HTML report (no Streamlit needed)
    fas version         print the package version
"""

from __future__ import annotations

import argparse
import sys

from fas import __version__


def _demo(data_path: str | None = None, write_summary: bool = True) -> None:
    from fas.examples.synthetic_pipeline import run_demo

    run_demo(data_path=data_path, write_summary=write_summary)


def _product_build(args) -> None:
    from fas.product.build import product_build

    summary = product_build(
        data_path=args.data, allow_download=not args.no_download, seed=args.seed,
        sb_competition=args.sb_competition, sb_season=args.sb_season,
        sb_team=args.sb_team, sb_max_matches=args.sb_max_matches,
        sb_per_season=args.sb_per_season)
    print("\nRun the UI with:\n    python -m fas.cli ui")
    if summary["is_synthetic"]:
        print("(Displayed data is deterministic synthetic demo data.)")


def _ui(data_root: str, port: int, no_download: bool, seed: int) -> int:
    from pathlib import Path

    from fas.product.loader import artifacts_present

    if not artifacts_present(data_root):
        print("Product artifacts not found — building them first ...")
        from fas.product.loader import ensure_artifacts

        ensure_artifacts(data_root, allow_download=not no_download, seed=seed, verbose=True)

    try:
        import streamlit  # noqa: F401
    except ImportError:
        print("Streamlit is not installed. Either install it:\n"
              "    pip install streamlit\n"
              "or build a static HTML report instead:\n"
              "    python -m fas.cli report")
        return 1

    import subprocess

    app = Path(__file__).parent / "ui" / "app.py"
    print(f"Launching Streamlit UI at http://localhost:{port} ...")
    return subprocess.call([
        sys.executable, "-m", "streamlit", "run", str(app),
        "--server.port", str(port), "--server.headless", "true",
    ])


def _report(data_root: str, no_download: bool, seed: int) -> None:
    from fas.product.loader import ensure_artifacts
    from fas.ui.report import build_report

    ensure_artifacts(data_root, allow_download=not no_download, seed=seed, verbose=True)
    path = build_report(data_root)
    print(f"Static HTML report written to:\n    {path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="fas", description="Football Analytics System")
    sub = parser.add_subparsers(dest="cmd")

    demo = sub.add_parser("demo", help="run the offline end-to-end pipeline")
    demo.add_argument("--data", help="canonical actions file (.csv, .json, .parquet)")
    demo.add_argument("--no-summary", action="store_true",
                      help="do not write data/processed/demo_summary.json")

    pb = sub.add_parser("product-build", help="materialize the full product artifact set")
    pb.add_argument("--data", help="canonical actions file to ingest")
    pb.add_argument("--no-download", action="store_true",
                    help="never attempt a StatsBomb download; use local/synthetic only")
    pb.add_argument("--seed", type=int, default=7, help="synthetic-data seed")
    pb.add_argument("--sb-competition", default=None,
                    help="limit real data to one competition (default: all)")
    pb.add_argument("--sb-season", default=None,
                    help="limit real data to one season (default: all)")
    pb.add_argument("--sb-team", default=None,
                    help="limit real data to one team (default: all teams)")
    pb.add_argument("--sb-max-matches", type=int, default=60,
                    help="cap total real matches ingested (0 = all; slow)")
    pb.add_argument("--sb-per-season", type=int, default=4,
                    help="matches sampled per competition-season for breadth")

    ui = sub.add_parser("ui", help="launch the Streamlit analytics workspace")
    ui.add_argument("--data-root", default="data", help="artifact root (default: data)")
    ui.add_argument("--port", type=int, default=8501)
    ui.add_argument("--no-download", action="store_true")
    ui.add_argument("--seed", type=int, default=7)

    rep = sub.add_parser("report", help="build a static HTML report (no Streamlit)")
    rep.add_argument("--data-root", default="data")
    rep.add_argument("--no-download", action="store_true")
    rep.add_argument("--seed", type=int, default=7)

    sub.add_parser("version", help="print version")

    args = parser.parse_args(argv)
    if args.cmd == "version":
        print(__version__)
    elif args.cmd == "demo":
        _demo(data_path=args.data, write_summary=not args.no_summary)
    elif args.cmd == "product-build":
        _product_build(args)
    elif args.cmd == "ui":
        return _ui(args.data_root, args.port, args.no_download, args.seed)
    elif args.cmd == "report":
        _report(args.data_root, args.no_download, args.seed)
    else:
        parser.print_help()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
