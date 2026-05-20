"""Command-line entry point for fas.

    fas demo          run the local-data or synthetic end-to-end pipeline
    fas version       print the package version
"""

from __future__ import annotations

import argparse

from fas import __version__


def _demo(data_path: str | None = None, write_summary: bool = True) -> None:
    """Run the offline pipeline and print a compact summary."""
    from fas.examples.synthetic_pipeline import run_demo

    run_demo(data_path=data_path, write_summary=write_summary)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="fas", description="Football Analytics System")
    sub = parser.add_subparsers(dest="cmd")
    demo = sub.add_parser("demo", help="run the offline end-to-end pipeline")
    demo.add_argument("--data", help="canonical actions file (.csv, .json, .parquet)")
    demo.add_argument(
        "--no-summary",
        action="store_true",
        help="do not write data/processed/demo_summary.json",
    )
    sub.add_parser("version", help="print version")

    args = parser.parse_args(argv)
    if args.cmd == "version":
        print(__version__)
    elif args.cmd == "demo":
        _demo(data_path=args.data, write_summary=not args.no_summary)
    else:
        parser.print_help()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
