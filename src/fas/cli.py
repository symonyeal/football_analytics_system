"""Command-line entry point for fas.

    fas demo          run the synthetic end-to-end pipeline (no network/data)
    fas version       print the package version
"""

from __future__ import annotations

import argparse

from fas import __version__


def _demo() -> None:
    """Run the offline synthetic pipeline (graph -> xT -> PVS -> squad MILP)."""
    from fas.examples.synthetic_pipeline import run_demo

    run_demo()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="fas", description="Football Analytics System")
    sub = parser.add_subparsers(dest="cmd")
    sub.add_parser("demo", help="run the offline synthetic end-to-end pipeline")
    sub.add_parser("version", help="print version")

    args = parser.parse_args(argv)
    if args.cmd == "version":
        print(__version__)
    elif args.cmd == "demo":
        _demo()
    else:
        parser.print_help()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
