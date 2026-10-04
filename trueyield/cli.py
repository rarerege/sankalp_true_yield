"""Command line: python -m trueyield run | fetch-public | templates"""
from __future__ import annotations

import argparse
import sys

from .config import load_config, path_of
from .ingest import SystemsFileError, ensure_templates


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):            # ₹ and – on Windows consoles
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(prog="trueyield", description="Independent verification of rooftop solar yield.")
    parser.add_argument("command", choices=["run", "fetch-public", "templates"])
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)

    if args.command == "templates":
        made = ensure_templates(path_of(cfg, "systems_csv"), path_of(cfg, "events_csv"))
        print("Created: " + (", ".join(made) if made else "nothing (both files already exist)"))
        return 0
    if args.command == "fetch-public":
        from .fetch_public import fetch_public
        print(fetch_public(cfg))
        return 0

    from . import pipeline, report
    try:
        run = pipeline.run_all(cfg)
    except SystemsFileError as exc:
        print("\nSTOP: site details are missing and will not be guessed.\n"
              f"  {exc}\n  Fill in {path_of(cfg, 'systems_csv')} and run again.", file=sys.stderr)
        return 2
    except ConnectionError as exc:
        print(f"\nSTOP: data/raw/ is empty and the public dataset could not be fetched ({exc}).\n"
              "  No data are generated in its place. Put inverter files in data/raw/ or retry later.", file=sys.stderr)
        return 3
    written = report.write_all(run)

    for warning in run["economics_warnings"]:
        print(f"NOTE: {warning}; rupee figures are marked not available.")
    for res in run["results"]:
        for note in res["notes"]:
            print(f"NOTE [{res['system_id']}]: {note}")
    print("\nslide6_values.md")
    for label, lines in written["slide"]:
        print(f" {label}")
        for i, line in enumerate(lines, 1):
            print(f"  {i}. {line}")
    for head in written["headline_systems"]:
        if head in written["charts"]:
            print(f"\nCharts for system {head}:")
            for path in written["charts"][head].values():
                print(f"  {path}")
    print(f"\nAll outputs are in {path_of(cfg, 'results_dir')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
