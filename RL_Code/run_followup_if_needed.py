from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def _load_terminal_summary(summary_path: Path) -> dict[str, object] | None:
    if not summary_path.exists():
        return None
    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if payload.get("status") in {"passed", "failed"} else None


def _should_run_followup(summary_path: Path, threshold: float) -> tuple[bool, str]:
    payload = _load_terminal_summary(summary_path)
    if payload is None:
        return True, "The preceding candidate did not produce a full-budget summary."

    scores = payload.get("scores", {})
    try:
        no_maintenance_gap = float(scores["baseline_minus_no_maintenance"])
        risk_comp_gap = float(scores["baseline_minus_risk_comp"])
    except (KeyError, TypeError, ValueError):
        return True, "The preceding full-budget summary lacks comparable score gaps."

    if no_maintenance_gap < threshold or risk_comp_gap < threshold:
        return (
            True,
            "At least one paired full-budget gap is below "
            f"{threshold:.3f}: baseline-no-maintenance={no_maintenance_gap:.6f}, "
            f"baseline-risk-comp={risk_comp_gap:.6f}.",
        )
    return (
        False,
        "Both paired full-budget gaps meet the threshold: "
        f"baseline-no-maintenance={no_maintenance_gap:.6f}, "
        f"baseline-risk-comp={risk_comp_gap:.6f}.",
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a follow-up hazard candidate only when the prior candidate lacks separation."
    )
    parser.add_argument("--source-run-dir", type=Path, required=True)
    parser.add_argument(
        "--completion-dir",
        type=Path,
        help="Directory where SUCCESS.json or NO_SUCCESS.txt is written; defaults to source-run-dir.",
    )
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--threshold", type=float, default=0.05)
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    args = parser.parse_args()

    source_dir = args.source_run_dir.resolve()
    completion_dir = (
        args.completion_dir.resolve() if args.completion_dir is not None else source_dir
    )
    full_summary = source_dir / "full_summary.json"
    terminal_markers = [completion_dir / "SUCCESS.json", completion_dir / "NO_SUCCESS.txt"]
    print(
        f"Waiting for preceding candidate at {source_dir}; "
        f"threshold={args.threshold:.3f}",
        flush=True,
    )
    while (
        _load_terminal_summary(full_summary) is None
        and not any(path.exists() for path in terminal_markers)
    ):
        time.sleep(max(5.0, float(args.poll_seconds)))

    should_run, reason = _should_run_followup(full_summary, float(args.threshold))
    print(reason, flush=True)
    if not should_run:
        print("Follow-up skipped: separation threshold was met.", flush=True)
        return 0

    script_dir = Path(__file__).resolve().parent
    command = [
        sys.executable,
        "-u",
        str(script_dir / "hazard_sweep_experiment.py"),
        "--candidate",
        str(args.candidate),
    ]
    print(f"Starting follow-up: {' '.join(command)}", flush=True)
    return subprocess.run(command, cwd=script_dir.parent.parent).returncode


if __name__ == "__main__":
    raise SystemExit(main())
