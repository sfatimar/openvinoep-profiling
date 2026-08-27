#!/usr/bin/env python3
"""Benchmark a ResNet model with the Microsoft WinML CLI.

Drives the three-step winml-cli workflow against the `winml_cli` virtual
environment created by setup_winml_cli.bat:

    1. inspect   -- confirm the source model is recognised before downloading it
    2. build     -- export / optimize into an ONNX model directory
    3. perf      -- benchmark the exported model

The headline invocation this script wraps is:

    winml perf -m resnet_out/model.onnx --device auto --iterations 50 --monitor

Usage
-----
    python resnet_perf_example.py                       # full pipeline
    python resnet_perf_example.py --skip-build          # re-benchmark an existing export
    python resnet_perf_example.py --device npu          # pin the device
    python resnet_perf_example.py --iterations 200      # longer run
    python resnet_perf_example.py --via-uv              # invoke through `uv run --active`

Reference: https://github.com/microsoft/winml-cli
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_VENV = "winml_cli"
DEFAULT_MODEL_ID = "microsoft/resnet-50"
DEFAULT_OUT_DIR = "resnet_out"

# winml's progress bars use Unicode block glyphs, but a Windows console defaults to a
# legacy code page (cp1252 here). Without this, echoing child output raises
# UnicodeEncodeError. Done at import so it is in effect for any entry point, not just main().
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    except (AttributeError, ValueError):  # not a TextIOWrapper (e.g. redirected/wrapped)
        pass


# --------------------------------------------------------------------------- #
# environment resolution
# --------------------------------------------------------------------------- #

def find_winml(venv_dir: Path) -> Path:
    """Locate the `winml` entry point inside the virtual environment.

    The env is named `winml_cli` rather than `.venv`, so `uv run winml` will not
    auto-discover it. Calling the entry point directly sidesteps that entirely.
    """
    candidates = [
        venv_dir / "Scripts" / "winml.exe",   # Windows
        venv_dir / "Scripts" / "winml",
        venv_dir / "bin" / "winml",           # POSIX
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate

    sys.exit(
        f"error: no 'winml' entry point found under {venv_dir}\n"
        f"       Run setup_winml_cli.bat first (or pass --venv if the env lives elsewhere)."
    )


def build_command(args: list[str], winml: Path, venv_dir: Path, via_uv: bool) -> tuple[list[str], dict[str, str]]:
    """Return (argv, env) for a winml subcommand.

    Two invocation styles:
      - direct  (default): call the venv's winml executable. No uv dependency at run time.
      - via_uv:            `uv run --active winml ...` with VIRTUAL_ENV pointed at the env,
                           matching the form documented in the winml-cli README.
    """
    env = os.environ.copy()

    # winml draws progress bars with Unicode block glyphs. Force UTF-8 on the child so
    # its output is decodable regardless of the console code page -- on a cp1252 console
    # reading stdout otherwise dies with UnicodeDecodeError partway through a download.
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"

    if not via_uv:
        return [str(winml), *args], env

    if shutil.which("uv") is None:
        sys.exit("error: --via-uv requested but 'uv' is not on PATH")

    # `uv run` looks for `.venv` by default; --active makes it honour VIRTUAL_ENV instead.
    env["VIRTUAL_ENV"] = str(venv_dir)
    return ["uv", "run", "--active", "winml", *args], env


# --------------------------------------------------------------------------- #
# step runner
# --------------------------------------------------------------------------- #

def run_step(title: str, args: list[str], *, winml: Path, venv_dir: Path,
             via_uv: bool, log_path: Path | None, allow_failure: bool = False) -> int:
    """Run one winml subcommand, streaming output to the console and optionally a log."""
    argv, env = build_command(args, winml, venv_dir, via_uv)

    print(f"\n{'=' * 78}\n== {title}\n== $ {' '.join(argv)}\n{'=' * 78}", flush=True)

    started = dt.datetime.now()
    lines: list[str] = []

    # Stream rather than capture-then-print: `--monitor` emits progress during the run,
    # and a long benchmark should not look hung.
    # Decode as UTF-8 explicitly; `text=True` alone would use the locale encoding.
    # errors="replace" keeps a stray byte from aborting a long-running benchmark.
    with subprocess.Popen(
        argv, cwd=HERE, env=env, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True, bufsize=1,
        encoding="utf-8", errors="replace",
    ) as proc:
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            lines.append(line)
        rc = proc.wait()

    elapsed = (dt.datetime.now() - started).total_seconds()
    print(f"\n-- {title}: exit={rc} elapsed={elapsed:.1f}s", flush=True)

    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        header = (
            f"# {title}\n"
            f"# command : {' '.join(argv)}\n"
            f"# started : {started.isoformat(timespec='seconds')}\n"
            f"# elapsed : {elapsed:.1f}s\n"
            f"# exit    : {rc}\n\n"
        )
        log_path.write_text(header + "".join(lines), encoding="utf-8")
        print(f"-- log: {log_path}", flush=True)

    if rc != 0 and not allow_failure:
        sys.exit(f"error: '{title}' failed with exit code {rc}")

    return rc


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    p.add_argument("--venv", default=DEFAULT_VENV,
                   help=f"virtual environment directory (default: {DEFAULT_VENV})")
    p.add_argument("--model-id", default=DEFAULT_MODEL_ID,
                   help=f"source model to export (default: {DEFAULT_MODEL_ID})")
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR,
                   help=f"export output directory (default: {DEFAULT_OUT_DIR})")
    p.add_argument("--model-path", default=None,
                   help="benchmark this ONNX file directly instead of <out-dir>/model.onnx")

    p.add_argument("--device", default="npu",
                   help="target device: auto | npu | gpu | cpu (default: auto, which "
                        "prefers NPU, then GPU, then CPU)")
    p.add_argument("--iterations", type=int, default=1,
                   help="benchmark iterations (default: 1)")

    # The README shows `--monitor` as a bare flag, so it is treated as one here.
    # If a future CLI revision gives it a value (e.g. a run label), switch this to
    # `p.add_argument("--monitor", nargs="?", const=True)` and pass it through below.
    p.add_argument("--monitor", dest="monitor", action="store_true", default=True,
                   help="enable resource monitoring during the run (default: on)")
    p.add_argument("--no-monitor", dest="monitor", action="store_false",
                   help="disable resource monitoring")

    p.add_argument("--quant", dest="no_quant", action="store_false", default=True,
                   help="enable quantization during build (default: disabled, i.e. --no-quant)")
    p.add_argument("--skip-inspect", action="store_true",
                   help="skip the inspect step")
    p.add_argument("--skip-build", action="store_true",
                   help="skip export/build and benchmark the existing model")
    p.add_argument("--skip-sys", action="store_true",
                   help="skip the device/EP enumeration step")

    p.add_argument("--log-dir", default="logs",
                   help="directory for per-step logs (default: logs; '-' to disable)")
    p.add_argument("--via-uv", action="store_true",
                   help="invoke through 'uv run --active winml' instead of the venv executable")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    venv_dir = (HERE / args.venv).resolve()
    winml = find_winml(venv_dir)

    out_dir = (HERE / args.out_dir).resolve()
    model_path = Path(args.model_path).resolve() if args.model_path else out_dir / "model.onnx"

    stamp = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = None if args.log_dir == "-" else (HERE / args.log_dir / f"resnet_{stamp}").resolve()

    common = dict(winml=winml, venv_dir=venv_dir, via_uv=args.via_uv)

    print(f"winml    : {winml}")
    print(f"model id : {args.model_id}")
    print(f"model    : {model_path}")
    print(f"device   : {args.device}   iterations: {args.iterations}   monitor: {args.monitor}")
    print(f"logs     : {log_dir if log_dir else '(disabled)'}")

    def log_for(name: str) -> Path | None:
        return None if log_dir is None else log_dir / f"{name}.log"

    # 1. Devices and execution providers -- records the hardware the numbers came from.
    #    Non-fatal: enumeration can fail on older Windows builds while perf still works.
    if not args.skip_sys:
        run_step("winml sys (devices / EPs)",
                 ["sys", "--list-device", "--list-ep"],
                 log_path=log_for("01_sys"), allow_failure=True, **common)

    # 2. Inspect before building -- catches unsupported architectures early, before
    #    paying the download and export cost.
    if not args.skip_inspect:
        run_step(f"winml inspect ({args.model_id})",
                 ["inspect", "-m", args.model_id],
                 log_path=log_for("02_inspect"), **common)

    # 3. Build: export -> optimize -> (quantize).
    if args.skip_build:
        print(f"\n-- skipping build (--skip-build)")
    else:
        build_args = ["build", "-m", args.model_id, "-o", f"{args.out_dir}/"]
        if args.no_quant:
            build_args.append("--no-quant")
        run_step(f"winml build ({args.model_id} -> {args.out_dir}/)",
                 build_args, log_path=log_for("03_build"), **common)

    if not model_path.is_file():
        sys.exit(
            f"error: model not found: {model_path}\n"
            f"       The build step may have written a different filename -- inspect "
            f"{out_dir} and pass --model-path explicitly."
        )

    # 4. Benchmark.
    perf_args = [
        "perf",
        "-m", str(model_path),
        "--device", args.device,
        "--iterations", str(args.iterations),
    ]
    if args.monitor:
        perf_args.append("--monitor")

    run_step(f"winml perf ({model_path.name} on {args.device}, {args.iterations} iterations)",
             perf_args, log_path=log_for("04_perf"), **common)

    print(f"\n{'=' * 78}")
    print("Done.")
    if log_dir is not None:
        print(f"Logs: {log_dir}")
    print(f"{'=' * 78}")


if __name__ == "__main__":
    main()
