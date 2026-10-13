# -*- coding: utf-8 -*-
"""bed_runner.py - run a GRID of bed arms in sequence on one card: the launch point for a rented card (a local queue may drive
bed_train.py directly). A small grid spec expands into bed_train.py invocations; finished invocations are remembered in
<out>/runner_done.txt so a restart resumes; every run prints its place in the grid, the minutes it took and the time left; a stop
file <out>/STOP ends the grid between runs (never mid-run); each finished run can ship its record and weights to a Hub repo.

From Python (a notebook cell):
    from geolip.nano_beatrix.bed_runner import run
    run({"arms": "castor-both,pollux-both", "seeds": "1,2", "ctx": 4096, "mb": 8, "steps": 12000, "frames": "dyn", ...},
        out="/content/out", push="AbstractPhil/alephllm-nano-beatrix-training")
Each run is a child process whose lines are read as they come: a grid bar counts the runs, a step bar follows the trainer's own
progress lines (step, bits a byte, tok/s, time left), the job-size and gauge lines are written through the bars, and the whole log
goes to <out>/runner.log. The bars are tqdm.auto (notebook widgets in Colab, text in a terminal); without tqdm the lines print plain.

From the command line:
    python -m geolip.nano_beatrix.bed_runner --out=DIR --arms=smr-fixed,altn-fixed --seeds=1,2 --ctx=1024 --steps=12000 --mb=32 \\
        [--frames=dyn --sizes=32,128,256 --ask=any --facts=0,64,256 ...] [--push=AbstractPhil/alephllm-nano-beatrix-training] [--dry=1]
Grid keys (a comma list expands into the product): arms, seeds, ctx, steps, mb, frames, ask, facts, register, tags, n_lookup, n_bulk,
frame_records, nest, overlap, records, hub_K, hub_D, hub_const, plant, lr, cues. Everything else passes through to bed_train.py as given
(--sizes keeps its own comma list). Shipping needs HF_TOKEN in the environment; without it the runner only prints the plan."""
from __future__ import annotations

import itertools
import os
import re
import subprocess
import sys
import time

GRID_KEYS = ("arms", "seeds", "ctx", "steps", "mb", "frames", "ask", "facts", "register", "tags", "n_lookup", "n_bulk", "frame_records",
             "nest", "overlap", "records", "hub_K", "hub_D", "hub_const", "plant", "lr", "cues")
SINGULAR = {"arms": "arm", "seeds": "seed"}
RUNNER_KEYS = ("out", "dry", "push", "python")
STEP_RE = re.compile(r"step (\d+)/(\d+) loss [\d.]+ ema [\d.]+ \(([\d.]+) bpb\).*?\| (\d+)k tok/s \| spent ([\d.]+) min, left ([\d.]+) min")


def parse_args(argv: list) -> dict:
    return {a.split("=")[0].lstrip("-"): a.split("=", 1)[1] for a in argv if "=" in a}


def expand(args: dict) -> list:
    """The grid: every grid key's comma list, the product in a fixed order; -> a list of bed_train argument lists (without --out)."""
    axes, fixed = [], []
    for k, v in args.items():
        if k in RUNNER_KEYS:
            continue
        v = str(v)
        name = SINGULAR.get(k, k)
        if k in GRID_KEYS and "," in v:
            axes.append([(name, x.strip()) for x in v.split(",") if x.strip()])
        else:
            fixed.append((name, v))
    runs = []
    for combo in itertools.product(*axes) if axes else [()]:
        kv = dict(fixed)
        kv.update(dict(combo))
        runs.append([f"--{k}={v}" for k, v in kv.items()])
    return runs


def run_key(run: list) -> str:
    return " ".join(sorted(run))


def run_label(run: list) -> str:
    """A short name for a run's bar: its arm, seed and register (when framed)."""
    kv = dict(a.lstrip("-").split("=", 1) for a in run)
    parts = [kv.get("arm", "?"), f"s{kv.get('seed', '?')}"]
    if kv.get("frames"):
        parts.append(kv.get("register", "caption"))
    return " ".join(parts)


def parse_step(line: str) -> dict | None:
    """The trainer's progress line -> step, total, bits a byte, k tok/s, minutes spent, minutes left; None for any other line."""
    m = STEP_RE.search(line)
    if not m:
        return None
    return {"step": int(m.group(1)), "total": int(m.group(2)), "bpb": float(m.group(3)), "tok_s": int(m.group(4)),
            "spent": float(m.group(5)), "left": float(m.group(6))}


def wrote_path(stdout: str) -> str | None:
    """The record the trainer wrote (its last 'wrote <path>' line), or None."""
    m = re.findall(r"wrote (\S+\.json)", stdout)
    return m[-1] if m else None


def ship(repo: str, record: str, out: str, log) -> None:
    """Ship one finished run: its record to bed/reports/ and its weights folder to bed/weights/<tag>/ on the Hub (HF_TOKEN set)."""
    from huggingface_hub import HfApi
    api = HfApi(token=os.environ["HF_TOKEN"])
    name = os.path.basename(record)
    tag = name[4:-5]
    api.upload_file(path_or_fileobj=record, path_in_repo=f"bed/reports/{name}", repo_id=repo, commit_message=f"bed: {tag} record")
    wdir = os.path.join(out, "bed", tag)
    if os.path.isdir(wdir):
        api.upload_folder(folder_path=wdir, path_in_repo=f"bed/weights/{tag}", repo_id=repo, commit_message=f"bed: {tag} weights")
    log(f"shipped {tag} to {repo}")


def _tqdm():
    try:
        from tqdm.auto import tqdm
        return tqdm
    except ImportError:
        return None


def _emit(bar, text: str) -> None:
    """Print through the bar when there is one, and never let the display cost the run."""
    try:
        (bar.write if bar is not None else print)(text)
    except Exception:  # noqa: BLE001
        print(text)


def run(spec: dict, out: str, push: str = "", python: str | None = None, dry: bool = False) -> dict:
    """Run the grid `spec` (the command-line keys; values as strings or numbers; comma lists expand into the product) under `out`.
    Resumes from <out>/runner_done.txt, stops between runs at <out>/STOP, ships each finished run when `push` names a Hub repo and
    HF_TOKEN is set. Returns {"runs", "done_before", "ran": [records], "failed": rc or None, "stopped": bool}."""
    py = python or sys.executable
    os.makedirs(out, exist_ok=True)
    done_f, logf = os.path.join(out, "runner_done.txt"), os.path.join(out, "runner.log")
    done = set(open(done_f, encoding="utf-8").read().splitlines()) if os.path.exists(done_f) else set()
    runs = expand({k: v for k, v in spec.items() if k not in RUNNER_KEYS})
    todo = [r for r in runs if run_key(r) not in done]
    tqdm = _tqdm()
    t0 = time.time()

    def log(msg: str, bar=None) -> None:
        line = f"[{time.strftime('%H:%M:%S')} +{time.time() - t0:6.0f}s] {msg}"
        _emit(bar, line)
        with open(logf, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    result = {"runs": len(runs), "done_before": len(runs) - len(todo), "ran": [], "failed": None, "stopped": False}
    log(f"grid: {len(runs)} runs, {len(runs) - len(todo)} done before, {len(todo)} to run; out {out}; push {push or 'no'}")
    for i, r in enumerate(todo):
        log(f"  {i + 1}/{len(todo)}: {' '.join(r)}")
    if dry or not todo:
        return result
    if push and not os.environ.get("HF_TOKEN"):
        log("no HF_TOKEN in the environment: the runs will not ship; set it or drop push")
        result["failed"] = 2
        return result
    gbar = tqdm(total=len(todo), unit="run", dynamic_ncols=True, desc="grid") if tqdm else None
    took = []
    try:
        for i, r in enumerate(todo):
            if os.path.exists(os.path.join(out, "STOP")):
                log("STOP present; the grid ends here", gbar)
                result["stopped"] = True
                break
            log(f"run {i + 1}/{len(todo)}: {' '.join(r)}", gbar)
            t1, sbar, lines = time.time(), None, []
            cmd = [py, "-X", "utf8", "-m", "geolip.nano_beatrix.bed_train", f"--out={out}", *r]
            p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", bufsize=1)
            try:
                for raw in p.stdout:
                    line = raw.rstrip("\n")
                    lines.append(line)
                    with open(logf, "a", encoding="utf-8") as f:
                        f.write(line + "\n")
                    st = parse_step(line)
                    if st is None:
                        _emit(sbar or gbar, line)                      # the job size, the gauges, the write: shown through the bars
                        continue
                    if sbar is None and tqdm:
                        sbar = tqdm(total=st["total"], unit="step", dynamic_ncols=True, desc=run_label(r), leave=False)
                    if sbar is not None:
                        sbar.n = st["step"]
                        sbar.set_postfix_str(f"{st['bpb']:.3f} bpb, {st['tok_s']}k tok/s, {st['left']:.1f} min left", refresh=False)
                        sbar.refresh()
                    elif st["step"] % 1000 == 1:                       # no tqdm: one plain line per 1,000 steps
                        print(line, flush=True)
                p.wait()
            except KeyboardInterrupt:
                p.terminate()
                p.wait()
                log("interrupted: the running run is lost (the bed has no mid-run checkpoint); the finished runs are remembered", gbar)
                result["stopped"] = True
                raise
            finally:
                if sbar is not None:
                    sbar.close()
            if p.returncode != 0:
                _emit(gbar, "\n".join(lines[-30:]))
                log(f"run failed rc={p.returncode}; the grid stops (fix, then run again: the finished runs are remembered)", gbar)
                result["failed"] = p.returncode
                break
            took.append(time.time() - t1)
            rec = wrote_path("\n".join(lines))
            if push and rec:
                ship(push, rec, out, lambda m: log(m, gbar))
            with open(done_f, "a", encoding="utf-8") as f:
                f.write(run_key(r) + "\n")
            result["ran"].append(rec or run_key(r))
            if gbar is not None:
                gbar.update(1)
            left = (len(todo) - i - 1) * sum(took) / len(took)
            log(f"run {i + 1}/{len(todo)} done in {took[-1] / 60:.1f} min; {len(todo) - i - 1} left, about {left / 60:.0f} min", gbar)
    finally:
        if gbar is not None:
            gbar.close()
    if result["failed"] is None and not result["stopped"]:
        try:
            from .bed_table import render
            with open(os.path.join(out, "bed", "README.md"), "w", encoding="utf-8") as f:
                f.write(render(os.path.join(out, "bed"), pairs=True))
            log("results index written to bed/README.md")
        except Exception as e:  # noqa: BLE001
            log(f"results index not written: {e}")
        log("grid done")
    return result


def main(argv=None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    spec = {k: v for k, v in args.items() if k not in RUNNER_KEYS}
    res = run(spec, args["out"], push=args.get("push", ""), python=args.get("python"), dry=args.get("dry", "0") == "1")
    if res["failed"]:
        return int(res["failed"])
    return 3 if res["stopped"] else 0


if __name__ == "__main__":
    sys.exit(main())
