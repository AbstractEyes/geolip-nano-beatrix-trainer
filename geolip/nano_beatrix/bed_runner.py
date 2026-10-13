# -*- coding: utf-8 -*-
"""bed_runner.py - run a GRID of bed arms in sequence on one card: the launch point for a rented card (a local queue may drive
bed_train.py directly). A small grid spec expands into bed_train.py invocations; finished invocations are remembered in
<out>/runner_done.txt so a restart resumes; every run prints its place in the grid, the minutes it took and the time left; a stop
file <out>/STOP ends the grid between runs (never mid-run); each finished run can ship its record and weights to a Hub repo.
Usage:
    python -m geolip.nano_beatrix.bed_runner --out=DIR --arms=smr-fixed,altn-fixed --seeds=1,2 --ctx=1024 --steps=12000 --mb=32 \\
        [--frames=dyn --sizes=32,128,256 --ask=any --facts=0,64,256 ...] [--push=AbstractPhil/alephllm-nano-beatrix-training] [--dry=1]
Grid keys (a comma list expands into the product): arms, seeds, ctx, steps, mb, frames, ask, facts, register, tags, n_lookup, n_bulk,
frame_records, nest, overlap, records, hub_K, hub_D, hub_const, plant, lr, cues. Everything else passes through to bed_train.py as given
(--sizes keeps its own comma list). --push needs HF_TOKEN in the environment; without it the runner only prints the plan."""
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


def parse_args(argv: list) -> dict:
    return {a.split("=")[0].lstrip("-"): a.split("=", 1)[1] for a in argv if "=" in a}


def expand(args: dict) -> list:
    """The grid: every grid key's comma list, the product in a fixed order; -> a list of bed_train argument lists (without --out)."""
    axes, fixed = [], []
    for k, v in args.items():
        if k in RUNNER_KEYS:
            continue
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


def main(argv=None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    out = args["out"]
    dry = args.get("dry", "0") == "1"
    push = args.get("push", "")
    py = args.get("python", sys.executable)
    os.makedirs(out, exist_ok=True)
    done_f = os.path.join(out, "runner_done.txt")
    done = set(open(done_f, encoding="utf-8").read().splitlines()) if os.path.exists(done_f) else set()
    runs = expand(args)
    todo = [r for r in runs if run_key(r) not in done]
    t0 = time.time()

    def log(msg):
        print(f"[{time.strftime('%H:%M:%S')} +{time.time() - t0:6.0f}s] {msg}", flush=True)

    log(f"grid: {len(runs)} runs, {len(runs) - len(todo)} done before, {len(todo)} to run; out {out}; push {push or 'no'}")
    for i, r in enumerate(todo):
        log(f"  {i + 1}/{len(todo)}: {' '.join(r)}")
    if dry:
        return 0
    if push and not os.environ.get("HF_TOKEN"):
        log("no HF_TOKEN in the environment: the runs will not ship; set it or drop --push")
        return 2
    took = []
    for i, r in enumerate(todo):
        if os.path.exists(os.path.join(out, "STOP")):
            log("STOP present; the grid ends here")
            return 3
        log(f"run {i + 1}/{len(todo)}: {' '.join(r)}")
        t1 = time.time()
        p = subprocess.run([py, "-X", "utf8", "-m", "geolip.nano_beatrix.bed_train", f"--out={out}", *r], capture_output=True, text=True)
        sys.stdout.write(p.stdout[-4000:])
        if p.returncode != 0:
            sys.stdout.write(p.stderr[-4000:])
            log(f"run failed rc={p.returncode}; the grid stops (fix, then rerun: the finished runs are remembered)")
            return p.returncode
        took.append(time.time() - t1)
        rec = wrote_path(p.stdout)
        if push and rec:
            ship(push, rec, out, log)
        with open(done_f, "a", encoding="utf-8") as f:
            f.write(run_key(r) + "\n")
        left = (len(todo) - i - 1) * sum(took) / len(took)
        log(f"run {i + 1}/{len(todo)} done in {took[-1] / 60:.1f} min; {len(todo) - i - 1} left, about {left / 60:.0f} min")
    try:
        from . import bed_table  # noqa: F401
        p = subprocess.run([py, "-X", "utf8", "-m", "geolip.nano_beatrix.bed_table", os.path.join(out, "bed"), "--pairs"], capture_output=True, text=True)
        with open(os.path.join(out, "bed", "README.md"), "w", encoding="utf-8") as f:
            f.write(p.stdout)
        log("results index written to bed/README.md")
    except Exception as e:  # noqa: BLE001
        log(f"results index not written: {e}")
    log("grid done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
