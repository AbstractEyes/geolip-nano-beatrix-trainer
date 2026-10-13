# -*- coding: utf-8 -*-
"""bed_runner.py - run a GRID of bed arms on one card: the launch point for a rented card (a local queue may drive bed_train.py
directly). A small grid spec expands into bed_train.py invocations; finished invocations are remembered in <out>/runner_done.txt so a
restart resumes; every run prints its place in the grid, the minutes it took and the time left; a stop file <out>/STOP lets the running
runs finish and starts no new one; each finished run can ship its record and weights to a Hub repo. Several runs can share the card at
once (`parallel`): a 9M craft at 8 x 4,096 takes about 5 GB and a fraction of a large card's compute, so the grid's wall time falls
until the card or the CPU-side row composition saturates; the per-run tok/s on the step bars is the measurement.

From Python (a notebook cell):
    from geolip.nano_beatrix.bed_runner import run
    run({"arms": "castor-both,pollux-both", "seeds": "1,2", "ctx": 4096, "mb": 8, "steps": 12000, "frames": "dyn", ...},
        out="/content/out", push="AbstractPhil/alephllm-nano-beatrix-training", parallel=4)
Each run is a child process whose lines are read as they come: a grid bar counts the runs, a step bar per running child follows the
trainer's own progress lines (step, bits a byte, tok/s, time left), the job-size and gauge lines are written through the bars with the
run's label, and the whole log goes to <out>/runner.log. The bars are tqdm.auto (notebook widgets in Colab, text in a terminal);
without tqdm the lines print plain.

From the command line:
    python -m geolip.nano_beatrix.bed_runner --out=DIR --arms=smr-fixed,altn-fixed --seeds=1,2 --ctx=1024 --steps=12000 --mb=32 \\
        [--frames=dyn --sizes=32,128,256 --ask=any --facts=0,64,256 ...] [--push=AbstractPhil/alephllm-nano-beatrix-training] [--parallel=4] [--dry=1]
Grid keys (a comma list expands into the product): arms, seeds, ctx, steps, mb, frames, ask, facts, register, tags, n_lookup, n_bulk,
frame_records, nest, overlap, records, hub_K, hub_D, hub_const, plant, lr, cues. Everything else passes through to bed_train.py as given
(--sizes keeps its own comma list). Shipping needs HF_TOKEN in the environment; without it the runner only prints the plan."""
from __future__ import annotations

import itertools
import os
import re
import subprocess
import sys
import threading
import time

GRID_KEYS = ("arms", "seeds", "ctx", "steps", "mb", "frames", "ask", "facts", "register", "tags", "n_lookup", "n_bulk", "frame_records",
             "nest", "overlap", "records", "hub_K", "hub_D", "hub_const", "plant", "lr", "cues")
SINGULAR = {"arms": "arm", "seeds": "seed"}
RUNNER_KEYS = ("out", "dry", "push", "python", "parallel")
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


class _Child:
    """One running bed_train.py: the process, a reader thread that follows its lines, its step bar."""

    def __init__(self, run_args: list, cmd: list, slot: int, tqdm, logf: str, gbar) -> None:
        self.run, self.label, self.slot, self.lines, self.t0 = run_args, run_label(run_args), slot, [], time.time()
        self.tqdm, self.logf, self.gbar, self.bar = tqdm, logf, gbar, None
        self.p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", bufsize=1)
        self.th = threading.Thread(target=self._read, daemon=True)
        self.th.start()

    def _read(self) -> None:
        for raw in self.p.stdout:
            line = raw.rstrip("\n")
            self.lines.append(line)
            with open(self.logf, "a", encoding="utf-8") as f:
                f.write(f"[{self.label}] {line}\n")
            st = parse_step(line)
            if st is None:
                _emit(self.bar or self.gbar, f"[{self.label}] {line}")       # the job size, the gauges, the write: shown through the bars
                continue
            if self.bar is None and self.tqdm:
                self.bar = self.tqdm(total=st["total"], unit="step", dynamic_ncols=True, desc=self.label, position=self.slot + 1, leave=False)
            if self.bar is not None:
                self.bar.n = st["step"]
                self.bar.set_postfix_str(f"{st['bpb']:.3f} bpb, {st['tok_s']}k tok/s, {st['left']:.1f} min left", refresh=False)
                self.bar.refresh()
            elif st["step"] % 1000 == 1:                                      # no tqdm: one plain line per 1,000 steps
                print(f"[{self.label}] {line}", flush=True)

    def finished(self) -> bool:
        return self.p.poll() is not None

    def close(self) -> None:
        self.th.join(timeout=10)
        if self.bar is not None:
            self.bar.close()


def run(spec: dict, out: str, push: str = "", python: str | None = None, dry: bool = False, parallel: int = 1) -> dict:
    """Run the grid `spec` (the command-line keys; values as strings or numbers; comma lists expand into the product) under `out`, up to
    `parallel` runs at once on the card. Resumes from <out>/runner_done.txt; <out>/STOP starts no new run and lets the running ones finish;
    ships each finished run when `push` names a Hub repo and HF_TOKEN is set. Returns {"runs", "done_before", "ran": [records],
    "failed": rc or None, "stopped": bool}."""
    py = python or sys.executable
    parallel = max(1, int(parallel))
    os.makedirs(out, exist_ok=True)
    done_f, logf, stop_f = os.path.join(out, "runner_done.txt"), os.path.join(out, "runner.log"), os.path.join(out, "STOP")
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
    log(f"grid: {len(runs)} runs, {len(runs) - len(todo)} done before, {len(todo)} to run, {parallel} at a time; out {out}; push {push or 'no'}")
    for i, r in enumerate(todo):
        log(f"  {i + 1}/{len(todo)}: {' '.join(r)}")
    if dry or not todo:
        return result
    if push and not os.environ.get("HF_TOKEN"):
        log("no HF_TOKEN in the environment: the runs will not ship; set it or drop push")
        result["failed"] = 2
        return result
    gbar = tqdm(total=len(todo), unit="run", dynamic_ncols=True, desc="grid", position=0) if tqdm else None
    pending, active, took, halt = list(todo), [], [], False
    try:
        while pending or active:
            while pending and len(active) < parallel and not halt:
                if os.path.exists(stop_f):
                    log("STOP present: no new run starts; the running ones finish", gbar)
                    result["stopped"] = halt = True
                    break
                r = pending.pop(0)
                slot = min(set(range(parallel)) - {c.slot for c in active})
                log(f"run {len(todo) - len(pending)}/{len(todo)} starts: {' '.join(r)}", gbar)
                active.append(_Child(r, [py, "-X", "utf8", "-m", "geolip.nano_beatrix.bed_train", f"--out={out}", *r], slot, tqdm, logf, gbar))
            time.sleep(0.5)
            for c in [c for c in active if c.finished()]:
                active.remove(c)
                c.close()
                rc = c.p.returncode
                if rc != 0:
                    _emit(gbar, "\n".join(c.lines[-30:]))
                    log(f"{c.label} failed rc={rc}; no new run starts (fix, then run again: the finished runs are remembered)", gbar)
                    result["failed"] = rc
                    halt = True
                    continue
                took.append(time.time() - c.t0)
                rec = wrote_path("\n".join(c.lines))
                if push and rec:
                    ship(push, rec, out, lambda m: log(m, gbar))
                with open(done_f, "a", encoding="utf-8") as f:
                    f.write(run_key(c.run) + "\n")
                result["ran"].append(rec or run_key(c.run))
                if gbar is not None:
                    gbar.update(1)
                left = (len(pending) + len(active)) * (sum(took) / len(took)) / parallel
                log(f"{c.label} done in {took[-1] / 60:.1f} min; {len(pending) + len(active)} left, about {left / 60:.0f} min", gbar)
    except KeyboardInterrupt:
        for c in active:
            c.p.terminate()
        for c in active:
            c.p.wait()
            c.close()
        log("interrupted: the running runs are lost (the bed has no mid-run checkpoint); the finished runs are remembered", gbar)
        result["stopped"] = True
        raise
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
    res = run(spec, args["out"], push=args.get("push", ""), python=args.get("python"), dry=args.get("dry", "0") == "1",
              parallel=int(args.get("parallel", "1")))
    if res["failed"]:
        return int(res["failed"])
    return 3 if res["stopped"] else 0


if __name__ == "__main__":
    sys.exit(main())
