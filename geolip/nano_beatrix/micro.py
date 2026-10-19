# -*- coding: utf-8 -*-
"""micro.py - THE MICRO TIER'S THROUGHPUT SCAN (0.1.57; Phil 2026-10-19 11:5x: "May as well create a micro variant then, step up from nano
while we're at it."; plans/2026-10-19_nano_todo_frame_pretrain_plan.md section 7, plans/2026-10-19_pollux_oven_plan.md section 7). The
bodies live in bodies.SIZES and nano_v1.CRAFTS_MICRO; this module prices them, so the week of bytes a schooled body could see is a number
before any run is promised. For each craft: its parameter count (a CPU build), the model step on random bytes at several batch shapes
(nano_v1.bench: the 2s shape 16 x 4 and the alternatives; a shape that does not fit reads infinite), the shape picked as the v1 runs pick
theirs (nano_v1.pick_shape), then THE REAL TRAINER on the fineweb-edu stream for a few hundred steps at that shape (the data plane in its
child process, the clip, both optimizers, bf16 autocast, eager: the pace a long run would hold), timed from a warm step by the step clock
(the manifest's token count is advanced once a step; the startup and the exit checkpoint stay outside the window); bytes a second, the
card's memory, the bytes a day and a week at one notebook, the bytes a parameter those make, the hours the 2s diet would take. Nothing here
trains a body to keep: the scan's runs write under <out>/scan and are removed; the record ships to the repo's preflight/ folder when a
token is given (the weights never). The long-runs law (10-07): a run of days gets its engineering first; this is its first measurement."""
from __future__ import annotations

import json
import math
import os
import shutil
import statistics
import time

import torch

from . import __version__
from . import nano_v1 as V1
from .bodies import MILLIONS, SIZES

SCAN_NAMES = ("nano-pollux-dual-v1.1", "micro26-pollux-dual-v1.1", "micro45-pollux-dual-v1.1", "micro66-pollux-dual-v1.1")
SHAPES = ((16, 4), (8, 8), (32, 2))      # micro_batch x grad_accum at the 2s step (262,144 bytes); the first is the 2s shape
DAY, WEEK = 86_400.0, 7 * 86_400.0
STREAM = "fineweb-edu"                   # the scan's text: the diet's main stream (the held-out head is reserved by the library)


def diet_bytes() -> int:
    """The 2s diet's bytes (16.1 billion), from the plan itself."""
    return sum(ph["planned_tokens"] for ph in V1.diet())


def shape_of(name: str, ctx: int = V1.CTX) -> dict:
    """The craft's size, width, depth, heads and its parameter count at `ctx` (a CPU build of the body)."""
    from geolip.alephllm.model.alephlm import AlephLM
    size = V1.size_of(name)
    cfg = V1.craft_cfg(V1.ALL[name]["arm"], ctx, size=size)
    n = AlephLM(cfg).param_count()
    return dict(name=name, size=size, arm=V1.ALL[name]["arm"], d_model=cfg.d_model, n_layers=cfg.n_layers, n_heads=cfg.n_heads,
                hub_layers=list(cfg.hub_layers), hub_K=cfg.hub_K, hub_D=cfg.hub_D, head_K=cfg.head_K, head_D=cfg.head_D, params=int(n),
                millions=MILLIONS[size])


def projection(bytes_s: float, params: int, diet: int | None = None) -> dict:
    """What a pace buys: bytes a day and a week at one notebook, the bytes a parameter a week makes, the hours of the 2s diet and of 4x."""
    diet = diet or diet_bytes()
    if not bytes_s or not math.isfinite(bytes_s):
        return dict(day_B=0.0, week_B=0.0, bytes_per_param_week=0.0, diet_hours=float("inf"), diet4x_hours=float("inf"))
    return dict(day_B=bytes_s * DAY / 1e9, week_B=bytes_s * WEEK / 1e9, bytes_per_param_week=bytes_s * WEEK / params,
                diet_hours=diet / bytes_s / 3600, diet4x_hours=4 * diet / bytes_s / 3600)


def scan_preset(name: str, birth_file: str, micro_batch: int, grad_accum: int, steps: int, repo: str = V1.REPO):
    """The scan's preset: the craft's body (at its size) and recipe (nano_v1.preset's; the v1.1 readout on AdamW for a .1 craft), ONE phase
    of `steps` steps on the fineweb-edu stream, the eval, checkpoint, health and TensorBoard cadences past the steps, the weights-only
    start from the birth file at step 0."""
    from geolip.alephllm.presets import Preset, TrainConfig
    cfg = V1.craft_cfg(V1.ALL[name]["arm"], V1.CTX, size=V1.size_of(name))
    cfg.name = f"{name}-scan"
    far = 10 ** 9
    tc = TrainConfig(micro_batch=micro_batch, grad_accum=grad_accum, governor="", head_addr_frozen=False, phase_seed_offset=True,
                     eval_every=far, ckpt_every=far, tb_upload_every=far, health_every=far, log_every=50, **V1.train_of(name))
    tps = micro_batch * grad_accum * V1.CTX
    phases = [dict(name="scan_fineweb", dataset=STREAM, planned_tokens=steps * tps, status="planned")]
    return Preset(model=cfg, train=tc, hf_repo=repo, curriculum=phases, data_scale=1.0, init_from={"file": birth_file, "step": 0})


def stream_run(name: str, birth_file: str, micro_batch: int, grad_accum: int, steps: int, warm: int, out: str, device: str = "cuda",
               log=print) -> dict:
    """The real trainer for `steps` steps on the fineweb-edu stream at this shape, local only (no hub): the pace from the step clock
    after `warm` steps, the card's peak memory, the loss's running mean in bits a byte at the end. The run's folder is removed after."""
    assert 1 <= warm < steps, (warm, steps)
    folder = os.path.join(out, "scan", f"{name}_{micro_batch}x{grad_accum}")
    shutil.rmtree(folder, ignore_errors=True)
    p = scan_preset(name, birth_file, micro_batch, grad_accum, steps)
    t = V1.make_trainer(p, None, folder, process_streams=True, after_eval=None, device=device)
    times = []
    orig = t.manifest.add_tokens

    def add_tokens(n):                                   # called once a step, after the optimizers' step
        times.append(time.time())
        return orig(n)

    t.manifest.add_tokens = add_tokens
    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    try:
        t.train(max_steps=steps)
    finally:
        V1.close_stream(t)
    wall = time.time() - t0
    assert len(times) >= steps, f"{name}: the run stopped after {len(times)} steps, not {steps}"
    seg = times[warm - 1:steps]
    diffs = [b - a for a, b in zip(seg, seg[1:])]
    step_s = (seg[-1] - seg[0]) / len(diffs)
    tps = micro_batch * grad_accum * V1.CTX
    peak = torch.cuda.max_memory_allocated() / 2 ** 30 if device == "cuda" else 0.0
    ema = getattr(t, "_loss_ema", None)
    rec = dict(micro_batch=micro_batch, grad_accum=grad_accum, steps=steps, warm=warm, tokens_per_step=tps, step_s=step_s,
               step_s_median=statistics.median(diffs), step_s_max=max(diffs), steps_s=1.0 / step_s, bytes_s=tps / step_s, peak_gb=peak,
               wall_s=wall, startup_s=times[0] - t0, bpb_ema=None if ema is None else ema / math.log(2))
    del t
    if device == "cuda":
        torch.cuda.empty_cache()
    shutil.rmtree(folder, ignore_errors=True)
    log(f"[micro] {name}: {micro_batch} x {grad_accum}: {rec['bytes_s'] / 1e3:,.0f} kB/s ({rec['steps_s']:.2f} steps/s, median step "
        f"{rec['step_s_median']:.3f} s), peak {peak:.1f} GB, startup {rec['startup_s']:.0f} s, {steps} steps in {wall:.0f} s")
    return rec


def scan(out: str, names=SCAN_NAMES, shapes=SHAPES, steps: int = 300, warm: int = 30, device: str = "cuda", hf_token: str | None = None,
         repo: str = V1.REPO, log=print, bench_steps: int = 15, bench_warm: int = 5) -> dict:
    """The scan over `names`: the bench at every shape, the picked shape's stream run, the projection; the record written to
    <out>/micro_scan.json after every craft (a stopped scan keeps what it measured) and shipped to <repo>/preflight/ when a token is given.
    -> the record."""
    os.makedirs(out, exist_ok=True)
    names = list(names)
    card_gb = torch.cuda.get_device_properties(0).total_memory / 2 ** 30 if device == "cuda" else 0.0
    res = dict(utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), card=torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
               card_gb=card_gb, torch=torch.__version__, trainer=__version__, steps=steps, warm=warm, shapes=[list(s) for s in shapes],
               stream=STREAM, diet_bytes=diet_bytes(), crafts=[])
    path = os.path.join(out, "micro_scan.json")
    for name in names:
        sh = shape_of(name)
        log(f"[micro] {name}: {sh['params']:,} parameters (d{sh['d_model']}, {sh['n_layers']} blocks, {sh['n_heads']} heads, hubs "
            f"{sh['hub_layers']} at K{sh['hub_K']} D{sh['hub_D']})")
        birth = V1.write_birth(name, os.path.join(out, "births"))
        rows = []
        for mb, ga in shapes:
            r = V1.bench(name, birth, mb, ga, steps=bench_steps, warm=bench_warm, device=device)
            rows.append(r)
            log(f"[micro]   bench {mb} x {ga}: {r['tok_s'] / 1e3:,.0f} kB/s, peak {r['peak_gb']:.1f} GB")
        pick, why = V1.pick_shape(rows, card_gb, prefer=tuple(shapes[0]))
        log(f"[micro]   the shape: {why}")
        run = stream_run(name, birth, pick[0], pick[1], steps, warm, out, device, log)
        proj = projection(run["bytes_s"], sh["params"], res["diet_bytes"])
        res["crafts"].append(dict(sh, bench=rows, pick=dict(micro_batch=pick[0], grad_accum=pick[1], why=why), stream=run, **proj))
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(res, fh, indent=1)
        log(scan_table(res))
    if hf_token:
        res["shipped"] = ship(res, hf_token, repo)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(res, fh, indent=1)
        log(f"[micro] shipped {repo}/{res['shipped']}")
    return res


def scan_table(res: dict) -> str:
    """The record as one table: a row a craft."""
    head = ("| craft | params | shape | kB/s | steps/s | peak GB | B a day | B a week | bytes a param a week | 2s diet (h) | 4x (h) |\n"
            "|---|---|---|---|---|---|---|---|---|---|---|")
    rows = []
    for c in res.get("crafts", []):
        s = c["stream"]
        rows.append(f"| {c['name']} | {c['params'] / 1e6:.1f} M | {s['micro_batch']} x {s['grad_accum']} | {s['bytes_s'] / 1e3:,.0f} | "
                    f"{s['steps_s']:.2f} | {s['peak_gb']:.1f} | {c['day_B']:.1f} | {c['week_B']:.0f} | {c['bytes_per_param_week']:,.0f} | "
                    f"{c['diet_hours']:.1f} | {c['diet4x_hours']:.1f} |")
    return "\n".join([head] + rows)


def ship(res: dict, hf_token: str, repo: str = V1.REPO) -> str:
    """The record to <repo>/preflight/micro_scan_<utc>.json. -> the path in the repo."""
    import io
    from huggingface_hub import HfApi
    stamp = res["utc"].replace("-", "").replace(":", "").replace("T", "_").rstrip("Z")
    path = f"preflight/micro_scan_{stamp}.json"
    HfApi(token=hf_token).upload_file(path_or_fileobj=io.BytesIO(json.dumps(res, indent=1).encode("utf-8")), path_in_repo=path,
                                      repo_id=repo, commit_message=f"micro scan on {res.get('card', '?')}: {len(res['crafts'])} crafts")
    return path
