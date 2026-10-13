# -*- coding: utf-8 -*-
"""bed_timing.py - THE TIME COST OF EVERY PROTOTYPE. Under identical conditions on the same card, back to back: every arm of the nano
bed (the attention form x the bank form), the last-block fix modes on the bed craft, and the two teacher forms at their full size
(the splat and the softmax twin, random weights: the cost is the architecture's). For each: params, the training step (forward + backward + the optimizer step; bf16 autocast; pure Adam) in ms and
tokens/s, the inference forward (no grad, bf16) in ms per row and tokens/s, the peak memory of each.
Usage (through card_gate / run_cf): bed_timing.py --out=DIR [--mb=32 --ctx=1024 --steps=10 --teachers=1 --fixes=1 --smoke=1
                                                           --arms=sm-free,hubr-free,hubd-free,hub-free --big=1]
  --arms = a filter (default every arm); --big=1 also times the big-address variant (2 books of 64 @ 32) of every hub arm.
Output: DIR/bed/bed_timing[_smoke][_ctx<N>][_mb<M>].json (+ the table printed); the name carries ctx / mb when not 1,024 / 32."""
import json
import math
import os
import sys
import time

import torch
import torch.nn.functional as F

from geolip.alephllm.presets import AlephLMConfig, PRESETS  # noqa: E402
from geolip.alephllm.model.alephlm import AlephLM  # noqa: E402

ARGS = {a.split("=")[0].lstrip("-"): a.split("=", 1)[1] for a in sys.argv[1:] if "=" in a}
OUT = ARGS["out"]
SMOKE = ARGS.get("smoke", "0") == "1"
MB = int(ARGS.get("mb", "8" if SMOKE else "32"))
CTX = int(ARGS.get("ctx", "1024"))
STEPS = int(ARGS.get("steps", "3" if SMOKE else "10"))
TEACHERS = ARGS.get("teachers", "1") == "1"
FIXES = ARGS.get("fixes", "1") == "1"
ARMS = [a for a in ARGS.get("arms", "").split(",") if a]
BIG = ARGS.get("big", "0") == "1"
DEV = "cuda"
torch.cuda.set_per_process_memory_fraction(float(os.environ.get("BATTERY_GPU_FRAC", "0.35")))
torch.backends.cuda.matmul.allow_tf32 = True
BED = os.path.join(OUT, "bed")
os.makedirs(BED, exist_ok=True)
T0 = time.time()


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')} +{time.time() - T0:6.0f}s] {msg}", flush=True)


def craft_cfg(arm, fix="", big=False):
    """The bed's arm configs (bed_train.py's table, verbatim) with an optional last-block fix; big = the big address (2 x 64 @ 32)."""
    attn, bank = arm.split("-")
    L = 6
    base = dict(name=f"bed-{arm}", d_model=256, n_layers=L, n_heads=8, context=CTX, hub_layers=(), hub_K=32, hub_D=16, tau=0.1,
                bank_experts=7, bank_ff=256, head_K=64, head_D=16, gate_init=-3.0, hub_chunk=128)
    if big:
        base.update(hub_K=64, hub_D=32, hub_const=2)
    A = {"sm": {}, "smr": dict(attn_rotary=True), "smrn": dict(attn_rotary=True, pos_table=False),
         "hub": dict(hub_layers=tuple(range(L))), "hubr": dict(hub_layers=tuple(range(L)), hub_rotary=True),
         "hubrn": dict(hub_layers=tuple(range(L)), hub_rotary=True, pos_table=False),
         "hubd": dict(hub_layers=tuple(range(L)), hub_dual=True),
         "alt": dict(hub_layers=tuple(range(0, L, 2)), attn_rotary=True),
         "altn": dict(hub_layers=tuple(range(0, L, 2)), attn_rotary=True, hub_rotary=True)}
    Bk = {"free": "", "fixed": "fixed", "both": "add"}
    return AlephLMConfig(**{**base, **A[attn]}, bank_byte_table=Bk[bank], bank_last_fix=fix, fix_warm=2)


def sync():
    torch.cuda.synchronize()


def time_train(model, mb, ctx, steps):
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=1e-4, betas=(0.9, 0.95))
    model.train()
    x = torch.randint(0, 256, (mb, ctx + 1), device=DEV)
    torch.cuda.reset_peak_memory_stats()

    def step():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x[:, :-1])[0]
        loss = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), x[:, 1:].reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
    for _ in range(3):
        step()
    sync()
    t = time.perf_counter()
    for _ in range(steps):
        step()
    sync()
    ms = (time.perf_counter() - t) / steps * 1e3
    return dict(train_ms_per_step=ms, train_tok_per_s=mb * ctx / (ms / 1e3), train_peak_gb=torch.cuda.max_memory_allocated() / 2**30)


@torch.no_grad()
def time_infer(model, mb, ctx, steps):
    model.eval()
    x = torch.randint(0, 256, (mb, ctx), device=DEV)
    torch.cuda.reset_peak_memory_stats()
    for _ in range(2):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            model(x)
    sync()
    t = time.perf_counter()
    for _ in range(steps):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            model(x)
    sync()
    ms = (time.perf_counter() - t) / steps * 1e3
    return dict(infer_ms_per_batch=ms, infer_ms_per_row=ms / mb, infer_tok_per_s=mb * ctx / (ms / 1e3), infer_peak_gb=torch.cuda.max_memory_allocated() / 2**30)


R = {"mb": MB, "ctx": CTX, "steps": STEPS, "device": torch.cuda.get_device_name(0), "arms": {}, "fixes": {}, "teachers": {}}
ATTN = ["sm", "smr", "smrn", "hub", "hubr", "hubrn", "hubd", "alt", "altn"]
BANK = ["free", "fixed", "both"]
arms = [f"{a}-{b}" for a in ATTN for b in BANK]
if SMOKE:
    arms = ["sm-free", "hubr-both"]
if ARMS:
    arms = ARMS
jobs = []
for arm in arms:
    jobs.append((arm, False))
    if BIG and arm.split("-")[0] in ("hub", "hubr", "hubrn", "hubd", "alt", "altn"):
        jobs.append((arm, True))
for arm, big in jobs:
    key = arm + (" K64D32c2" if big else "")
    torch.manual_seed(0)
    m = AlephLM(craft_cfg(arm, big=big)).to(DEV)
    r = dict(params=m.param_count(), big=big)
    r.update(time_train(m, MB, CTX, STEPS))
    r.update(time_infer(m, 16, CTX, STEPS))
    R["arms"][key] = r
    log(f"{key:20s} {r['params'] / 1e6:5.2f}M | train {r['train_ms_per_step']:7.1f} ms/step {r['train_tok_per_s'] / 1e3:6.0f}k tok/s peak {r['train_peak_gb']:.2f} GB | "
        f"infer {r['infer_ms_per_row']:6.2f} ms/row {r['infer_tok_per_s'] / 1e3:6.0f}k tok/s")
    del m
    torch.cuda.empty_cache()
FIX_MODES = ["register"] if SMOKE else ["register", "temperature", "gain", "normwrite"]
if not FIXES or "sm-free" not in R["arms"]:
    FIX_MODES = []
for fix in FIX_MODES:
    torch.manual_seed(0)
    m = AlephLM(craft_cfg("sm-free", fix=fix)).to(DEV)
    r = dict(params=m.param_count())
    r.update(time_train(m, MB, CTX, STEPS))
    r.update(time_infer(m, 16, CTX, STEPS))
    R["fixes"][fix] = r
    base = R["arms"]["sm-free"]
    log(f"fix {fix:12s} {r['params'] / 1e6:5.2f}M | train {r['train_ms_per_step']:7.1f} ms/step ({r['train_ms_per_step'] / base['train_ms_per_step'] - 1:+.1%}) | infer {r['infer_ms_per_row']:6.2f} ms/row ({r['infer_ms_per_row'] / base['infer_ms_per_row'] - 1:+.1%})")
    del m
    torch.cuda.empty_cache()
if TEACHERS:
    # the two 2s forms at their own context, inference only (random weights: the architecture's cost); the register added in eval
    for name, cfgname, fix in (("splat 2s", "mini-beatrix-2s", ""), ("softmax twin", "mini-beatrix-2s-control-attn-fp16", ""),
                               ("splat 2s + register", "mini-beatrix-2s", "register"), ("softmax twin + register", "mini-beatrix-2s-control-attn-fp16", "register")):
        import copy
        cfg = copy.deepcopy(PRESETS[cfgname].model)
        cfg.bank_last_fix = fix
        torch.manual_seed(0)
        m = AlephLM(cfg).to(DEV)
        tctx = 1024 if SMOKE else 2048
        r = dict(params=m.param_count(), ctx=tctx)
        r.update(time_infer(m, 2, tctx, 2 if SMOKE else 5))
        R["teachers"][name] = r
        log(f"{name:24s} {r['params'] / 1e6:6.1f}M ctx {tctx} | infer {r['infer_ms_per_row']:7.1f} ms/row {r['infer_tok_per_s'] / 1e3:6.1f}k tok/s peak {r['infer_peak_gb']:.2f} GB")
        del m
        torch.cuda.empty_cache()
R["wall_s"] = time.time() - T0
path = os.path.join(BED, "bed_timing" + ("_smoke" if SMOKE else "") + (f"_ctx{CTX}" if CTX != 1024 else "") + (f"_mb{MB}" if MB != 32 and not SMOKE else "") + ".json")
json.dump(R, open(path, "w", encoding="utf-8"), indent=1)
log(f"wrote {path}")
