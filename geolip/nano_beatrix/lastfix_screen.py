# -*- coding: utf-8 -*-
"""lastfix_screen.py - the last block's write under a bound, screened on the nano v1 bodies before their 2s train.

Under Beatrix 2s's recipe the v1 crafts turned most of their last block's output onto one fixed direction within their first 1,145
steps (the energy share along it 0.24 at birth -> 0.81; the last bank's write 25-36x larger; the last block's rank 65 -> 13-16), the
form the larger 2s crafts ended in (0.98-0.99 of every byte, the final norm's bias set against it). The library carries four bounds on
that write (AlephLMConfig.bank_last_fix, model/lastfix.py):

  register     the carrier (the write along its slowly tracked mean direction) becomes a declared number a_t: the stream keeps
               c_t + gamma a_t b with one detached bounded gain a step (gamma = clamp(a_ref / a_bar, 1/4, 4)), the head reads a_t on its
               own zero-born path
  temperature  the carrier leaves the stream; the head divides its logits by a per-byte temperature read from a_t
  gain         the whole write times one detached bounded gain a step (its mean size held to the warm-in's)
  normwrite    the write standardized over the row's positions (causal running statistics): the comparison arm, built to fail
               (normalizing removes the per-byte confidence the direction carries)

Each arm trains one v1 body from the v1 craft's own start through the start of the 2s diet (warmup_wikitext 0.3B + fineweb_main
1.0B = 4,960 steps) with the 2s recipe; every 500 steps the full eval (bits a byte on the fineweb-edu holdout) and the big-direction
gauge (the 2s tool's reads) land in the run's TensorBoard and the arm's record; at the end the drift memory on random bytes (the
exit's H, the 2s tool's estimator). run() goes arm by arm and resumes from the repo; table() compares the arms.
"""
from __future__ import annotations

import json
import math
import os
import time

import numpy as np
import torch

from . import nano_v1
from .bodies import craft_cfg

REPO = "AbstractPhil/alephllm-nano-beatrix-training"
PREFIX = "v1_lastfix"
FINEWEB_TOKENS = 1_000_000_000
EVAL_EVERY = 500
ARMS = [
    dict(name="c-bare-s3", craft="nano-castor-v1", seed=3, mode=""),
    dict(name="c-register", craft="nano-castor-v1", seed=3, mode="register"),
    dict(name="c-temperature", craft="nano-castor-v1", seed=3, mode="temperature"),
    dict(name="c-gain", craft="nano-castor-v1", seed=3, mode="gain"),
    dict(name="c-bare-s4", craft="nano-castor-v1", seed=4, mode=""),
    dict(name="c-normwrite", craft="nano-castor-v1", seed=3, mode="normwrite"),
    dict(name="pdc-bare", craft="nano-pollux-dual-canon-v1", seed=1, mode=""),
    dict(name="pdc-register", craft="nano-pollux-dual-canon-v1", seed=1, mode="register"),
    dict(name="pdc-temperature", craft="nano-pollux-dual-canon-v1", seed=1, mode="temperature"),
]
BY_NAME = {a["name"]: a for a in ARMS}
GAUGE_DOCS, GAUGE_CTX, IID_SEQS = 16, 2048, 24
SCALES = np.array([16, 32, 64, 128, 256, 512, 1024])


# ------------------------------------------------------------------ the arms
def arm_cfg(arm: dict):
    cfg = craft_cfg(nano_v1.CRAFTS[arm["craft"]]["arm"], nano_v1.CTX)
    cfg.name = f"{PREFIX}/{arm['name']}"
    cfg.bank_last_fix = arm["mode"]
    return cfg


def screen_diet(scale: float = 1.0) -> list:
    """The start of the 2s diet: warmup_wikitext 0.3B and the first 1.0B of fineweb_main (4,960 steps at 262,144 bytes)."""
    ph = nano_v1.diet()[:2]
    ph[1]["planned_tokens"] = FINEWEB_TOKENS
    if scale != 1.0:
        for p in ph:
            p["planned_tokens"] = max(1, int(p["planned_tokens"] * scale))
    return ph


def arm_state(arm: dict) -> dict:
    """The arm's start: the v1 craft's start (the bed birth of its body and seed, carried to 4,096) and, for a fix arm, the fix's own
    tensors from a fresh build of the fixed craft at the same seed (its statistics at zero; the register's head path and the
    temperature born silent: the gain and the register are the plain craft at birth)."""
    from geolip.alephllm.model.alephlm import AlephLM
    body = nano_v1.CRAFTS[arm["craft"]]["arm"]
    sd = nano_v1.birth_state(body, arm["seed"])
    if arm["mode"]:
        torch.manual_seed(arm["seed"])
        full = AlephLM(arm_cfg(arm)).state_dict()
        fix = {k: v.detach().clone().contiguous() for k, v in full.items() if ".fix." in k}
        assert fix and set(full) == set(sd) | set(fix), sorted(set(full) ^ (set(sd) | set(fix)))[:6]
        sd.update(fix)
    return sd


def write_arm_birth(arm: dict, folder: str) -> str:
    """<folder>/<arm>_birth.safetensors (written once; a present file is checked against a fresh build). -> the path."""
    from safetensors.torch import load_file, save_file
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"{arm['name']}_birth.safetensors")
    sd = arm_state(arm)
    if os.path.exists(path):
        old = load_file(path)
        assert set(old) == set(sd) and all(torch.equal(old[k], sd[k]) for k in sd), f"{path} is not the start of {arm['name']}"
        return path
    save_file(sd, path + ".part", metadata={"arm": arm["name"], "mode": arm["mode"], "seed": str(arm["seed"])})
    os.replace(path + ".part", path)
    return path


def arm_preset(arm: dict, birth_file: str, micro_batch: int = 16, grad_accum: int = 4, repo: str = REPO, diet_scale: float = 1.0,
               **train_over):
    """The library preset of an arm: the v1 body with its fix, the 2s recipe, the start of the 2s diet, the full eval every 500 steps,
    the weights-only start from the arm's start file."""
    from geolip.alephllm.presets import Preset, TrainConfig
    if diet_scale == 1.0:
        assert micro_batch * grad_accum * nano_v1.CTX == nano_v1.TOKENS_PER_STEP, f"the 2s step is {nano_v1.TOKENS_PER_STEP:,} tokens"
    over = dict(eval_every=EVAL_EVERY)
    over.update(train_over)
    tc = TrainConfig(micro_batch=micro_batch, grad_accum=grad_accum, governor="", head_addr_frozen=False, phase_seed_offset=True,
                     **over)
    return Preset(model=arm_cfg(arm), train=tc, hf_repo=repo, curriculum=screen_diet(diet_scale), data_scale=1.0,
                  init_from={"file": birth_file, "step": 0})


# ------------------------------------------------------------------ the gauges
def gauge_docs(heldout: str, n: int = GAUGE_DOCS, ctx: int = GAUGE_CTX) -> list:
    """The first n held-out documents of at least ctx + 64 bytes, DOC in front (the 2s big-direction tool's rows)."""
    from geolip.alephllm.data.special_tokens import DOC
    docs = []
    for line in open(heldout, encoding="utf-8"):
        b = json.loads(line)["text"].encode("utf-8")
        if len(b) >= ctx + 64:
            docs.append([DOC] + list(b[:ctx - 1]))
        if len(docs) >= n:
            break
    return docs


@torch.no_grad()
def _states(model, rows: list, batch: int = 4) -> dict:
    """Every block's output, the final norm's output and the last bank's write on the rows (fp32, the first position excluded)."""
    caps = {}
    L = len(model.blocks)
    hs = [b.register_forward_hook(lambda m, i, o, k=k: caps.__setitem__(k, o.detach())) for k, b in enumerate(model.blocks)]
    hs.append(model.nf.register_forward_hook(lambda m, i, o: caps.__setitem__("nf", o.detach())))
    hs.append(model.blocks[-1].bank.register_forward_hook(lambda m, i, o: caps.__setitem__("bank", o.detach())))
    dev = next(model.parameters()).device
    keep = {k: [] for k in list(range(L)) + ["nf", "bank"]}
    try:
        for i0 in range(0, len(rows), batch):
            model(torch.tensor(rows[i0:i0 + batch], dtype=torch.long, device=dev))
            for k in keep:
                keep[k].append(caps[k][:, 1:, :].float().reshape(-1, caps[k].shape[-1]).cpu())
    finally:
        for h in hs:
            h.remove()
    return {k: torch.cat(v, 0) for k, v in keep.items()}


def lastwrite_gauge(model, docs: list) -> dict:
    """The big-direction reads (the 2s tool's definitions): u = the last block's uncentered mean output direction; the energy share
    along u at every block and after the final norm; the per-byte share at the last block; the last bank write's own mean-direction
    share, size and rank; every block's centered rank (the trainer census's); the last block's top principal share; the final norm's
    gain at u's top channels and its bias against u; the fix's statistics."""
    from geolip.alephllm.train.instruments import effective_rank
    was = model.training
    model.eval()
    try:
        X = _states(model, docs)
    finally:
        model.train(was)
    L = len(model.blocks)
    last = X[L - 1]
    u = last.mean(0)
    u = u / u.norm()

    def share(M, d):
        return float(((M @ d) ** 2).sum() / (M ** 2).sum())

    def rank(M):
        torch.manual_seed(0)
        return effective_rank(M)
    c = last - last.mean(0, keepdim=True)
    sv = torch.linalg.svdvals(c[torch.randperm(c.shape[0], generator=torch.Generator().manual_seed(0))[:8192]])
    ub = X["bank"].mean(0)
    ub = ub / ub.norm()
    top = torch.topk(u.abs(), 16).indices
    w, b = model.nf.weight.detach().float().cpu(), model.nf.bias.detach().float().cpu()
    fix = model.last_fix
    out = {"share_u": {str(k): share(X[k], u) for k in list(range(L)) + ["nf"]},
           "byte_share_last": float((((last @ u) ** 2) / (last ** 2).sum(1).clamp_min(1e-30)).mean()),
           "rank": {str(k): rank(X[k]) for k in range(L)},
           "pc1_last": float(sv[0] ** 2 / (sv ** 2).sum()),
           "last_size": float(last.norm(dim=1).mean()),
           "bank": {"share_own_mean": share(X["bank"], ub), "share_u": share(X["bank"], u), "rank": rank(X["bank"]),
                    "size": float(X["bank"].norm(dim=1).mean())},
           "norm": {"gain_top16": float(w[top].mean()), "gain_all": float(w.mean()),
                    "bias_u_r": float(np.corrcoef(b.numpy(), u.numpy())[0, 1]) if float(b.std()) > 0 else float("nan")},
           "fix": fix.stats() if fix is not None else None,
           "_u": u}
    return out


def gauge_scalars(g: dict) -> dict:
    """The gauge -> TensorBoard scalars (lastwrite/...)."""
    L = len(g["rank"])
    sc = {"lastwrite/share_last": g["share_u"][str(L - 1)], "lastwrite/share_after_norm": g["share_u"]["nf"],
          "lastwrite/byte_share_last": g["byte_share_last"], "lastwrite/rank_last": g["rank"][str(L - 1)],
          "lastwrite/pc1_last": g["pc1_last"], "lastwrite/size_last": g["last_size"],
          "lastwrite/bank_share": g["bank"]["share_own_mean"], "lastwrite/bank_size": g["bank"]["size"], "lastwrite/bank_rank": g["bank"]["rank"],
          "lastwrite/norm_gain_top16": g["norm"]["gain_top16"], "lastwrite/norm_bias_u_r": g["norm"]["bias_u_r"]}
    for k, v in g["share_u"].items():
        sc[f"lastwrite/share_by_block/{k}"] = v
    for k, v in g["rank"].items():
        sc[f"lastwrite/rank_by_block/{k}"] = v
    if g.get("fix"):
        for k in ("gamma", "a_bar", "a_ref", "tau_at_mean"):
            if k in g["fix"]:
                sc[f"lastwrite/fix_{k}"] = g["fix"][k]
    return {k: float(v) for k, v in sc.items() if v is not None and math.isfinite(float(v))}


# the drift memory on random bytes (the 2s tool's estimator: DFA-1 per sequence, the pooled fit, the octave slopes, the bootstrap median)
def _dfa_f2(x, order):
    y = np.cumsum(x - x.mean())
    out = np.empty(len(SCALES))
    for i, s in enumerate(SCALES):
        nw = len(y) // s
        if nw == 0:                                       # a series shorter than the scale reads nothing there
            out[i] = np.nan
            continue
        Y = y[:nw * s].reshape(nw, s).T
        V = np.vander(np.arange(s, dtype=np.float64), order + 1)
        coef, *_ = np.linalg.lstsq(V, Y, rcond=None)
        out[i] = ((Y - V @ coef) ** 2).mean()
    return out


def _fit(xs, ys):
    lx, ly = np.log(np.asarray(xs, dtype=np.float64)), np.log(np.asarray(ys, dtype=np.float64))
    ok = np.isfinite(ly)
    if ok.sum() < 3:
        return float("nan")
    return float(np.polyfit(lx[ok], ly[ok], 1)[0])


def _analyze(series):
    Z = [(x - x.mean()) / max(x.std(), 1e-300) for x in (np.asarray(s, dtype=np.float64) for s in series)]
    C1 = np.sqrt(np.stack([_dfa_f2(z, 1) for z in Z]))
    per = [_fit(SCALES, c) for c in C1]
    pooled = np.sqrt((C1 ** 2).mean(0))
    lx, ly = np.log(SCALES.astype(np.float64)), np.log(pooled)
    g = np.random.default_rng(0)
    meds = np.median(np.asarray(per)[g.integers(0, len(per), (2000, len(per)))], axis=1)
    return {"H_median": float(np.median(per)), "H_ci95": [float(np.quantile(meds, .025)), float(np.quantile(meds, .975))],
            "H_pooled": _fit(SCALES, pooled), "octave_slopes": [float((ly[i + 2] - ly[i]) / (lx[i + 2] - lx[i])) for i in range(len(lx) - 2)]}


def exit_memory(model, u: torch.Tensor, n: int = IID_SEQS, length: int = 2048) -> dict:
    """The exit's H: the last block's output along u on iid printable bytes (the 2s tool's read), and a shuffled control."""
    from geolip.alephllm.data.special_tokens import DOC
    g = np.random.default_rng(777)
    rows = [[DOC] + g.integers(32, 127, size=length - 1).tolist() for _ in range(n)]
    was = model.training
    model.eval()
    try:
        X = _states(model, rows)
    finally:
        model.train(was)
    a = (X[len(model.blocks) - 1] @ u).numpy().reshape(n, length - 1)
    return {"exit": _analyze(list(a)), "shuffled": _analyze([np.random.default_rng(1).permutation(s) for s in a])}


# ------------------------------------------------------------------ the run
def run(out: str, hf_token: str | None = None, names=None, micro_batch: int = 16, grad_accum: int = 4, repo: str = REPO,
        max_hours: float | None = None, diet_scale: float = 1.0, device=None, **train_over) -> dict:
    """The screen, arm by arm (each resumes from its record in the repo; a finished arm is read from its record and skipped).
    -> {arm: its record}; each record also lands at <out>/lastfix/<arm>.json and in the repo at <PREFIX>/<arm>/screen_record.json."""
    from .bed_corpus import write_heldout
    names = list(names or [a["name"] for a in ARMS])
    t_end = None if max_hours is None else time.time() + max_hours * 3600
    held = write_heldout(os.path.join(out, "bed"))
    docs = gauge_docs(held)
    recs = {}
    os.makedirs(os.path.join(out, "lastfix"), exist_ok=True)
    for n in names:
        arm = BY_NAME[n]
        rec_p = os.path.join(out, "lastfix", f"{n}.json")
        rec = json.load(open(rec_p, encoding="utf-8")) if os.path.exists(rec_p) else None
        if rec and rec.get("done"):
            recs[n] = rec
            continue
        if t_end is not None and time.time() >= t_end:
            print(f"[lastfix] the session's wall budget is spent before {n}", flush=True)
            break
        p = arm_preset(arm, write_arm_birth(arm, os.path.join(out, "births")), micro_batch, grad_accum, repo, diet_scale, **train_over)
        box = {"rec": rec or {"arm": arm, "gauges": []}}

        def hook(t, box=box, n=n):
            rec = box["rec"]
            g = lastwrite_gauge(t.raw_model, docs)
            ledger = (getattr(t, "_last_eval", {}) or {}).get("ledger") or {}
            sc = gauge_scalars(g)
            for k, v in sc.items():
                t.writer.add_scalar(k, v, t.step)
            t.writer.flush()
            g.pop("_u")
            rec["gauges"] = [x for x in rec["gauges"] if x["step"] != t.step] + [dict(step=t.step, val_bpb=ledger.get("bpb_full"), **g)]
            print(f"[lastfix] {n} step {t.step:,}: val {ledger.get('bpb_full', float('nan')):.4f} bits a byte | share along the last "
                  f"block's direction {sc['lastwrite/share_last']:.3f} (after the norm {sc['lastwrite/share_after_norm']:.3f}) | last bank "
                  f"write {sc['lastwrite/bank_share']:.3f} on its direction, size {sc['lastwrite/bank_size']:.3g} | last block rank "
                  f"{sc['lastwrite/rank_last']:.0f}" + (f" | gain {g['fix']['gamma']:.3f}" if g.get("fix") else ""), flush=True)

        t = nano_v1.make_trainer(p, hf_token, os.path.join(out, "runs"), True, hook, device)
        if not box["rec"]["gauges"] and t.hub.api is not None:           # a fresh machine: the record so far from the repo
            prev = t.hub.download("screen_record.json")
            if prev:
                box["rec"] = json.load(open(prev, encoding="utf-8"))
        rec = box["rec"]
        if rec.get("done"):                                                # finished in an earlier session
            json.dump(rec, open(rec_p, "w", encoding="utf-8"), indent=1, default=float)
            recs[n] = rec
            continue
        print(f"[lastfix] {n}: {arm['craft']} seed {arm['seed']}, last-write fix {arm['mode'] or 'none (bare)'}, "
              f"{t.raw_model.param_count():,} parameters, step {t.step:,}", flush=True)
        stopped = None
        t0 = time.time()
        while nano_v1.phase_index(t) is not None:
            left = None if t_end is None else (t_end - time.time()) / 3600
            if left is not None and left <= 0:
                stopped = "the session's wall budget"
                break
            try:
                t.train(stop_at_boundary=True, max_hours=left)
            except FloatingPointError as e:
                rec["failed"] = f"step {t.step:,}: {e}"
                print(f"[lastfix] {n} STOPPED (non-finite): {e}", flush=True)
                break
            if t._interrupted or t._guard_halt or t._last_boundary is None:
                stopped = "a manual stop" if t._interrupted else ("a guard halt" if t._guard_halt else "the session's wall budget")
                break
            nano_v1.close_stream(t)
            if t.step % t.tc.eval_every != 0:
                t._full_eval()                                             # the boundary's eval (and the gauge after it)
        nano_v1.close_stream(t)
        if stopped is None and "failed" not in rec:
            g = lastwrite_gauge(t.raw_model, docs)
            rec["exit_memory"] = exit_memory(t.raw_model, g["_u"])
            rec["done"] = True
            rec["steps"], rec["tokens"] = t.step, t.manifest.tokens_seen
            print(f"[lastfix] {n} done at step {t.step:,}: the exit's H on random bytes {rec['exit_memory']['exit']['H_median']:.3f} "
                  f"{[round(x, 3) for x in rec['exit_memory']['exit']['H_ci95']]} (shuffled {rec['exit_memory']['shuffled']['H_median']:.3f})", flush=True)
        rec["wall_s"] = rec.get("wall_s", 0) + round(time.time() - t0, 1)
        json.dump(rec, open(rec_p, "w", encoding="utf-8"), indent=1, default=float)
        t.hub.upload_bytes(json.dumps(rec, indent=1, default=float).encode(), "screen_record.json")
        t.writer.flush()
        t.hub.upload_tensorboard(t.tb_dir)
        recs[n] = rec
        if stopped:
            print(f"[lastfix] stopped: {stopped}", flush=True)
            break
    return recs


def _at(rec: dict, step: int | None):
    gs = sorted(rec.get("gauges") or [], key=lambda x: x["step"])
    if not gs:
        return None
    if step is None:
        return gs[-1]
    return next((x for x in gs if x["step"] == step), None)


def table(recs: dict, step: int | None = None) -> str:
    """The arms side by side at a step (default each arm's last gauge): the direction, the size, the rank, the bits, the drift memory."""
    base = recs.get("c-bare-s3")
    rows = ["| arm | fix | step | val bits a byte | vs bare s3 | share on the last block's direction | after the norm | last bank write: "
            "share / size | last block rank | exit H on random bytes |", "|---|---|---|---|---|---|---|---|---|---|"]
    for n, r in recs.items():
        g = _at(r, step)
        if g is None:
            continue
        L = len(g["rank"])
        b = _at(base, g["step"]) if base else None
        d = "-" if (b is None or g.get("val_bpb") is None or b.get("val_bpb") is None) else f"{g['val_bpb'] - b['val_bpb']:+.4f}"
        h = (r.get("exit_memory") or {}).get("exit") or {}
        hs = "-" if not h else f"{h['H_median']:.3f} [{h['H_ci95'][0]:.3f}, {h['H_ci95'][1]:.3f}]"
        rows.append(f"| {n} | {r['arm']['mode'] or 'none'} | {g['step']:,} | {g['val_bpb'] if g.get('val_bpb') is not None else float('nan'):.4f} | {d} | "
                    f"{g['share_u'][str(L - 1)]:.3f} | {g['share_u']['nf']:.3f} | {g['bank']['share_own_mean']:.3f} / {g['bank']['size']:.3g} | "
                    f"{g['rank'][str(L - 1)]:.0f} | {hs} |")
    return "\n".join(rows)
