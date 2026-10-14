# -*- coding: utf-8 -*-
"""bed_pair.py - how alike two crafts are on the same held-out rows, and the helpers that read a stored pair the same way.

A pair run (bed_train --pair) trains two crafts on the very same rows and writes these reads into both records; a pair trained apart (two
solo runs of one seed) is read by the same functions from its stored weights (load_craft, heldout_rows), so the two cases compare one to
one. The reads: the KL between the crafts' next-byte distributions each way (bits a byte), how often their greedy next bytes agree, and the
debiased linear CKA between their residual streams after each block and after the final norm (Nguyen, Raghu and Kornblith 2021: the
unbiased HSIC estimator of Song et al. 2012)."""
from __future__ import annotations

import json
import math
import os
import re

import numpy as np
import torch
import torch.nn.functional as F

from geolip.alephllm.data.special_tokens import DOC    # the document byte every trainer row starts with

LN2 = math.log(2.0)


def hsic1(K, L):
    """The unbiased HSIC estimator of two Gram matrices of the same n items (n > 3)."""
    n = K.shape[0]
    K, L = K.clone(), L.clone()
    K.fill_diagonal_(0)
    L.fill_diagonal_(0)
    t1 = (K * L).sum()
    t2 = K.sum() * L.sum() / ((n - 1) * (n - 2))
    t3 = 2.0 / (n - 2) * (K.sum(0) @ L.sum(1))
    return (t1 + t2 - t3) / (n * (n - 3))


def cka_debiased(X, Y) -> float:
    """Debiased linear CKA of two representations (n, d) and (n, d') of the same n items: fp64, the features centered."""
    X, Y = X.double(), Y.double()
    X, Y = X - X.mean(0, keepdim=True), Y - Y.mean(0, keepdim=True)
    K, L = X @ X.T, Y @ Y.T
    return float(hsic1(K, L) / torch.sqrt(hsic1(K, K) * hsic1(L, L)).clamp_min(1e-30))


def rows_tensor(docs: list, ctx: int, dev) -> torch.Tensor:
    """The held-out bits' rows of the trainer: each document's first ctx bytes behind a DOC byte -> the input (n, ctx) on dev."""
    rows = [bytes([DOC]) + d[:ctx] for d in docs]
    return torch.tensor(np.stack([np.frombuffer(r, dtype=np.uint8) for r in rows]).astype(np.int64), device=dev)[:, :-1]


def heldout_rows(path: str, ctx: int, n_docs: int = 32) -> list:
    """The trainer's evaluation documents from its held-out file: the documents of at least ctx + 64 bytes, in order, the first n_docs
    (the trainer's eval_docs start the same way, so its pair reads and these read the same rows)."""
    docs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            b = json.loads(line)["text"].encode("utf-8")
            if len(b) >= ctx + 64:
                docs.append(b)
                if len(docs) == n_docs:
                    break
    return docs


@torch.no_grad()
def twin_reads(ma, mb, x: torch.Tensor, every: int = 8, names: tuple = ("a", "b")) -> dict:
    """The pair's reads on the rows x (n, ctx): KL(a || b) and KL(b || a) between the next-byte distributions (bits a byte, every
    position), the greedy next byte's agreement, and the debiased linear CKA between the residual streams after each block and after the
    final norm at every `every`-th position. Both crafts in eval mode; nothing in them changes."""
    def run(m):
        m.eval()
        acts = {}

        def keep(key):
            def hook(mod, inp, out):
                acts[key] = out.detach().float()[:, ::every].reshape(-1, out.shape[-1])
            return hook
        hooks = [b.register_forward_hook(keep(f"block{i}")) for i, b in enumerate(m.blocks)] + [m.nf.register_forward_hook(keep("final"))]
        try:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lg = m(x)[0].float()
        finally:
            for h in hooks:
                h.remove()
        return F.log_softmax(lg, dim=-1), acts
    la, aa = run(ma)
    lb, ab = run(mb)
    kl = [float((p.exp() * (p - q)).sum(-1).mean()) / LN2 for p, q in ((la, lb), (lb, la))]
    agree = float((la.argmax(-1) == lb.argmax(-1)).float().mean())
    cka = {k: cka_debiased(aa[k], ab[k]) for k in aa if k in ab}
    return dict(crafts=list(names), rows=int(x.shape[0]), positions_every=every, vectors=int(next(iter(aa.values())).shape[0]),
                kl_bits=kl, greedy_agree=agree, cka=cka)


def load_craft(folder: str, ctx: int, dev):
    """A stored craft (a weights folder: config.json + model.safetensors) as the trainer builds it, at context ctx (a longer context
    keeps the trained position rows and zeroes the new ones, as a continuation does)."""
    from safetensors.torch import load_file
    from geolip.alephllm.model.alephlm import AlephLM
    from geolip.alephllm.presets import AlephLMConfig
    c = json.load(open(os.path.join(folder, "config.json"), encoding="utf-8"))
    c = {k: (tuple(v) if isinstance(v, list) else v) for k, v in c.items()}
    c["context"] = ctx
    m = AlephLM(AlephLMConfig(**c))
    sd = load_file(os.path.join(folder, "model.safetensors"))
    pos = sd.get("embed.pos")
    if pos is not None and pos.shape[1] != ctx:
        new_pos = torch.zeros(1, ctx, pos.shape[2], dtype=pos.dtype)
        n_keep = min(ctx, pos.shape[1])
        new_pos[:, :n_keep] = pos[:, :n_keep]
        sd["embed.pos"] = new_pos
    m.load_state_dict(sd, strict=True)
    return m.to(dev).eval()


# ------------------------------------------------------------------ the pair notebook's read
PAIR_PIECE = re.compile(r"_pair-[a-z]+-[a-z]+-[tsm][va](?:w[0-9.e-]+)?(?:f[0-9]+)?")


def solo_tag(tag: str) -> str:
    """The tag of the same craft trained alone: a pair tag without its pair piece."""
    return PAIR_PIECE.sub("", tag)


def fetch_record(tag: str, out: str, repo: str):
    """A run's record: <out>/bed/bed_<tag>.json, else the Hub's bed/reports/ (tokenless) -> dict, or None when it is in neither."""
    p = os.path.join(out, "bed", f"bed_{tag}.json")
    if not os.path.exists(p):
        try:
            from huggingface_hub import hf_hub_download
            p = hf_hub_download(repo, f"bed/reports/bed_{tag}.json", token=False, local_dir=os.path.join(out, "_hub"))
        except Exception:  # noqa: BLE001
            return None
    return json.load(open(p, encoding="utf-8"))


def fetch_weights(tag: str, out: str, repo: str):
    """A run's weights folder: <out>/bed/<tag>/, else the Hub's bed/weights/<tag>/ (tokenless) -> the folder, or None."""
    d = os.path.join(out, "bed", tag)
    if os.path.exists(os.path.join(d, "model.safetensors")):
        return d
    try:
        from huggingface_hub import hf_hub_download
        for f in ("config.json", "model.safetensors"):
            p = hf_hub_download(repo, f"bed/weights/{tag}/{f}", token=False, local_dir=os.path.join(out, "_hub"))
        return os.path.dirname(p)
    except Exception:  # noqa: BLE001
        return None


def craft_reads(rec: dict) -> dict:
    """One record's reads for the pair table: the copy (the whole value pooled over the lookup groups) and the first step the near
    records' bits went under the uniform rate, the strict questions, the text, the 1,024-byte frame's first letter and whole value, and
    the block a continuation started from."""
    from .bed_frames import band_fall, pooled_fetch
    whole = pooled_fetch(rec.get("fetch_by_frame"))
    asks = rec.get("fetch_by_frame_asks") or {}
    nested, over = pooled_fetch(asks.get("nested")), pooled_fetch(asks.get("overlap"))
    g = (rec.get("fetch_by_frame") or {}).get("data/1024/depth0") or {}
    form = rec.get("formation") or {}
    init = rec.get("init") or ""
    far = band_fall(rec.get("curve") or [], "512", 800)          # the online bits of records more than 512 bytes back, the last 800 steps
    return dict(whole=None if whole is None else whole["full_value"], first_under=form.get("first_step_under"), formed=form.get("formed"),
                nested=None if nested is None else nested["full_value"], redefined=None if over is None else over["full_value"],
                text=rec.get("heldout_bpb"), first1k=g.get("first_byte"), whole1k=g.get("full_value"), n1k=g.get("n"),
                far=None if far is None else far["last"], after=init.rsplit("_st", 1)[-1] if "_st" in init else "")


def _f(x, nd=2):
    return "-" if x is None else f"{x:.{nd}f}"


def solo_twins(ta: str, tp: str, ctx: int, out: str, repo: str, dev=None):
    """How alike two solo crafts are, read exactly as a pair run reads its twins (the same 32 held-out documents at the crafts' context);
    cached in <out>/pair_solo_twins.json. -> the reads, or None when a craft or the held-out file is missing."""
    cache_p = os.path.join(out, "pair_solo_twins.json")
    cache = json.load(open(cache_p, encoding="utf-8")) if os.path.exists(cache_p) else {}
    key = f"{ta}|{tp}"
    if key in cache:
        return cache[key]
    held = next((p for p in (os.path.join(out, "heldout_fineweb_edu_head2048.jsonl"), os.path.join(out, "bed", "heldout_fineweb_edu_head2048.jsonl"))
                 if os.path.exists(p)), None)
    wa, wp = fetch_weights(ta, out, repo), fetch_weights(tp, out, repo)
    if held is None or wa is None or wp is None:
        return None
    dev = dev or ("cuda" if torch.cuda.is_available() else "cpu")
    ma, mp = load_craft(wa, ctx, dev), load_craft(wp, ctx, dev)
    r = twin_reads(ma, mp, rows_tensor(heldout_rows(held, ctx, 32), ctx, dev), names=(ta.split("_s")[0], tp.split("_s")[0]))
    del ma, mp
    cache[key] = r
    json.dump(cache, open(cache_p, "w", encoding="utf-8"), indent=1)
    return r


def report(rows: list, out: str, repo: str, dev=None) -> str:
    """The pair notebook's read (markdown): rows = [(form label, stage 'A' or 'B', seed, the arm's pair tag, the partner's pair tag)].
    One table a stage: for each seed, the two twins alone (the solo runs of the same settings, the pair tags without their pair piece),
    then each form's two paired twins; each twin's reads (craft_reads) and how alike it and its twin are (the pair's record, or
    solo_twins for the solo twins)."""
    lines = []
    for st in ("A", "B"):
        sel = [r for r in rows if r[1] == st]
        if not sel:
            continue
        lines.append(f"### Stage {st}: " + ("the first stage (1,024 x 32, 12,000 steps)" if st == "A" else "the long-window block (4,096 x 8, 4,000 steps)"))
        timing = "the copy's first step under the rate" if st == "A" else "far records' bits, the last 800 steps"
        lines.append(f"| seed | trained | twin | copy (whole value) | {timing} | nested question | redefined key | text "
                     "| 1,024-byte frame: first letter / whole value | twins: KL each way (bits) | twins: next bytes agree | twins: CKA middle / last block / final |")
        lines.append("|" + "---|" * 12)
        for seed in sorted({r[2] for r in sel}):
            mine = [r for r in sel if r[2] == seed]
            groups = [("alone (the train)", solo_tag(mine[0][3]), solo_tag(mine[0][4]), False)] + [(lab, t, p, True) for lab, _, _, t, p in mine]
            for lab, ta, tp, paired in groups:
                ra, rp = fetch_record(ta, out, repo), fetch_record(tp, out, repo)
                if ra is None or rp is None:
                    lines.append(f"| {seed} | {lab} | - | not on disk or on the Hub yet | | | | | | | | |")
                    continue
                tw = (ra.get("pair") or {}).get("twins") if paired else solo_twins(ta, tp, int(ra["ctx"]), out, repo, dev)
                blocks = sorted(k for k in tw["cka"] if k.startswith("block")) if tw else []
                mid = tw["cka"][f"block{len(blocks) // 2 - 1}"] if blocks else None
                last = tw["cka"][f"block{len(blocks) - 1}"] if blocks else None
                kl = "-" if tw is None else f"{tw['kl_bits'][0]:.3f} / {tw['kl_bits'][1]:.3f}"
                agree = "-" if tw is None else f"{tw['greedy_agree']:.3f}"
                final = None if tw is None else tw["cka"].get("final")
                for rec in (ra, rp):
                    c = craft_reads(rec)
                    note = f" (after {c['after']})" if (not paired and st == "B" and c["after"] not in ("", "A1")) else ""
                    when = (c["first_under"] or "-") if st == "A" else _f(c["far"])
                    lines.append(f"| {seed} | {lab}{note} | {rec['arm'].split('-')[0].capitalize()} | {_f(c['whole'])} | {when} "
                                 f"| {_f(c['nested'])} | {_f(c['redefined'])} | {_f(c['text'], 4)} | {_f(c['first1k'])} / {_f(c['whole1k'])} "
                                 f"| {kl} | {agree} | {_f(mid, 3)} / {_f(last, 3)} / {_f(final, 3)} |")
        lines.append("")
    return "\n".join(lines)
