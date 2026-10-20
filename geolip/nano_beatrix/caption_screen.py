# -*- coding: utf-8 -*-
"""caption_screen.py - THE CAPTION READS on a stored craft (0.1.55; plans/2026-10-19_caption_finetune_plan.md THE READS). On held pack
documents never trained: (1) THE EXTRACTION READ, the word-meaning mark: the caption register's head, the caption alone, a newline, the
question for one of the document's records; the value has to come from the caption's wording. (2) THE COPY READ: the head, the frame
holding the document's records in a shuffled order, a newline, the caption, a newline, the same question: the near read on real records.
Both reads on the same document, record and caption; each under greedy decoding (the whole value right, the first byte right) and
teacher-forced (the value's bits); per key and per kind; examples. Record <out>/bed/caption_<init>.json. Run as a child by
nano_v1.caption_stored (the card's budget from BATTERY_GPU_FRAC, the device from BATTERY_DEV)."""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import torch

from geolip.alephllm.data.special_tokens import CUE, END, SEP

from . import caption_pack as cp
from . import frames_data as fd
from .gate_screen import span_bits


@torch.no_grad()
def greedy(logits_fn, prompts: list, dev, max_new: int, batch: int = 16, ctx: int | None = None) -> list:
    """Greedy continuations of the prompts, each up to max_new bytes, stopped at END (not included); right-padded batches, every row read
    at its own last position. ctx (0.1.69): the window; a row that has reached it stops (an answer that never says END once grew a row to
    one byte past the position table; a value that fits ends before the window does, so only run-on answers are cut). -> list of bytes."""
    out = []
    for i0 in range(0, len(prompts), batch):
        chunk = [bytearray(p) for p in prompts[i0:i0 + batch]]
        done = [False] * len(chunk)
        gen = [bytearray() for _ in chunk]
        for _ in range(max_new):
            if ctx is not None:
                for j, c in enumerate(chunk):
                    if len(c) >= ctx:
                        done[j] = True
                if all(done):
                    break
            L = max(len(c) for c in chunk)
            xb = np.full((len(chunk), L), 32, dtype=np.uint8)
            for j, c in enumerate(chunk):
                xb[j, :len(c)] = np.frombuffer(bytes(c), dtype=np.uint8)
            lg = logits_fn(torch.tensor(xb.astype(np.int64), device=dev)).float()
            last = torch.tensor([len(c) - 1 for c in chunk], device=dev)
            nxt = lg[torch.arange(len(chunk), device=dev), last].argmax(-1).cpu().tolist()
            for j, b in enumerate(nxt):
                if done[j]:
                    continue
                if b == END:
                    done[j] = True
                    continue
                gen[j].append(b)
                chunk[j].append(b)
            if all(done):
                break
        out += [bytes(g) for g in gen]
    return out


def build_items(pack, n_rows: int, seed: int, ctx: int) -> list:
    """Held documents with records, in a seeded order: one caption and one framed record each; the two prompts (extraction, copy)."""
    rng = np.random.default_rng([seed, 55])
    items = []
    for i in rng.permutation(pack.n).tolist():
        if len(items) >= n_rows:
            break
        recs, caps = pack.records(i), pack.captions(i)
        if not recs or not caps:
            continue
        cap = caps[int(rng.integers(len(caps)))]
        perm = rng.permutation(len(recs)).tolist()
        fb, kept = fd.caption_frame(rng, [recs[j] for j in perm])
        key, val = kept[int(rng.integers(len(kept)))]
        q = bytes([CUE]) + key + bytes([SEP])
        ex = bytes(fd.CAPTION_HEAD) + cap + b"\n" + q
        co = bytes(fd.CAPTION_HEAD) + fb + b"\n" + cap + b"\n" + q
        if len(co) + len(val) + 1 > ctx:
            continue
        items.append(dict(doc=i, kind=pack.kind(i), key=key, val=val, ex=ex, co=co, caption_bytes=len(cap), n_records=len(recs), framed=len(kept)))
    return items


def _mean(xs):
    return float(np.mean(xs)) if len(xs) else None


def _by(items: list, vals: list, field: str) -> dict:
    """The mean of vals by a field of the items (key decoded, or kind). -> {name: dict(n, whole)}."""
    out = {}
    for it, v in zip(items, vals):
        name = it[field].decode("ascii", "replace") if isinstance(it[field], bytes) else it[field]
        d = out.setdefault(name, dict(n=0, whole=0.0))
        d["n"] += 1
        d["whole"] += v
    for d in out.values():
        d["whole"] = d["whole"] / d["n"]
    return out


def prior_of(train, n_docs: int = 50000, seed: int = 1) -> tuple:
    """0.1.56: the commonest training value of each key over a seeded sample of training documents: the prior a body answers with when it
    does not read the caption (the first caption phase's lesson: the craft and its control both read 0.17 whole from the caption alone,
    the prior). Ties by the count, then the value. -> ({key bytes: value bytes}, documents tallied)."""
    rng = np.random.default_rng([seed, 56])
    idx = rng.permutation(train.n)[:n_docs].tolist() if train.n > n_docs else list(range(train.n))
    tally = {}
    for i in idx:
        for k, v in train.records(i):
            d = tally.setdefault(k, {})
            d[v] = d.get(v, 0) + 1
    return {k: max(d.items(), key=lambda kv: (kv[1], kv[0]))[0] for k, d in tally.items()}, len(idx)


def read(logits_fn, dev, pack, n_rows: int, seed: int, ctx: int, log=print, train=None) -> dict:
    """The two reads over the held items, and with `train` (the pack's training split) the prior-only baseline on the same items.
    -> dict(extraction=..., copy=..., prior=..., extraction_over_prior=..., n=...)."""
    items = build_items(pack, n_rows, seed, ctx)
    assert items, "no held document with records: the pack's held split is empty or has no attributes (a bigger pack)"
    res = dict(n=len(items), kinds={k: sum(1 for it in items if it["kind"] == k) for k in sorted({it["kind"] for it in items})})
    if train is not None:                                                   # 0.1.56: the prior beside every extraction number
        major, n_docs = prior_of(train)
        pw = [float(major.get(it["key"]) == it["val"]) for it in items]
        asked = {it["key"] for it in items}
        res["prior"] = dict(n=len(items), n_docs=n_docs, whole=_mean(pw), by_key=_by(items, pw, "key"), by_kind=_by(items, pw, "kind"),
                            majority={k.decode("ascii", "replace"): v.decode("ascii", "replace") for k, v in major.items() if k in asked})
        log(f"[caption] prior: each key's commonest training value answers {res['prior']['whole']:.3f} of the {len(items)} held questions whole "
            f"({n_docs:,} training documents tallied)")
    for form, field in (("extraction", "ex"), ("copy", "co")):
        t0 = time.time()
        prompts = [it[field] for it in items]
        gens = greedy(logits_fn, prompts, dev, cp.VAL_MAX + 1)
        bits = span_bits(logits_fn, [(p + it["val"] + bytes([END]), len(p)) for p, it in zip(prompts, items)], dev)
        whole = [float(g == it["val"]) for g, it in zip(gens, items)]
        first = [float(g[:1] == it["val"][:1]) for g, it in zip(gens, items)]
        vbits = [float(sum(b[:len(it["val"])])) for b, it in zip(bits, items)]
        groups = {}
        for it, w, f, v in zip(items, whole, first, vbits):
            for gk in (("key", it["key"].decode("ascii", "replace")), ("kind", it["kind"])):
                d = groups.setdefault(gk, dict(n=0, whole=0.0, first_byte=0.0, value_bits=0.0))
                d["n"] += 1
                d["whole"] += w
                d["first_byte"] += f
                d["value_bits"] += v
        for d in groups.values():
            for k in ("whole", "first_byte", "value_bits"):
                d[k] = d[k] / d["n"]
        res[form] = dict(n=len(items), whole=_mean(whole), first_byte=_mean(first), value_bits=_mean(vbits),
                         by_key={k: v for (g, k), v in groups.items() if g == "key"}, by_kind={k: v for (g, k), v in groups.items() if g == "kind"},
                         examples=[dict(kind=it["kind"], key=it["key"].decode("ascii", "replace"), value=it["val"].decode("ascii", "replace"),
                                        read=g.decode("utf-8", "replace"), right=bool(w)) for it, g, w in list(zip(items, gens, whole))[:16]],
                         seconds=round(time.time() - t0, 1))
        log(f"[caption] {form}: whole {res[form]['whole']:.3f}, first byte {res[form]['first_byte']:.3f}, value bits {res[form]['value_bits']:.2f} over {len(items)} rows ({res[form]['seconds']} s)")
    if "prior" in res:
        res["extraction_over_prior"] = res["extraction"]["whole"] - res["prior"]["whole"]
        log(f"[caption] extraction over the prior: {res['extraction_over_prior']:+.3f}")
    return res


def main():
    args = {a.split("=")[0].lstrip("-"): a.split("=", 1)[1] for a in sys.argv[1:] if "=" in a}
    out, init = args["out"], args["init"]
    n_rows, seed, ctx = int(args.get("rows", "256")), int(args.get("seed", "1")), int(args.get("ctx", "4096"))
    pack_dir = args.get("pack") or os.path.join(out, "caption_pack")
    dev = torch.device(os.environ.get("BATTERY_DEV", "cuda" if torch.cuda.is_available() else "cpu"))
    bed = os.path.join(out, "bed")
    rec_path = os.path.join(bed, f"caption_{init}.json")
    t0 = time.time()
    from geolip.nano_beatrix.bed_pair import load_craft
    model = load_craft(os.path.join(bed, init), ctx, dev)
    if dev.type == "cuda" and os.environ.get("BATTERY_GPU_FRAC"):
        torch.cuda.set_per_process_memory_fraction(float(os.environ["BATTERY_GPU_FRAC"]))

    def logits_fn(x):
        if dev.type == "cuda":
            with torch.autocast("cuda", dtype=torch.bfloat16):
                return model(x)[0]
        return model(x)[0]

    pack, train = cp.load(pack_dir, "held"), cp.load(pack_dir, "train")
    print(f"[caption] {init}: {pack.n} held documents in {pack_dir} ({train.n:,} training documents for the prior); up to {n_rows} rows; {dev}", flush=True)
    rec = read(logits_fn, dev, pack, n_rows, seed, ctx, log=lambda s: print(s, flush=True), train=train)
    rec.update(init=init, seed=seed, ctx=ctx, pack=pack_dir, seconds=round(time.time() - t0, 1), device=str(dev))
    os.makedirs(bed, exist_ok=True)
    json.dump(rec, open(rec_path, "w", encoding="utf-8"), indent=1)
    print(f"[caption] done: extraction whole {rec['extraction']['whole']:.3f}, copy whole {rec['copy']['whole']:.3f}; {rec['seconds']:.0f} s -> {rec_path}", flush=True)


if __name__ == "__main__":
    main()
