# -*- coding: utf-8 -*-
"""thought_screen.py - THE THOUGHT READS on a stored craft (0.1.67; plans/2026-10-20_thought_castor_run_plan.md; the dataset plan's section 8).
On the thought pack's HELD rows (never trained: the series the loader holds out) rendered by thought_frames, every question of a row asked
at the end of its document and the answer generated greedily to END: (1) THE RECORD READ, needs 'record': the value stands in the spec
frame; (2) THE CHAIN READ, needs 'chain': a value the thoughts derive and state, read in layout A (the chain present) against layout C
(the wordings only: the body must think unaided); (3) THE UNKNOWN READ, needs 'unknown': the body answers 'unknown'; (4) THE TRAVEL
READ: the same questions in layout D, the layout sealed from training (the question first, then the chain, the spec and the wordings).
Beside them THE PRIOR (each key's commonest training value answers the question) and the same reads on a seeded sample of TRAINED rows in
layout A (how much is recall of trained rows). Every number whole (the value right byte for byte), with the first byte and the value's
teacher-forced bits; the seconds per read. Record <out>/bed/thought_<init>.json. Run as a child by nano_v1.thought_live / thought_stored
(the card's budget from BATTERY_GPU_FRAC, the device from BATTERY_DEV)."""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import torch

from geolip.alephllm.data.special_tokens import END

from . import thought_frames as TF
from .caption_screen import greedy
from .gate_screen import span_bits

VAL_MAX = 24                        # a value is at most 20 bytes (the dataset's rule); the greedy read runs to 24 bytes or END
LAYOUT_READS = ("A", "C", "D")      # A the trained layout with the chain; C the frameless control; D the sealed layout (the travel read)
NEEDS = ("record", "chain", "unknown")
UNKNOWN = b"unknown"


def items_of(pack, n_rows: int, seed: int, ctx: int, layouts=LAYOUT_READS, max_q: int = 4) -> list:
    """The pack's rows in a seeded order (up to n_rows), each of their questions (up to max_q, in the row's order) in each layout: the
    document rendered with that question answered, cut right after the question's key (the prompt), the value, the needs, the depth,
    the domain, the layout. A prompt that would not fit the context with its value is kept as a skip (counted, never read)."""
    rng = np.random.default_rng([seed, 67])
    items = []
    for i in rng.permutation(pack.n).tolist()[:n_rows]:
        row = pack.row(i)
        for qi, q in enumerate((row.get("questions") or [])[:max_q]):
            val = str(q.get("value", "")).encode("ascii", "replace")
            key = str(q.get("key", ""))
            if not val or not key:
                continue
            for lay in layouts:
                doc, _ = TF.render(np.random.default_rng([seed, i, qi]), row, lay, ask=qi)
                prompt = TF.prompt_of(doc, key)
                if prompt is None or len(prompt) + len(val) + 1 > ctx:
                    items.append(dict(skip=True, layout=lay, depth=str(row.get("depth", ""))))
                    continue
                items.append(dict(doc=i, q=qi, layout=lay, key=key, val=val, needs=str(q.get("needs") or "record"),
                                  depth=str(row.get("depth", "")), domain=str(row.get("domain", "")), prompt=prompt))
    return items


def prior_of(train, n_rows: int = 50000, seed: int = 1) -> tuple:
    """The commonest training value of each key over the training rows' records (thought_frames.records on the spec): the prior a body
    answers with when it reads nothing (the caption phase's lesson). Ties by the count, then the value. -> ({key: value}, rows tallied)."""
    rng = np.random.default_rng([seed, 68])
    idx = rng.permutation(train.n)[:n_rows].tolist() if train.n > n_rows else list(range(train.n))
    tally = {}
    for i in idx:
        for k, v in TF.records(train.row(i).get("spec") or {}):
            d = tally.setdefault(k, {})
            d[v] = d.get(v, 0) + 1
    return {k: max(d.items(), key=lambda kv: (kv[1], kv[0]))[0] for k, d in tally.items()}, len(idx)


def _mean(xs):
    return float(np.mean(xs)) if len(xs) else None


def _group(items: list) -> dict:
    """The reads by layout and needs: n, whole, first_byte, value_bits, said_unknown (the share of answers that are the word unknown)."""
    out = {}
    for it in items:
        d = out.setdefault(it["layout"], {}).setdefault(it["needs"], dict(n=0, whole=0.0, first_byte=0.0, value_bits=0.0, said_unknown=0.0))
        d["n"] += 1
        for k, f in (("whole", "whole"), ("first_byte", "first"), ("value_bits", "vbits"), ("said_unknown", "said_unknown")):
            d[k] += it[f]
    for lays in out.values():
        for d in lays.values():
            for k in ("whole", "first_byte", "value_bits", "said_unknown"):
                d[k] = d[k] / d["n"]
    return out


def _score(logits_fn, dev, items: list) -> None:
    """The greedy answer, the whole and first-byte matches, the value's teacher-forced bits and the unknown flag, written into the items."""
    prompts = [it["prompt"] for it in items]
    gens = greedy(logits_fn, prompts, dev, VAL_MAX)
    bits = span_bits(logits_fn, [(p + it["val"] + bytes([END]), len(p)) for p, it in zip(prompts, items)], dev)
    for it, g, b in zip(items, gens, bits):
        it["gen"], it["whole"], it["first"] = g, float(g == it["val"]), float(g[:1] == it["val"][:1])
        it["vbits"] = float(sum(b[:len(it["val"])]))
        it["said_unknown"] = float(g == UNKNOWN)


def _get(groups: dict, lay: str, needs: str, field: str = "whole"):
    return ((groups.get(lay) or {}).get(needs) or {}).get(field)


def headline_of(groups: dict, prior: dict, trained: dict) -> dict:
    """The numbers the boundary row keeps: the record, chain and unknown reads in A, the chain read in C, the record and chain reads in D,
    the false-unknown share on record questions in A, the prior's record and chain answers, the trained rows' record and chain reads."""
    return dict(record_A=_get(groups, "A", "record"), chain_A=_get(groups, "A", "chain"), unknown_A=_get(groups, "A", "unknown", "said_unknown"),
                false_unknown_A=_get(groups, "A", "record", "said_unknown"), chain_C=_get(groups, "C", "chain"), record_C=_get(groups, "C", "record"),
                record_D=_get(groups, "D", "record"), chain_D=_get(groups, "D", "chain"), unknown_D=_get(groups, "D", "unknown", "said_unknown"),
                prior_record=prior.get("record"), prior_chain=prior.get("chain"),
                trained_record_A=_get(trained, "A", "record"), trained_chain_A=_get(trained, "A", "chain"))


def read(logits_fn, dev, held, n_rows: int, seed: int, ctx: int, log=print, train=None, train_rows: int = 64) -> dict:
    """The reads over the held rows in the three layouts, the prior beside them and the trained rows' reads in layout A.
    -> dict(n_rows, n, skipped, layouts, by_depth, prior, trained, headline, examples, seconds)."""
    t0 = time.time()
    all_items = items_of(held, n_rows, seed, ctx)
    items = [it for it in all_items if not it.get("skip")]
    skipped = {}
    for it in all_items:
        if it.get("skip"):
            skipped[it["layout"]] = skipped.get(it["layout"], 0) + 1
    assert items, "no held row fits the context: the pack's held split is empty"
    _score(logits_fn, dev, items)
    groups = _group(items)
    by_depth = {}
    for dp in sorted({it["depth"] for it in items}):
        by_depth[dp] = _group([it for it in items if it["depth"] == dp])
    res = dict(n_rows=min(n_rows, held.n), held_rows=held.n, n=len(items), skipped=skipped, layouts=groups, by_depth=by_depth)
    prior, tr_items = {}, []
    if train is not None:
        major, n_docs = prior_of(train)
        for needs in NEEDS:
            sel = [it for it in items if it["layout"] == "A" and it["needs"] == needs]
            prior[needs] = _mean([float(str(major.get(it["key"], "")).encode("ascii", "replace") == it["val"]) for it in sel])
        res["prior"] = dict(prior, n_docs=n_docs)
        log(f"[thought] the prior (each key's commonest value over {n_docs:,} training rows) answers record {prior.get('record')}, chain {prior.get('chain')}")
        tr_items = [it for it in items_of(train, train_rows, seed + 1, ctx, layouts=("A",)) if not it.get("skip")]
        if tr_items:
            _score(logits_fn, dev, tr_items)
            res["trained"] = _group(tr_items)
            res["trained"]["n"] = len(tr_items)
    res["headline"] = headline_of(groups, prior, res.get("trained") or {})
    res["examples"] = [dict(layout=it["layout"], needs=it["needs"], depth=it["depth"], key=it["key"], value=it["val"].decode("ascii", "replace"),
                            read=it["gen"].decode("utf-8", "replace"), right=bool(it["whole"])) for it in items[:30]]
    res["seconds"] = round(time.time() - t0, 1)
    res["seconds_per_item"] = round(res["seconds"] / max(1, len(items) + len(tr_items)), 4)
    h = res["headline"]
    log(f"[thought] A: record {h['record_A']}, chain {h['chain_A']}, unknown answered {h['unknown_A']} (false unknown on records {h['false_unknown_A']}); "
        f"C: chain {h['chain_C']}, record {h['record_C']}; D (sealed): record {h['record_D']}, chain {h['chain_D']}; trained rows A: record {h['trained_record_A']}, "
        f"chain {h['trained_chain_A']}; {len(items)} held items ({skipped} skipped for length) in {res['seconds']} s")
    return res


def main():
    args = {a.split("=")[0].lstrip("-"): a.split("=", 1)[1] for a in sys.argv[1:] if "=" in a}
    out, init = args["out"], args["init"]
    n_rows, seed, ctx = int(args.get("rows", "192")), int(args.get("seed", "1")), int(args.get("ctx", "4096"))
    pack_dir = args.get("pack") or os.path.join(out, "thought_pack")
    dev = torch.device(os.environ.get("BATTERY_DEV", "cuda" if torch.cuda.is_available() else "cpu"))
    bed = os.path.join(out, "bed")
    rec_path = os.path.join(bed, f"thought_{init}.json")
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

    held, train = TF.load(pack_dir, "held"), TF.load(pack_dir, "train")
    print(f"[thought] {init}: {held.n} held rows in {pack_dir} ({train.n:,} training rows for the prior and the trained-row read); up to {n_rows} rows; {dev}", flush=True)
    rec = read(logits_fn, dev, held, n_rows, seed, ctx, log=lambda s: print(s, flush=True), train=train)
    rec.update(init=init, seed=seed, ctx=ctx, pack=pack_dir, total_seconds=round(time.time() - t0, 1), device=str(dev))
    os.makedirs(bed, exist_ok=True)
    json.dump(rec, open(rec_path, "w", encoding="utf-8"), indent=1)
    print(f"[thought] done: {rec['total_seconds']:.0f} s -> {rec_path}", flush=True)


if __name__ == "__main__":
    main()
