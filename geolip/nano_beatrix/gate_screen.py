# -*- coding: utf-8 -*-
"""THE GATE SCREEN (0.1.51): the body's own loss decides which of its notes to keep.

The reading-with-notes protocol of bed_train.carry_gauge is kept as it is: a held-out row is read in 64-byte grid steps; at each grid
point the model's greedy next byte decides whether a frame opens (DATA), and an opened frame is generated greedily to END and inserted at
the grid point with the row's tail cut by the same length. This module adds, for every note the body writes at grid point p (L bytes),
three teacher-forced reads of the H bytes of text that follow it:
  WITH      the row as the body left it (the note in place);
  WITHOUT   the same text with the note removed (the continuation from p as the plain row has it);
  WITHHELD  the note with its value bytes replaced by spaces (the key and the frame kept).
gain_without = bits(WITHOUT) - bits(WITH); gain_withheld = bits(WITHHELD) - bits(WITH); both summed over the H bytes. A note is KEPT at
(H, m) when both gains reach m bits. H and m are scanned (horizons 64/128/256/512, margins 0.25/0.5/1/2/4 by default), never set.
A RANDOM-NOTE CONTROL runs beside: a one-record 32-byte frame with a key and a value from the bed's vocabulary, inserted at the same
positions and scored the same way; the control must be able to fail.

Modes: `self` reads the note gauge's rows (held-out documents holding a usable fact, in file order; frames_data.note_row with no note
and no carries: the fact in the text, the question in the last 512 bytes, the notes read from byte 128); `plain` reads the carry
gauge's plain rows (frames_data.plain_row: no fact selection, no question). The horizon never crosses the question.

A child command like bed_train (gauge_stored's convention): --out=<folder> --init=<weights folder under out/bed> --mode=self|plain
[--rows=192 --horizons=64,128,256,512 --margins=0.25,0.5,1,2,4 --seed=1 --ctx=4096 --dump=1]. It writes <out>/bed/gate_<init>_<mode>.json
and, with dump, one JSON line a note beside it. The record: n rows, notes proposed a row, kept a row and the kept share at every (H, m),
the gain distributions (median, p90) on all notes and on the kept at (256, 1), the share of kept notes whose value appears verbatim in
the preceding 1,024 bytes, the share whose key the extractor also finds in the row's text, the content classes, ten kept and ten dropped
examples at (256, 1), the random control's numbers, and the wall time.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from geolip.alephllm.data.special_tokens import CUE, DATA, DOC, END, SEP
from geolip.nano_beatrix import bed_frames as bf
from geolip.nano_beatrix import frames_data as fd

HORIZONS = (64, 128, 256, 512)
MARGINS = (0.25, 0.5, 1.0, 2.0, 4.0)
REF = (256, 1.0)                 # the setting the shares and the examples are reported at
LOOKBACK = 1024                  # a note about the text: its value appears verbatim in this many bytes before it
LN2 = math.log(2)


# ------------------------------------------------------------------ the note's parts
def parse_note(gen: bytes) -> dict | None:
    """The first record of a generated frame: key and value (CUE key SEP value, the value up to the next CUE or END, trailing spaces
    dropped) and the value's byte span inside `gen`. None when the frame holds no CUE ... SEP record."""
    i = gen.find(bytes([CUE]))
    if i < 0:
        return None
    j = gen.find(bytes([SEP]), i + 1)
    if j < 0:
        return None
    k_end = len(gen)
    for stop in (bytes([CUE]), bytes([END])):
        s = gen.find(stop, j + 1)
        if s >= 0:
            k_end = min(k_end, s)
    raw = gen[j + 1:k_end]
    val = raw.rstrip(b" ")
    if not val:
        return None
    return dict(key=gen[i + 1:j], value=val, v0=j + 1, v1=j + 1 + len(val))


def withhold(gen: bytes, parts: dict | None) -> bytes:
    """The note with its value bytes replaced by spaces (the key and the frame kept); an unparsed note comes back unchanged."""
    if parts is None:
        return gen
    return gen[:parts["v0"]] + b" " * (parts["v1"] - parts["v0"]) + gen[parts["v1"]:]


def random_note(rng, size: int = 32) -> bytes:
    """The control's note: a one-record sized frame of `size` bytes with a key and a value from the bed's vocabulary, spaces as slack."""
    key = bf.key_name(int(rng.integers(16000)))
    val = bf.value(rng, "letters", 6)
    state = dict(frames=[], defs={}, facts=None, key_iter=iter(()), fact_iter=None, sized=True, sliced=32)
    return bf.build_frame(rng, bf.spec("data", size, records=1, keys=[key], values=[val]), 0, bf.Filler(b" " * 4096, 0), "caption", state)


# ------------------------------------------------------------------ the variants and their bits
def variants(row: bytes, p: int, note: bytes, horizon: int, control: bytes | None = None) -> dict:
    """The rows to score for a note of len(note) bytes standing at p in `row` (the row as the body left it), with `horizon` bytes of
    text after it: with, without, withheld, and the control's with and withheld when a control note is given. Every variant ends with
    the same `horizon` text bytes; `score_from` is the index of the first of them in each variant."""
    L = len(note)
    assert row[p:p + L] == note, "the note is not where the caller says"
    text = row[p + L:p + L + horizon]
    assert len(text) == horizon, (len(text), horizon)
    head = row[:p]
    out = {"with": (head + note + text, p + L), "without": (head + text, p), "withheld": (head + withhold(note, parse_note(note)) + text, p + L)}
    if control is not None:
        out["control"] = (head + control + text, p + len(control))
        out["control_withheld"] = (head + withhold(control, parse_note(control)) + text, p + len(control))
    return out


@torch.no_grad()
def span_bits(logits_fn, items: list, dev, batch: int = 8) -> list:
    """items: [(row bytes, score_from)]; -> for each, the per-byte bits (float list) of the bytes from score_from to the row's end,
    teacher-forced (the byte at t predicted from the bytes before t). Right-padded batches; each row read at its own positions."""
    out = [None] * len(items)
    order = sorted(range(len(items)), key=lambda i: -len(items[i][0]))
    for i0 in range(0, len(order), batch):
        idx = order[i0:i0 + batch]
        L = max(len(items[i][0]) for i in idx)
        xb = np.full((len(idx), L), 32, dtype=np.uint8)
        for j, i in enumerate(idx):
            r = items[i][0]
            xb[j, :len(r)] = np.frombuffer(r, dtype=np.uint8)
        x = torch.tensor(xb.astype(np.int64), device=dev)
        lg = logits_fn(x[:, :-1]).float()                                   # [B, L - 1, 256]: the prediction for byte t sits at t - 1
        nll = F.cross_entropy(lg.reshape(-1, lg.shape[-1]), x[:, 1:].reshape(-1), reduction="none").view(len(idx), L - 1)
        for j, i in enumerate(idx):
            r, s = items[i]
            out[i] = (nll[j, s - 1:len(r) - 1] / LN2).cpu().tolist()
    return out


def gains(bits: dict, horizons=HORIZONS) -> dict:
    """bits: name -> per-byte bits over the scored span (all variants the same length). -> per horizon H (None where the span is
    shorter): gain_without, gain_withheld, and the control's two when present."""
    n = len(bits["with"])
    out = {}
    for H in horizons:
        if n < H:
            out[H] = None
            continue
        w = sum(bits["with"][:H])
        g = dict(gain_without=sum(bits["without"][:H]) - w, gain_withheld=sum(bits["withheld"][:H]) - w, bits_with=w)
        if "control" in bits:
            c = sum(bits["control"][:H])
            g.update(control_gain_without=sum(bits["without"][:H]) - c, control_gain_withheld=sum(bits["control_withheld"][:H]) - c)
        out[H] = g
    return out


def kept(g: dict | None, m: float, prefix: str = "") -> bool | None:
    if g is None:
        return None
    return bool(g[prefix + "gain_without"] >= m and g[prefix + "gain_withheld"] >= m)


# ------------------------------------------------------------------ the rows
def heldout_docs(out: str, ctx: int) -> list:
    """The held-out documents as bed_train reads them (the same file, the same order, the same length floor)."""
    bed = os.path.join(out, "bed")
    held = [p for p in (os.path.join(out, "heldout_fineweb_edu_head2048.jsonl"), os.path.join(bed, "heldout_fineweb_edu_head2048.jsonl")) if os.path.exists(p)]
    assert held, f"no heldout_fineweb_edu_head2048.jsonl in {out} or {bed}"
    docs = []
    with open(held[0], encoding="utf-8") as f:
        for line in f:
            b = json.loads(line)["text"].encode("utf-8")
            if len(b) >= ctx + 64:
                docs.append(b)
    return docs


def build_rows(docs: list, mode: str, n_rows: int, ctx: int, seed: int) -> tuple:
    """-> (rows as bytearrays, metas, n_scanned). self: frames_data.note_row with no note and no carries over documents holding a fact
    (the note gauge's rows, file order); plain: frames_data.plain_row over the first n_rows documents (the carry gauge's plain rows)."""
    rng = np.random.default_rng([seed, 5151])                             # the gauges' own stream: the same rows as theirs
    rows, metas, scanned = [], [], 0
    form = fd.NOTE
    if mode == "self":
        for doc in docs:
            if len(rows) >= n_rows:
                break
            scanned += 1
            rm = fd.note_row(rng, bytes([DOC]) + doc, form, ctx, notes=False, carries=False)
            if rm is None:
                continue
            r, m = rm
            rows.append(bytearray(r))
            metas.append(dict(start=form["min_pos"], q0=m["q0"], v0=m["v0"], vlen=m["vlen"], fact_end=m["fact_end"], fact_key=m["early_keys"][0],
                              fact_val=m["early_vals"][0], text=bytes(r[:m["q0"]])))
        assert len(rows) == n_rows, f"only {len(rows)} held-out documents of {scanned} hold a usable fact; rows {n_rows}"
    else:
        assert mode == "plain", mode
        assert len(docs) >= n_rows, (len(docs), n_rows)
        for doc in docs[:n_rows]:
            scanned += 1
            r = fd.plain_row(rng, bytes([DOC]) + doc, ctx)
            rows.append(bytearray(r))
            metas.append(dict(start=fd.PLAIN_START, q0=ctx + 1 - form["late"], fact_end=None, fact_key=None, fact_val=None, text=bytes(r)))
    return rows, metas, scanned


# ------------------------------------------------------------------ the screen
@torch.no_grad()
def screen(logits_fn, rows: list, metas: list, dev, seed: int = 1, ctx: int = 4096, horizons=HORIZONS, margins=MARGINS, grid: int = 64,
           max_gen: int | None = None, log=print) -> tuple:
    """The grid protocol with the gate's reads at every note. -> (notes: one dict a note with its gains and flags, reads: grid reads a row)."""
    from geolip.alephllm.data.special_tokens import DATA as _DATA
    form = fd.NOTE
    g = grid
    max_gen = max_gen or (bf.frame_bytes(form["size"], form["sliced"]) + 1)
    crng = np.random.default_rng([seed, 7171])
    n = len(rows)
    p = [m["start"] for m in metas]
    inserted = [0] * n
    q_end = [m["q0"] for m in metas]
    # the room the insertions have: the filler after the answer on a question row (the question and its value must stay inside the
    # window, as bed_train keeps them), the late region on a plain row
    tail_room = [(ctx + 1 - (m["v0"] + m["vlen"] + 1)) if m.get("v0") is not None else form["late"] for m in metas]
    notes, reads = [], [0] * n

    def greedy(prompts):
        out_b, out_p = [], []
        for i0 in range(0, len(prompts), 16):
            chunk = prompts[i0:i0 + 16]
            L = max(len(pm) for pm in chunk)
            xb = np.full((len(chunk), L), 32, dtype=np.uint8)
            for j, pm in enumerate(chunk):
                xb[j, :len(pm)] = np.frombuffer(pm, dtype=np.uint8)
            lg = logits_fn(torch.tensor(xb.astype(np.int64), device=dev)).float()
            last = torch.tensor([len(pm) - 1 for pm in chunk], device=dev)
            at = lg[torch.arange(len(chunk), device=dev), last]
            out_b += at.argmax(-1).cpu().tolist()
            out_p += F.softmax(at, -1)[:, _DATA].cpu().tolist()
        return out_b, out_p

    active = list(range(n))
    t0 = time.time()
    tick = 0
    while active:
        nb, pr = greedy([bytes(rows[i][:p[i]]) for i in active])
        for i in active:
            reads[i] += 1
        openers = [i for i, b in zip(active, nb) if b == _DATA]
        gens = {i: bytearray([_DATA]) for i in openers}
        live = list(openers)
        while live:
            nxt, _ = greedy([bytes(rows[i][:p[i]]) + bytes(gens[i]) for i in live])
            for i, b in zip(live, nxt):
                gens[i].append(int(b))
            live = [i for i in live if len(gens[i]) < max_gen and gens[i][-1] != END]
        items, owners = [], []
        for i in openers:
            full = bytes(gens[i])
            gen = full[:min(len(full), tail_room[i] - inserted[i])]
            fd.insert_carry(rows[i], p[i], gen, ctx)
            inserted[i] += len(gen)
            avail = q_end[i] + inserted[i] - (p[i] + len(gen))                   # text bytes before the question (self) or the late region (plain)
            H = max(h for h in horizons if h <= avail) if avail >= min(horizons) else 0
            parts = parse_note(gen)
            ctrl = random_note(crng)
            ctrl = ctrl[:len(gen)] if len(ctrl) > len(gen) and len(gen) < 8 else ctrl
            rec = dict(row=i, pos=p[i], length=len(gen), generated=full.decode("latin-1"), inserted=gen.decode("latin-1"),
                       ended_on_end=bool(full and full[-1] == END), cut=len(gen) < len(full), parsed=parts is not None,
                       key=(parts["key"].decode("latin-1") if parts else None), value=(parts["value"].decode("latin-1") if parts else None),
                       horizon_max=H, gains=None, control=ctrl.decode("latin-1"))
            if parts is not None:
                before = bytes(rows[i][max(0, p[i] - LOOKBACK):p[i]])
                rec["value_in_text_before"] = parts["value"].lower() in before.lower() and len(parts["value"]) >= 3
                facts = fd.extract_facts(metas[i]["text"], form)
                keys = {k.lower() for _, k, _, _ in facts}
                rec["key_found_by_extractor"] = parts["key"].lower() in keys
                rec["holds_fact_key"] = (metas[i]["fact_key"] is not None and parts["key"].decode("latin-1").lower() == metas[i]["fact_key"].lower())
                rec["holds_fact_value"] = (metas[i]["fact_val"] is not None and parts["value"].decode("latin-1").lower() == metas[i]["fact_val"].lower())
            else:
                rec.update(value_in_text_before=False, key_found_by_extractor=False, holds_fact_key=False, holds_fact_value=False)
            if H:
                v = variants(bytes(rows[i]), p[i], gen, H, ctrl)
                names = list(v)
                items += [v[k] for k in names]
                owners.append((rec, names))
            notes.append(rec)
            p[i] = fd.next_grid(p[i], len(gen), g) - g
        if items:
            bits = span_bits(logits_fn, items, dev)
            k = 0
            for rec, names in owners:
                rec["gains"] = {str(H): gg for H, gg in gains({nm: bits[k + j] for j, nm in enumerate(names)}, horizons).items()}
                k += len(names)
        for i in active:
            p[i] += g
        active = [i for i in active if p[i] < q_end[i] + inserted[i] - g and inserted[i] < tail_room[i]]
        tick += 1
        if tick % 8 == 0:
            log(f"[gate] tick {tick}: {len(active)} rows active, {len(notes)} notes, {time.time() - t0:.0f} s")
    return notes, reads


def summarize(notes: list, n_rows: int, mode: str, horizons=HORIZONS, margins=MARGINS, ref=REF) -> dict:
    """The record's numbers from the notes."""
    scored = [r for r in notes if r["gains"] is not None]
    table = {}
    for H in horizons:
        for m in margins:
            ks = [kept(r["gains"].get(str(H)), m) for r in scored]
            cs = [kept(r["gains"].get(str(H)), m, "control_") for r in scored]
            have = [k for k in ks if k is not None]
            chave = [c for c in cs if c is not None]
            table[f"{H}|{m}"] = dict(scored=len(have), kept=sum(have), kept_a_row=sum(have) / n_rows, kept_share=(sum(have) / len(have) if have else None),
                                   control_kept=sum(chave), control_kept_share=(sum(chave) / len(chave) if chave else None))
    Hr, mr = ref
    at = [r for r in scored if r["gains"].get(str(Hr)) is not None]
    kept_at = [r for r in at if kept(r["gains"][str(Hr)], mr)]
    dropped_at = [r for r in at if not kept(r["gains"][str(Hr)], mr)]

    def dist(vals):
        if not vals:
            return None
        a = np.array(vals, dtype=np.float64)
        return dict(n=len(a), median=float(np.median(a)), p90=float(np.percentile(a, 90)), mean=float(a.mean()), share_positive=float((a > 0).mean()))

    def share(rs, key):
        return (sum(1 for r in rs if r.get(key)) / len(rs)) if rs else None

    ex = lambda r: dict(row=r["row"], pos=r["pos"], note=r["inserted"], key=r["key"], value=r["value"], gain_without=r["gains"][str(Hr)]["gain_without"],
                        gain_withheld=r["gains"][str(Hr)]["gain_withheld"], value_in_text_before=r.get("value_in_text_before"))
    return dict(mode=mode, rows=n_rows, notes=len(notes), notes_a_row=len(notes) / n_rows, rows_with_notes=len({r["row"] for r in notes}) / n_rows,
                parsed_share=(sum(1 for r in notes if r["parsed"]) / len(notes) if notes else None), scored=len(scored), ref=dict(horizon=Hr, margin=mr),
                table=table,
                gain_without_all=dist([r["gains"][str(Hr)]["gain_without"] for r in at]), gain_withheld_all=dist([r["gains"][str(Hr)]["gain_withheld"] for r in at]),
                gain_without_kept=dist([r["gains"][str(Hr)]["gain_without"] for r in kept_at]), gain_withheld_kept=dist([r["gains"][str(Hr)]["gain_withheld"] for r in kept_at]),
                control_gain_without_all=dist([r["gains"][str(Hr)]["control_gain_without"] for r in at]),
                kept_value_in_text_before=share(kept_at, "value_in_text_before"), kept_key_found_by_extractor=share(kept_at, "key_found_by_extractor"),
                kept_holds_fact_key=share(kept_at, "holds_fact_key"), all_value_in_text_before=share(at, "value_in_text_before"),
                kept_examples=[ex(r) for r in kept_at[:10]], dropped_examples=[ex(r) for r in dropped_at[:10]])


def main():
    args = {a.split("=")[0].lstrip("-"): a.split("=", 1)[1] for a in sys.argv[1:] if "=" in a}
    out, init, mode = args["out"], args["init"], args.get("mode", "self")
    n_rows, seed, ctx = int(args.get("rows", "192")), int(args.get("seed", "1")), int(args.get("ctx", "4096"))
    horizons = tuple(int(x) for x in args.get("horizons", ",".join(map(str, HORIZONS))).split(","))
    margins = tuple(float(x) for x in args.get("margins", ",".join(map(str, MARGINS))).split(","))
    dump = args.get("dump", "1") == "1"
    dev = torch.device(os.environ.get("BATTERY_DEV", "cuda" if torch.cuda.is_available() else "cpu"))
    bed = os.path.join(out, "bed")
    rec_path = os.path.join(bed, f"gate_{init}_{mode}.json")
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

    docs = heldout_docs(out, ctx)
    rows, metas, scanned = build_rows(docs, mode, n_rows, ctx, seed)
    print(f"[gate] {init} {mode}: {len(rows)} rows ({scanned} scanned); horizons {horizons}; margins {margins}; {dev}", flush=True)
    notes, reads = screen(logits_fn, rows, metas, dev, seed=seed, ctx=ctx, horizons=horizons, margins=margins, log=lambda s: print(s, flush=True))
    rec = summarize(notes, len(rows), mode, horizons, margins)
    rec.update(init=init, seed=seed, ctx=ctx, scanned=scanned, grid_reads_a_row=sum(reads) / len(rows), seconds=time.time() - t0, device=str(dev),
               horizons=list(horizons), margins=list(margins))
    os.makedirs(bed, exist_ok=True)
    if dump:
        dp = os.path.join(bed, f"gate_{init}_{mode}_notes.jsonl")
        with open(dp, "w", encoding="utf-8") as fh:
            for r in notes:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        rec["dump"] = dp
    json.dump(rec, open(rec_path, "w", encoding="utf-8"), indent=1)
    t = rec["table"].get(f"{REF[0]}|{REF[1]}", {})
    print(f"[gate] done: {rec['notes_a_row']:.2f} notes a row; at ({REF[0]}, {REF[1]}) kept {t.get('kept_a_row', 0):.2f} a row, share {t.get('kept_share')}, "
          f"control share {t.get('control_kept_share')}; {rec['seconds']:.0f} s -> {rec_path}", flush=True)


if __name__ == "__main__":
    main()
