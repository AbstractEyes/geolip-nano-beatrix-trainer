# -*- coding: utf-8 -*-
"""bed_train.py - THE NANO BED: a small byte craft trained from birth on fineweb-edu text with planted records (or framed rows,
bed_frames.py), to measure what a craft this small holds and how fast it trains. One ARM per run = the attention form x the bank's
dispatch form (the library's flags):
  attention  sm    softmax + the learned position table                       smr   softmax + rotary (the table kept)
             smrn  softmax + rotary, no table                                  hub   every block a splat hub (the chunked scan with an aleph address)
             hubr  hubs with the address rotary (RoPE on the oriented codes)   alt   hub / softmax+rotary alternating
             hubrn hubs with the address rotary, no position table             hubd  hubs with two address masses a block
             altn  alternating, both rotaries
             castor = smr and pollux = altn: the names of the nano twin pair (Castor the mortal twin whose memory grows with the
             window, Pollux the immortal whose hub half keeps a constant decode memory)
  bank       free  the aleph address alone (the current form)   fixed  an installed byte-class table REPLACES the amplitudes (one
             expert per class)                                   both   the address PLUS the installed table (learned)
Gauges: the online value-byte loss by distance band during training; at the end on held-out documents: bits a byte by byte class,
the planted fetch at 64 / 256 / 512 / 896 bytes (teacher-forced value NLL, greedy first byte, all five), the dispatch census by byte
class (block 0 / middle / last), the dispatch-off toggle. Pure Adam. Progress lines carry the job size, the time spent and the time left.
Usage (through card_gate / run_cf): bed_train.py --out=DIR --arm=sm-free --seed=1 [--steps=3000 --mb=32 --ctx=1024 --lr=1e-3 --smoke=1 --prefetch=0]
Framed rows: --frames=dyn [--sizes=32,128,256 --frame_records=4 --nest=0.3 --overlap=0.0 --ask=any --sized=1 --sliced=32] (bed_frames.py;
--sized=1 names every frame's size in its open tag, --sliced=32 builds every larger frame as closed 32-byte slices; both off by default)
[--gauge_asks=nested,overlap]: after the main gauge, the held-out fetch asked each named question strictly (the diet may ask any)
A pair run: [--pair=ARM2 --xc=one|mutual --xc_pos=value|all --xc_w=1 --xc_from=0 --pair_init=TAG2]: a second craft trains beside the
first on the very same rows, each with its own Adam and the same schedule, plus a cross term: the KL from one craft's next-byte
distribution (held fixed) to the other's, at the asked values' bytes or at every byte, a byte's term weighted like its own target; "one":
the partner learns from this arm (this arm trains exactly as alone), "mutual": each from the other. Both crafts start from the run's seed,
so with --xc_w=0 each trains exactly as it would alone. Two records (the partner's first), each with the pair's reads (twin_reads).
Data: DIR/bed/bed_corpus.bin + bed_corpus_offsets.npy (bed_corpus.py); held-out heldout_fineweb_edu_head2048.jsonl in DIR or DIR/bed (the reserved head).
Output: DIR/bed/bed_<arm>_s<seed>[_smoke].json; the weights DIR/bed/<arm>_s<seed>/model.safetensors + config.json"""
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from geolip.alephllm.presets import AlephLMConfig  # noqa: E402
from geolip.alephllm.model.alephlm import AlephLM  # noqa: E402
from geolip.alephllm.model.bank import BYTE_CLASSES, byte_class_table  # noqa: E402
from geolip.alephllm.data.special_tokens import DOC  # noqa: E402
try:
    from . import bed_frames as bf  # noqa: E402  (the package)
except ImportError:                                      # run as a file (run_cf.py): the script's own folder on the path
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import bed_frames as bf  # noqa: E402
try:
    from .names import partner_args, run_tag  # noqa: E402  (the package)
except ImportError:                                      # run as a file: the folder is on the path (above)
    from names import partner_args, run_tag  # noqa: E402
try:
    from . import bed_pair  # noqa: E402  (the package)
except ImportError:                                      # run as a file: the folder is on the path (above)
    import bed_pair  # noqa: E402

ARGS = {a.split("=")[0].lstrip("-"): a.split("=", 1)[1] for a in sys.argv[1:] if "=" in a}
OUT, ARM, SEED = ARGS["out"], ARGS["arm"], int(ARGS.get("seed", "1"))
SMOKE = ARGS.get("smoke", "0") == "1"
STEPS = int(ARGS.get("steps", "20" if SMOKE else "3000"))
MB = int(ARGS.get("mb", "8" if SMOKE else "32"))
CTX = int(ARGS.get("ctx", "1024"))
LR = float(ARGS.get("lr", "1e-3"))
WARM = int(ARGS.get("warmup", "5" if SMOKE else "100"))
P_PLANT = float(ARGS.get("plant", "0.25"))
N_EVAL = int(ARGS.get("eval_docs", "16" if SMOKE else "128"))
N_TRIALS = int(ARGS.get("trials", "8" if SMOKE else "64"))
DISTS = [int(x) for x in ARGS.get("dists", "64,256,512,896").split(",")]
# the framed rows: "" = the planted rows as before (bit-identical); "A" = the fixed reference layout;
# "dyn" = dynamic sectors of generic sizes (bed_frames.py)
FRAMES = ARGS.get("frames", "")
assert FRAMES in ("", "A", "dyn"), FRAMES
REGISTER = ARGS.get("register", "caption")
TAGS = ARGS.get("tags", "mode")
N_LOOKUP = int(ARGS.get("n_lookup", "2"))
N_BULK = int(ARGS.get("n_bulk", "0"))
N_FACTS = int(ARGS.get("facts", "0"))                       # a fact table recurring across training (the in-weights axis)
ASK = ARGS.get("ask", "lookup" if FRAMES == "A" else "any")
SIZES = [int(x) for x in ARGS.get("sizes", "32,128,256").split(",")]
N_REC_FRAME = int(ARGS.get("frame_records", "4"))         # the most records a dynamic lookup frame carries
NEST_P = float(ARGS.get("nest", "0.3"))
OVERLAP_P = float(ARGS.get("overlap", "0.0"))
# 0.1.23: the strict questions read by the gauge alone. A curriculum's diet asks any (a craft trained on a strict ask learns a stricter
# opening grammar); the nested and the redefined-key questions are then the gauge's: each named ask is read on its own held-out rows after
# the main gauge (fetch_by_frame_asks). A gauge row's plan draws the asked feature at least as often as the scrutiny settings of record
# (nesting 0.6, a key defined twice in 0.3 of the rows): the redraw until the feature appears stays exact, and a craft trained on that ask
# with those settings is gauged on the very same rows. Empty by default (the record unchanged).
GAUGE_ASKS = [a for a in ARGS.get("gauge_asks", "").split(",") if a]
assert all(a in ("nested", "overlap") for a in GAUGE_ASKS) and (not GAUGE_ASKS or FRAMES == "dyn"), f"--gauge_asks={GAUGE_ASKS}: nested, overlap (--frames=dyn)"
GAUGE_FLOOR = {"nested": ("nest_p", 0.6), "overlap": ("overlap_p", 0.3)}
# 0.1.21: the sized frame (every open tag names its frame's generic size in one payload byte) and the sliced frame (every frame larger
# than the slice size built as closed slices of that size; a sliced frame is a sized frame); both off by default (the rows bit-identical)
SIZED = ARGS.get("sized", "0") == "1"
SLICED = int(ARGS.get("sliced", "0"))
if SIZED or SLICED:
    assert FRAMES == "dyn", "--sized and --sliced shape the dynamic frames (--frames=dyn)"
    assert SIZED or not SLICED, "--sliced needs --sized=1 (a slice is a size-typed frame)"
    _gen = set(SIZES) | {max(32, s // 4) for s in SIZES}                      # the plan's frame sizes and its nested child sizes
    assert _gen <= set(bf.SIZE_LETTER), f"a sized frame's size must be one of {sorted(bf.SIZE_LETTER)}: {sorted(_gen)}"
    assert not SLICED or (SLICED in bf.SIZE_LETTER and all(s % SLICED == 0 for s in _gen)), \
        f"--sliced={SLICED}: every frame size must be a whole number of slices: {sorted(_gen)}"
LOG_EVERY = int(ARGS.get("log_every", "5" if SMOKE else "50"))
PREFETCH = ARGS.get("prefetch", "1") == "1"   # the next step's rows compose in a worker thread while this step runs on the card;
                                              # the batches are identical either way (every step seeds its own rng from [SEED, step])
DEV = os.environ.get("BATTERY_DEV", "cuda")
BED = os.path.join(OUT, "bed")
os.makedirs(BED, exist_ok=True)
TAG = run_tag(ARGS)        # the record and weights names: one function with the runner's resume (names.run_tag)
INIT = ARGS.get("init", "")        # a previous run's weights folder under DIR/bed: its config.json sets the craft, the context may change
# 0.1.24: the pair run (module docstring). The partner is built from the run's seed like this arm, so with --xc_w=0 each craft starts,
# reads its rows and trains exactly as it would alone: the parity the pair's comparisons with solo runs rest on.
PAIR = ARGS.get("pair", "")
PAIR_INIT = ARGS.get("pair_init", "")      # the partner's starting craft (a continuation pair): a weights folder under DIR/bed
XC = ARGS.get("xc", "one")                 # "one": the partner learns from this arm; "mutual": each craft from the other
XC_POS = ARGS.get("xc_pos", "value")       # "value": the asked values' bytes (the lookups' answers); "all": every predicted byte
XC_W = float(ARGS.get("xc_w", "1.0"))      # the cross term's weight a byte: 1 = the byte's own target's weight (the loss is a mean)
XC_FROM = int(ARGS.get("xc_from", "0"))    # the cross term's first step
assert "pair_role" not in ARGS, "--pair_role is the partner's name piece, set by the run itself"
assert not PAIR or (PAIR != ARM and XC in ("one", "mutual") and XC_POS in ("value", "all")), \
    f"--pair={PAIR!r} --xc={XC!r} --xc_pos={XC_POS!r}: the partner is another arm; xc one or mutual; xc_pos value or all"
assert not PAIR_INIT or PAIR, "--pair_init names the partner's starting craft (--pair)"
PAIR_TAG = run_tag(partner_args(ARGS)) if PAIR else ""     # the partner's own name: its arm and starting craft, this arm, its role
T0 = time.time()
LN2 = math.log(2.0)
BANDS = [(0, 128), (128, 256), (256, 512), (512, 100000)]
N_BANDS = len(BANDS) + 1                                   # + the weights source (a value with no distance; the framed rows)
BAND_LABELS = [f"d{lo}+" for lo, _ in BANDS] + ["w"]
CLS_TABLE = byte_class_table().numpy()


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')} +{time.time() - T0:6.0f}s] {msg}", flush=True)


# ------------------------------------------------------------------ the arms
def craft_cfg(arm: str) -> AlephLMConfig:
    attn, bank = arm.split("-")
    L = 6
    base = dict(name=f"bed-{arm}", d_model=256, n_layers=L, n_heads=8, context=CTX, hub_layers=(), hub_K=32, hub_D=16, tau=0.1,
                bank_experts=7, bank_ff=256, head_K=64, head_D=16, gate_init=-3.0, hub_chunk=128)
    A = {"sm": {}, "smr": dict(attn_rotary=True), "smrn": dict(attn_rotary=True, pos_table=False),
         "hub": dict(hub_layers=tuple(range(L))), "hubr": dict(hub_layers=tuple(range(L)), hub_rotary=True),
         "hubrn": dict(hub_layers=tuple(range(L)), hub_rotary=True, pos_table=False),
         "hubd": dict(hub_layers=tuple(range(L)), hub_dual=True),
         "alt": dict(hub_layers=tuple(range(0, L, 2)), attn_rotary=True),
         "altn": dict(hub_layers=tuple(range(0, L, 2)), attn_rotary=True, hub_rotary=True)}
    A["castor"] = A["smr"]            # the twins' names: Castor = the rotary softmax body (the mortal twin, the growing memory)
    A["pollux"] = A["altn"]           # Pollux = the all-rotary alternating body (the immortal twin, the constant decode memory)
    # the dual-hub bodies: each hub keeps two address masses, one rotary (sharp near) and one plain (distance-free, far reach); with the
    # dense planted records the dual hub recalled the full value at every distance where a rotary-only hub lost the far records
    A["hubdb"] = dict(A["hubd"], hub_K=64, hub_D=32, hub_const=2)        # six dual hubs on the bigger address (two books of 64 in 32)
    A["altd"] = dict(hub_layers=tuple(range(0, L, 2)), attn_rotary=True, hub_dual=True)   # Pollux's layout with dual hubs
    A["d5s1"] = dict(hub_layers=(0, 1, 2, 4, 5), attn_rotary=True, hub_dual=True)        # five dual hubs, one rotary softmax block (3)
    A["d4s2"] = dict(hub_layers=(0, 2, 4, 5), attn_rotary=True, hub_dual=True)           # four dual hubs, two rotary softmax blocks (1, 3)
    Bk = {"free": "", "fixed": "fixed", "both": "add"}
    assert attn in A and bank in Bk, f"unknown arm {arm!r}: attention in {sorted(A)}, bank in {sorted(Bk)}"
    extra = {}
    if "rotary_base" in ARGS:
        extra["rotary_base"] = float(ARGS["rotary_base"])
    if "hub_rotary_frac" in ARGS:
        extra["hub_rotary_frac"] = float(ARGS["hub_rotary_frac"])
    for k in ("hub_K", "hub_D", "hub_const"):          # the hub's address size (the capacity question)
        if k in ARGS:
            extra[k] = int(ARGS[k])
    return AlephLMConfig(**{**base, **A[attn], **extra}, bank_byte_table=Bk[bank])


# ------------------------------------------------------------------ the planted record
CUES = [("code", "digits", 5), ("word", "letters", 6), ("key", "hex", 8), ("number", "digits", 7), ("token", "alnum", 6)]
# the dense composer's extra cue words: compose_row draws its cues from CUES, so --records above 5 plants five records
# a row; with --cues=16 it draws from CUES + EXTRA_CUES (plant() and the fetch keep the five, so every fetch stays comparable)
EXTRA_CUES = [("phrase", "letters", 6), ("label", "alnum", 6), ("serial", "digits", 6), ("pin", "digits", 5), ("badge", "hex", 6),
              ("ticket", "digits", 7), ("slot", "alnum", 5), ("tag", "letters", 5), ("mark", "hex", 7), ("sign", "letters", 7), ("note", "alnum", 7)]
N_CUES = int(ARGS.get("cues", "5"))
assert N_CUES in (5, 16), N_CUES
ROW_CUES = CUES if N_CUES == 5 else CUES + EXTRA_CUES
ALPH = {"digits": b"0123456789", "letters": b"abcdefghijklmnopqrstuvwxyz", "hex": b"0123456789abcdef",
        "alnum": b"abcdefghijklmnopqrstuvwxyz0123456789"}


def value(rng, kind, n):
    a = ALPH[kind]
    v = bytes(a[int(i)] for i in rng.integers(len(a), size=n))
    if kind == "digits" and v[0:1] == b"0":
        v = b"1" + v[1:]
    return v


def boundary(w, start, limit):
    """The first sentence start (after '. ' or a newline) in [start, limit), else the first byte after a space, else start."""
    for i in range(start, min(limit, len(w) - 1)):
        if (w[i - 1] == 32 and i >= 2 and w[i - 2] == 46) or w[i - 1] == 10:
            return i
    for i in range(start, min(limit, len(w) - 1)):
        if w[i - 1] == 32:
            return i
    return start


def snap(w, target, lim):
    """target moved to the next byte-after-space within 24 bytes (not past lim)."""
    for i in range(target, min(target + 24, lim)):
        if w[i - 1] == 32:
            return i
    return min(target, lim)


def plant(w: bytes, rng, dist=None, form=None):
    """Insert a record and its recurrence into the window w (len ctx+1 or longer), keep ctx+1 bytes.
    -> (bytes, meta) with meta: p (the plant start), q (the question start), v0 (the value's first byte index in the output),
    vlen, dist = q - p, form. dist None = drawn (half log-uniform 48..512, half uniform 512..the room); form None = drawn."""
    n = len(w)
    cue, kind, vn = CUES[int(rng.integers(len(CUES)))]
    form = form or ("single", "single", "single", "lockers", "update")[int(rng.integers(5))]
    p = boundary(w, 16, 160)
    if form == "single":
        v = value(rng, kind, vn)
        rec = f"The secret {cue} is ".encode() + v + b". "
        head, ques, vals = rec, rec, [v]
        prefix = len(f"The secret {cue} is ".encode())
    elif form == "lockers":
        lk = [str(int(x)) for x in rng.choice(np.arange(1000, 10000), size=4, replace=False)]
        vals = [value(rng, kind, vn) for _ in lk]
        head = b"".join(f"Locker {l} holds {cue} ".encode() + x + b". " for l, x in zip(lk, vals))
        a = int(rng.integers(4))
        ques = f"Locker {lk[a]} holds {cue} ".encode() + vals[a] + b". "
        prefix = len(f"Locker {lk[a]} holds {cue} ".encode())
        vals = [vals[a]]
    else:
        v1, v2 = value(rng, kind, vn), value(rng, kind, vn)
        head = f"The secret {cue} is ".encode() + v1 + b". "
        upd = (f"The secret {cue} is now ".encode() if rng.random() < 0.5 else f"The secret {cue} is ".encode()) + v2 + b". "
        ques = f"The secret {cue} is ".encode() + v2 + b". "
        prefix = len(f"The secret {cue} is ".encode())
        vals = [v2]
    Lh, Lq = len(head), len(ques)
    room = CTX + 1 - Lh - Lq - 2 - (len(upd) if form == "update" else 0)   # the question must end inside the window
    if room - p < Lh + 48:
        return None
    if dist is None:
        lo, hi = Lh + 48, room - p
        if hi <= 512 or rng.random() < 0.5:
            d = int(round(math.exp(rng.uniform(math.log(lo), math.log(min(512, hi))))))
        else:
            d = int(rng.integers(512, hi + 1))
    else:
        d = int(dist)
        if d < Lh + 8 or p + d > room:
            return None
    g = d - Lh                                     # the gap text between the record and the recurrence
    q0 = snap(w, p + g, min(p + g + 24, room - Lh)) if dist is None else p + g
    gap = w[p:q0]
    if form == "update":
        m = max(8, len(gap) // 2)
        m = snap(gap, m, len(gap)) if len(gap) > 40 else m
        gap = gap[:m] + upd + gap[m:]
    out = w[:p] + head + gap + ques + w[q0:]
    out = out[:CTX + 1]
    q = p + Lh + len(gap)
    v0 = q + prefix
    assert out[v0:v0 + len(vals[0])] == vals[0], "the planted value is not where the arithmetic says"
    return out, dict(p=p, q=q, v0=v0, vlen=len(vals[0]), dist=q - p, form=form)


def compose_row(rng, K):
    """THE DENSE ROW: K single-form records planted in web text, each recurring d bytes later (d drawn as plant() draws it), the cues distinct within
    the row so the records are distinguishable; the filler is corpus text. -> (bytes of CTX + 1, [meta...]) with meta as plant() gives."""
    cues = [ROW_CUES[i] for i in rng.permutation(len(ROW_CUES))[:K]]      # at most len(ROW_CUES) records a row (5 or 16)
    recs = []
    for cue, kind, vn in cues:
        v = value(rng, kind, vn)
        rec = f"The secret {cue} is ".encode() + v + b". "
        recs.append((rec, len(f"The secret {cue} is ".encode()), len(v)))
    Lmax = max(len(r[0]) for r in recs)
    events = []
    for k, (rec, prefix, vlen) in enumerate(recs):
        lo = 16 + k * (CTX // (2 * K))
        p = int(rng.integers(lo, lo + max(8, CTX // (2 * K) - len(rec))))
        room = CTX + 1 - len(rec) - 2
        hi = room - p - len(rec)
        if hi <= 48:
            continue
        if hi <= 512 or rng.random() < 0.5:
            d = int(round(math.exp(rng.uniform(math.log(48), math.log(min(512, hi))))))
        else:
            d = int(rng.integers(512, hi + 1))
        events.append((p, "plant", k))
        events.append((p + d, "ques", k))
    events.sort()
    src = packed[int(rng.integers(0, len(packed) - 2 * CTX - 4)):][:2 * CTX + 2].tobytes()
    out, cur, si, metas, planted_at = bytearray(), 0, 0, {}, {}
    for pos, kind, k in events:
        rec, prefix, vlen = recs[k]
        if pos < len(out):
            pos = len(out)                     # events never overlap: the later one moves after the earlier
        gap = pos - len(out)
        out += src[si:si + gap]
        si += gap
        if len(out) + len(rec) > CTX + 1:
            break
        start = len(out)
        out += rec
        if kind == "plant":
            planted_at[k] = start
        elif k in planted_at:
            metas[k] = dict(p=planted_at[k], q=start, v0=start + prefix, vlen=vlen, dist=start - planted_at[k], form="single")
    out += src[si:si + (CTX + 1 - len(out))]
    out = bytes(out[:CTX + 1])
    if len(out) < CTX + 1:
        out = out + b" " * (CTX + 1 - len(out))
    kept = []
    for k, m in metas.items():
        rec, prefix, vlen = recs[k]
        if m["v0"] + vlen <= len(out):
            assert out[m["v0"]:m["v0"] + vlen] == rec[prefix:prefix + vlen], "a dense record's value is not where the arithmetic says"
            kept.append(m)
    return out, kept


# ------------------------------------------------------------------ the framed rows (--frames)
FACTS = bf.fact_table(N_FACTS, REGISTER) if (FRAMES and N_FACTS) else None
FRAME_CFG = (dict(mode=FRAMES, register=REGISTER, tags=TAGS, n_lookup=N_LOOKUP, n_bulk=N_BULK, facts=N_FACTS, ask=ASK, sizes=SIZES,
                  frame_records=N_REC_FRAME, nest=NEST_P, overlap=OVERLAP_P) if FRAMES else None)
if SIZED:
    FRAME_CFG.update(sized=True, sliced=SLICED)
REC_BYTES = bf.capacity(CTX, "A", REGISTER)["record_bytes"] if FRAMES else 0


def cap(size: int) -> int:
    """The records a lookup frame of this size holds in the run's form (plain, sized or sliced; bed_frames.frame_capacity)."""
    return bf.frame_capacity(size, REC_BYTES, SIZED, SLICED)


def draw_plan(rng, nest_p=None, overlap_p=None):
    """One row's dynamic plan: frames of the generic sizes while they fit the window (a gap and the question reserved), lookup frames
    with as many records as fit up to N_REC_FRAME, a thinking frame nesting a smaller lookup frame with probability NEST_P, an
    overlapping pair of lookup frames in conflict with probability OVERLAP_P. -> (specs, has_nested, has_overlap). A frame's room and
    length follow the run's form (a sized open tag has two bytes; a sliced frame is its slices plus three tag bytes). nest_p and
    overlap_p replace the run's two probabilities for one draw (0.1.23: a strict-ask gauge row; None = the run's own)."""
    nest_p = NEST_P if nest_p is None else nest_p
    overlap_p = OVERLAP_P if overlap_p is None else overlap_p
    budget = CTX + 1 - 6 - 64 - (REC_BYTES + 2)
    specs, used, nested, overlap = [], 0, False, False
    if overlap_p and rng.random() < overlap_p:
        sz = SIZES[0]
        specs += bf.overlap_specs(rng, REGISTER, sz, sz, max(1, min(2, cap(sz))), "conflict")
        used += 2 * (bf.frame_bytes(sz, SLICED) + 64)
        overlap = True
    for _ in range(int(rng.integers(1, 4))):
        size = int(SIZES[int(rng.integers(len(SIZES)))])
        if used + bf.frame_bytes(size, SLICED) + 64 > budget:
            continue
        if rng.random() < 0.7:
            n = min(N_REC_FRAME, cap(size))
            specs.append(bf.spec("data", size, records=n) if n >= 1 else bf.spec("think", size))
        else:
            kids = []
            csize = max(32, size // 4)
            if rng.random() < nest_p and cap(csize) >= 1 and csize <= size - 4:
                kids = [bf.spec("data", csize, records=min(N_REC_FRAME, cap(csize)))]
                nested = True
            specs.append(bf.spec("think", size, children=kids))
        used += bf.frame_bytes(size, SLICED) + 64
    if not any(sp["kind"] == "data" or sp["children"] for sp in specs):     # nothing to ask: the smallest lookup frame
        sz = SIZES[0]
        specs.append(bf.spec("data", sz, records=max(1, min(N_REC_FRAME, cap(sz)))))
    return specs, nested, overlap


def frame_row(rng, src, strict=False, ask_as=None):
    """A framed row. Training (strict=False): the ask falls back to 'any' when the drawn plan lacks the asked feature, and the weights
    ask is asked half the time (the other half reads the frame). The held-out gauge (strict=True): the plan is redrawn until it has the
    feature and the ask is exactly ASK, or exactly `ask_as` (0.1.23, a strict-ask gauge: the plan draws the feature at least as often as
    GAUGE_FLOOR). -> (row bytes of CTX + 1, [meta])."""
    if FRAMES == "A":
        assert ask_as is None, "the strict-ask gauge reads the dynamic frames"
        ask = ASK
        if ask == "weights" and not strict and rng.random() < 0.5:
            ask = "lookup" if N_LOOKUP else "bulk"
        row, m = bf.compose(rng, src, CTX, "A", TAGS, REGISTER, N_LOOKUP, N_BULK, FACTS, ask)
        return row, [m]
    want, kw = ask_as or ASK, {}
    if ask_as:
        key, floor = GAUGE_FLOOR[ask_as]
        kw[key] = max(NEST_P if key == "nest_p" else OVERLAP_P, floor)
    for _ in range(64):
        specs, nested, overlap = draw_plan(rng, **kw)
        ask = want
        if (ask == "nested" and not nested) or (ask == "overlap" and not overlap) or (ask == "weights" and FACTS is None):
            ask = "any"
        elif ask == "weights" and not strict and rng.random() < 0.5:
            ask = "any"
        if strict and ask != want:
            continue
        row, m = bf.compose_dynamic(rng, src, CTX, specs, REGISTER, TAGS, ask, FACTS, sized=SIZED, sliced=SLICED)
        return row, [m]
    raise RuntimeError(f"no plan with the feature for ask={want!r} in 64 draws")


# ------------------------------------------------------------------ data
N_REC = int(ARGS.get("records", "1"))        # records per planted row (1 = plant(); >1 = compose_row, the dense recon regime)
corpus = np.fromfile(os.path.join(BED, "bed_corpus.bin"), dtype=np.uint8)
offs = np.load(os.path.join(BED, "bed_corpus_offsets.npy"))
packed = np.insert(corpus, offs[1:], np.uint8(DOC))           # DOC after every document, as the packer does
log(f"corpus {len(corpus) / 1e6:.1f} MB, {len(offs) - 1:,} documents; packed {len(packed) / 1e6:.1f} MB")
heldout = []
HELD = [p for p in (os.path.join(OUT, "heldout_fineweb_edu_head2048.jsonl"), os.path.join(BED, "heldout_fineweb_edu_head2048.jsonl")) if os.path.exists(p)]
assert HELD, f"no heldout_fineweb_edu_head2048.jsonl in {OUT} or {BED} (bed_corpus.py writes it beside the corpus)"
with open(HELD[0], encoding="utf-8") as f:
    for line in f:
        b = json.loads(line)["text"].encode("utf-8")
        if len(b) >= CTX + 64:
            heldout.append(b)
eval_docs = heldout[:N_EVAL]
fetch_docs = heldout[N_EVAL:N_EVAL + N_TRIALS]
assert len(fetch_docs) == N_TRIALS, (len(heldout), N_EVAL, N_TRIALS)
log(f"held-out: {len(eval_docs)} documents for bits a byte, {len(fetch_docs)} for the fetch (x {len(DISTS)} distances)")


PLANTED = [0, 0]   # records planted, rows drawn (the census of what the composer really gave)


def batch(step):
    """(x (MB, ctx), y (MB, ctx), vmask (MB, ctx) with the band id + 1 at the value bytes' target positions, 0 elsewhere)."""
    rng = np.random.default_rng([SEED, step])
    pos = rng.integers(0, len(packed) - CTX - 2, size=MB)
    rows, vm = [], np.zeros((MB, CTX), dtype=np.int64)
    for i, p in enumerate(pos):
        w = packed[p:p + CTX + 1].tobytes()
        metas = []
        if rng.random() < P_PLANT:
            if FRAMES:
                w, metas = frame_row(rng, packed[p:p + 2 * CTX + 2].tobytes())
            elif N_REC > 1:
                w, metas = compose_row(rng, N_REC)
            else:
                r = plant(w, rng)
                if r is not None:
                    w, meta = r
                    metas = [meta]
        PLANTED[0] += len(metas)
        PLANTED[1] += 1
        for meta in metas:
            band = len(BANDS) if meta["dist"] is None else next(k for k, (lo, hi) in enumerate(BANDS) if lo <= meta["dist"] < hi)
            lo = max(meta["v0"] - 1, 0)
            vm[i, lo:lo + meta["vlen"]] = band + 1
        rows.append(np.frombuffer(w, dtype=np.uint8))
    x = torch.from_numpy(np.stack(rows).astype(np.int64))
    return x[:, :-1], x[:, 1:], torch.from_numpy(vm)


# ------------------------------------------------------------------ the craft (two in a pair run)
torch.backends.cuda.matmul.allow_tf32 = True
if DEV == "cuda":
    torch.cuda.set_per_process_memory_fraction(float(os.environ.get("BATTERY_GPU_FRAC", "0.35")))   # the card_gate cap


class Craft:
    """One craft of the run: its config, its model, its pure Adam and its online gauges. Every craft is built right after seeding the
    generators with the run's seed, so a pair's partner starts exactly as it would alone. role: "" alone; in a pair "t" (it teaches its
    partner), "s" (it learns from its partner) or "m" (both ways)."""

    def __init__(self, arm: str, init: str, tag: str, role: str) -> None:
        self.arm, self.init, self.tag, self.role = arm, init, tag, role
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        if init:
            c = json.load(open(os.path.join(BED, init, "config.json"), encoding="utf-8"))
            c = {k: (tuple(v) if isinstance(v, list) else v) for k, v in c.items()}
            c["context"] = CTX
            self.cfg = AlephLMConfig(**c)
            assert self.cfg.name == f"bed-{arm}", (self.cfg.name, arm)
        else:
            self.cfg = craft_cfg(arm)
        self.model = AlephLM(self.cfg).to(DEV)
        if init:
            from safetensors.torch import load_file
            sd = load_file(os.path.join(BED, init, "model.safetensors"))
            pos = sd.get("embed.pos")
            if pos is not None and pos.shape[1] != CTX:      # the trained rows kept, the new positions zero (the rotary carries the phase)
                new_pos = torch.zeros(1, CTX, pos.shape[2], dtype=pos.dtype)
                n_keep = min(CTX, pos.shape[1])
                new_pos[:, :n_keep] = pos[:, :n_keep]
                sd["embed.pos"] = new_pos
            self.model.load_state_dict({k: v.to(DEV) for k, v in sd.items()}, strict=True)
            log(f"continued from {init}: {len(sd)} tensors; the position table {tuple(pos.shape) if pos is not None else None} -> {tuple(self.model.embed.pos.shape)}")
        self.n_params = self.model.param_count()
        self.params = [p for p in self.model.parameters() if p.requires_grad]
        self.opt = torch.optim.Adam(self.params, lr=LR, betas=(0.9, 0.95), eps=1e-8)      # pure Adam, never AdamW
        self.curve, self.ema, self.li, self.gn = [], None, None, None
        self.online = {k: [0.0, 0] for k in range(N_BANDS)}      # the value bytes' nats and count by band since step 1 (never reset)
        self.prev = {k: [0.0, 0] for k in range(N_BANDS)}        # the same at the previous progress line: the window since it is exact
        self.xc, self.xc_prev = [0.0, 0], [0.0, 0]               # the cross term this craft took (a pair run): its nats and bytes
        cfg = self.cfg
        log(f"arm {arm} seed {SEED}: {self.n_params / 1e6:.2f}M params ({sum(p.numel() for p in self.params) / 1e6:.2f}M trainable); "
            f"{STEPS} steps x {MB} x {CTX} = {STEPS * MB * CTX / 1e6:.0f}M tokens; hub layers {cfg.hub_layers}, attn_rotary {cfg.attn_rotary}, "
            f"hub_rotary {cfg.hub_rotary}, pos_table {cfg.pos_table}, bank table {cfg.bank_byte_table!r}")

    def takes_xc(self) -> bool:
        """Does this craft take the cross term (a pair run with a weight: the student of "one", both crafts of "mutual")?"""
        return bool(PAIR) and XC_W != 0 and self.role in ("s", "m")


crafts = [Craft(ARM, INIT, TAG, ("t" if XC == "one" else "m") if PAIR else "")]
if PAIR:
    crafts.append(Craft(PAIR, PAIR_INIT, PAIR_TAG, "s" if XC == "one" else "m"))
    log(f"pair: {PAIR} trains beside {ARM} on the same rows; the cross term {XC} ("
        + (f"{PAIR} learns from {ARM}" if XC == "one" else "each learns from the other")
        + f") at {'the asked values' if XC_POS == 'value' else 'every'} byte, weight {XC_W:g} a byte, from step {XC_FROM}")
model = crafts[0].model          # the gauges read the module's `model`; finish() points it at each craft in turn


def lr_at(step):
    if step < WARM:
        return LR * (step + 1) / WARM
    t = (step - WARM) / max(1, STEPS - WARM)
    return LR * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * t)))


def xc_kl(t_logits, s_logits, sel):
    """One step's cross term from a teacher craft to a student craft: KL(teacher || student) between their next-byte distributions at
    every predicted byte, the teacher's held fixed, kept at the selected bytes (sel: 1 at the asked values' bytes; None keeps every
    byte). -> (the term: the kept bytes' sum over the batch's byte count, so a kept byte's term weighs like its own target in the mean
    loss; the kept bytes' nats; their count)."""
    lq = F.log_softmax(t_logits.detach().float(), dim=-1)
    lp = F.log_softmax(s_logits.float(), dim=-1)
    kl = (lq.exp() * (lq - lp)).sum(-1)                         # (MB, CTX) nats
    if sel is not None:
        kl = kl * sel
    tot = kl.sum()
    return tot / kl.numel(), float(tot.detach()), (int(sel.sum()) if sel is not None else kl.numel())


t_train = time.time()
for _c in crafts:
    _c.model.train()
from concurrent.futures import ThreadPoolExecutor  # noqa: E402
_pool = ThreadPoolExecutor(max_workers=1) if PREFETCH else None       # one worker, submitted in order: the planted census stays exact
_next = _pool.submit(batch, 0) if PREFETCH else None
for step in range(STEPS):
    for _c in crafts:
        for g in _c.opt.param_groups:
            g["lr"] = lr_at(step)
    if PREFETCH:
        x, y, vm = _next.result()
        if step + 1 < STEPS:
            _next = _pool.submit(batch, step + 1)
    else:
        x, y, vm = batch(step)
    x, y, vm = x.to(DEV, non_blocking=True), y.to(DEV, non_blocking=True), vm.to(DEV, non_blocking=True)
    for _c in crafts:                                            # a pair: both crafts read the very same rows
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _c.logits = _c.model(x)[0]
        _c.nll = F.cross_entropy(_c.logits.float().reshape(-1, 256), y.reshape(-1), reduction="none").view(MB, CTX)
        _c.loss = _c.ce = _c.nll.mean()
    if PAIR and XC_W != 0 and step >= XC_FROM:                    # 0.1.24: the cross term (teacher, student) pairs of the step
        sel = (vm > 0).float() if XC_POS == "value" else None
        for t, s in [(crafts[0], crafts[1])] + ([(crafts[1], crafts[0])] if XC == "mutual" else []):
            term, nats, n = xc_kl(t.logits, s.logits, sel)
            s.loss = s.loss + XC_W * term
            s.xc[0] += nats
            s.xc[1] += n
    for _c in crafts:
        _c.opt.zero_grad(set_to_none=True)
        _c.loss.backward()
        _c.gn = torch.nn.utils.clip_grad_norm_(_c.params, 1.0)
        _c.opt.step()
        with torch.no_grad():
            for k in range(N_BANDS):
                sel = vm == (k + 1)
                if sel.any():
                    _c.online[k][0] += float(_c.nll[sel].sum())
                    _c.online[k][1] += int(sel.sum())
        _c.li = _c.ce.item()                                     # the craft's own loss (a student's cross term is read apart)
        _c.ema = _c.li if _c.ema is None else 0.98 * _c.ema + 0.02 * _c.li
        _c.logits = _c.nll = _c.loss = _c.ce = None
    if step % LOG_EVERY == 0 or step == STEPS - 1:
        el = time.time() - t_train
        rate = (step + 1) * MB * CTX / max(el, 1e-6)
        left = (STEPS - step - 1) * el / (step + 1)
        lab = [str(BANDS[k][0]) if k < len(BANDS) else "w" for k in range(N_BANDS)]
        said = []
        for _c in crafts:
            # 0.1.22: the window since the previous line, exact from the sums (the 0.1.21 line printed the mean since step 1: its reset sat
            # inside this branch on a step it can never reach, so every record's value_bits is a mean since step 1, and stays so for comparison)
            on, pv = _c.online, _c.prev
            win = {k: ((on[k][0] - pv[k][0]) / (on[k][1] - pv[k][1]) / LN2 if on[k][1] > pv[k][1] else None) for k in range(N_BANDS)}
            bands = " ".join(f"{BAND_LABELS[k]}:{win[k]:.2f}" if win[k] is not None else f"{BAND_LABELS[k]}:-" for k in range(N_BANDS))
            line = dict(step=step + 1, loss=_c.li, ema=_c.ema, lr=lr_at(step),
                        value_bits={lab[k]: (on[k][0] / on[k][1] / LN2 if on[k][1] else None) for k in range(N_BANDS)},   # since step 1
                        value_n={lab[k]: on[k][1] for k in range(N_BANDS)},                 # 0.1.22: the counts, so any window between lines is exact
                        value_bits_window={lab[k]: win[k] for k in range(N_BANDS)})         # 0.1.22: since the previous line
            xw = None
            if _c.takes_xc():                                    # 0.1.24: the cross term's KL a kept byte since the previous line (bits)
                xw = (_c.xc[0] - _c.xc_prev[0]) / (_c.xc[1] - _c.xc_prev[1]) / LN2 if _c.xc[1] > _c.xc_prev[1] else None
                line.update(xc_bits_window=xw, xc_n=_c.xc[1])
            _c.curve.append(line)
            _c.prev = {k: list(on[k]) for k in range(N_BANDS)}
            _c.xc_prev = list(_c.xc)
            said.append((_c, bands, xw))
        a, bands, _ = said[0]
        msg = (f"step {step + 1}/{STEPS} loss {a.li:.3f} ema {a.ema:.3f} ({a.ema / LN2:.3f} bpb) gn {float(a.gn):.2f} lr {lr_at(step):.2e} | "
               f"{rate / 1e3:.0f}k tok/s | spent {el / 60:.1f} min, left {left / 60:.1f} min | value bits since the last line {bands}")
        for _c, b, _ in said[1:]:
            msg += f" || {_c.arm} loss {_c.li:.3f} ema {_c.ema:.3f} ({_c.ema / LN2:.3f} bpb) gn {float(_c.gn):.2f} | value bits since the last line {b}"
        xs = [f"{_c.arm} {xw:.3f}" for _c, _, xw in said if xw is not None]
        if xs:
            msg += " | the cross term since the last line, bits a kept byte: " + ", ".join(xs)
        log(msg)
train_s = time.time() - t_train
if _pool is not None:
    _pool.shutdown()
log(f"training done in {train_s / 60:.1f} min")

# ------------------------------------------------------------------ the gauges (each reads the module's `model`: finish() sets it)


@torch.no_grad()
def logprobs(x):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        return F.log_softmax(model(x)[0].float(), dim=-1)


@torch.no_grad()
def heldout_bits(disable_bank=False):
    """Held-out bits a byte: the mean, by byte class, and BY POSITION in the window (the row frame of eval/position_bits.py; the
    eval rows are single documents behind a DOC byte, so the row frame is the document frame)."""
    from geolip.alephllm.eval import position_bits as pb
    tot, n = 0.0, 0
    cls = {c: [0.0, 0] for c in BYTE_CLASSES}
    pos_acc = None
    for i in range(0, len(eval_docs), 16):
        rows = [bytes([DOC]) + d[:CTX] for d in eval_docs[i:i + 16]]
        x = torch.tensor(np.stack([np.frombuffer(r, dtype=np.uint8) for r in rows]).astype(np.int64), device=DEV)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lg = model(x[:, :-1], disable_bank=disable_bank)[0].float()
        nll = F.cross_entropy(lg.reshape(-1, 256), x[:, 1:].reshape(-1), reduction="none").view(x.shape[0], CTX)
        pos_acc = pb.merge(pos_acc, pb.bits_by_position(nll))
        tgt = x[:, 1:].cpu().numpy()
        nl = nll.cpu().numpy()
        tot += float(nl.sum())
        n += nl.size
        cc = CLS_TABLE[tgt]
        for ci, c in enumerate(BYTE_CLASSES):
            sel = cc == ci
            if sel.any():
                cls[c][0] += float(nl[sel].sum())
                cls[c][1] += int(sel.sum())
    heldout_bits.by_position = pb.finish(pos_acc)
    return tot / n / LN2, {c: (v[0] / v[1] / LN2 if v[1] else None) for c, v in cls.items()}


@torch.no_grad()
def fetch(d):
    """The single-record frame at distance d on the fetch documents: teacher-forced value bits, greedy first byte, all five."""
    rng = np.random.default_rng([SEED, 777, d])
    rows, metas = [], []
    for doc in fetch_docs:
        w = bytes([DOC]) + doc[:CTX + 400]
        r = plant(w, rng, dist=d, form="single")
        if r is None:
            continue
        rows.append(r[0])
        metas.append(r[1])
    if not rows:
        return dict(n=0)
    x = torch.tensor(np.stack([np.frombuffer(r[:CTX + 1], dtype=np.uint8) for r in rows]).astype(np.int64), device=DEV)
    lp = logprobs(x[:, :-1])
    tgt = x[:, 1:]
    nll_bits, first, allv = [], 0, 0
    for i, m in enumerate(metas):
        v0, L = m["v0"], m["vlen"]
        pos = torch.arange(v0 - 1, v0 - 1 + L, device=DEV)
        nll_bits.append(float(-lp[i, pos, tgt[i, pos]].sum()) / LN2)
    # greedy: every row's prompt ends at its own value start; right-padded with spaces, each row read at its own last position
    gen = [b"" for _ in rows]
    for s in range(max(m["vlen"] for m in metas)):
        prompts = [rows[i][:metas[i]["v0"]] + gen[i] for i in range(len(rows))]
        L = max(len(p) for p in prompts)
        xb = np.full((len(rows), L), 32, dtype=np.uint8)
        for i, p in enumerate(prompts):
            xb[i, :len(p)] = np.frombuffer(p, dtype=np.uint8)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lg = model(torch.tensor(xb.astype(np.int64), device=DEV))[0].float()
        last = torch.tensor([len(p) - 1 for p in prompts], device=DEV)
        nxt = lg[torch.arange(len(rows), device=DEV), last].argmax(-1).cpu().numpy()
        for i in range(len(rows)):
            if len(gen[i]) < metas[i]["vlen"]:
                gen[i] += bytes([int(nxt[i])])
    for i, m in enumerate(metas):
        truth = rows[i][m["v0"]:m["v0"] + m["vlen"]]
        first += gen[i][:1] == truth[:1]
        allv += gen[i][:5] == truth[:5]
    n = len(rows)
    return dict(n=n, value_bits=float(np.mean(nll_bits)), first_byte=first / n, all_five=allv / n)


@torch.no_grad()
def dispatch_census():
    """|w| per expert by the INPUT byte's class at block 0 / middle / last on 16 held-out windows."""
    rows = [bytes([DOC]) + d[:CTX] for d in eval_docs[:16]]
    x = torch.tensor(np.stack([np.frombuffer(r, dtype=np.uint8) for r in rows]).astype(np.int64), device=DEV)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        model(x[:, :-1])
    cc = CLS_TABLE[x[:, :-1].cpu().numpy()]
    out = {}
    for name, bi in (("b0", 0), ("mid", len(model.blocks) // 2), ("last", len(model.blocks) - 1)):
        w = model.blocks[bi].bank.last_dispatch.float().cpu().numpy()        # (B, n, K)
        K = w.shape[-1]
        tab = {}
        for ci, c in enumerate(BYTE_CLASSES):
            sel = cc == ci
            if not sel.any():
                continue
            a = np.abs(w[sel]).mean(0)                                        # (K,)
            pn = a / max(a.sum(), 1e-12)
            tab[c] = dict(abs_mean=[round(float(v), 4) for v in a], dominant=int(a.argmax()),
                          home_share=round(float(pn[ci % K]), 4), entropy_bits=round(float(-(pn * np.log2(np.clip(pn, 1e-12, 1))).sum()), 3))
        out[name] = tab
    return out


planted_mean = PLANTED[0] / max(PLANTED[1], 1)
log(f"records planted a row (census of the training rows): {planted_mean:.2f} mean over {PLANTED[1]:,} rows (asked {N_REC}; the cue list holds {len(ROW_CUES)})")


@torch.no_grad()
def frame_gauges(disable_bank=False, ask_as=None):
    """Held-out framed rows (the eval and fetch documents as the filler, the plan drawn as in training, the ask strict; `ask_as` asks
    that question instead of the run's own): bits by frame (kind x size x segment, by depth, outside the frames) and the fetch grouped by
    the truth's source / frame size / depth (teacher-forced value bits, the greedy first byte, the whole value, the mean distance)."""
    rng = np.random.default_rng([SEED, 4242])
    rows, metas = [], []
    for doc in eval_docs + fetch_docs:
        r, m = frame_row(rng, bytes([DOC]) + doc, strict=True, ask_as=ask_as)
        rows.append(r)
        metas.append(m[0])
    acc, nll_rows = None, []
    for i in range(0, len(rows), 16):
        x = torch.tensor(np.stack([np.frombuffer(r, dtype=np.uint8) for r in rows[i:i + 16]]).astype(np.int64), device=DEV)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lg = model(x[:, :-1], disable_bank=disable_bank)[0].float()
        nll = F.cross_entropy(lg.reshape(-1, 256), x[:, 1:].reshape(-1), reduction="none").view(x.shape[0], CTX).cpu().numpy()
        acc = bf.bits_by_frame(nll, metas[i:i + 16], acc)
        nll_rows.append(nll)
    nll_all = np.concatenate(nll_rows)
    fb = bf.finish_frame_bits(acc)
    gen = [b"" for _ in rows]
    for _ in range(max(m["vlen"] for m in metas)):
        prompts = [rows[i][:metas[i]["v0"]] + gen[i] for i in range(len(rows))]
        L = max(len(p) for p in prompts)
        xb = np.full((len(rows), L), 32, dtype=np.uint8)
        for i, p in enumerate(prompts):
            xb[i, :len(p)] = np.frombuffer(p, dtype=np.uint8)
        for i0 in range(0, len(rows), 16):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lg = model(torch.tensor(xb[i0:i0 + 16].astype(np.int64), device=DEV), disable_bank=disable_bank)[0].float()
            last = torch.tensor([len(p) - 1 for p in prompts[i0:i0 + 16]], device=DEV)
            nxt = lg[torch.arange(lg.shape[0], device=DEV), last].argmax(-1).cpu().numpy()
            for j, i in enumerate(range(i0, min(i0 + 16, len(rows)))):
                if len(gen[i]) < metas[i]["vlen"]:
                    gen[i] += bytes([int(nxt[j])])
    groups = {}
    for i, m in enumerate(metas):
        v0, L = m["v0"], m["vlen"]
        truth = rows[i][v0:v0 + L]
        if m["source"] == "weights":
            g = "weights"
        elif "frames" in m:
            g = bf.frame_group(m["frames"][m["truth_frame"]])      # kind / generic size / depth of the frame the record belongs to
        else:
            g = m["source"]
        a = groups.setdefault(g, dict(n=0, value_bits=0.0, first_byte=0, full_value=0, dist=0.0))
        a["n"] += 1
        a["value_bits"] += float(nll_all[i, v0 - 1:v0 - 1 + L].sum()) / LN2
        a["first_byte"] += int(gen[i][:1] == truth[:1])
        a["full_value"] += int(gen[i][:L] == truth)
        a["dist"] += float(m["dist"] or 0)
    for a in groups.values():
        for k in ("value_bits", "first_byte", "full_value", "dist"):
            a[k] = a[k] / a["n"]
    return fb, groups


@torch.no_grad()
def canary_gauge():
    """The canary rows (bed_frames.canary_rows) under the model: bits a byte, whole and by position, for the plain, tag and frame rows."""
    from geolip.alephllm.eval import position_bits as pb
    rows, meta = bf.canary_rows(eval_docs, CTX, REGISTER, TAGS, SEED, sized=SIZED)     # the frame row in the craft's own grammar
    out = dict(meta)
    for kind, rs in rows.items():
        acc, tot, n = None, 0.0, 0
        for i in range(0, len(rs), 16):
            x = torch.tensor(np.stack([np.frombuffer(r, dtype=np.uint8) for r in rs[i:i + 16]]).astype(np.int64), device=DEV)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lg = model(x[:, :-1])[0].float()
            nll = F.cross_entropy(lg.reshape(-1, 256), x[:, 1:].reshape(-1), reduction="none").view(x.shape[0], CTX)
            acc = pb.merge(acc, pb.bits_by_position(nll))
            tot += float(nll.sum())
            n += nll.numel()
        out[kind] = dict(bits=tot / n / LN2, by_position=pb.finish(acc)["row"])
    return out


def finish(c: Craft) -> tuple:
    """One craft's gauges, then its weights. -> (its record, the record's path); the caller writes the record (a pair's records carry
    the pair's reads, taken after both crafts' gauges)."""
    global model
    model = c.model
    model.eval()
    if PAIR:
        log(f"the gauges of {c.arm}")
    bits, by_cls = heldout_bits()
    by_pos = heldout_bits.by_position
    collapse = bf.plain_collapse(by_pos["row"]) if by_pos and "row" in by_pos else None
    if collapse and collapse["flag"]:
        log(f"FLAG plain collapse: the windows {collapse['windows_over']} read above {collapse['threshold']} bits a byte (worst {collapse['worst_window']}"
            f" {collapse['worst_bits']:.2f}); the record carries it, the run ends normally")
    log("held-out bits by position (row): " + " ".join(f"{b} {v:.3f} (n {by_pos['row_n'][b]})" for b, v in by_pos["row"].items() if v is not None))
    bits_off, by_cls_off = heldout_bits(disable_bank=True)
    log(f"held-out {bits:.4f} bits a byte (dispatch off {bits_off:.4f}); by class " + " ".join(f"{k} {v:.3f}" for k, v in by_cls.items() if v is not None))
    fetch_res = {}
    for d in DISTS:
        fetch_res[str(d)] = fetch(d)
        r = fetch_res[str(d)]
        log(f"fetch at {d}: n {r.get('n')} value bits {r.get('value_bits', float('nan')):.2f} first {r.get('first_byte', float('nan')):.2f} all five {r.get('all_five', float('nan')):.2f}")
    census = dispatch_census()
    log("dispatch census (last block): " + " ".join(f"{k}:e{v['dominant']}/{v['home_share']:.2f}" for k, v in census["last"].items()))
    frame_bits, frame_fetch = frame_gauges() if FRAMES else (None, None)
    frame_fetch_bank_off = frame_gauges(disable_bank=True)[1] if FRAMES else None     # the same rows with the bank off: does the address carry the lookup?
    if FRAMES:
        log("held-out bits by frame (kind / size / segment; by depth; outside):\n" + bf.fmt_frame_bits(frame_bits))
        log("fetch by frame: " + " | ".join(f"{g}: n {a['n']} bits {a['value_bits']:.2f} first {a['first_byte']:.2f} full {a['full_value']:.2f} dist {a['dist']:.0f}"
                                            for g, a in sorted(frame_fetch.items())))
        log("fetch by frame, bank off: " + " | ".join(f"{g}: full {a['full_value']:.2f} bits {a['value_bits']:.2f}" for g, a in sorted(frame_fetch_bank_off.items())))
    frame_fetch_asks = {}
    for _a in GAUGE_ASKS:                      # 0.1.23: the strict questions the diet does not ask alone, read by the gauge
        try:
            frame_fetch_asks[_a] = frame_gauges(ask_as=_a)[1]
            log(f"fetch by frame, asked {_a} (the gauge's strict question): " + " | ".join(
                f"{g}: n {a['n']} bits {a['value_bits']:.2f} first {a['first_byte']:.2f} full {a['full_value']:.2f} dist {a['dist']:.0f}"
                for g, a in sorted(frame_fetch_asks[_a].items())))
        except RuntimeError as e:              # diagnostics collect, never raise (a plan without the feature in 64 draws)
            frame_fetch_asks[_a] = None
            log(f"fetch by frame, asked {_a}: not read ({e})")
    canary, formation = (canary_gauge(), bf.formation(c.curve, REGISTER, frame_fetch)) if FRAMES else (None, None)
    if FRAMES:
        nb = formation["near_band_value_bits_at_end"]
        log("canary rows (bits a byte): " + " | ".join(f"{k} {canary[k]['bits']:.3f}" for k in ("plain", "tag", "frame"))
            + f" | formation: {'FAILURE' if formation['failure'] else 'formed'} (near band {'-' if nb is None else f'{nb:.2f}'} against the uniform"
            + f" {formation['uniform_bits']:.2f}; first step under {formation['first_step_under']}; whole value zero everywhere {formation['whole_value_zero_everywhere']})")
        if formation["failure"]:
            log(f"FLAG formation failure: the near band sits at {'-' if nb is None else f'{nb:.2f}'} bits against the uniform {formation['uniform_bits']:.2f}"
                f" ({REGISTER}) and the whole value reads 0 in every lookup group; the record carries it, the run ends normally")
    # the weights + the record
    wdir = os.path.join(BED, c.tag)
    os.makedirs(wdir, exist_ok=True)
    from safetensors.torch import save_file
    save_file({k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}, os.path.join(wdir, "model.safetensors"))
    json.dump(c.cfg.to_dict(), open(os.path.join(wdir, "config.json"), "w", encoding="utf-8"), indent=1)
    R = dict(arm=c.arm, seed=SEED, smoke=SMOKE, steps=STEPS, micro_batch=MB, ctx=CTX, lr=LR, warmup=WARM, p_plant=P_PLANT, records=N_REC, params=c.n_params, init=c.init,
             config=c.cfg.to_dict(), tokens=STEPS * MB * CTX, train_minutes=train_s / 60, tok_per_s=STEPS * MB * CTX / train_s,
             final_loss_ema=c.ema, final_bpb_ema=c.ema / LN2, heldout_bpb=bits, heldout_bpb_by_class=by_cls, heldout_bpb_by_position=by_pos, dispatch_off_bpb=bits_off,
             records_planted_mean=planted_mean, cues=len(ROW_CUES),
             dispatch_off_by_class=by_cls_off, fetch=fetch_res, dispatch_census=census, curve=c.curve, wall_s=time.time() - T0,
             frames=FRAME_CFG, heldout_bits_by_frame=frame_bits, fetch_by_frame=frame_fetch, fetch_by_frame_bank_off=frame_fetch_bank_off, canary=canary, formation=formation, plain_collapse=collapse,
             weights=os.path.relpath(os.path.join(wdir, "model.safetensors"), OUT))
    if ARGS.get("stage"):
        R["stage"] = ARGS["stage"]           # 0.1.22: a curriculum block's stage name (its tag carries it in place of the starting craft's)
    if GAUGE_ASKS:
        R["fetch_by_frame_asks"] = frame_fetch_asks   # 0.1.23: {ask: the fetch groups asked strictly, or None when no plan carried the feature}
    return R, os.path.join(BED, f"bed_{c.tag}.json")


ROLE = {"t": "teacher", "s": "student", "m": "mutual"}
done = [finish(c) for c in (crafts[::-1] if PAIR else crafts)]      # a pair: the partner first (the run's own record is written last)
if PAIR:
    # how alike the pair ends on held-out text (bed_pair.twin_reads): the first 32 evaluation documents, the rows the held-out bits read;
    # a pair trained apart is read the same way from its stored weights (bed_pair.load_craft, bed_pair.heldout_rows)
    twins = bed_pair.twin_reads(crafts[0].model, crafts[1].model, bed_pair.rows_tensor(eval_docs[:32], CTX, DEV),
                                names=(crafts[0].arm, crafts[1].arm))
    log(f"the pair's reads on held-out text: KL {twins['crafts'][0]} -> {twins['crafts'][1]} {twins['kl_bits'][0]:.3f} and back {twins['kl_bits'][1]:.3f}"
        f" bits a byte; greedy next bytes agree {twins['greedy_agree']:.3f}; debiased CKA by block "
        + " ".join(f"{k} {v:.3f}" for k, v in twins["cka"].items()))
    for (R, _), c in zip(done, crafts[::-1]):
        o = crafts[1] if c is crafts[0] else crafts[0]
        R["pair"] = dict(partner=o.arm, partner_tag=o.tag, partner_init=o.init, role=ROLE[c.role], xc=XC, xc_pos=XC_POS, xc_w=XC_W,
                         xc_from=XC_FROM, xc_bits=(c.xc[0] / c.xc[1] / LN2 if c.xc[1] else None), xc_bytes=c.xc[1],
                         minutes="train_minutes and tok_per_s are the pair's: both crafts trained in them, on the same rows", twins=twins)
for R, path in done:
    json.dump(R, open(path, "w", encoding="utf-8"), indent=1)
    log(f"wrote {path} ({(time.time() - T0) / 60:.1f} min)")
