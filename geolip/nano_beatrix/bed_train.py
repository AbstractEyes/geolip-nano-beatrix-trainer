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
    from .names import run_tag  # noqa: E402  (the package)
except ImportError:                                      # run as a file: the folder is on the path (above)
    from names import run_tag  # noqa: E402

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
LOG_EVERY = int(ARGS.get("log_every", "5" if SMOKE else "50"))
PREFETCH = ARGS.get("prefetch", "1") == "1"   # the next step's rows compose in a worker thread while this step runs on the card;
                                              # the batches are identical either way (every step seeds its own rng from [SEED, step])
DEV = os.environ.get("BATTERY_DEV", "cuda")
BED = os.path.join(OUT, "bed")
os.makedirs(BED, exist_ok=True)
TAG = run_tag(ARGS)        # the record and weights names: one function with the runner's resume (names.run_tag)
INIT = ARGS.get("init", "")        # a previous run's weights folder under DIR/bed: its config.json sets the craft, the context may change
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
REC_BYTES = bf.capacity(CTX, "A", REGISTER)["record_bytes"] if FRAMES else 0


def draw_plan(rng):
    """One row's dynamic plan: frames of the generic sizes while they fit the window (a gap and the question reserved), lookup frames
    with as many records as fit up to N_REC_FRAME, a thinking frame nesting a smaller lookup frame with probability NEST_P, an
    overlapping pair of lookup frames in conflict with probability OVERLAP_P. -> (specs, has_nested, has_overlap)."""
    budget = CTX + 1 - 6 - 64 - (REC_BYTES + 2)
    specs, used, nested, overlap = [], 0, False, False
    if OVERLAP_P and rng.random() < OVERLAP_P:
        sz = SIZES[0]
        specs += bf.overlap_specs(rng, REGISTER, sz, sz, max(1, min(2, (sz - 2) // REC_BYTES)), "conflict")
        used += 2 * (sz + 64)
        overlap = True
    for _ in range(int(rng.integers(1, 4))):
        size = int(SIZES[int(rng.integers(len(SIZES)))])
        if used + size + 64 > budget:
            continue
        if rng.random() < 0.7:
            n = min(N_REC_FRAME, (size - 2) // REC_BYTES)
            specs.append(bf.spec("data", size, records=n) if n >= 1 else bf.spec("think", size))
        else:
            kids = []
            csize = max(32, size // 4)
            if rng.random() < NEST_P and (csize - 2) // REC_BYTES >= 1 and csize <= size - 4:
                kids = [bf.spec("data", csize, records=min(N_REC_FRAME, (csize - 2) // REC_BYTES))]
                nested = True
            specs.append(bf.spec("think", size, children=kids))
        used += size + 64
    if not any(sp["kind"] == "data" or sp["children"] for sp in specs):     # nothing to ask: the smallest lookup frame
        sz = SIZES[0]
        specs.append(bf.spec("data", sz, records=max(1, min(N_REC_FRAME, (sz - 2) // REC_BYTES))))
    return specs, nested, overlap


def frame_row(rng, src, strict=False):
    """A framed row. Training (strict=False): the ask falls back to 'any' when the drawn plan lacks the asked feature, and the weights
    ask is asked half the time (the other half reads the frame). The held-out gauge (strict=True): the plan is redrawn until it has the
    feature and the ask is exactly ASK. -> (row bytes of CTX + 1, [meta])."""
    if FRAMES == "A":
        ask = ASK
        if ask == "weights" and not strict and rng.random() < 0.5:
            ask = "lookup" if N_LOOKUP else "bulk"
        row, m = bf.compose(rng, src, CTX, "A", TAGS, REGISTER, N_LOOKUP, N_BULK, FACTS, ask)
        return row, [m]
    for _ in range(64):
        specs, nested, overlap = draw_plan(rng)
        ask = ASK
        if (ask == "nested" and not nested) or (ask == "overlap" and not overlap) or (ask == "weights" and FACTS is None):
            ask = "any"
        elif ask == "weights" and not strict and rng.random() < 0.5:
            ask = "any"
        if strict and ask != ASK:
            continue
        row, m = bf.compose_dynamic(rng, src, CTX, specs, REGISTER, TAGS, ask, FACTS)
        return row, [m]
    raise RuntimeError(f"no plan with the feature for ask={ASK!r} in 64 draws")


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


# ------------------------------------------------------------------ the craft
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cuda.matmul.allow_tf32 = True
if DEV == "cuda":
    torch.cuda.set_per_process_memory_fraction(float(os.environ.get("BATTERY_GPU_FRAC", "0.35")))   # the card_gate cap
if INIT:
    _c = json.load(open(os.path.join(BED, INIT, "config.json"), encoding="utf-8"))
    _c = {k: (tuple(v) if isinstance(v, list) else v) for k, v in _c.items()}
    _c["context"] = CTX
    cfg = AlephLMConfig(**_c)
    assert cfg.name == f"bed-{ARM}", (cfg.name, ARM)
else:
    cfg = craft_cfg(ARM)
model = AlephLM(cfg).to(DEV)
if INIT:
    from safetensors.torch import load_file as _load_file
    sd = _load_file(os.path.join(BED, INIT, "model.safetensors"))
    pos = sd.get("embed.pos")
    if pos is not None and pos.shape[1] != CTX:          # the trained rows kept, the new positions zero (the rotary carries the phase)
        new_pos = torch.zeros(1, CTX, pos.shape[2], dtype=pos.dtype)
        n_keep = min(CTX, pos.shape[1])
        new_pos[:, :n_keep] = pos[:, :n_keep]
        sd["embed.pos"] = new_pos
    model.load_state_dict({k: v.to(DEV) for k, v in sd.items()}, strict=True)
    log(f"continued from {INIT}: {len(sd)} tensors; the position table {tuple(pos.shape) if pos is not None else None} -> {tuple(model.embed.pos.shape)}")
n_params = model.param_count()
params = [p for p in model.parameters() if p.requires_grad]
opt = torch.optim.Adam(params, lr=LR, betas=(0.9, 0.95), eps=1e-8)      # pure Adam, never AdamW
log(f"arm {ARM} seed {SEED}: {n_params / 1e6:.2f}M params ({sum(p.numel() for p in params) / 1e6:.2f}M trainable); "
    f"{STEPS} steps x {MB} x {CTX} = {STEPS * MB * CTX / 1e6:.0f}M tokens; hub layers {cfg.hub_layers}, attn_rotary {cfg.attn_rotary}, "
    f"hub_rotary {cfg.hub_rotary}, pos_table {cfg.pos_table}, bank table {cfg.bank_byte_table!r}")


def lr_at(step):
    if step < WARM:
        return LR * (step + 1) / WARM
    t = (step - WARM) / max(1, STEPS - WARM)
    return LR * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * t)))


curve, online = [], {k: [0.0, 0] for k in range(N_BANDS)}
ema = None
t_train = time.time()
model.train()
from concurrent.futures import ThreadPoolExecutor  # noqa: E402
_pool = ThreadPoolExecutor(max_workers=1) if PREFETCH else None       # one worker, submitted in order: the planted census stays exact
_next = _pool.submit(batch, 0) if PREFETCH else None
for step in range(STEPS):
    for g in opt.param_groups:
        g["lr"] = lr_at(step)
    if PREFETCH:
        x, y, vm = _next.result()
        if step + 1 < STEPS:
            _next = _pool.submit(batch, step + 1)
    else:
        x, y, vm = batch(step)
    x, y, vm = x.to(DEV, non_blocking=True), y.to(DEV, non_blocking=True), vm.to(DEV, non_blocking=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(x)[0]
    nll = F.cross_entropy(logits.float().reshape(-1, 256), y.reshape(-1), reduction="none").view(MB, CTX)
    loss = nll.mean()
    opt.zero_grad(set_to_none=True)
    loss.backward()
    gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
    opt.step()
    with torch.no_grad():
        for k in range(N_BANDS):
            sel = vm == (k + 1)
            if sel.any():
                online[k][0] += float(nll[sel].sum())
                online[k][1] += int(sel.sum())
    li = loss.item()
    ema = li if ema is None else 0.98 * ema + 0.02 * li
    if step % LOG_EVERY == 0 or step == STEPS - 1:
        el = time.time() - t_train
        rate = (step + 1) * MB * CTX / max(el, 1e-6)
        left = (STEPS - step - 1) * el / (step + 1)
        bands = " ".join(f"{BAND_LABELS[k]}:{online[k][0] / online[k][1] / LN2:.2f}" if online[k][1] else f"{BAND_LABELS[k]}:-" for k in range(N_BANDS))
        log(f"step {step + 1}/{STEPS} loss {li:.3f} ema {ema:.3f} ({ema / LN2:.3f} bpb) gn {float(gn):.2f} lr {lr_at(step):.2e} | "
            f"{rate / 1e3:.0f}k tok/s | spent {el / 60:.1f} min, left {left / 60:.1f} min | value bits {bands}")
        curve.append(dict(step=step + 1, loss=li, ema=ema, lr=lr_at(step), value_bits={(str(BANDS[k][0]) if k < len(BANDS) else "w"): (online[k][0] / online[k][1] / LN2 if online[k][1] else None) for k in range(N_BANDS)}))
        if step % (LOG_EVERY * 4) == LOG_EVERY * 4 - 1:
            online = {k: [0.0, 0] for k in range(N_BANDS)}
train_s = time.time() - t_train
if _pool is not None:
    _pool.shutdown()
log(f"training done in {train_s / 60:.1f} min")

# ------------------------------------------------------------------ the gauges
model.eval()


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
bits, by_cls = heldout_bits()
by_pos = heldout_bits.by_position
collapse = bf.plain_collapse(by_pos["row"]) if by_pos and "row" in by_pos else None
if collapse and collapse["flag"]:
    log(f"FLAG plain collapse: the windows {collapse['windows_over']} read above {collapse['threshold']} bits a byte (worst {collapse['worst_window']}"
        f" {collapse['worst_bits']:.2f}); the record carries it, the run ends normally")
log("held-out bits by position (row): " + " ".join(f"{b} {v:.3f} (n {by_pos['row_n'][b]})" for b, v in by_pos["row"].items() if v is not None))
bits_off, by_cls_off = heldout_bits(disable_bank=True)
log(f"held-out {bits:.4f} bits a byte (dispatch off {bits_off:.4f}); by class " + " ".join(f"{c} {v:.3f}" for c, v in by_cls.items() if v is not None))
fetch_res = {}
for d in DISTS:
    fetch_res[str(d)] = fetch(d)
    r = fetch_res[str(d)]
    log(f"fetch at {d}: n {r.get('n')} value bits {r.get('value_bits', float('nan')):.2f} first {r.get('first_byte', float('nan')):.2f} all five {r.get('all_five', float('nan')):.2f}")
census = dispatch_census()
log("dispatch census (last block): " + " ".join(f"{c}:e{v['dominant']}/{v['home_share']:.2f}" for c, v in census["last"].items()))


@torch.no_grad()
def frame_gauges():
    """Held-out framed rows (the eval and fetch documents as the filler, the plan drawn as in training, the ask strict): bits by frame
    (kind x size x segment, by depth, outside the frames) and the fetch grouped by the truth's source / frame size / depth (teacher-forced
    value bits, the greedy first byte, the whole value, the mean distance)."""
    rng = np.random.default_rng([SEED, 4242])
    rows, metas = [], []
    for doc in eval_docs + fetch_docs:
        r, m = frame_row(rng, bytes([DOC]) + doc, strict=True)
        rows.append(r)
        metas.append(m[0])
    acc, nll_rows = None, []
    for i in range(0, len(rows), 16):
        x = torch.tensor(np.stack([np.frombuffer(r, dtype=np.uint8) for r in rows[i:i + 16]]).astype(np.int64), device=DEV)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lg = model(x[:, :-1])[0].float()
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
                lg = model(torch.tensor(xb[i0:i0 + 16].astype(np.int64), device=DEV))[0].float()
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
            f = m["frames"][m["truth_frame"]]
            g = f"{f['kind']}/{bf.size_bucket(f['size'])}/depth{f['depth']}"
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


frame_bits, frame_fetch = frame_gauges() if FRAMES else (None, None)
if FRAMES:
    log("held-out bits by frame (kind / size / segment; by depth; outside):\n" + bf.fmt_frame_bits(frame_bits))
    log("fetch by frame: " + " | ".join(f"{g}: n {a['n']} bits {a['value_bits']:.2f} first {a['first_byte']:.2f} full {a['full_value']:.2f} dist {a['dist']:.0f}"
                                        for g, a in sorted(frame_fetch.items())))


@torch.no_grad()
def canary_gauge():
    """The canary rows (bed_frames.canary_rows) under the model: bits a byte, whole and by position, for the plain, tag and frame rows."""
    from geolip.alephllm.eval import position_bits as pb
    rows, meta = bf.canary_rows(eval_docs, CTX, REGISTER, TAGS, SEED)
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


canary, formation = (canary_gauge(), bf.formation(curve, REGISTER, frame_fetch)) if FRAMES else (None, None)
if FRAMES:
    nb = formation["near_band_value_bits_at_end"]
    log("canary rows (bits a byte): " + " | ".join(f"{k} {canary[k]['bits']:.3f}" for k in ("plain", "tag", "frame"))
        + f" | formation: {'FAILURE' if formation['failure'] else 'formed'} (near band {'-' if nb is None else f'{nb:.2f}'} against the uniform"
        + f" {formation['uniform_bits']:.2f}; first step under {formation['first_step_under']}; whole value zero everywhere {formation['whole_value_zero_everywhere']})")
    if formation["failure"]:
        log(f"FLAG formation failure: the near band sits at {'-' if nb is None else f'{nb:.2f}'} bits against the uniform {formation['uniform_bits']:.2f}"
            f" ({REGISTER}) and the whole value reads 0 in every lookup group; the record carries it, the run ends normally")

# ------------------------------------------------------------------ the weights + the record
wdir = os.path.join(BED, TAG)
os.makedirs(wdir, exist_ok=True)
from safetensors.torch import save_file  # noqa: E402
save_file({k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}, os.path.join(wdir, "model.safetensors"))
json.dump(cfg.to_dict(), open(os.path.join(wdir, "config.json"), "w", encoding="utf-8"), indent=1)
R = dict(arm=ARM, seed=SEED, smoke=SMOKE, steps=STEPS, micro_batch=MB, ctx=CTX, lr=LR, warmup=WARM, p_plant=P_PLANT, records=N_REC, params=n_params, init=INIT,
         config=cfg.to_dict(), tokens=STEPS * MB * CTX, train_minutes=train_s / 60, tok_per_s=STEPS * MB * CTX / train_s,
         final_loss_ema=ema, final_bpb_ema=ema / LN2, heldout_bpb=bits, heldout_bpb_by_class=by_cls, heldout_bpb_by_position=by_pos, dispatch_off_bpb=bits_off,
         records_planted_mean=planted_mean, cues=len(ROW_CUES),
         dispatch_off_by_class=by_cls_off, fetch=fetch_res, dispatch_census=census, curve=curve, wall_s=time.time() - T0,
         frames=FRAME_CFG, heldout_bits_by_frame=frame_bits, fetch_by_frame=frame_fetch, canary=canary, formation=formation, plain_collapse=collapse,
         weights=os.path.relpath(os.path.join(wdir, "model.safetensors"), OUT))
path = os.path.join(BED, f"bed_{TAG}.json")
json.dump(R, open(path, "w", encoding="utf-8"), indent=1)
log(f"wrote {path} ({(time.time() - T0) / 60:.1f} min)")
