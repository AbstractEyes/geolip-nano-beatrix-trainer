# -*- coding: utf-8 -*-
"""bed_frames.py - THE FRAMED ROW for the nano bed: rows built from frames, to measure what a small craft holds and how it reads
compartments of its window (a lookup frame, a thinking space, bulk lookup after; sectors of generic sizes rather than fixed ranges).
A composer the bed imports when --frames=<layout> is given (bed_train.py; the default rows stay bit-identical). The registry's specials
carry the structure (geolip/alephllm/data/special_tokens.py; all thirteen invalid-UTF-8 bytes are allocated there, so NO new byte):
    DOC opens the row | a register tag names the kind of content | DATA ... END is a lookup frame of records CUE key SEP value |
    THINK ... END is the thinking space (filler text in the capacity bed) | the question is CUE key SEP, the answer the value, then END.
Layouts (byte ranges in the window; the last compartment runs to the window end):
    A: head 0-8 | lookup 8-81 | think 81-401 | bulk 401-end   (the first example, kept as a fixed reference)
Register tags (caption / math / code / identity): scheme "mode" = MODE + one ASCII byte (c m k i; no registry change) or
scheme "esc" = ESC + a slot (0x02 CAPTION, 0x03 MATH, 0x04 CODE, 0x05 IDENT; an append-only grant if chosen). Identity is also the
SYS block already, so the identity register is optional.
Three sources of the asked record: "lookup" (the small frame), "bulk" (the big frame), "weights" (a fact table recurring across
training rows; at the question the fact is ABSENT from the window: recall from the weights = what the craft holds).
Self-test (pure numpy, no model, no card):  python bed_frames.py --selftest
"""
from __future__ import annotations

import os
import sys

import numpy as np

if __name__ == "__main__" and "--selftest" in sys.argv:          # the self-test imports the registry straight from the worktree
    _wt = os.environ.get("ALEPHLLM_WT", "")
    if _wt:
        sys.path.insert(0, _wt)
from geolip.alephllm.data.special_tokens import CUE, DATA, DOC, END, ESC, MODE, SEP, THINK, decode_visible  # noqa: E402

LAYOUTS = {"A": (("head", 0, 8), ("lookup", 8, 81), ("think", 81, 401), ("bulk", 401, None))}
# register: (the MODE payload byte, the ESC slot, the value alphabet, the value length)
REGISTERS = {"caption": (b"c", 0x02, "letters", 6), "math": (b"m", 0x03, "digits", 6), "code": (b"k", 0x04, "hex", 8),
             "identity": (b"i", 0x05, "alnum", 6)}
ALPH = {"digits": b"0123456789", "letters": b"abcdefghijklmnopqrstuvwxyz", "hex": b"0123456789abcdef",
        "alnum": b"abcdefghijklmnopqrstuvwxyz0123456789"}
WORDS = ["code", "word", "key", "number", "token", "phrase", "label", "serial", "pin", "badge", "ticket", "slot", "tag", "mark", "sign", "note"]
SPECIALS = {DOC, DATA, END, THINK, CUE, SEP, MODE, ESC}


def tag_bytes(register: str, scheme: str = "mode") -> bytes:
    payload, slot = REGISTERS[register][0], REGISTERS[register][1]
    if scheme == "mode":
        return bytes([MODE]) + payload
    assert scheme == "esc", scheme
    return bytes([ESC, slot])


def value(rng, kind: str, n: int) -> bytes:
    a = ALPH[kind]
    v = bytes(a[int(i)] for i in rng.integers(len(a), size=n))
    if kind == "digits" and v[:1] == b"0":
        v = b"1" + v[1:]
    return v


def key_name(idx: int) -> bytes:
    """Distinct ASCII keys: a cue word + a three-digit number (16 x 1000 keys)."""
    return f"{WORDS[idx % len(WORDS)]}{idx // len(WORDS):03d}".encode()


def record_bytes(key: bytes, val: bytes) -> bytes:
    return bytes([CUE]) + key + bytes([SEP]) + val


def fact_table(m: int, register: str, seed: int = 2026) -> list:
    """M fixed (key, value) facts for the in-weights capacity axis; the same table for every seed of the craft (the content is the
    experiment, the seed is the craft)."""
    rng = np.random.default_rng([seed, m])
    kind, vn = REGISTERS[register][2], REGISTERS[register][3]
    return [(key_name(i), value(rng, kind, vn)) for i in range(m)]


def clean_filler(src: bytes, n: int) -> bytes:
    """n bytes of corpus text with the specials replaced by newlines (a DOC inside a frame would be a boundary inside a frame)."""
    b = bytearray(src[:n])
    for i, c in enumerate(b):
        if c in SPECIALS:
            b[i] = 10
    if len(b) < n:
        b += b" " * (n - len(b))
    return bytes(b)


def capacity(ctx: int, layout: str, register: str) -> dict:
    """How many records fit: the lookup frame and the bulk frame at this window (the question and its answer reserved)."""
    comps = dict((c[0], (c[1], ctx + 1 if c[2] is None else c[2])) for c in LAYOUTS[layout])
    vn = REGISTERS[register][3]
    klen = max(len(w) for w in WORDS) + 3            # the longest key (the words differ in length; the arithmetic takes the longest)
    rec = 1 + klen + 1 + vn
    lk = comps["lookup"]
    bk = comps["bulk"]
    q = 1 + klen + 1 + vn + 1                        # CUE key SEP value END
    return dict(record_bytes=rec, lookup_max=(lk[1] - lk[0] - 2) // rec, bulk_max=(bk[1] - bk[0] - 2 - q - 8) // rec)


def compose(rng, src: bytes, ctx: int, layout: str = "A", scheme: str = "mode", register: str = "caption", n_lookup: int = 2,
            n_bulk: int = 0, facts: list | None = None, ask: str = "lookup") -> tuple:
    """-> (row bytes of length ctx + 1, meta). meta: v0 (the value's first byte index), vlen, dist (v0 minus the asked record's start;
    None for the weights source), source, key, register, n_lookup, n_bulk, comps (name -> (lo, hi)), bands (the gauge edges)."""
    L = LAYOUTS[layout]
    comps = {c[0]: (c[1], ctx + 1 if c[2] is None else c[2]) for c in L}
    kind, vn = REGISTERS[register][2], REGISTERS[register][3]
    cap = capacity(ctx, layout, register)
    assert 0 <= n_lookup <= cap["lookup_max"], (n_lookup, cap)
    assert 0 <= n_bulk <= cap["bulk_max"], (n_bulk, cap)
    assert ask in ("lookup", "bulk", "weights"), ask
    # the records: fresh pairs, or draws from the fact table (the asked fact absent from the window in the weights source)
    n_rec = n_lookup + n_bulk
    if facts is None:
        assert ask != "weights", "the weights source needs a fact table"
        idx = rng.permutation(16000)[:n_rec]
        recs = [(key_name(int(i)), value(rng, kind, vn)) for i in idx]
        asked = None
    else:
        perm = rng.permutation(len(facts))
        if ask == "weights":
            asked = facts[int(perm[0])]
            recs = [facts[int(i)] for i in perm[1:1 + n_rec]]
        else:
            recs = [facts[int(i)] for i in perm[:n_rec]]
            asked = None
    look, bulk = recs[:n_lookup], recs[n_lookup:]
    if asked is None:
        pool = look if ask == "lookup" else bulk
        assert pool, f"nothing to ask from the {ask} frame"
        asked = pool[int(rng.integers(len(pool)))]
    # head: DOC, the register tag, SEP, the task byte q, SEP, padding to the compartment
    h0, h1 = comps["head"]
    head = bytes([DOC]) + tag_bytes(register, scheme) + bytes([SEP]) + b"q" + bytes([SEP])
    assert len(head) <= h1 - h0, (len(head), h1 - h0)
    head = head + b" " * (h1 - h0 - len(head))
    # the lookup frame: DATA records END, padded
    l0, l1 = comps["lookup"]
    body = b"".join(record_bytes(k, v) for k, v in look)
    frame = bytes([DATA]) + body + b" " * (l1 - l0 - 2 - len(body)) + bytes([END])
    assert len(frame) == l1 - l0
    rec_pos = {}
    off = l0 + 1
    for k, v in look:
        rec_pos[k] = off
        off += len(record_bytes(k, v))
    # the thinking space: THINK filler END
    t0, t1 = comps["think"]
    think = bytes([THINK]) + clean_filler(src, t1 - t0 - 2) + bytes([END])
    # the bulk frame, the question, the answer, filler to the end
    b0, b1 = comps["bulk"]
    out = bytearray(head + frame + think)
    assert len(out) == b0
    if bulk:
        out += bytes([DATA])
        for k, v in bulk:
            rec_pos[k] = len(out)
            out += record_bytes(k, v)
        out += bytes([END])
    q0 = len(out)
    out += bytes([CUE]) + asked[0] + bytes([SEP])
    v0 = len(out)
    out += asked[1] + bytes([END])
    tail = ctx + 1 - len(out)
    assert tail >= 0, (len(out), ctx + 1)
    out += clean_filler(src[t1:], tail)
    row = bytes(out)
    assert len(row) == ctx + 1 and row[v0:v0 + len(asked[1])] == asked[1], "the framed value is not where the arithmetic says"
    src_name = "weights" if ask == "weights" else ("lookup" if asked[0] in dict(look) else "bulk")
    dist = None if src_name == "weights" else v0 - rec_pos[asked[0]]
    # the layout's compartments as frames too, so the frame masks and bits by frame read layout A like a dynamic row
    frames = [dict(kind="data", size=l1 - l0, start=l0, end=l1, depth=0, n_records=len(look), keys=[k.decode() for k, _ in look], children=0),
              dict(kind="think", size=t1 - t0, start=t0, end=t1, depth=0, n_records=0, keys=[], children=0)]
    if bulk:
        frames.append(dict(kind="data", size=q0 - b0, start=b0, end=q0, depth=0, n_records=len(bulk), keys=[k.decode() for k, _ in bulk], children=0))
    truth_frame = None if src_name == "weights" else (0 if src_name == "lookup" else 2)
    meta = dict(v0=v0, vlen=len(asked[1]), dist=dist, source=src_name, key=asked[0].decode(), register=register, scheme=scheme,
                n_lookup=n_lookup, n_bulk=n_bulk, q0=q0, comps=comps, bands=bands_for(ctx, layout), frames=frames, truth_frame=truth_frame,
                truth_depth=None if truth_frame is None else 0, n_defs=0 if truth_frame is None else 1)
    return row, meta


def bands_for(ctx: int, layout: str = "A") -> tuple:
    """The gauge edges for bits by position (eval/position_bits.py takes custom bands): the compartment boundaries, then the window."""
    edges = [c[1] for c in LAYOUTS[layout]] + [ctx]
    for e in (1024, 2048, 4096):
        if edges[-2] < e < ctx:
            edges.insert(-1, e)
    return tuple(edges)


# ------------------------------------------------------------------ the dynamic mode: sectors of generic sizes, not locked spaces
FRAME_SIZES = (32, 128, 256, 512, 1024)
KIND_TAG = {"data": DATA, "think": THINK}
KIND_ID = {"none": 0, "data": 1, "think": 2}
SEG_ID = {"outside": 0, "open": 1, "head": 2, "body": 3, "close": 4}
SIZE_BUCKETS = (32, 64, 128, 256, 512, 1024, 2048, 4096)
# THE SIZED FRAME (0.1.21): the open tag carries one payload byte naming the frame's generic size, as the register tag is MODE + one
# letter (no new special byte; all thirteen are allocated). The craft can then count to the close: END becomes predictable at every
# size the way it already is at 32 bytes, where the records fill the frame.
SIZE_LETTER = {32: b"a", 64: b"b", 128: b"c", 256: b"d", 512: b"e", 1024: b"f", 2048: b"g"}


def size_bucket(size: int) -> int:
    """The smallest generic size at or above a frame's byte size: the gauges group by it (a layout-A bulk frame of 121-123 bytes
    reads as 128; the dynamic frames keep their generic sizes)."""
    return next((b for b in SIZE_BUCKETS if b >= size), SIZE_BUCKETS[-1])


def frame_generic(f: dict) -> int:
    """A frame's generic size: the one its sized open tag names (a sliced frame is its slices plus three tag bytes, so its byte size
    is above the class it names), else the size bucket of its byte size."""
    return f.get("generic") or size_bucket(f["size"])


def frame_group(f: dict) -> str:
    """The fetch's group for a truth frame: kind / generic size / nesting depth of the frame the record belongs to. A slice reads as
    the frame it cuts (its 'unit' and 'ldepth'); every other frame as itself, so the plain and sized forms keep their 0.1.20 groups."""
    return f"{f['kind']}/{f.get('unit') or frame_generic(f)}/depth{f.get('ldepth', f['depth'])}"


def open_tag(kind: str, generic: int, sized: bool = False) -> bytes:
    """A frame's open tag: the kind's special (DATA or THINK), and when sized one payload byte naming the generic size (a = 32,
    b = 64, c = 128, d = 256, e = 512, f = 1,024, g = 2,048)."""
    if not sized:
        return bytes([KIND_TAG[kind]])
    assert generic in SIZE_LETTER, f"a sized frame needs a generic size in {sorted(SIZE_LETTER)}, not {generic}"
    return bytes([KIND_TAG[kind]]) + SIZE_LETTER[generic]


def frame_capacity(size: int, rec_bytes: int, sized: bool = False, sliced: int = 0) -> int:
    """The records a lookup frame of `size` holds at `rec_bytes` a record: (size - 2) // rec plain, (size - 3) // rec sized (the open
    tag has two bytes); sliced into slices of `sliced` bytes, (size // sliced) x (sliced - 3) // rec (a frame no larger than a slice
    is one slice)."""
    if sliced and size > sliced:
        return (size // sliced) * ((sliced - 3) // rec_bytes)
    return (size - (3 if (sized or sliced) else 2)) // rec_bytes


def frame_bytes(size: int, sliced: int = 0) -> int:
    """A frame's length in the row: its size, or for a sliced frame its slices plus the open tag and END around them (size + 3)."""
    return size + 3 if (sliced and size > sliced) else size


class Filler:
    """Corpus text handed out in order from a start offset, the specials cleaned, wrapping at the end."""

    def __init__(self, src: bytes, start: int = 0):
        self.src, self.i = src, start % max(1, len(src))

    def take(self, n: int) -> bytes:
        out = bytearray()
        while len(out) < n:
            chunk = self.src[self.i:self.i + n - len(out)]
            if not chunk:
                self.i = 0
                chunk = self.src[:n - len(out)]
            out += chunk
            self.i += len(chunk)
        return clean_filler(bytes(out), n)


def spec(kind: str, size: int, records: int = 0, children=(), keys=None, values=None) -> dict:
    """A frame to build: kind data | think, its size in bytes (the open tag + the body + END), the records to write in it, the child
    frames nested in its body, explicit keys (an overlapping frame repeats another frame's keys) and values (None = fresh)."""
    assert kind in KIND_TAG and size >= 4, (kind, size)
    return dict(kind=kind, size=int(size), records=int(records), children=list(children), keys=keys, values=values)


def _records_for(rng, sp, register, state):
    kind, vn = REGISTERS[register][2], REGISTERS[register][3]
    if sp["keys"] is not None:
        keys = list(sp["keys"])
        vals = list(sp["values"]) if sp["values"] is not None else [value(rng, kind, vn) for _ in keys]
        return list(zip(keys, vals))
    if state["facts"] is not None:
        return [state["facts"][int(next(state["fact_iter"]))] for _ in range(sp["records"])]
    return [(key_name(int(next(state["key_iter"]))), value(rng, kind, vn)) for _ in range(sp["records"])]


def build_frame(rng, sp, base, fill, register, state, depth=0) -> bytes:
    """The frame's bytes (len == sp['size']): the open tag, the records, the children after random gaps, filler, END. Appends the
    frame's meta to state['frames'] and every record's definition to state['defs'] (key -> [(position, value, frame index, depth,
    owner)]; the owner is the frame whose own records these are). state['sized'] gives the open tag its size byte (the meta then
    carries 'generic' and 'open', the tag's length); state['sliced'] = s builds a frame larger than s bytes as slices (build_sliced)."""
    sized, sliced = state.get("sized", False), state.get("sliced", 0)
    if sliced and sp["size"] > sliced:
        return build_sliced(rng, sp, base, fill, register, state, depth)
    size = sp["size"]
    tag = open_tag(sp["kind"], size, sized)
    room = size - len(tag) - 1
    body = bytearray()
    recs = _records_for(rng, sp, register, state)
    fi = len(state["frames"])
    state["frames"].append(None)                                   # the slot; the children append after the parent's index
    for k, v in recs:
        rb = record_bytes(k, v)
        assert len(body) + len(rb) <= room, f"a {sp['kind']} frame of {size} cannot hold {len(recs)} records"
        state["defs"].setdefault(k, []).append((base + len(tag) + len(body), v, fi, depth, fi))
        body += rb
    kids = sp["children"]
    need = sum(frame_bytes(c["size"], sliced) for c in kids)
    assert len(body) + need <= room, f"the children ({need} bytes) do not fit a {sp['kind']} frame of {size} after {len(body)} bytes of records"
    for c in kids:
        slack = room - len(body) - need
        gap = int(rng.integers(0, slack + 1)) if slack > 0 else 0
        body += fill.take(gap)
        body += build_frame(rng, c, base + len(tag) + len(body), fill, register, state, depth + 1)
        need -= frame_bytes(c["size"], sliced)
    body += fill.take(room - len(body))
    meta = dict(kind=sp["kind"], size=size, start=base, end=base + size, depth=depth, n_records=len(recs),
                keys=[k.decode() for k, _ in recs], children=len(kids))
    if sized:
        meta.update(generic=size, open=len(tag))
    state["frames"][fi] = meta
    return tag + bytes(body) + bytes([END])


def build_sliced(rng, sp, base, fill, register, state, depth=0) -> bytes:
    """A frame larger than the slice size s, built as slices (Phil 2026-10-13: one tag for a larger size, sliced into smaller sizes
    that carry the closures): its sized open tag, then size / s closed slices of s bytes back to back with no filler between them,
    then END (size + 3 bytes in all; the letter names the size of the slices it holds). A lookup frame's own records fill its first
    slices in order, as many a slice as fit; its other slices carry filler, a thinking frame's slices always. A child frame becomes a
    run of child-size / s slices of the child's kind at a slice-aligned place after the parent's record slices (the gap drawn in whole
    slices; the child's records in the run's first slices). Records are defined at the depth of the frame they belong to (a slice is a
    cut of its frame, not a frame inside it); the slices are frames of their own one level down for the masks and the bits by frame,
    each carrying the generic size ('unit') and nesting depth ('ldepth') of the frame it cuts, so the fetch groups a record by the
    frame it belongs to (frame_group)."""
    s, S = state["sliced"], sp["size"]
    assert S % s == 0, f"a sliced frame of {S} bytes is not a whole number of {s}-byte slices"
    k, room = S // s, s - 3                                         # a slice's body: its two-byte open tag and END around it
    tag = open_tag(sp["kind"], S, True)
    fi = len(state["frames"])
    state["frames"].append(None)                                   # the slot; the slices append after the parent's index

    def pack(recs):
        """The records in slice order, as many a slice as fit."""
        out, cur, used = [], [], 0
        for kk, vv in recs:
            n = len(record_bytes(kk, vv))
            assert n <= room, f"a record of {n} bytes does not fit a {s}-byte slice"
            if used + n > room:
                out.append(cur)
                cur, used = [], 0
            cur.append((kk, vv))
            used += n
        return out + ([cur] if cur else [])

    recs = _records_for(rng, sp, register, state)
    slots = [(sp["kind"], r, S, depth, fi) for r in pack(recs)]     # (kind, records, unit, ldepth, owner) per slice
    assert len(slots) <= k, f"{len(recs)} records need {len(slots)} slices; a {S}-byte frame has {k}"
    kids = sp["children"]
    for c in kids:
        assert c["size"] % s == 0 and not c["children"], f"a child of a sliced frame is a whole number of slices with no children ({c['size']})"
    need = sum(c["size"] // s for c in kids)
    assert len(slots) + need <= k, f"the children ({need} slices) do not fit a sliced frame of {k} slices after {len(slots)} record slices"
    for c in kids:
        slack = k - len(slots) - need
        gap = int(rng.integers(0, slack + 1)) if slack > 0 else 0
        slots += [(sp["kind"], [], S, depth, fi)] * gap
        crecs, kc = pack(_records_for(rng, c, register, state)), c["size"] // s
        assert len(crecs) <= kc, f"a child of {c['size']} bytes holds {kc} slices, its records need {len(crecs)}"
        owner = ("run", fi, len(slots))                            # a child run is never a top-level frame
        slots += [(c["kind"], crecs[j] if j < len(crecs) else [], c["size"], depth + 1, owner) for j in range(kc)]
        need -= kc
    slots += [(sp["kind"], [], S, depth, fi)] * (k - len(slots))
    body = bytearray()
    for kind, rs, unit, ld, owner in slots:
        sb = base + len(tag) + len(body)
        si = len(state["frames"])
        state["frames"].append(None)
        st = open_tag(kind, s, True)
        sbody = bytearray()
        for kk, vv in rs:
            state["defs"].setdefault(kk, []).append((sb + len(st) + len(sbody), vv, si, ld, owner))
            sbody += record_bytes(kk, vv)
        sbody += fill.take(room - len(sbody))
        state["frames"][si] = dict(kind=kind, size=s, start=sb, end=sb + s, depth=depth + 1, n_records=len(rs), keys=[kk.decode() for kk, _ in rs],
                                   children=0, generic=s, open=len(st), unit=unit, ldepth=ld, slice_of=fi)
        body += st + bytes(sbody) + bytes([END])
    assert len(body) == S, (len(body), S)
    state["frames"][fi] = dict(kind=sp["kind"], size=S + 3, start=base, end=base + S + 3, depth=depth, n_records=len(recs),
                               keys=[kk.decode() for kk, _ in recs], children=len(kids), generic=S, open=len(tag), sliced=s, slices=k)
    return tag + bytes(body) + bytes([END])


def compose_dynamic(rng, src: bytes, ctx: int, specs: list, register: str = "caption", scheme: str = "mode", ask: str = "any",
                    facts: list | None = None, gap_max: int = 64, sized: bool = False, sliced: int = 0) -> tuple:
    """The dynamic row: DOC, the register tag, SEP q SEP, the frames in order after gaps of 0..gap_max filler bytes, the question
    CUE key SEP, the value, END, filler to the window end. ask: any (a uniform key), nested (a key defined at depth >= 1), overlap (a key
    defined more than once; the truth is the LATEST definition before the question), first / last (a key of that frame), weights (a fact
    absent from every frame; needs facts). sized: every open tag names its frame's generic size (one payload byte); sliced = s (needs
    sized): every frame larger than s bytes is built as closed s-byte slices (build_sliced). -> (row, meta): v0, vlen, dist (v0 minus
    the truth's position; None for weights), key, truth_frame, truth_depth, n_defs, source ('frame' | 'weights'), frames (the list),
    scheme, register, ctx, q0 (and sized / sliced when set)."""
    assert not sliced or sized, "a sliced frame is a sized frame: sliced needs sized"
    fill = Filler(src, int(rng.integers(0, max(1, len(src) - 1))))
    state = dict(frames=[], defs={}, facts=None, key_iter=iter(rng.permutation(16000).tolist()), fact_iter=None, sized=bool(sized),
                 sliced=int(sliced))
    asked_fact = None
    if facts is not None:
        perm = rng.permutation(len(facts)).tolist()
        if ask == "weights":
            asked_fact, perm = facts[perm[0]], perm[1:]
        state["facts"], state["fact_iter"] = facts, iter(perm)
    else:
        assert ask != "weights", "the weights source needs a fact table"
    out = bytearray(bytes([DOC]) + tag_bytes(register, scheme) + bytes([SEP]) + b"q" + bytes([SEP]))
    for sp in specs:
        out += fill.take(int(rng.integers(0, gap_max + 1)))
        out += build_frame(rng, sp, len(out), fill, register, state)
    defs = state["defs"]
    if asked_fact is not None:
        key, val, truth = asked_fact[0], asked_fact[1], None
        assert key not in defs, "the asked fact is in a frame"
    else:
        if ask == "any":
            cands = list(defs)
        elif ask == "nested":
            cands = [k for k, d in defs.items() if d[-1][3] >= 1]
        elif ask == "overlap":
            cands = [k for k, d in defs.items() if len(d) > 1]
        elif ask in ("first", "last") and sliced:                  # sliced: a key of the first / last top-level frame's own records
            tops = [i for i, f in enumerate(state["frames"]) if f["depth"] == 0]
            fi = tops[0] if ask == "first" else tops[-1]
            cands = [k for k, d in defs.items() if d[-1][4] == fi]
        elif ask in ("first", "last"):
            fi = 0 if ask == "first" else len(state["frames"]) - 1
            cands = [k for k, d in defs.items() if d[-1][2] == fi]
        else:
            raise ValueError(ask)
        assert cands, f"nothing to ask for {ask!r}"
        key = cands[int(rng.integers(len(cands)))]
        truth = defs[key][-1]                                      # the latest definition wins
        val = truth[1]
    out += fill.take(int(rng.integers(0, gap_max + 1)))
    q0 = len(out)
    out += bytes([CUE]) + key + bytes([SEP])
    v0 = len(out)
    out += val + bytes([END])
    tail = ctx + 1 - len(out)
    assert tail >= 0, f"the row overflows the window by {-tail} bytes"
    out += fill.take(tail)
    row = bytes(out)
    assert len(row) == ctx + 1 and row[v0:v0 + len(val)] == val
    meta = dict(v0=v0, vlen=len(val), dist=None if truth is None else v0 - truth[0], key=key.decode(),
                truth_frame=None if truth is None else truth[2], truth_depth=None if truth is None else truth[3],
                n_defs=0 if truth is None else len(defs[key]), source="weights" if truth is None else "frame",
                frames=state["frames"], scheme=scheme, register=register, ctx=ctx, q0=q0)
    if sized:
        meta["sized"] = True
    if sliced:
        meta["sliced"] = int(sliced)
    return row, meta


def overlap_specs(rng, register: str, size1: int, size2: int, k: int, mode: str = "conflict") -> list:
    """Two lookup frames sharing k keys: the same values (redundant) or fresh values in the second (conflict; the truth is the second)."""
    kind, vn = REGISTERS[register][2], REGISTERS[register][3]
    keys = [key_name(int(i)) for i in rng.permutation(16000)[:k]]
    vals1 = [value(rng, kind, vn) for _ in keys]
    vals2 = vals1 if mode == "same" else [value(rng, kind, vn) for _ in keys]
    return [spec("data", size1, keys=keys, values=vals1), spec("data", size2, keys=keys, values=vals2)]


def frame_masks(meta: dict, ctx: int) -> dict:
    """Per position of the row (length ctx + 1): the innermost frame's kind id, its generic size (frame_generic), its depth + 1 (0
    outside), and the segment (1 the open tag, two bytes when sized; 2 the first eight body bytes; 3 the body; 4 the close tag; 0 outside).
    A sliced frame shows its own open tag and END; its slices paint the rest."""
    n = ctx + 1
    kind, size, depth, seg = (np.zeros(n, dtype=np.int64) for _ in range(4))
    for f in sorted(meta["frames"], key=lambda f: f["depth"]):
        s, e, o = f["start"], f["end"], f.get("open", 1)
        kind[s:e], size[s:e], depth[s:e] = KIND_ID[f["kind"]], frame_generic(f), f["depth"] + 1
        seg[s:e] = SEG_ID["body"]
        seg[s:s + o] = SEG_ID["open"]
        seg[s + o:min(s + o + 8, e - 1)] = SEG_ID["head"]
        seg[e - 1] = SEG_ID["close"]
    return dict(kind=kind, size=size, depth=depth, seg=seg)


def bits_by_frame(nll, metas: list, acc: dict | None = None) -> dict:
    """Accumulate per-target nats (nll: (B, ctx); the target in column i is the row position i + 1) by (kind, size, segment) and by
    (kind, depth), plus the bytes outside every frame. acc[key] = [sum of nats, count]; finish_frame_bits() gives bits a byte."""
    acc = {} if acc is None else acc
    nll = np.asarray(nll, dtype=np.float64)
    B, T = nll.shape
    for b in range(B):
        m = frame_masks(metas[b], T)
        kind, size, depth, seg = (m[k][1:T + 1] for k in ("kind", "size", "depth", "seg"))
        inside = depth > 0
        for kd in np.unique(kind[inside]):
            for sz in np.unique(size[(kind == kd) & inside]):
                for sg in np.unique(seg[(kind == kd) & (size == sz)]):
                    sel = (kind == kd) & (size == sz) & (seg == sg)
                    a = acc.setdefault(("kss", int(kd), int(sz), int(sg)), [0.0, 0])
                    a[0] += float(nll[b, sel].sum())
                    a[1] += int(sel.sum())
            for dp in np.unique(depth[(kind == kd) & inside]):
                sel = (kind == kd) & (depth == dp)
                a = acc.setdefault(("kd", int(kd), int(dp)), [0.0, 0])
                a[0] += float(nll[b, sel].sum())
                a[1] += int(sel.sum())
        a = acc.setdefault(("outside",), [0.0, 0])
        a[0] += float(nll[b, ~inside].sum())
        a[1] += int((~inside).sum())
    return acc


def finish_frame_bits(acc: dict) -> dict:
    ln2 = float(np.log(2.0))
    kid = {v: k for k, v in KIND_ID.items()}
    sid = {v: k for k, v in SEG_ID.items()}
    out = dict(by_kind_size_segment={}, by_kind_depth={}, outside=None)
    for k, (s, n) in acc.items():
        cell = dict(bits=(s / n / ln2 if n else None), n=n)
        if k[0] == "kss":
            out["by_kind_size_segment"][f"{kid[k[1]]}/{k[2]}/{sid[k[3]]}"] = cell
        elif k[0] == "kd":
            out["by_kind_depth"][f"{kid[k[1]]}/depth{k[2]}"] = cell
        else:
            out["outside"] = cell
    return out


def fmt_frame_bits(res: dict) -> str:
    rows = ["| frame kind / size / segment | bits a byte | bytes |", "|---|---|---|"]
    for k, v in sorted(res["by_kind_size_segment"].items()):
        rows.append(f"| {k} | {v['bits']:.3f} | {v['n']:,} |")
    for k, v in sorted(res["by_kind_depth"].items()):
        rows.append(f"| {k} | {v['bits']:.3f} | {v['n']:,} |")
    if res.get("outside"):
        rows.append(f"| outside the frames | {res['outside']['bits']:.3f} | {res['outside']['n']:,} |")
    return "\n".join(rows)


def uniform_bits(register: str) -> float:
    """The register's uniform rate: log2 of its value alphabet (letters 4.70, digits 3.32, hex 4.00 bits a byte). A copy that never
    formed sits at it; the formation verdict reads the near band's online value bits against it."""
    return float(np.log2(len(ALPH[REGISTERS[register][2]])))


def canary_rows(docs: list, ctx: int, register: str = "caption", scheme: str = "mode", seed: int = 1, sized: bool = False) -> tuple:
    """The plain-document canary rows (ctx + 1 bytes each) for every document, three kinds: plain = DOC + text (the only row a diet
    composes); tag = DOC + the register tag + SEP q SEP + text (a question cue before any frame); frame = the same with one flat 32-byte
    lookup frame of one record before the text (the same frame in every row; sized like the craft's own frames when it was trained on
    sized frames). -> ({kind: [row bytes]}, meta). A craft that reads the plain row and breaks on the others learned its opening grammar
    strictly; one that breaks on the plain row lost the bare document."""
    rng = np.random.default_rng([seed, 4343])
    prefix = tag_bytes(register, scheme) + bytes([SEP]) + b"q" + bytes([SEP])
    state = dict(frames=[], defs={}, facts=None, key_iter=iter(rng.permutation(16000).tolist()), fact_iter=None, sized=bool(sized))
    frame = build_frame(rng, spec("data", 32, records=1), 1 + len(prefix), Filler(docs[-1], 0), register, state)
    assert len(frame) == 32 and frame[0] == DATA and frame[-1] == END, (len(frame), frame[:1], frame[-1:])
    rows = {"plain": [], "tag": [], "frame": []}
    for d in docs:
        rows["plain"].append(bytes([DOC]) + d[:ctx])
        rows["tag"].append(bytes([DOC]) + prefix + d[:ctx - len(prefix)])
        rows["frame"].append(bytes([DOC]) + prefix + frame + d[:ctx - len(prefix) - len(frame)])
    for kind, rs in rows.items():
        assert all(len(r) == ctx + 1 for r in rs), (kind, [len(r) for r in rs if len(r) != ctx + 1][:3])
    meta = dict(prefix_bytes=len(prefix), frame_bytes=len(frame), frame_records=1, register=register, scheme=scheme)
    if sized:
        meta["sized"] = True
    return rows, meta


def near_band_windows(curve: list, window: int = 400, band: str = "0", min_n: int = 8) -> tuple:
    """The near band's online value bits as (step, bits) read over the last `window` steps at every progress line, exact from the
    cumulative counts a 0.1.22 curve carries (value_n; at least `min_n` values in a window); a curve without counts (records before
    0.1.22) gives its means since step 1, as recorded. -> (points, basis)."""
    pts = [(c["step"], c["value_bits"].get(band), (c.get("value_n") or {}).get(band)) for c in curve if (c.get("value_bits") or {}).get(band) is not None]
    if not pts or any(n is None for _, _, n in pts):
        return [(s, b) for s, b, _ in pts], "the mean since step 1 (a curve before 0.1.22)"
    out = []
    for i, (s, b, n) in enumerate(pts):
        base = [p for p in pts[:i] if p[0] <= s - window]                       # the window starts after the last line this far back
        s0, b0, n0 = base[-1] if base else (0, 0.0, 0)                          # (or at the run start)
        if n - n0 >= min_n:
            out.append((s, (b * n - b0 * n0) / (n - n0)))
    return out, f"the last {window} steps"


def band_fall(curve: list, band: str = "512", window: int = 800, min_n: int = 8) -> dict | None:
    """Is a band's online value still improving at a run's end? Its mean value bits over the last `window` steps against the `window`
    before them, exact from a 0.1.22 curve's counts (value_n): a window starts after the last progress line that far back (or at the run
    start), as in near_band_windows. -> dict(last, before, fall = before - last, steps = the three line steps, n = the two windows'
    value counts), or None when the curve has no counts, or either window holds fewer than `min_n` values (a curriculum's stage-B rule
    reads the far band: the records more than 512 bytes back)."""
    pts = [(c["step"], (c.get("value_bits") or {}).get(band), (c.get("value_n") or {}).get(band)) for c in curve]
    pts = [(0, 0.0, 0)] + [(s, b, n) for s, b, n in pts if b is not None and n is not None]
    if len(pts) < 3 or any(c.get("value_n") is None for c in curve):
        return None
    s2, b2, n2 = pts[-1]
    base1 = [p for p in pts[:-1] if p[0] <= s2 - window]
    if not base1:
        return None
    s1, b1, n1 = base1[-1]
    base0 = [p for p in pts if p[0] <= s1 - window]
    if not base0:
        return None
    s0, b0, n0 = base0[-1]
    if n2 - n1 < min_n or n1 - n0 < min_n:
        return None
    last, before = (b2 * n2 - b1 * n1) / (n2 - n1), (b1 * n1 - b0 * n0) / (n1 - n0)
    return dict(last=last, before=before, fall=before - last, steps=(s0, s1, s2), n=(n1 - n0, n2 - n1), band=band, window=window)


def pooled_fetch(groups: dict | None, kind: str = "data/") -> dict | None:
    """The fetch pooled over the lookup groups (every frame size and depth), weighted by their trials: the whole value, the first byte,
    the value bits, the trials. None when no group was read."""
    g = [v for k, v in (groups or {}).items() if k.startswith(kind) and v.get("n")]
    n = sum(v["n"] for v in g)
    if not n:
        return None
    return dict(full_value=sum(v["full_value"] * v["n"] for v in g) / n, first_byte=sum(v["first_byte"] * v["n"] for v in g) / n,
                value_bits=sum(v["value_bits"] * v["n"] for v in g) / n, n=n)


def formation(curve: list, register: str, fetch_groups: dict | None, margin: float = 0.2, floor: float = 0.05, min_trials: int = 8,
              window: int = 400) -> dict:
    """Did the copy form? A formation FAILURE when every gauged lookup group of `min_trials` trials or more reads the whole value at or
    under `floor` (a 6-letter value written whole from its frame). The near band's online value bits (the value's bytes at the question
    when the frame sits within 128 bytes) are reported against the register's uniform rate, read over the last `window` steps at the end and
    at the first progress line under the rate by `margin` (near_band_windows: exact on a 0.1.22 curve; a curve before 0.1.22 only has the
    mean since step 1, and its 'at the end' and 'first step under' are that mean's), and are NOT required for the verdict: a copy can form
    misaligned (near band 2.26 bits, the greedy first byte wrong in every trial, 2026-10-13) or never form. A verdict in the record, never a halt."""
    u = uniform_bits(register)
    near, basis = near_band_windows(curve, window)
    last = near[-1][1] if near else None
    first_under = next((s for s, v in near if v < u - margin), None)
    at_rate = last is None or last > u - margin
    lookup = {g: a["full_value"] for g, a in (fetch_groups or {}).items() if g.startswith("data/") and a.get("n", 0) >= min_trials}
    whole_zero = bool(lookup) and all(v <= floor for v in lookup.values())
    failure = bool(whole_zero)
    return dict(register=register, uniform_bits=u, margin=margin, near_band_value_bits_at_end=last, first_step_under=first_under,
                at_the_rate=bool(at_rate), whole_value_zero_everywhere=whole_zero, failure=failure, formed=not failure, floor=floor,
                min_trials=min_trials, groups_judged=len(lookup), near_band_basis=basis)


def plain_collapse(by_position_row: dict, threshold: float = 2.5, head: str = "0-64") -> dict:
    """The plain-collapse flag of the curriculum plan: the held-out bits a byte by window of the target position; a collapse when any
    window past the head reads above the threshold (a Pollux seed-1 math craft at 4,096 read 4.3 / 3.4 in its middle windows while its
    head and its frames were normal, 2026-10-13). -> flag, the windows over, the worst window past the head."""
    past = {w: b for w, b in by_position_row.items() if w != head and b is not None}
    over = [w for w, b in past.items() if b > threshold]
    worst = max(past, key=past.get) if past else None
    return dict(flag=bool(over), threshold=threshold, windows_over=over, worst_window=worst, worst_bits=(past[worst] if worst else None))


def _check_dynamic(row, m, ctx):
    assert len(row) == ctx + 1 and row[0] == DOC
    for f in m["frames"]:
        assert row[f["start"]] == KIND_TAG[f["kind"]] and row[f["end"] - 1] == END and f["end"] - f["start"] == f["size"], f
        if f.get("generic") is not None:                          # sized: the payload byte names the generic size
            assert row[f["start"] + 1:f["start"] + 2] == SIZE_LETTER[f["generic"]], f
        if f["depth"] >= 1:
            assert any(p["depth"] == f["depth"] - 1 and p["start"] < f["start"] and f["end"] <= p["end"] - 1 for p in m["frames"]), f
    assert row[m["q0"]] == CUE and row[m["v0"] - 1] == SEP and row[m["v0"] + m["vlen"]] == END
    assert int((frame_masks(m, ctx)["seg"] == SEG_ID["open"]).sum()) == sum(f.get("open", 1) for f in m["frames"])


def _check_sliced(row, m):
    """Every sliced frame: its open tag, then exactly size / s closed slices tiling its body with no byte between them, then END."""
    n = 0
    for i, f in enumerate(m["frames"]):
        if not f.get("sliced"):
            continue
        s, k = f["sliced"], f["slices"]
        kids = [g for g in m["frames"] if g.get("slice_of") == i]
        assert len(kids) == k and f["size"] == k * s + 3 and row[f["end"] - 1] == END, f
        for j, g in enumerate(sorted(kids, key=lambda g: g["start"])):
            assert g["start"] == f["start"] + 2 + j * s and g["end"] - g["start"] == s, (j, g)
            assert row[g["start"]] == KIND_TAG[g["kind"]] and row[g["start"] + 1:g["start"] + 2] == SIZE_LETTER[s] and row[g["end"] - 1] == END
        n += 1
    return n


def _selftest_dynamic(rng, src):
    n, metas = 0, []
    for ctx in (1024, 4096):
        for scheme in ("mode", "esc"):
            for reg in ("caption", "code"):
                specs = [spec("data", 32, records=1), spec("think", 256, children=[spec("data", 64, records=2)]), spec("data", 128, records=4)]
                for ask in ("any", "nested", "first", "last"):
                    row, m = compose_dynamic(rng, src, ctx, specs, reg, scheme, ask)
                    _check_dynamic(row, m, ctx)
                    if ask == "nested":
                        assert m["truth_depth"] == 1, m["truth_depth"]
                    n += 1
                    if ctx == 1024:
                        metas.append(m)
                facts = fact_table(128, reg)
                row, m = compose_dynamic(rng, src, ctx, [spec("data", 128, records=3), spec("data", 256, records=6)], reg, scheme, "weights", facts)
                assert m["source"] == "weights" and m["key"].encode() not in row[:m["q0"]] and m["dist"] is None
                _check_dynamic(row, m, ctx)
                n += 1
                row, m = compose_dynamic(rng, src, ctx, overlap_specs(rng, reg, 128, 128, 3, "conflict"), reg, scheme, "overlap")
                assert m["n_defs"] == 2 and m["truth_frame"] == 1
                k = m["key"].encode()
                i1 = row.index(bytes([CUE]) + k + bytes([SEP]))
                assert row[i1 + len(k) + 2:i1 + len(k) + 2 + m["vlen"]] != row[m["v0"]:m["v0"] + m["vlen"]], "the conflict's truth is not the latest"
                _check_dynamic(row, m, ctx)
                n += 1
    res = finish_frame_bits(bits_by_frame(np.ones((len(metas), 1024)), metas))
    cells = list(res["by_kind_size_segment"].values()) + list(res["by_kind_depth"].values()) + [res["outside"]]
    assert all(abs(c["bits"] - 1 / np.log(2.0)) < 1e-9 for c in cells)
    assert sum(v["n"] for v in res["by_kind_depth"].values()) + res["outside"]["n"] == len(metas) * 1024
    print(f"dynamic: composed {n} rows (sizes 32 / 64 / 128 / 256, a nested lookup inside a thinking frame, the weights source, an overlap"
          f" in conflict) at 1,024 and 4,096 across 2 schemes x 2 registers: every assert held; bits by frame sums to the window")
    print(fmt_frame_bits(res))


def _selftest_sized(rng, src):
    n = 0
    for sliced in (0, 32):
        for ctx in (1024, 4096):
            specs = [spec("data", 32, records=1), spec("think", 256, children=[spec("data", 64, records=2)]), spec("data", 128, records=4)]
            if ctx == 4096:
                specs += [spec("think", 1024, children=[spec("data", 256, records=4)]), spec("data", 512, records=8)]
            for ask in ("any", "nested", "first", "last"):
                row, m = compose_dynamic(rng, src, ctx, specs, "caption", "mode", ask, sized=True, sliced=sliced)
                _check_dynamic(row, m, ctx)
                n_sl = _check_sliced(row, m)
                assert n_sl == (0 if not sliced else (2 if ctx == 1024 else 4)), n_sl     # every frame above 32 bytes
                if ask == "nested":
                    assert m["truth_depth"] == 1 and frame_group(m["frames"][m["truth_frame"]]).endswith("/depth1"), m["truth_frame"]
                n += 1
    print(f"sized and sliced: composed {n} rows (plain sized; sliced into 32-byte slices) at 1,024 and 4,096: every open tag names its size,"
          f" every sliced frame is its slices back to back and its END")


def _selftest():
    rng = np.random.default_rng(7)
    src = bytes(rng.integers(97, 123, size=20000).astype(np.uint8)) + bytes([DOC]) + b"the next document begins here. " * 400
    print("capacity (records that fit):")
    for ctx in (1024, 2048, 4096):
        for reg in ("caption", "math", "code"):
            print(f"  ctx {ctx} {reg}: {capacity(ctx, 'A', reg)}")
    n = 0
    for ctx in (1024, 4096):
        cap = capacity(ctx, "A", "caption")
        for scheme in ("mode", "esc"):
            for reg in REGISTERS:
                capr = capacity(ctx, "A", reg)
                for (nl, nb, ask) in ((2, 0, "lookup"), (4, 8, "bulk"), (1, cap["bulk_max"], "bulk"), (3, 5, "weights")):
                    facts = fact_table(256, reg) if ask == "weights" or nb > 20 else None
                    row, m = compose(rng, src, ctx, "A", scheme, reg, min(nl, capr["lookup_max"]), min(nb, capr["bulk_max"]), facts, ask)
                    comps = m["comps"]
                    assert row[0] == DOC and row[comps["lookup"][0]] == DATA and row[comps["lookup"][1] - 1] == END
                    assert row[comps["think"][0]] == THINK and row[comps["think"][1] - 1] == END
                    assert row[m["q0"]] == CUE and row[m["v0"] - 1] == SEP and row[m["v0"] + m["vlen"]] == END
                    assert row[1] == (MODE if scheme == "mode" else ESC)
                    think_body = row[comps["think"][0] + 1:comps["think"][1] - 1]
                    assert not (set(think_body) & SPECIALS), "a special inside the thinking space"
                    if ask == "weights":
                        assert m["key"].encode() not in row[:m["q0"]], "the asked fact is present in the window"
                        assert m["dist"] is None
                    else:
                        assert m["source"] == ask and m["dist"] > 0, m
                    n += 1
    print(f"composed {n} rows at 1,024 and 4,096 across 2 schemes x 4 registers x 4 asks: every assert held")
    row, m = compose(rng, src, 1024, "A", "mode", "math", 3, 4, None, "bulk")
    print("a sample row (layout A, math, 3 lookup + 4 bulk, asked from bulk):")
    print("  " + decode_visible(row[:m["v0"] + m["vlen"] + 1])[:720].replace("\n", " "))
    print(f"  meta: source {m['source']} key {m['key']} dist {m['dist']} v0 {m['v0']} bands {m['bands']}")
    print("bands:", bands_for(1024), bands_for(4096))
    _selftest_dynamic(rng, src)
    _selftest_sized(rng, src)


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _selftest()
    else:
        print(__doc__)
