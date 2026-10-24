# -*- coding: utf-8 -*-
"""grid_rows.py - THE ADDRESSED GRID ROW (layout "AG", 0.1.70): compartments at fixed positions of the window, each holding one
addressed lookup frame, then an index the craft writes and the questions. The row is 32 slots of 128 bytes (context 4,096):
    slot 0        the head: DOC, MODE c, SEP, g, SEP, then corpus text
    slots 1-24    24 COMPARTMENTS: an addressed lookup frame DATA a1 a2 CUE key SEP value END (a1 a2 = the compartment's address,
                  two digits '0'-'7'), then corpus text to the slot's end
    slots 25-27   THE INDEX: DATA, one entry CUE key SEP a1 a2 for each question in the questions' order, END, corpus text
    slots 28-31   THE QUESTIONS: CUE key SEP value END for each asked record, corpus text
The address space is 64 addresses (two digits of 8); a row carries 24 of them, sorted (a compartment's slot moves with the row, its
address does not). Keys are bed_frames.key_name (distinct in the row), values six lowercase letters (the caption register).

THE STREAMS (the anchored rotary's per-token address, anchored_rope.py): every input position carries a key stream and a query stream
of two digits (0-7, 8 = the null anchor). A token inside compartment c (its whole slot) carries c's digits as key and as query; a
question's answer-producing positions (its SEP and its six value bytes, which predict the six value bytes and the END) carry the null
anchor as key and THE ADDRESS THEY ASK FOR as query; every other token carries the null anchor for both.

Positions in the meta are row positions (0 .. ctx); the input x is row[:-1] and the target y is row[1:], so the target of input
position t is row[t + 1]. Self-test (pure numpy): python grid_rows.py --selftest"""
from __future__ import annotations

import sys
import zlib

import numpy as np

try:
    from . import bed_frames as bf  # noqa: E402  (the package)
except ImportError:                                      # run as a file: the folder on the path
    import os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import bed_frames as bf  # noqa: E402
from geolip.alephllm.data.special_tokens import CUE, DATA, DOC, END, MODE, SEP  # noqa: E402

SLOT = 128
CTX = 4096
N_SLOTS = CTX // SLOT
N_COMP = 24
COMP_SLOTS = tuple(range(1, 1 + N_COMP))
INDEX_LO, INDEX_HI = 25 * SLOT, 28 * SLOT
QUEST_LO, QUEST_HI = 28 * SLOT, 32 * SLOT
RADIX, LEVELS = 8, 2
N_ADDR = RADIX ** LEVELS
NULL = RADIX                                      # the null anchor's digit
VLEN = bf.REGISTERS["caption"][3]
HEAD = bytes([DOC, MODE]) + b"c" + bytes([SEP]) + b"g" + bytes([SEP])
# the segments of a row's target bytes (bits_by_segment): what the byte at row position t + 1 belongs to
SEGMENTS = ("head", "frame", "comp_text", "index_entry", "index_text", "q_answer", "q_other", "q_text")


def crc(tag: str) -> int:
    return zlib.crc32(tag.encode())


def digits(a: int) -> tuple:
    """An address 0..63 as its two digits (first, second)."""
    return (a // RADIX, a % RADIX)


ONE_BYTE0 = 0x80                                  # 0.1.74: the one-byte address form writes address a as the byte 0x80 + a (0x80-0xBF)


def addr_bytes(a: int, abytes: int = 2) -> bytes:
    """An address as the row writes it: two digit bytes '0'-'7' (the default), or (abytes=1, 0.1.74) ONE byte 0x80 + a, unique in its
    row and carried by the grid rows otherwise only inside the filler's multi-byte UTF-8 characters."""
    if abytes == 1:
        return bytes([ONE_BYTE0 + a])
    d1, d2 = digits(a)
    return bytes([48 + d1, 48 + d2])


def decode_written(b, abytes: int = 2) -> np.ndarray:
    """0.1.74: written address bytes (..., abytes) -> digits (..., 2): a digit byte outside '0'-'7' is the null anchor at its level (the
    two-byte form); a byte outside 0x80-0xBF is the null anchor at both levels (the one-byte form)."""
    b = np.asarray(b, dtype=np.int64)
    if abytes == 1:
        a = b[..., 0] - ONE_BYTE0
        ok = (a >= 0) & (a < N_ADDR)
        return np.stack([np.where(ok, a // RADIX, NULL), np.where(ok, a % RADIX, NULL)], -1)
    w = b - 48
    return np.where((w >= 0) & (w < RADIX), w, NULL)


def owned_order(n_owned: int = 48) -> tuple:
    """The members' address sets (the merge order): a fixed permutation of the 64 addresses by crc; the first n_owned are owned in that
    order, the rest are never owned (the unowned set at every merged count)."""
    perm = np.random.default_rng(crc("SCALE|owned")).permutation(N_ADDR)
    return tuple(int(a) for a in perm[:n_owned]), tuple(sorted(int(a) for a in perm[n_owned:]))


def hard_keys(rng) -> list:
    """0.1.75: THE KEY-HARD ROW's keys: the factorial of their parts - 2 words x 2 hundreds digits x 2 tens digits x
    3 units digits = 24 distinct keys, so every key has a neighbour that differs from it in exactly one part, for each of the four parts (a
    match on any part of the key is right on at most half of such a row's entries; only the whole key on all), in a drawn order.
    -> [key bytes] (N_COMP)."""
    nw = len(bf.WORDS)
    w = rng.choice(nw, size=2, replace=False)
    h = rng.choice(10, size=2, replace=False)
    t = rng.choice(10, size=2, replace=False)
    u = rng.choice(10, size=3, replace=False)
    idx = [int(a) + nw * (100 * int(b) + 10 * int(c) + int(d)) for a in w for b in h for c in t for d in u]   # key_name's index
    assert len(idx) == N_COMP and len(set(idx)) == N_COMP
    return [bf.key_name(idx[int(i)]) for i in rng.permutation(N_COMP)]


def compose_ag(rng, src: bytes, ctx: int = CTX, addrs=None, asked=None, swap_index: bool = False, abytes: int = 2, keys=None) -> tuple:
    """One addressed grid row. rng: the row's numpy Generator; src: corpus text (at least ctx + 1 bytes; specials become newlines).
    addrs: the row's 24 addresses (None = 24 distinct draws of the 64); they are sorted onto slots 1-24. asked: the records the questions
    ask, as indices into the sorted compartments, in question order (None = all 24 in a drawn order). swap_index: one pair of index
    entries carries the other's address (a corrupted index; the gauge's control that must fail), the questions unchanged. abytes (0.1.74):
    the address as two digit bytes (2, the default: the rows of 0.1.73 byte for byte) or as one byte 0x80 + a (1), in the frame and in
    the index entry alike; the draws and the streams are the same either way. keys (0.1.75): the compartments' 24 keys in slot order
    (hard_keys: the key-hard row); None = the standard draw. The standard draw is made either way, so every other draw of the row is the
    same; a row with given keys says so in its meta ("hard").
    -> (row bytes of ctx + 1, meta)."""
    assert abytes in (1, 2), abytes
    assert ctx == CTX, f"the addressed grid is laid out for a {CTX}-byte window, got {ctx}"
    if addrs is None:
        addrs = rng.choice(N_ADDR, size=N_COMP, replace=False)
    addrs = sorted(int(a) for a in addrs)
    assert len(addrs) == N_COMP and len(set(addrs)) == N_COMP and all(0 <= a < N_ADDR for a in addrs), addrs
    std = [bf.key_name(int(i)) for i in rng.permutation(16000)[:N_COMP]]
    hard = keys is not None
    keys = std if not hard else [bytes(k) for k in keys]
    assert len(keys) == N_COMP and len(set(keys)) == N_COMP, "a row's 24 keys are distinct"
    vals = [bf.value(rng, "letters", VLEN) for _ in range(N_COMP)]
    order = [int(i) for i in (rng.permutation(N_COMP) if asked is None else asked)]
    assert order and len(set(order)) == len(order) and all(0 <= i < N_COMP for i in order), order
    row = bytearray(bf.clean_filler(src, ctx + 1))
    row[0:len(HEAD)] = HEAD
    recs = []
    for j, (a, s) in enumerate(zip(addrs, COMP_SLOTS)):
        frame = bytes([DATA]) + addr_bytes(a, abytes) + bytes([CUE]) + keys[j] + bytes([SEP]) + vals[j] + bytes([END])
        lo = s * SLOT
        assert len(frame) <= SLOT
        row[lo:lo + len(frame)] = frame
        recs.append(dict(addr=a, slot=s, lo=lo, hi=lo + SLOT, key=keys[j].decode(), value=vals[j].decode(),
                         frame_end=lo + len(frame), v0=lo + 2 + abytes + len(keys[j]) + 1))
    shown = list(order)
    if swap_index:                                      # a corrupted index: two entries trade their addresses
        assert len(order) >= 2
        i, j = (int(x) for x in rng.choice(len(order), size=2, replace=False))
        shown[i], shown[j] = order[j], order[i]
    entries, p = [], INDEX_LO + 1
    ix = bytearray([DATA])
    for qi, o in enumerate(order):
        e = bytes([CUE]) + keys[o] + bytes([SEP]) + addr_bytes(recs[shown[qi]]["addr"], abytes)
        entries.append(dict(rec=o, lo=p, a_pos=p + len(e) - abytes, shown=shown[qi]))
        ix += e
        p += len(e)
    ix += bytes([END])
    assert INDEX_LO + len(ix) <= INDEX_HI, len(ix)
    row[INDEX_LO:INDEX_LO + len(ix)] = ix
    quest, p = [], QUEST_LO
    qb = bytearray()
    for o in order:
        q = bytes([CUE]) + keys[o] + bytes([SEP]) + vals[o] + bytes([END])
        quest.append(dict(rec=o, lo=p, sep=p + 1 + len(keys[o]), v0=p + 2 + len(keys[o]), end=p + 2 + len(keys[o]) + VLEN))
        qb += q
        p += len(q)
    assert QUEST_LO + len(qb) <= QUEST_HI, len(qb)
    row[QUEST_LO:QUEST_LO + len(qb)] = qb
    row = bytes(row[:ctx + 1])
    meta = dict(layout="AG", addrs=addrs, recs=recs, index=entries, index_end=INDEX_LO + len(ix), questions=quest,
                quest_end=QUEST_LO + len(qb), swap_index=swap_index)
    if abytes != 2:                                     # 0.1.74: the one-byte form names itself; the default meta is 0.1.73's
        meta["abytes"] = abytes
    if hard:                                            # 0.1.75: a row with given keys (the key-hard row) names itself
        meta["hard"] = True
    return row, meta


def streams(meta: dict, ctx: int = CTX, request=None) -> tuple:
    """The row's address streams for the input positions 0 .. ctx - 1: (dk, dq), each int8 (ctx, 2). request: the address each question
    asks for, in question order (None = the true address; a digit outside 0-7 = the null anchor at that level)."""
    dk = np.full((ctx, LEVELS), NULL, dtype=np.int8)
    for r in meta["recs"]:
        dk[r["lo"]:r["hi"]] = digits(r["addr"])
    dq = dk.copy()
    for qi, q in enumerate(meta["questions"]):
        req = digits(meta["recs"][q["rec"]]["addr"]) if request is None else tuple(request[qi])
        req = tuple(int(d) if 0 <= int(d) < RADIX else NULL for d in req)
        span = slice(q["sep"], q["sep"] + VLEN + 1)              # the SEP and the six value bytes (they predict the value and the END)
        dk[span] = NULL
        dq[span] = req
    return dk, dq


def answer_positions(meta: dict) -> list:
    """For each question (in order): the input positions whose targets are its six value bytes and its END."""
    return [list(range(q["sep"], q["sep"] + VLEN + 1)) for q in meta["questions"]]


def abytes_of(meta: dict) -> int:
    """0.1.74: the row's address form (2 = two digit bytes, the default; 1 = one byte)."""
    return meta.get("abytes", 2)


def index_positions(meta: dict) -> list:
    """For each index entry (in order): the input positions whose targets are its address bytes (two digits; one byte in the one-byte
    form). The first is always the entry's SEP."""
    ab = abytes_of(meta)
    return [tuple(e["a_pos"] - 1 + i for i in range(ab)) for e in meta["index"]]


def head_targets(meta: dict) -> tuple:
    """The address head's reads: each question's SEP (an input position) and the asked address's two digits. -> (positions, digits)."""
    pos = [q["sep"] for q in meta["questions"]]
    dig = [digits(meta["recs"][q["rec"]]["addr"]) for q in meta["questions"]]
    return pos, dig


def index_head_targets(meta: dict) -> tuple:
    """0.1.72: the address head's reads at the index: each entry's SEP (an input position; its target is the entry's first address byte)
    and the address of the record that holds the entry's key, as two digits (the truth, whatever address the entry shows).
    -> (positions, digits). At an entry's SEP both streams carry the null anchor (the index lies outside every compartment) and nothing
    earlier in the row names the key's address but its record."""
    pos = [e["a_pos"] - 1 for e in meta["index"]]
    dig = [digits(meta["recs"][e["rec"]]["addr"]) for e in meta["index"]]
    return pos, dig


def written_request(pred_row: np.ndarray, meta: dict) -> list:
    """The address the craft wrote for each question: its greedy bytes at the index entry's address positions (pred_row: the argmax
    at every input position of the teacher-forced row), as digits (decode_written: a byte outside the form's range is the null anchor)."""
    ab = abytes_of(meta)
    return [tuple(int(d) for d in decode_written([int(pred_row[p]) for p in ps], ab)) for ps in index_positions(meta)]


def row_request(row: bytes, meta: dict) -> list:
    """0.1.72: the address each index entry of `row` names, as digits (decode_written): on a free-running row (free_running) the craft's
    own index, in question order."""
    ab = abytes_of(meta)
    return [tuple(int(d) for d in decode_written([row[p + 1] for p in ps], ab)) for ps in index_positions(meta)]


def put_index_bytes(row: bytes, meta: dict, pred_row: np.ndarray) -> bytes:
    """0.1.72: the row with each index entry's address bytes replaced by the craft's greedy bytes there (pred_row: the argmax at every
    input position of this row; the byte at row position t + 1 is predicted at input position t). Everything else stays as composed."""
    b = bytearray(row)
    for ps in index_positions(meta):
        for p in ps:
            b[p + 1] = int(pred_row[p])
    return bytes(b)


def free_running(ag: list, greedy, max_passes: int | None = None) -> tuple:
    """0.1.72: THE FREE-RUNNING INDEX (the craft's own index bytes in the row). For rows ag [(row, meta)]:
        b(0) = the row as composed;   b(k + 1) = b(k) with its index address bytes replaced by pass k's greedy bytes there;
        stop at the first k with b(k + 1) = b(k), or after max_passes passes (default: the row's address byte count)
    Each address byte depends only on the bytes before it, so pass k fixes at least the k-th address byte for good: the fixed point is the
    sequential greedy writing (entry by entry, digit by digit) whatever the start, reached within the byte count's passes. Keys, records,
    questions and values stay as composed. greedy(ag_now) -> np.ndarray (len(ag_now), >= the index's end) the argmax at every input
    position; every pass sends ALL the rows (a fixed batch: a row's bytes never depend on which rows are still moving).
    -> (the rows at the fixed point [(row, meta)], passes taken, converged: per row, b(k + 1) = b(k) seen or the byte count reached)."""
    n_bytes = max(abytes_of(m) * len(m["index"]) for _, m in ag)
    limit = n_bytes if max_passes is None else int(max_passes)
    cur = [r for r, _ in ag]
    metas = [m for _, m in ag]
    done = [False] * len(ag)
    passes = 0
    while passes < limit and not all(done):
        pred = greedy(list(zip(cur, metas)))
        passes += 1
        for i in range(len(cur)):
            nxt = put_index_bytes(cur[i], metas[i], pred[i])
            done[i] = done[i] or nxt == cur[i]
            cur[i] = nxt
    conv = [d or passes >= abytes_of(m) * len(m["index"]) for d, m in zip(done, metas)]
    return list(zip(cur, metas)), passes, conv


def decode_scores(s, thr: float = 0.5) -> list:
    """0.1.72: THE STREAM DECODE of per-digit scores s (Q, LEVELS, RADIX) in [0, 1]: a level's digit is d when s(d) > thr for exactly one
    digit d, else the null anchor (each digit judged by its own score against one fixed number; no digit is compared with another).
    -> [(d1, d2)] per query."""
    s = np.asarray(s, dtype=np.float64)
    above = s > thr
    one = above.sum(-1) == 1
    dig = np.where(one, above.argmax(-1), NULL)
    return [tuple(int(v) for v in row) for row in dig]


def segments(meta: dict, ctx: int = CTX) -> np.ndarray:
    """The segment id (SEGMENTS) of every target byte (the byte at row position t + 1 for input position t)."""
    seg = np.full(ctx + 1, SEGMENTS.index("q_text"), dtype=np.int8)
    seg[:SLOT] = SEGMENTS.index("head")
    for r in meta["recs"]:
        seg[r["lo"]:r["hi"]] = SEGMENTS.index("comp_text")
        seg[r["lo"]:r["frame_end"]] = SEGMENTS.index("frame")
    seg[INDEX_LO:INDEX_HI] = SEGMENTS.index("index_text")
    seg[INDEX_LO:meta["index_end"]] = SEGMENTS.index("index_entry")
    seg[QUEST_LO:meta["quest_end"]] = SEGMENTS.index("q_other")
    for q in meta["questions"]:
        seg[q["v0"]:q["end"] + 1] = SEGMENTS.index("q_answer")
    return seg[1:]


def selftest(n: int = 200) -> None:
    """Every row's arithmetic: the frames, the index and the questions where the meta says, the streams' spans, the answers' targets."""
    rng = np.random.default_rng(1)
    src = bytes(rng.integers(32, 127, size=CTX + 1, dtype=np.uint8))
    for t in range(n):
        drawn = {}
        for ab in (2, 1):                               # 0.1.74: both address forms from the same draws
            r = np.random.default_rng([7, t])
            swap = t % 5 == 4
            asked = None if t % 3 else list(r.permutation(N_COMP)[:18])
            row, m = compose_ag(r, src, addrs=None, asked=asked, swap_index=swap, abytes=ab)
            assert len(row) == CTX + 1 and row[:len(HEAD)] == HEAD and abytes_of(m) == ab
            for rec in m["recs"]:
                assert row[rec["lo"]] == DATA and row[rec["lo"] + 1:rec["lo"] + 1 + ab] == addr_bytes(rec["addr"], ab)
                assert row[rec["v0"]:rec["v0"] + VLEN].decode() == rec["value"] and row[rec["v0"] + VLEN] == END
                assert row[rec["v0"] - 1] == SEP and row[rec["lo"] + 1 + ab] == CUE
            assert [rec["slot"] for rec in m["recs"]] == list(COMP_SLOTS) and m["addrs"] == sorted(m["addrs"])
            x, y = row[:-1], row[1:]
            for qi, (q, e) in enumerate(zip(m["questions"], m["index"])):
                rec = m["recs"][q["rec"]]
                assert row[q["lo"]] == CUE and row[q["sep"]] == SEP and row[q["v0"]:q["v0"] + VLEN].decode() == rec["value"]
                assert row[q["end"]] == END and row[q["lo"] + 1:q["sep"]].decode() == rec["key"]
                shown = m["recs"][e["shown"]]
                assert row[e["a_pos"]:e["a_pos"] + ab] == addr_bytes(shown["addr"], ab) and row[e["a_pos"] - 1] == SEP
                assert (e["shown"] == e["rec"]) or swap
                tg = [y[p] for p in answer_positions(m)[qi]]
                assert bytes(tg) == rec["value"].encode() + bytes([END]), "the answer positions' targets are the value and the END"
                ps = index_positions(m)[qi]
                assert len(ps) == ab and bytes(y[p] for p in ps) == addr_bytes(shown["addr"], ab)
                assert tuple(int(d) for d in decode_written([y[p] for p in ps], ab)) == digits(shown["addr"])
            if ab == 1:                                 # the address bytes are unique in the row, one byte each, in 0x80-0xBF
                ab_bytes = [row[rec["lo"] + 1] for rec in m["recs"]]
                assert len(set(ab_bytes)) == N_COMP and all(ONE_BYTE0 <= b < ONE_BYTE0 + N_ADDR for b in ab_bytes)
            dk, dq = streams(m)
            for rec in m["recs"]:
                assert (dk[rec["lo"]:rec["hi"]] == digits(rec["addr"])).all()
            for q in m["questions"]:
                sp = slice(q["sep"], q["sep"] + VLEN + 1)
                assert (dk[sp] == NULL).all() and (dq[sp] == digits(m["recs"][q["rec"]]["addr"])).all()
            assert (dk[:SLOT] == NULL).all() and (dk[INDEX_LO:INDEX_HI] == NULL).all()
            assert (dq[INDEX_LO:INDEX_HI] == NULL).all()
            seg = segments(m)
            assert seg.shape == (CTX,)
            assert len(m["questions"]) == (N_COMP if asked is None else 18)
            drawn[ab] = (m["addrs"], [rec["key"] for rec in m["recs"]], [rec["value"] for rec in m["recs"]],
                         [q["rec"] for q in m["questions"]], [e["shown"] for e in m["index"]], dk, dq)
        a2, a1 = drawn[2], drawn[1]
        assert all((np.array_equal(u, v) if isinstance(u, np.ndarray) else u == v) for u, v in zip(a2, a1)), "the two forms share every draw"
    own, never = owned_order()
    assert len(own) == 48 and len(never) == 16 and not set(own) & set(never)
    for t in range(n // 4):                             # 0.1.75: the key-hard rows, both address forms
        kh = hard_keys(np.random.default_rng([11, t]))
        parts = [(k[:-3], k[-3], k[-2], k[-1]) for k in kh]
        assert len(set(kh)) == N_COMP and len({p[0] for p in parts}) == 2 and len({p[1] for p in parts}) == 2
        assert len({p[2] for p in parts}) == 2 and len({p[3] for p in parts}) == 3
        for p in parts:                                 # a neighbour that differs in exactly one part, for each of the four parts
            for i in range(4):
                assert any(q[i] != p[i] and all(q[j] == p[j] for j in range(4) if j != i) for q in parts), (p, i)
        for ab in (2, 1):
            row0, m0 = compose_ag(np.random.default_rng([7, t]), src, abytes=ab)
            row1, m1 = compose_ag(np.random.default_rng([7, t]), src, abytes=ab, keys=kh)
            assert m1["hard"] and "hard" not in m0 and [r["key"].encode() for r in m1["recs"]] == kh
            same = lambda m: ([(r["addr"], r["slot"], r["value"]) for r in m["recs"]], [q["rec"] for q in m["questions"]],
                              [e["shown"] for e in m["index"]])
            assert same(m0) == same(m1), "a hard row differs from the standard row on its draw only in the keys"
            dk0, dq0 = streams(m0)
            dk1, dq1 = streams(m1)
            assert (dk0[:INDEX_HI] == dk1[:INDEX_HI]).all(), "the compartments' streams do not depend on the keys"
            for q in m1["questions"]:
                sp = slice(q["sep"], q["sep"] + VLEN + 1)
                assert (dq1[sp] == digits(m1["recs"][q["rec"]]["addr"])).all()
    print(f"grid_rows self-test: {n} rows x 2 address forms OK (frames, index, questions, streams, segments; the forms share every draw "
          f"and stream; 1 in 5 with a swapped index; 1 in 3 asking 18); {n // 4} key-hard rows x 2 forms OK (24 distinct keys, a one-part "
          f"neighbour for every part; the row's other draws the standard row's)")


if __name__ == "__main__" and "--selftest" in sys.argv:
    selftest()
