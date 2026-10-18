# -*- coding: utf-8 -*-
"""frames_data.py - the nano bed's framed rows as a library data stream (0.1.35), for a phase of the library's curriculum.

A framed row is the bed's dynamic row (bed_frames.compose_dynamic): the register tag, one to three frames of lookup records and
thinking filler, the question and the value. The row plan is bed_train.draw_plan / frame_row with its settings passed in (the bed's
module reads them from its command line); tests/test_frames_continuation.py checks the two produce the same rows.

The generator rows: every row is drawn from a local text corpus (bed_corpus.py's files) at a position drawn first from the row's own
seed, then a coin decides framed (probability p_framed) or plain; a framed row composes from the text at that position (its filler),
a plain row is the text itself. Two streams that differ only in p_framed therefore read the very same text, row for row: the frames
arm and its control. Every row is exactly ROW_BYTES = 4,096 bytes, so with the packer's document byte after it one row fills one
4,097-id block (context 4,096): no frame is ever cut across two blocks. Rows are built on access (a resumed stream skips the rows it
already consumed without composing them).

Registered names (added, nothing existing changed): the generators nano-frames-p75 / nano-frames-p0 and the mixes of the same names
with "-mix" appended (a curriculum mix validates on the fineweb-edu holdout, the same held-out text as every other phase). The corpus
folder reaches the stream's child process through the environment variable NANO_FRAMES_CORPUS."""
from __future__ import annotations

import os
import re
from collections.abc import Mapping

import numpy as np

from . import bed_frames as bf
from geolip.alephllm.data.special_tokens import CUE, DOC, END, SEP

ROW_BYTES = 4096                       # one row = one 4,097-id block with the packer's document byte
ROW_CTX = ROW_BYTES - 1                # the bed's composer writes ctx + 1 bytes
CORPUS_ENV = "NANO_FRAMES_CORPUS"
# the frames phase's form: the bed's long-window stage with its confirmed picks (the sliced frame at 32 bytes, eight records a frame,
# the caption register), nesting 0.6 and a key defined twice in 0.3 of the rows, the question asked of any record
FORM = dict(register="caption", tags="mode", sizes=(32, 128, 256, 512, 1024), frame_records=8, nest=0.6, overlap=0.3, ask="any",
            sized=True, sliced=32)
P_FRAMED = 0.75                        # three rows in four framed, the bed's dose
FRAMES_MIX, CONTROL_MIX = "nano-frames-p75-mix", "nano-frames-p0-mix"
# 0.1.37: THE CARRY ROW, the self-written frames prototype (form A, first instance; plans/2026-10-17_self_frames_prototype_plan.md):
# the register tag, ONE lookup frame of 32 or 64 bytes (one or two records) within the first early_gap bytes, then plain text with
# CARRIES on the 64-byte grid every carry_every bytes, each a copy of the early frame (the same records in the same order), the
# question in the last `late` bytes asking any early record, the value, END, filler. The asked record's LATEST definition is its last
# carry, so the question's distance to it stays within the reach the v1.1 crafts showed in the frame form (about a thousand bytes).
# carries=False builds the gauge's row: the same early frame and question with no carries, where the model writes its own
# (bed_train's carry gauge: reading with notes on the grid).
CARRY = dict(register="caption", tags="mode", early_sizes=(32, 64), early_gap=64, grid=64, carry_every=(384, 640), late=512,
             sized=True, sliced=32)
CARRY_MIX = "nano-frames-carry-mix"


# ------------------------------------------------------------------ the row plan (bed_train.draw_plan / frame_row, settings passed in)
def rec_bytes(form: dict = FORM, ctx: int = ROW_CTX) -> int:
    return bf.capacity(ctx, "A", form["register"])["record_bytes"]


def _cap(size: int, form: dict, rb: int) -> int:
    return bf.frame_capacity(size, rb, form["sized"], form["sliced"])


def draw_plan(rng, form: dict = FORM, ctx: int = ROW_CTX) -> tuple:
    """One row's dynamic plan (bed_train.draw_plan): frames of the form's sizes while they fit the window, lookup frames with as many
    records as fit up to frame_records, a thinking frame nesting a quarter-size lookup frame with probability nest, an overlapping pair
    in conflict with probability overlap. -> (specs, nested, overlap)."""
    rb = rec_bytes(form, ctx)
    sizes, sliced = list(form["sizes"]), form["sliced"]
    budget = ctx + 1 - 6 - 64 - (rb + 2)
    specs, used, nested, overlap = [], 0, False, False
    if form["overlap"] and rng.random() < form["overlap"]:
        sz = sizes[0]
        specs += bf.overlap_specs(rng, form["register"], sz, sz, max(1, min(2, _cap(sz, form, rb))), "conflict")
        used += 2 * (bf.frame_bytes(sz, sliced) + 64)
        overlap = True
    for _ in range(int(rng.integers(1, 4))):
        size = int(sizes[int(rng.integers(len(sizes)))])
        if used + bf.frame_bytes(size, sliced) + 64 > budget:
            continue
        if rng.random() < 0.7:
            n = min(form["frame_records"], _cap(size, form, rb))
            specs.append(bf.spec("data", size, records=n) if n >= 1 else bf.spec("think", size))
        else:
            kids = []
            csize = max(32, size // 4)
            if rng.random() < form["nest"] and _cap(csize, form, rb) >= 1 and csize <= size - 4:
                kids = [bf.spec("data", csize, records=min(form["frame_records"], _cap(csize, form, rb)))]
                nested = True
            specs.append(bf.spec("think", size, children=kids))
        used += bf.frame_bytes(size, sliced) + 64
    if not any(sp["kind"] == "data" or sp["children"] for sp in specs):       # nothing to ask: the smallest lookup frame
        sz = sizes[0]
        specs.append(bf.spec("data", sz, records=max(1, min(form["frame_records"], _cap(sz, form, rb)))))
    return specs, nested, overlap


def frame_row(rng, src: bytes, form: dict = FORM, ctx: int = ROW_CTX) -> tuple:
    """A training row of the dynamic frames asking any record (bed_train.frame_row with ask any). -> (row bytes of ctx + 1, meta)."""
    assert form["ask"] == "any", "the frames phase asks any record (the strict questions are the gauge's)"
    specs, _, _ = draw_plan(rng, form, ctx)
    return bf.compose_dynamic(rng, src, ctx, specs, form["register"], form["tags"], "any", None,
                              sized=form["sized"], sliced=form["sliced"])


# ------------------------------------------------------------------ the carry rows (0.1.37)
def carry_plan(rng, form: dict = CARRY, ctx: int = ROW_CTX) -> tuple:
    """The row's draws before a byte is written, so the row with and without its carries shares its early frame, its carry places and
    its question: (the early frame's size, the gap before it, the carry positions on the grid, the question's offset in the late region,
    the asked record's index)."""
    size = int(form["early_sizes"][int(rng.integers(len(form["early_sizes"])))])
    gap = int(rng.integers(0, form["early_gap"] + 1))
    g, lo, hi = form["grid"], form["carry_every"][0] // form["grid"], form["carry_every"][1] // form["grid"]
    cb = bf.frame_bytes(size, form["sliced"])                       # a carry is the early frame's length
    head = 6 + gap + cb                                              # DOC, the tag, SEP q SEP, the gap, the early frame
    q_lo = ctx + 1 - form["late"]
    positions, pos = [], head
    while True:
        pos = -(-pos // g) * g + int(rng.integers(lo, hi + 1)) * g   # the next grid point at or past pos, then the drawn interval
        if pos + cb + g > q_lo:                                      # a carry ends at least one grid step before the late region
            break
        positions.append(pos)
        pos += cb
    rb = bf.capacity(ctx, "A", form["register"])["record_bytes"]
    q_off = int(rng.integers(g, form["late"] - (rb + 2) - 1))
    n = max(1, bf.frame_capacity(size, rb, form["sized"], form["sliced"]))
    return size, gap, positions, q_off, int(rng.integers(n))


def carry_row(rng, src: bytes, form: dict = CARRY, ctx: int = ROW_CTX, carries: bool = True) -> tuple:
    """THE CARRY ROW (the module's head). -> (row of ctx + 1 bytes, meta): compose_dynamic's fields (v0, vlen, dist to the latest
    definition, key, truth_frame, truth_depth, n_defs, source, frames, scheme, register, ctx, q0, sized, sliced) and the carry form's:
    early_size, early_start, early_end, early_keys, carries (the positions written), planned (the positions drawn), grid."""
    fill = bf.Filler(src, int(rng.integers(0, max(1, len(src) - 1))))
    rb = bf.capacity(ctx, "A", form["register"])["record_bytes"]
    size, gap, positions, q_off, ask_i = carry_plan(rng, form, ctx)
    n = max(1, bf.frame_capacity(size, rb, form["sized"], form["sliced"]))
    state = dict(frames=[], defs={}, facts=None, key_iter=iter(rng.permutation(16000).tolist()), fact_iter=None, sized=form["sized"],
                 sliced=form["sliced"])
    pad = bf.Filler(b" " * 4096, 0)                                  # the frames' slack is spaces: a carry is the early frame byte for byte
    out = bytearray(bytes([DOC]) + bf.tag_bytes(form["register"], form["tags"]) + bytes([SEP]) + b"q" + bytes([SEP]))
    out += fill.take(gap)
    early_start = len(out)
    out += bf.build_frame(rng, bf.spec("data", size, records=n), len(out), pad, form["register"], state)
    early_end = len(out)
    recs = sorted((d[0][0], k, d[0][1]) for k, d in state["defs"].items())      # the early records in the order they were written
    keys, vals = [k for _, k, _ in recs], [v for _, _, v in recs]
    assert len(keys) == n, (len(keys), n)
    written = []
    if carries:
        for pos in positions:
            out += fill.take(pos - len(out))
            out += bf.build_frame(rng, bf.spec("data", size, records=n, keys=keys, values=vals), len(out), pad, form["register"], state)
            written.append(pos)
    q0 = ctx + 1 - form["late"] + q_off
    out += fill.take(q0 - len(out))
    key = keys[ask_i]
    truth = state["defs"][key][-1]                                   # the latest definition: the last carry when carries were written
    val = truth[1]
    out += bytes([CUE]) + key + bytes([SEP])
    v0 = len(out)
    out += val + bytes([END])
    tail = ctx + 1 - len(out)
    assert tail >= 0, f"the carry row overflows the window by {-tail} bytes"
    out += fill.take(tail)
    row = bytes(out)
    assert len(row) == ctx + 1 and row[v0:v0 + len(val)] == val
    meta = dict(v0=v0, vlen=len(val), dist=v0 - truth[0], key=key.decode(), truth_frame=truth[2], truth_depth=truth[3],
                n_defs=len(state["defs"][key]), source="frame", frames=state["frames"], scheme=form["tags"], register=form["register"],
                ctx=ctx, q0=q0, sized=True, sliced=int(form["sliced"]), early_size=size, early_start=early_start, early_end=early_end,
                early_keys=[k.decode() for k in keys], early_vals=[v.decode() for v in vals], carries=written, planned=positions, grid=form["grid"])
    return row, meta


PLAIN_START = 128    # 0.1.39: the plain-row read's first grid point (the carry rows' early frames end between 41 and 137 bytes: their first read is at 64, 128 or 192)


def plain_row(rng, src: bytes, ctx: int = ROW_CTX) -> bytes:
    """THE PLAIN ROW (0.1.39, the gate screen's premise): DOC and ctx bytes of the source taken as the carry row takes its text (bf.Filler at a
    drawn offset), no early frame, no carries, no question; the gauge reads it on the grid from PLAIN_START over the carry rows' span and
    records what the model opens where no question follows."""
    fill = bf.Filler(src, int(rng.integers(0, max(1, len(src) - 1))))
    row = bytes([DOC]) + fill.take(ctx)
    assert len(row) == ctx + 1
    return row


def insert_carry(row: bytearray, pos: int, gen: bytes, ctx: int = ROW_CTX) -> None:
    """The gauge's insertion: the model's carry `gen` enters the row at `pos`, and the row's tail is cut by the same length, so the
    window stays ctx + 1 bytes and the question moves right by len(gen) (the caller keeps the insertions within the tail's filler)."""
    row[pos:pos] = gen
    del row[ctx + 1:]
    assert len(row) == ctx + 1


# ------------------------------------------------------------------ the note rows (0.1.41)
# THE NOTE ROW (form A, second instance; plans/2026-10-18_note_row_plan.md): the frame's content is a fact the text itself states.
# A deterministic extractor (extract_facts) finds definitional facts in the row's own text: a definition (X is a Y), an abbreviation
# (Full Name (ABBR): the abbreviation asked, the name answered), a quantity, a capital, a place, the founding year of an institution.
# No pattern reads a person's name with a birth, an age or an address: the facts are about terms, places, quantities and institutions.
# The composer writes ONE 32-byte lookup frame holding the fact (key: value) at the first grid point after the sentence that states it,
# carries restating it on the grid every carry_every bytes, and the question in the last `late` bytes asking the key. The question's
# distance to the fact follows where the fact falls in the text, so one data form trains the near and the far read. notes=False builds
# the gauge's SELF row (no note, no carries: the body must notice the fact and write it); notes=True, carries=False the PLANTED row (the
# note given, the carries the body's own). A text with no usable fact gives None (the stream's row stays plain).
NOTE = dict(register="caption", tags="mode", size=32, grid=64, carry_every=(384, 640), late=512, sized=True, sliced=32, min_pos=128,
            key_max=20, val_max=20, rec_max=27)
NOTE_MIX = "nano-frames-note-mix"
# 0.1.49: THE MIXED PHASE (plans/2026-10-18_mixed_phase_arm_plan.md). The frames-then-carry arm lost the near read in the carry phase's
# first 0.2B bytes (the data form, not the habit), and the register fade showed that a register's lookup survives only if that register
# is in the mix. One phase holds both forms and the three registers: a framed row (the bed's dose; the same coin and text position as
# every other arm's row i) is a carry row two times in three and a frames row once, its register drawn caption one time in two, math and
# code one in four each (the value alphabets: letters, digits, hex); the plain rows stay. A THIRD generator ([seed, i, 2]) draws the kind
# and the register, so the composer's own generator is untouched: a caption carry row here is the carry arm's row i byte for byte, a
# caption frames row the frames arm's. The phase is 1.5 billion bytes (nano_v1.CONT_TOKENS_OF), so its carry rows total the carry
# arm's 0.75 billion (the far habit's dose kept) while the frames rows add 0.375 billion.
# 0.1.50: p_doc_start. The first-64 read (journal 2026-10-18 14:4x) found the framed streams' head tax to be the opening: a plain row is
# a slice taken mid-document and never begins with DOC, so a window-initial DOC announced a frame with probability 1.0 and a plain
# document's first byte cost 17 to 22 bits against 5 to 6 for the controls. Half of the mixed phase's plain rows begin at a document
# boundary (DOC + the document from its first byte); the other half keep the mid-document start the diet's windows have.
MIXED = dict(p_carry=2 / 3, registers=(("caption", 0.5), ("math", 0.25), ("code", 0.25)), carry=CARRY, frames=FORM, p_doc_start=0.5)
MIXED_MIX = "nano-frames-mixed-mix"
_TERM = r"([A-Z][a-z]{2,}(?: (?:[A-Z][a-z]{2,}|[a-z]{2,})){0,2})"
_UNITS = (r"km|kilometres|kilometers|miles|metres|meters|feet|kg|kilograms|tons|tonnes|percent|%|acres|hectares|million|billion|years"
          r"|days|degrees|species|members|students|employees")
_NOTE_PATTERNS = (
    ("def", re.compile(_TERM + r" (?:is|are) (?:a|an|the) ([a-z]{3,})(?=[ ,.;:])")),
    ("abbr", re.compile(r"([A-Z][a-z]{2,}(?: [A-Z][a-z]{2,}){1,3}) \(([A-Z]{2,6})\)")),
    ("qty", re.compile(_TERM + r" (?:is|was|measures|covers|spans|weighs|has) (?:about |approximately |roughly |nearly |over |around |some )?"
                       r"(\d[\d,]*(?:\.\d+)?(?: ?(?:" + _UNITS + r"))?)\b")),
    ("capital", re.compile(r"[Tt]he capital (?:city )?of ([A-Z][a-z]{2,}(?: [A-Z][a-z]{2,})?) is ([A-Z][a-z]{2,}(?: [A-Z][a-z]{2,})?)\b")),
    ("where", re.compile(_TERM + r" is (?:located|situated|found) in (?:the )?([A-Z][a-z]{2,}(?: [A-Z][a-z]{2,})?)\b")),
    ("year", re.compile(_TERM + r" was (?:founded|established|built|completed|published|invented|discovered|opened|launched|introduced|created|formed) in (\d{4})\b")),
)
_NOTE_STOP = frozenset(("The This That These Those There Here It Its In On At For From With Into Over Under After Before During Since Until "
                        "While However When Where Which What Who Whom Whose Why How If But And Also Although Because Though Thus Therefore "
                        "Instead Perhaps Maybe Still Even Only Both All Any Each Every Some Many Most Other Another Such Much More Less New Old "
                        "One Two Three First Second Third Finally Next Then Now Today Yesterday Tomorrow Our Your Their His Her My We You They "
                        "He She As So Yes No Not Please Note See Use Like Unlike Once Again Often Sometimes Usually Always Never Above Below "
                        "Following According Despite Without Within Among Between Through Across Around About Just Well Indeed Rather "
                        "Nevertheless Meanwhile Moreover Furthermore Nonetheless Otherwise Yet Unless Whether Whereas").split())


def extract_facts(text: bytes, form: dict = NOTE) -> list:
    """The row's facts as a deterministic function of its text: [(end, key, value, kind)] in text order, one per key; end = the byte
    after the part of the sentence that states the fact. A leading stop word (a sentence starter, a pronoun) is dropped from the term;
    keys and values are ASCII letters, digits, spaces and a few marks within the frame's record room (key_max, val_max, rec_max)."""
    s = text.decode("latin-1")
    out, seen = [], set()
    for kind, pat in _NOTE_PATTERNS:
        for m in pat.finditer(s):
            words = m.group(1).strip().split(" ")
            while words and words[0] in _NOTE_STOP:                    # a leading stop word is not part of the term
                words.pop(0)
            term, other = " ".join(words), m.group(2).strip()
            key, val = (other, term) if kind == "abbr" else (term, other)   # an abbreviation is asked, its name answered
            if not key or not val or key in seen or key in _NOTE_STOP:
                continue
            if not (2 <= len(key) <= form["key_max"] and 1 <= len(val) <= form["val_max"] and len(key) + len(val) <= form["rec_max"]):
                continue
            if not all(c.isascii() and (c.isalnum() or c in " ,.%") for c in key + val):
                continue
            seen.add(key)
            out.append((m.end(), key.encode(), val.encode(), kind))
    out.sort(key=lambda t: t[0])
    return out


def note_bytes(key: bytes, val: bytes, form: dict = NOTE) -> bytes:
    """The note as the composer writes it: the sized lookup frame of form["size"] bytes holding the one record, its slack spaces
    (deterministic given the record; the exactness reference of the gauge's two note modes)."""
    state = dict(frames=[], defs={}, facts=None, key_iter=iter(()), fact_iter=None, sized=form["sized"], sliced=form["sliced"])
    pad = bf.Filler(b" " * 4096, 0)
    return bf.build_frame(np.random.default_rng(0), bf.spec("data", form["size"], records=1, keys=[key], values=[val]), 0, pad,
                          form["register"], state)


def note_row(rng, src: bytes, form: dict = NOTE, ctx: int = ROW_CTX, notes: bool = True, carries: bool = True):
    """THE NOTE ROW (the section's head). -> (row of ctx + 1 bytes, meta) or None when the text holds no usable fact. meta: the carry
    row's fields (v0, vlen, dist to the latest definition, key, q0, early_start / early_end = the note's span or None, early_keys,
    early_vals, carries, planned, grid, ...) and the note form's: fact_end (the byte after the fact in the row), fact_kind, note_pos
    (the grid point the note takes), fact_dist (the question's distance to the fact), note_bytes (the note as written), notes."""
    fill = bf.Filler(src, int(rng.integers(0, max(1, len(src) - 1))))
    head = bytes([DOC]) + bf.tag_bytes(form["register"], form["tags"]) + bytes([SEP]) + b"q" + bytes([SEP])
    text = fill.take(ctx)
    g, nb, H = form["grid"], bf.frame_bytes(form["size"], form["sliced"]), len(head)
    q_lo = ctx + 1 - form["late"]
    usable = [f for f in extract_facts(text, form) if H + f[0] >= form["min_pos"] and -(-(H + f[0]) // g) * g + nb + g <= q_lo]
    if not usable:
        return None
    e, key, val, kind = usable[int(rng.integers(len(usable)))]
    fact_end = H + e
    note_pos = -(-fact_end // g) * g
    qlen = 1 + len(key) + 1 + len(val) + 1                             # CUE key SEP value END: the real record's own length (0.1.42)
    q0 = q_lo + int(rng.integers(g, form["late"] - qlen - 1))        # the question and its value end at least two bytes inside the window
    lo, hi = form["carry_every"][0] // g, form["carry_every"][1] // g
    positions, pos = [], note_pos + nb
    while True:
        pos = -(-pos // g) * g + int(rng.integers(lo, hi + 1)) * g
        if pos + nb + g > q_lo:
            break
        positions.append(pos)
        pos += nb
    state = dict(frames=[], defs={}, facts=None, key_iter=iter(()), fact_iter=None, sized=form["sized"], sliced=form["sliced"])
    pad = bf.Filler(b" " * 4096, 0)
    out, cur = bytearray(head), [0]

    def text_to(target: int) -> None:
        need = target - len(out)
        assert need >= 0, (target, len(out))
        out.extend(text[cur[0]:cur[0] + need])
        cur[0] += need

    note_start = note_end = None
    written = []
    if notes:
        text_to(note_pos)
        note_start = len(out)
        out += bf.build_frame(rng, bf.spec("data", form["size"], records=1, keys=[key], values=[val]), len(out), pad, form["register"], state)
        note_end = len(out)
        if carries:
            for p_ in positions:
                text_to(p_)
                out += bf.build_frame(rng, bf.spec("data", form["size"], records=1, keys=[key], values=[val]), len(out), pad, form["register"], state)
                written.append(p_)
    text_to(q0)
    out += bytes([CUE]) + key + bytes([SEP])
    v0 = len(out)
    out += val + bytes([END])
    text_to(ctx + 1)
    row = bytes(out)
    assert len(row) == ctx + 1 and row[v0:v0 + len(val)] == val
    latest = written[-1] if written else (note_start if notes else fact_end)
    meta = dict(v0=v0, vlen=len(val), dist=v0 - latest, key=key.decode(), truth_frame=None, truth_depth=0, n_defs=(1 + len(written)) if notes else 0,
                source="frame", frames=state["frames"], scheme=form["tags"], register=form["register"], ctx=ctx, q0=q0, sized=True,
                sliced=int(form["sliced"]), early_size=form["size"], early_start=note_start, early_end=note_end, early_keys=[key.decode()],
                early_vals=[val.decode()], carries=written, planned=positions, grid=g, fact_end=fact_end, fact_kind=kind, note_pos=note_pos,
                fact_dist=q0 - fact_end, note_bytes=note_bytes(key, val, form), notes=notes)
    return row, meta


# ------------------------------------------------------------------ the corpus and the rows
_CORPUS: dict = {}


_DOC_STARTS = {}


def doc_start_row(packed: np.ndarray, p: int) -> bytes:
    """0.1.50: the plain row that begins at the first document boundary at or after p (the packer's DOC byte, then the next document
    from its first byte); the row at p itself when no whole row fits after that boundary."""
    key = id(packed)
    if key not in _DOC_STARTS:
        _DOC_STARTS[key] = np.flatnonzero(packed == DOC)
    starts = _DOC_STARTS[key]
    j = int(np.searchsorted(starts, p))
    if j < len(starts) and int(starts[j]) + ROW_BYTES <= len(packed):
        d = int(starts[j])
        return packed[d:d + ROW_BYTES].tobytes()
    return packed[p:p + ROW_BYTES].tobytes()


def mixed_draw(rng, form: dict = MIXED) -> tuple:
    """0.1.49: the kind and the register of one framed row of the mixed phase, from the row's third generator ([seed, i, 2]).
    -> ('carry' | 'frames', register)."""
    kind = "carry" if rng.random() < form["p_carry"] else "frames"
    u, acc = rng.random(), 0.0
    for reg, w in form["registers"]:
        acc += w
        if u < acc:
            return kind, reg
    return kind, form["registers"][-1][0]


def load_corpus(folder: str | None = None) -> np.ndarray:
    """The corpus text with the document byte after every document (as the packer joins documents), cached per folder."""
    from geolip.alephllm.data.special_tokens import DOC
    folder = folder or os.environ.get(CORPUS_ENV, "")
    assert folder and os.path.exists(os.path.join(folder, "bed_corpus.bin")), \
        f"the frames corpus is missing: set {CORPUS_ENV} to a folder holding bed_corpus.bin (bed_corpus.build)"
    if folder not in _CORPUS:
        corpus = np.fromfile(os.path.join(folder, "bed_corpus.bin"), dtype=np.uint8)
        offs = np.load(os.path.join(folder, "bed_corpus_offsets.npy"))
        _CORPUS[folder] = np.insert(corpus, offs[1:], np.uint8(DOC))
    return _CORPUS[folder]


def row_bytes(packed: np.ndarray, seed: int, i: int, p_framed: float, form: dict = FORM, carry: bool = False) -> tuple:
    """Row i of the stream at `seed`: the position and the coin from the row's own generator ([seed, i]), the framing from a second one
    ([seed, i, 1]), so the position and the coin are the same whatever p_framed is (and whatever the framed form: carry=True composes
    the carry row of `form` in place of the frames row, 0.1.37). -> (ROW_BYTES bytes, framed?)."""
    rng = np.random.default_rng([seed, i])
    p = int(rng.integers(0, len(packed) - 2 * ROW_BYTES - 2))
    framed = bool(rng.random() < p_framed)
    if framed and carry == "note":                                 # 0.1.41: the note rows; a text with no usable fact stays plain
        nr = note_row(np.random.default_rng([seed, i, 1]), packed[p:p + 2 * ROW_BYTES].tobytes(), form, ROW_CTX)
        row, framed = (nr[0], True) if nr else (packed[p:p + ROW_BYTES].tobytes(), False)
    elif framed and carry == "mixed":                              # 0.1.49: the mixed phase: carry rows and frames rows, three registers
        kind, reg = mixed_draw(np.random.default_rng([seed, i, 2]), form)
        src, rng2 = packed[p:p + 2 * ROW_BYTES].tobytes(), np.random.default_rng([seed, i, 1])
        if kind == "carry":
            row, _ = carry_row(rng2, src, dict(form["carry"], register=reg), ROW_CTX)
        else:
            row, _ = frame_row(rng2, src, dict(form["frames"], register=reg), ROW_CTX)
    elif framed and carry:
        row, _ = carry_row(np.random.default_rng([seed, i, 1]), packed[p:p + 2 * ROW_BYTES].tobytes(), form, ROW_CTX)
    elif framed:
        row, _ = frame_row(np.random.default_rng([seed, i, 1]), packed[p:p + 2 * ROW_BYTES].tobytes(), form, ROW_CTX)
    else:
        row = packed[p:p + ROW_BYTES].tobytes()
        if carry == "mixed" and form.get("p_doc_start") and np.random.default_rng([seed, i, 2]).random() < form["p_doc_start"]:
            row = doc_start_row(packed, p)                         # 0.1.50: the mixed phase's document-start plain rows (the third generator's coin)
    assert len(row) == ROW_BYTES, len(row)
    return row, framed


class _LazyRow(Mapping):
    """A generator row built on access: the stream reads row['ids']; a resume's skip only advances past it."""

    def __init__(self, packed, seed, i, p_framed, form=FORM, carry=False):
        self._a = (packed, seed, i, p_framed, form, carry)

    def __getitem__(self, k):
        if k != "ids":
            raise KeyError(k)
        row, _ = row_bytes(*self._a)
        return np.frombuffer(row, dtype=np.uint8).astype(np.int64)

    def __iter__(self):
        return iter(("ids",))

    def __len__(self):
        return 1


def rows(seed: int, p_framed: float, form: dict = FORM, carry: bool = False):
    """The infinite generator a library stream opens (gen(seed)): row i at a time, built on access."""
    packed = load_corpus()
    i = 0
    while True:
        yield _LazyRow(packed, int(seed), i, p_framed, form, carry)
        i += 1


def _rows_p75(seed):
    return rows(seed, P_FRAMED)


def _rows_p0(seed):
    return rows(seed, 0.0)


def _rows_carry(seed):
    return rows(seed, P_FRAMED, CARRY, True)        # 0.1.37: the carry rows at the frames dose, the same text row for row


def _rows_note(seed):
    return rows(seed, P_FRAMED, NOTE, "note")       # 0.1.41: the note rows at the frames dose (a text without a usable fact stays plain)


def _rows_mixed(seed):
    return rows(seed, P_FRAMED, MIXED, "mixed")     # 0.1.49: the mixed phase at the frames dose


def register() -> None:
    """Add the generators and their mixes to the library's registries (names only added; an existing name is refused unless it is
    already ours)."""
    from geolip.alephllm.data import curriculum, streams
    for name, fn, mix in (("nano-frames-p75", _rows_p75, FRAMES_MIX), ("nano-frames-p0", _rows_p0, CONTROL_MIX),
                          ("nano-frames-carry", _rows_carry, CARRY_MIX), ("nano-frames-note", _rows_note, NOTE_MIX),
                          ("nano-frames-mixed", _rows_mixed, MIXED_MIX)):
        gen = name.replace("-", "_")
        entry = dict(path=None, generator=gen, ids_rows=True)
        assert streams.REGISTRY.get(name, entry) == entry, f"{name} is already a different registry entry"
        assert streams._GENERATORS.get(gen, fn) is fn, f"{gen} is already a different generator"
        assert curriculum.CURRICULUM_MIXES.get(mix, [(name, 1.0)]) == [(name, 1.0)], f"{mix} is already a different mix"
        streams.REGISTRY[name] = entry
        streams._GENERATORS[gen] = fn
        curriculum.CURRICULUM_MIXES[mix] = [(name, 1.0)]


register()

def next_grid(pos: int, length: int, grid: int = CARRY["grid"]) -> int:
    """The first grid point at or after an insertion of `length` bytes at `pos` (0.1.38): every training carry sits on the row's
    64-byte grid as the row stands with its earlier carries in it, so the reading-with-notes gauge reads the grid of the row as it now
    stands after writing its own carry (0.1.37 stepped off the grid by the carry's length and never opened a second carry)."""
    return -(-(pos + length) // grid) * grid
