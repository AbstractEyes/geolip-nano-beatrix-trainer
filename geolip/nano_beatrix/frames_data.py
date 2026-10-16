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
from collections.abc import Mapping

import numpy as np

from . import bed_frames as bf

ROW_BYTES = 4096                       # one row = one 4,097-id block with the packer's document byte
ROW_CTX = ROW_BYTES - 1                # the bed's composer writes ctx + 1 bytes
CORPUS_ENV = "NANO_FRAMES_CORPUS"
# the frames phase's form: the bed's long-window stage with its confirmed picks (the sliced frame at 32 bytes, eight records a frame,
# the caption register), nesting 0.6 and a key defined twice in 0.3 of the rows, the question asked of any record
FORM = dict(register="caption", tags="mode", sizes=(32, 128, 256, 512, 1024), frame_records=8, nest=0.6, overlap=0.3, ask="any",
            sized=True, sliced=32)
P_FRAMED = 0.75                        # three rows in four framed, the bed's dose
FRAMES_MIX, CONTROL_MIX = "nano-frames-p75-mix", "nano-frames-p0-mix"


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


# ------------------------------------------------------------------ the corpus and the rows
_CORPUS: dict = {}


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


def row_bytes(packed: np.ndarray, seed: int, i: int, p_framed: float, form: dict = FORM) -> tuple:
    """Row i of the stream at `seed`: the position and the coin from the row's own generator ([seed, i]), the framing from a second one
    ([seed, i, 1]), so the position and the coin are the same whatever p_framed is. -> (ROW_BYTES bytes, framed?)."""
    rng = np.random.default_rng([seed, i])
    p = int(rng.integers(0, len(packed) - 2 * ROW_BYTES - 2))
    framed = bool(rng.random() < p_framed)
    if framed:
        row, _ = frame_row(np.random.default_rng([seed, i, 1]), packed[p:p + 2 * ROW_BYTES].tobytes(), form, ROW_CTX)
    else:
        row = packed[p:p + ROW_BYTES].tobytes()
    assert len(row) == ROW_BYTES, len(row)
    return row, framed


class _LazyRow(Mapping):
    """A generator row built on access: the stream reads row['ids']; a resume's skip only advances past it."""

    def __init__(self, packed, seed, i, p_framed):
        self._a = (packed, seed, i, p_framed)

    def __getitem__(self, k):
        if k != "ids":
            raise KeyError(k)
        row, _ = row_bytes(*self._a)
        return np.frombuffer(row, dtype=np.uint8).astype(np.int64)

    def __iter__(self):
        return iter(("ids",))

    def __len__(self):
        return 1


def rows(seed: int, p_framed: float):
    """The infinite generator a library stream opens (gen(seed)): row i at a time, built on access."""
    packed = load_corpus()
    i = 0
    while True:
        yield _LazyRow(packed, int(seed), i, p_framed)
        i += 1


def _rows_p75(seed):
    return rows(seed, P_FRAMED)


def _rows_p0(seed):
    return rows(seed, 0.0)


def register() -> None:
    """Add the generators and their mixes to the library's registries (names only added; an existing name is refused unless it is
    already ours)."""
    from geolip.alephllm.data import curriculum, streams
    for name, fn, mix in (("nano-frames-p75", _rows_p75, FRAMES_MIX), ("nano-frames-p0", _rows_p0, CONTROL_MIX)):
        gen = name.replace("-", "_")
        entry = dict(path=None, generator=gen, ids_rows=True)
        assert streams.REGISTRY.get(name, entry) == entry, f"{name} is already a different registry entry"
        assert streams._GENERATORS.get(gen, fn) is fn, f"{gen} is already a different generator"
        assert curriculum.CURRICULUM_MIXES.get(mix, [(name, 1.0)]) == [(name, 1.0)], f"{mix} is already a different mix"
        streams.REGISTRY[name] = entry
        streams._GENERATORS[gen] = fn
        curriculum.CURRICULUM_MIXES[mix] = [(name, 1.0)]


register()
