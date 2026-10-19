# -*- coding: utf-8 -*-
"""framed_diet.py - THE FRAMED DIET (0.1.58; Phil 2026-10-19 12:0x: a nano "specifically pretrained to understand frames, rather than just
documents", its frame behaviors not "lost over time like they are currently"; plans/2026-10-19_nano_todo_frame_pretrain_plan.md).

Every nano so far met frames in a continuation of 1.0 to 1.5 billion bytes after a 16.1 billion byte plain diet, and a later phase
displaced an earlier form (the caption phase moved the old flat-frame copy from 0.81 to 0.47). Here the frames enter the diet itself:
any library training stream is wrapped so that a fraction p of its rows is FRAMED, from the first step of the first phase to the last
step of the last, in four forms by a seeded draw, in the mixed phase's three registers:
  carry rows       a record planted in the first bytes, carried every 384 to 640 bytes through the text, the question at the far end
                   (frames_data.CARRY; the far habit)
  frames rows      sized, sliced system frames nested and overlapping in the text, any record asked (frames_data.FORM; the near read)
  caption windows  the caption pack's documents: the trie frames of a caption's systems, the frame withheld in half the documents, the
                   question asked (frames_data.CAPTION2; word meaning into records)
  note rows        a fact of the text itself framed as a note (frames_data.NOTE); a text with no usable fact falls back to its plain row
A framed row takes TWO of the stream's own rows as its text (the composers want twice a row of source); a plain row is the stream's
row verbatim. The row's coin comes from its own generator ([seed, i]), its kind and register from a second ([seed, i, 2]), its
composition from a third ([seed, i, 1]), the mixed phase's arrangement, so row i is the same whatever happened before it, and a resume
is exact: the state is the row counter, the kinds' counts and the inner stream's own state (the library's packed and mix states nest
inside). The wrapper is built in the training stream's child process (nano_v1.ProcessStream with framed=<form>); the phase's dataset
name stays the plain one, so the manifest, the validation stream (the fineweb holdout) and the boundary names match the control's.
The forms: FORMS["p50"] (half the rows framed, the plan's proposal), "p25" and "p75"; the kinds' shares and the registers are one
table (KINDS, the mixed phase's registers)."""
from __future__ import annotations

import numpy as np
import torch

from geolip.alephllm.data.special_tokens import DOC
from . import frames_data as FD

KINDS = (("carry", 0.35), ("frames", 0.25), ("caption", 0.25), ("note", 0.15))
FRAMED = dict(p_framed=0.5, kinds=KINDS, registers=FD.MIXED["registers"], carry=FD.CARRY, frames=FD.FORM, caption=FD.CAPTION2, note=FD.NOTE)
FORMS = {"p50": FRAMED, "p25": dict(FRAMED, p_framed=0.25), "p75": dict(FRAMED, p_framed=0.75)}
COUNTED = ("plain", "carry", "frames", "caption", "note", "note_plain")     # note_plain: a note row whose text held no usable fact (plain)
ROW_IDS = FD.ROW_BYTES + 1                                                    # the stream's row: context + 1 ids at the v1 context


def _pick(rng, table) -> str:
    u, acc = rng.random(), 0.0
    for k, w in table:
        acc += w
        if u < acc:
            return k
    return table[-1][0]


def framed_draw(rng, form: dict = FRAMED) -> tuple:
    """The kind and the register of one framed row, from the row's second generator ([seed, i, 2]). -> (kind, register)."""
    return _pick(rng, form["kinds"]), _pick(rng, form["registers"])


def framed_row(rng_kind, rng_comp, src: bytes, form: dict = FRAMED, pack_fn=None) -> tuple:
    """One framed row from `src` (two rows of the stream's text, 2 x ROW_BYTES). pack_fn: returns the caption pack (called for a caption
    row only). -> (ROW_BYTES bytes, the kind), or (None, "note_plain") when a note row found no usable fact."""
    assert len(src) >= 2 * FD.ROW_BYTES, len(src)
    kind, reg = framed_draw(rng_kind, form)
    if kind == "carry":
        row, _ = FD.carry_row(rng_comp, src, dict(form["carry"], register=reg), FD.ROW_CTX)
    elif kind == "frames":
        row, _ = FD.frame_row(rng_comp, src, dict(form["frames"], register=reg), FD.ROW_CTX)
    elif kind == "caption":
        from . import caption_pack as cp
        row, _ = FD.caption_window(rng_comp, pack_fn() if pack_fn else cp.load(), form["caption"], FD.ROW_CTX)
    else:
        nr = FD.note_row(rng_comp, src, form["note"], FD.ROW_CTX)
        if nr is None:
            return None, "note_plain"
        row = nr[0]
    assert len(row) == FD.ROW_BYTES, (kind, len(row))
    return row, kind


class FramedStream:
    """A library training stream with a fraction of its rows framed (the module's head). inner: the library stream opened at micro-batch
    1 (its rows are consumed one at a time: one for a plain row, two for a framed row); micro_batch: the rows a batch; seed: the phase's
    seed (the coins); form: one of FORMS; pack: the caption pack (loaded from NANO_CAPTION_PACK at the first caption row when None)."""

    def __init__(self, inner, micro_batch: int, seed: int, form: dict = FRAMED, pack=None):
        self.inner, self.micro_batch, self.seed, self.form = inner, int(micro_batch), int(seed), form
        self.dataset = inner.dataset
        self.i = 0
        self.counts = {k: 0 for k in COUNTED}
        self._pack = pack

    def _pack_fn(self):
        if self._pack is None:
            from . import caption_pack as cp
            self._pack = cp.load()
        return self._pack

    @property
    def rows_consumed(self):
        return self.inner.rows_consumed

    @property
    def epoch(self):
        return self.inner.epoch

    def _inner_row(self) -> np.ndarray:
        b = self.inner.next_batch()
        b = b.numpy() if hasattr(b, "numpy") else np.asarray(b)
        assert b.shape[0] == 1, f"the inner stream must run at micro-batch 1, not {b.shape[0]}"
        assert b.shape[1] == ROW_IDS, f"the framed diet runs at the v1 context ({FD.ROW_BYTES}); the stream's rows are {b.shape[1]} ids"
        return b[0]

    def row(self) -> tuple:
        """The next row (context + 1 ids) and its kind."""
        i = self.i
        self.i += 1
        if np.random.default_rng([self.seed, i]).random() >= self.form["p_framed"]:
            self.counts["plain"] += 1
            return self._inner_row(), "plain"
        a, b = self._inner_row(), self._inner_row()
        src = np.concatenate([a[:FD.ROW_BYTES], b[:FD.ROW_BYTES]]).astype(np.uint8).tobytes()
        row, kind = framed_row(np.random.default_rng([self.seed, i, 2]), np.random.default_rng([self.seed, i, 1]), src, self.form,
                               self._pack_fn)
        self.counts[kind] += 1
        if row is None:
            return a, kind
        ids = np.empty(ROW_IDS, dtype=np.int64)
        ids[:FD.ROW_BYTES] = np.frombuffer(row, dtype=np.uint8)
        ids[FD.ROW_BYTES] = DOC                             # the packer's document byte after the row, as the frames streams' rows carry it
        return ids, kind

    def next_batch(self) -> torch.Tensor:
        return torch.from_numpy(np.stack([self.row()[0] for _ in range(self.micro_batch)]).astype(np.int64))

    def state_dict(self) -> dict:
        return {"framed": True, "i": self.i, "counts": dict(self.counts), "inner": self.inner.state_dict()}

    def load_state_dict(self, st: dict):
        assert st.get("framed"), "not a framed stream's state"
        self.i = int(st["i"])
        self.counts = {k: int((st.get("counts") or {}).get(k, 0)) for k in COUNTED}
        self.inner.load_state_dict(st["inner"])


def share_table(counts: dict) -> str:
    """The kinds' counts as one line (the preflight and the manifest)."""
    n = sum(counts.values())
    return ", ".join(f"{k} {v:,} ({v / max(1, n):.1%})" for k, v in counts.items() if v) + f" of {n:,} rows"
