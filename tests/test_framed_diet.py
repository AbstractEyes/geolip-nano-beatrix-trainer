# -*- coding: utf-8 -*-
"""THE FRAMED DIET (0.1.58): the draw's shares; the framed stream over a fake inner stream (rows of the right shape, the kinds counted, two
rows consumed a framed row, a resume bit for bit); the stream in the trainer's child process; the v1.2 crafts (the control's plan, the
gauges, the version, the card); the marks' arithmetic."""
import os

import numpy as np
import pytest
import torch

from geolip.nano_beatrix import framed_diet as FDT
from geolip.nano_beatrix import frames_data as FD
from geolip.nano_beatrix import nano_v1
from geolip.nano_beatrix import caption_pack as cp
from geolip.alephllm.data.special_tokens import DOC
from test_caption import pack_dir  # noqa: F401  (the module's small caption pack)
from test_carry import corpus  # noqa: F401  (test_caption's fixture needs it)


class FakeInner:
    """Rows of readable text, row k distinct; micro-batch 1; state = the counter."""

    dataset = "fake-text"

    def __init__(self, k=0):
        self.k = k
        self.rows_consumed, self.epoch = k, 0
        words = ("The river bends east of the old mill where the miller kept nine goats and a ledger of debts. " * 60).encode()
        self._text = words

    def next_batch(self):
        k = self.k
        self.k += 1
        self.rows_consumed = self.k
        body = (f"Row {k}: the town of Alder has {k + 3} bridges and the mayor is Ann Pole. ".encode() + self._text)[:FDT.ROW_IDS]
        return torch.from_numpy(np.frombuffer(body, dtype=np.uint8).astype(np.int64)[None, :])

    def state_dict(self):
        return {"k": self.k}

    def load_state_dict(self, st):
        self.k = int(st["k"])
        self.rows_consumed = self.k


def test_the_draw_follows_the_table():
    rng = np.random.default_rng(3)
    kinds, regs = {}, {}
    for _ in range(20_000):
        k, r = FDT.framed_draw(rng)
        kinds[k] = kinds.get(k, 0) + 1
        regs[r] = regs.get(r, 0) + 1
    for k, w in FDT.KINDS:
        assert abs(kinds[k] / 20_000 - w) < 0.015, (k, kinds[k])
    for r, w in FD.MIXED["registers"]:
        assert abs(regs[r] / 20_000 - w) < 0.015, (r, regs[r])
    assert FDT.FORMS["p50"]["p_framed"] == 0.5 and FDT.FORMS["p25"]["p_framed"] == 0.25 and FDT.FORMS["p75"]["p_framed"] == 0.75
    assert FDT.FRAMED["caption"] is FD.CAPTION2 and FDT.FRAMED["carry"] is FD.CARRY and FDT.FRAMED["frames"] is FD.FORM and FDT.FRAMED["note"] is FD.NOTE


def test_the_framed_stream_rows_kinds_and_consumption(pack_dir):  # noqa: F811
    pack = cp.load(pack_dir)
    s = FDT.FramedStream(FakeInner(), 4, seed=5, form=FDT.FORMS["p50"], pack=pack)
    kinds = []
    for _ in range(10):
        b = s.next_batch()
        assert b.shape == (4, FDT.ROW_IDS) and b.dtype == torch.int64 and int(b.min()) >= 0 and int(b.max()) < 256
    rows = [s.row() for _ in range(60)]
    kinds = [k for _, k in rows]
    n_plain = sum(1 for k in kinds if k == "plain")
    framed = [(r, k) for r, k in rows if k != "plain"]
    assert 0 < n_plain < 60 and len(framed) > 10 and {k for _, k in framed} <= set(FDT.COUNTED) - {"plain"}
    assert sum(s.counts.values()) == 100 and s.counts["plain"] >= n_plain
    assert s.inner.k == s.counts["plain"] + 2 * (100 - s.counts["plain"])          # a framed row (the note fallback too) takes two rows
    for r, k in framed:
        if k == "note_plain":
            assert r[:4].tolist() == list(b"Row ")                                   # the first of its two rows, plain
        else:
            assert r[FD.ROW_BYTES] == DOC and r[0] == DOC and r[:4].tolist() != list(b"Row ")
    assert {k for _, k in framed} >= {"carry", "frames", "caption"}


def test_the_resume_is_exact(pack_dir):  # noqa: F811
    pack = cp.load(pack_dir)
    a = FDT.FramedStream(FakeInner(), 1, seed=9, pack=pack)
    straight = [a.row()[0] for _ in range(40)]
    b = FDT.FramedStream(FakeInner(), 1, seed=9, pack=pack)
    for _ in range(23):
        b.row()
    st = b.state_dict()
    assert st["framed"] and st["i"] == 23 and st["inner"] == {"k": b.inner.k}
    c = FDT.FramedStream(FakeInner(), 1, seed=9, pack=pack)
    c.load_state_dict(st)
    resumed = [c.row()[0] for _ in range(17)]
    assert all(np.array_equal(x, y) for x, y in zip(straight[23:], resumed)) and c.i == 40 and c.inner.k == a.inner.k
    assert c.counts == a.counts
    with pytest.raises(AssertionError):
        c.load_state_dict({"i": 3})


def test_the_stream_in_the_child_process(pack_dir, monkeypatch):  # noqa: F811
    monkeypatch.setenv(cp.PACK_ENV, str(pack_dir))
    ps = nano_v1.ProcessStream("synthetic", "byte-trigram", FD.ROW_BYTES, 2, seed=11, depth=2, framed="p75")
    try:
        b1, b2 = ps.next_batch(), ps.next_batch()
        st = ps.state_dict()
    finally:
        ps.close()
    assert b1.shape == b2.shape == (2, FDT.ROW_IDS) and st["framed"] and st["i"] == 4 and sum(st["counts"].values()) == 4
    assert st["counts"]["plain"] < 4 and "inner" in st
    ps2 = nano_v1.ProcessStream("synthetic", "byte-trigram", FD.ROW_BYTES, 2, seed=11, state=st, depth=2, framed="p75")
    try:
        b3 = ps2.next_batch()
    finally:
        ps2.close()
    ps3 = nano_v1.ProcessStream("synthetic", "byte-trigram", FD.ROW_BYTES, 2, seed=11, depth=2, framed="p75")
    try:
        ps3.next_batch(), ps3.next_batch()
        b3s = ps3.next_batch()
    finally:
        ps3.close()
    assert torch.equal(b3, b3s)                                                         # the resumed child = the straight child


def test_the_v12_crafts_their_plan_and_their_gauges():
    c = nano_v1.ALL["nano-pollux-dual-v1.2"]
    assert c["arm"] == "altd-both" and c["seed"] == 1 and c["framed"] == "p50" and c["train"] == nano_v1.V11_TRAIN and c["tag"] == "v12"
    assert nano_v1.ALL["nano-pollux-dual-v1.2-p25"]["framed"] == "p25" and nano_v1.ALL["nano-pollux-dual-v1.2-p75"]["framed"] == "p75"
    assert nano_v1.diet_of("nano-pollux-dual-v1.2") == nano_v1.diet_of("nano-pollux-dual-v1.1")      # the control's phases, names and bytes
    assert nano_v1.plan_steps("nano-pollux-dual-v1.2", nano_v1.TOKENS_PER_STEP) == 61_422 and nano_v1.start_step("nano-pollux-dual-v1.2") == 0
    assert nano_v1.version_of(["nano-pollux-dual-v1.2"]) == "v1.2" and nano_v1.version_of(["nano-pollux-dual-v1.1"]) == "v1.1"
    assert nano_v1.size_of("nano-pollux-dual-v1.2") == "nano" and nano_v1.body_of("nano-pollux-dual-v1.2") == "nano-pollux-dual-v1.2"
    assert nano_v1.arm_text("nano-pollux-dual-v1.2") is None and nano_v1.framed_of("nano-pollux-dual-v1.2") == "p50" and nano_v1.framed_of("nano-pollux-dual-v1.1") is None
    g = nano_v1.gauge_args("nano-pollux-dual-v1.2", "v12pd_100", "out")
    assert g["carry_gauge"] == "1" and g["note_gauge"] == "1" and "size" not in g
    assert "carry_gauge" not in nano_v1.gauge_args("nano-pollux-dual-v1.1", "v11pd_100", "out")
    assert nano_v1.gauge_init("nano-pollux-dual-v1.2", 100) == "v12pd_100"
    card = nano_v1.card_in_training(["nano-pollux-dual-v1.2"])
    assert "Castor and Pollux, v1.2" in card and "framed" in card and "colab/nano_v1_2.ipynb" in card and "One small" in card
    p = nano_v1.preset("nano-pollux-dual-v1.2", "birth.safetensors")
    assert p.train.decay_params == nano_v1.READOUT and [ph["name"] for ph in p.curriculum][:2] == ["warmup_wikitext", "fineweb_main"]


def test_the_marks_arithmetic():
    rows = [dict(phase="warmup_wikitext", step=1145, val_bpb=2.31, bed={"bed/frame/copy_whole": 0.10, "bed/carry/full_value": 0.0}, caption=None),
            dict(phase="fineweb_main", step=20219, val_bpb=1.30, bed={"bed/frame/copy_whole": 0.85, "bed/carry/full_value": 0.40}, caption=dict(extraction=0.30, prior=0.22, over_prior=0.08, copy=0.9)),
            dict(phase="anneal_mix", step=61422, val_bpb=1.31, bed={"bed/frame/copy_whole": 0.70, "bed/carry/full_value": 0.55}, caption=dict(extraction=0.45, prior=0.22, over_prior=0.23, copy=0.95))]
    ctrl = [dict(phase="warmup_wikitext", val_bpb=2.308), dict(phase="fineweb_main", val_bpb=1.2925), dict(phase="anneal_mix", val_bpb=1.3057)]
    m = nano_v1.v12_marks(rows, ctrl)
    copy = m["reads"]["bed/frame/copy_whole"]
    assert copy["peak"] == 0.85 and copy["end"] == 0.70 and copy["kept"] is False and copy["peak_phase"] == "fineweb_main"
    carry = m["reads"]["bed/carry/full_value"]
    assert carry["peak"] == carry["end"] == 0.55 and carry["kept"] is True
    assert m["reads"]["caption/over_prior"]["end"] == 0.23 and m["reads"]["caption/over_prior"]["kept"] is True
    assert m["text"]["anneal_mix"]["delta"] == pytest.approx(0.0043, abs=1e-6) and m["text"]["fineweb_main"]["quiet"] is True
    assert m["frames_kept"] is False and m["text_kept"] is True and m["meaning"] is True
    t = nano_v1.v12_table(rows, ctrl)
    assert t.count("\n") >= 4 and "| anneal_mix |" in t and "1.3057" in t and "0.23" in t
    assert os.path.basename(__file__) == "test_framed_diet.py"
