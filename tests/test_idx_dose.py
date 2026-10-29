# -*- coding: utf-8 -*-
"""0.1.101: the index entries' dose (--idx_dose). The term reads the cross-entropy at each index entry's SEP + levels - 1, the input
position whose target is the entry's last address byte (the fourth digit at 4x4): on bed_train's own batches (AG48 and AG96 at 4x4, AG at
8x2, key-hard rows included) every such target is the digit byte of the address of the record that holds the entry's key, and the step
reads exactly those positions. The names carry _id<100 x lambda> after the address head's piece (off names nothing) and the runner passes
the dose. CPU only; the Hub is never touched."""
import os

import pytest
import torch

from geolip.nano_beatrix import bed_runner as br
from geolip.nano_beatrix import grid_rows as gr
from geolip.nano_beatrix.names import run_tag
from test_scale import _bed_batch

SRC = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "geolip", "nano_beatrix", "bed_train.py"), encoding="utf-8").read()


@pytest.mark.parametrize("layout,code", [("AG48", "4x4"), ("AG96", "4x4"), ("AG", "8x2")])
def test_the_dose_reads_each_entrys_last_address_byte(layout, code):
    r, n = gr.CODES[code]
    batch = _bed_batch(4, 2, layout, "std", GRID_CODE=code, GR_RADIX=r, GR_LEVELS=n)
    x, y, _, aux = batch(3)
    rows, pos = aux["ihead_row"], aux["ihead_pos"] + (n - 1)          # the step's expression (asserted in the source below)
    metas = aux["metas"]
    assert len(rows) == sum(len(m["index"]) for m in metas) == 4 * gr.lay(layout)["ask"] == 96   # one entry a question, 24 a row
    want = torch.tensor([48 + gr.digits(m["recs"][e["rec"]]["addr"], code)[-1] for m in metas for e in m["index"]])
    assert torch.equal(y[rows, pos], want), (layout, code)
    assert bool(((y[rows, pos] >= 48) & (y[rows, pos] < 48 + r)).all())
    last = torch.tensor([p[-1] for m in metas for p in gr.index_positions(m)])
    assert torch.equal(pos, last)                                       # grid_rows' own positions of the entries' address bytes
    nll = torch.rand(x.shape, dtype=torch.float64)
    term = nll[rows, pos].mean()
    by_hand = sum(float(nll[int(a), int(b)]) for a, b in zip(rows, pos)) / len(rows)
    assert abs(float(term) - by_hand) < 1e-12


def test_the_step_reads_those_positions_and_the_default_adds_nothing():
    assert 'IDX_DOSE = float(ARGS.get("idx_dose", "0"))' in SRC
    assert '_dr, _dp = aux["ihead_row"].to(DEV), aux["ihead_pos"].to(DEV) + (GR_LEVELS - 1)' in SRC
    assert "_c.dterm = _c.nll[_dr, _dp].mean()" in SRC and "_c.loss = _c.loss + IDX_DOSE * _c.dterm" in SRC
    for line in SRC.split("\n"):                                       # every dose line sits under an IDX_DOSE > 0 guard or is the parse
        if "dterm" in line and "if IDX_DOSE > 0" not in line:
            assert line.startswith(" " * 12), line


def test_the_names_carry_the_dose_only_when_set():
    base = {"arm": "alhena-both", "seed": "1", "steps": "24000", "ctx": "4096", "grid": "ag", "grid_p": "0.5", "rotary": "anchored",
            "head": "digit", "head_lam": "1", "grid_comps": "48", "grid_code": "4x4", "series": "a48dose"}
    t0 = run_tag(dict(base))
    assert "_id" not in t0 and run_tag(dict(base, idx_dose="0")) == t0
    t1, t2 = run_tag(dict(base, idx_dose="1")), run_tag(dict(base, idx_dose="0.1"))
    assert t1 == t0.replace("_ahdigit1_a48dose", "_ahdigit1_id100_a48dose") and t2 == t0.replace("_ahdigit1_a48dose", "_ahdigit1_id10_a48dose")
    assert run_tag(dict(base, idx_dose="1", head_at="both")).endswith("_ahdigit1i1_id100_a48dose")


def test_the_runner_passes_the_dose():
    runs = br.expand({"arms": "alhena-both", "seeds": "1", "steps": "24000", "idx_dose": "0.1,1"})
    assert [[a for a in r if a.startswith("--idx_dose=")] for r in runs] == [["--idx_dose=0.1"], ["--idx_dose=1"]]
