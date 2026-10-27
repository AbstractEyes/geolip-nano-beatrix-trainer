# -*- coding: utf-8 -*-
"""The address code (0.1.91): an address is L digits of radix R, written as L bytes '0' + d and carried as L streams, the null anchor R.
At every code the eight anchor pairs carry L levels of R / 2 pairs at 2 pi k / (R + 1), so over a level's pairs sum_k cos(w_k d) is R / 2
at d = 0 and -1/2 at every other difference (the R + 1 symbols a simplex); the 8x2 code is 0.1.90's rotation, field and tables exactly;
the 4x4 code turns each of its four levels by its own digit, the null anchor by nothing. The hold's record rule reads DATA and the code's
digits before the CUE: on the bed's own 4x4 batches (AG96 and AG48, key-hard rows included) it holds every record's coded SEP and nothing
else, and the 8x2 rule holds none of them. The batch at the 4x4 code streams four levels (every non-grid row at the null anchor 4) and
its head targets are four digits. The names carry _ac44 (the 8x2 code names nothing) and the runner passes and names the code. CPU only;
the Hub is never touched."""
import math

import pytest
import torch

from geolip.nano_beatrix import anchored_rope as ar
from geolip.nano_beatrix import bed_runner as br
from geolip.nano_beatrix import grid_rows as gr
from geolip.nano_beatrix import key_code as kc
from geolip.nano_beatrix import key_rotary as kr
from geolip.nano_beatrix.names import run_tag
from test_records_hold import _grid_truth
from test_scale import _bed_batch
from test_value_addr import STEM, W_BASE


@pytest.mark.parametrize("code", ["8x2", "4x4"])
def test_each_level_of_the_anchor_pairs_is_a_simplex(code):
    r, n = gr.CODES[code]
    w, kind = ar.frequencies(4, code)
    assert len(w) == ar.PAIRS and int((kind >= 2).sum()) == 8 and int(kind.max()) == 1 + n
    for lev in range(n):
        wl = w[kind == 2 + lev]
        assert len(wl) == r // 2
        for d in range(-r, r + 1):
            s = float(torch.cos(wl * d).sum())
            assert abs(s - (r / 2 if d == 0 else -0.5)) < 1e-9, (code, lev, d, s)


def test_the_default_code_is_0190s_rotation_field_and_tables():
    w0, k0 = ar.frequencies(4)
    assert torch.equal(w0[8:12], torch.tensor([2 * math.pi * k / 9 for k in range(1, 5)], dtype=torch.float64))
    assert torch.equal(w0[8:12], w0[12:16]) and k0[8:].tolist() == [2] * 4 + [3] * 4
    sp = ar.spec(4)
    assert sp == ar.spec(4, "8x2") and "code" not in sp and (sp["levels"], sp["anchor_freqs"], sp["null"]) == (2, 4, 8)
    assert list(sp) == ["form", "pairs", "rel_pairs", "content_pairs", "anchor_freqs", "levels", "ntk_scale", "base", "null", "w", "kind"]
    s4 = ar.spec(4, "4x4")
    assert s4["code"] == "4x4" and (s4["levels"], s4["anchor_freqs"], s4["null"]) == (4, 2, 4)
    assert ar.content_of(s4) == 4 and ar.code_of(s4) == "4x4" and ar.code_of(sp) == "8x2"
    assert tuple(ar.ValueAddress(16, ar.AnchorStreams(4)).table.shape) == (2, 8, 16)
    assert tuple(ar.ValueAddress(16, ar.AnchorStreams(4, "4x4")).table.shape) == (4, 4, 16)


def test_the_4x4_streams_turn_each_level_by_its_digit_and_the_null_anchor_by_nothing():
    st = ar.AnchorStreams(4, "4x4")
    dk = torch.tensor([[[0, 1, 2, 3], [4, 4, 4, 4], [3, 0, 4, 1]]])
    dq = torch.tensor([[[3, 2, 1, 0], [4, 4, 4, 4], [3, 0, 2, 1]]])
    st.set(dk, dq)
    aq, ak = st.angles(1, 3, 0, "cpu")
    for lev in range(4):
        sel = st.kind == 2 + lev
        for p in range(3):
            assert torch.allclose(ak[0, 0, p, sel], (dk[0, p, lev] - 4).float() * st.w[sel])
            assert torch.allclose(aq[0, 0, p, sel], (dq[0, p, lev] - 4).float() * st.w[sel])
        assert torch.all(ak[0, 0, 1, sel] == 0) and torch.all(aq[0, 0, 1, sel] == 0)
    assert torch.equal(aq[..., :8], ak[..., :8])                                  # the shared pairs: the sequence index alone
    with pytest.raises(AssertionError):
        st.set(torch.zeros(1, 3, 2, dtype=torch.long), torch.zeros(1, 3, 2, dtype=torch.long))   # two levels at the 4x4 code
    va = ar.ValueAddress(16, st)
    with torch.no_grad():
        va.table.copy_(torch.randn(4, 4, 16, generator=torch.Generator().manual_seed(0)))
        y = va.add(torch.zeros(1, 3, 16))
    assert torch.allclose(y[0, 0], va.table[0, 0] + va.table[1, 1] + va.table[2, 2] + va.table[3, 3])
    assert torch.equal(y[0, 1], torch.zeros(16))                                  # the null anchor adds nothing
    assert torch.allclose(y[0, 2], va.table[0, 3] + va.table[1, 0] + va.table[3, 1])   # a null digit at one level adds nothing there


@pytest.mark.parametrize("layout", ["AG96", "AG48"])
def test_the_hold_finds_every_record_on_the_beds_4x4_batches(layout):
    batch = _bed_batch(4, 2, layout, "std", GRID_CODE="4x4", GR_RADIX=4, GR_LEVELS=4)
    L = gr.lay(layout)
    n = hard = 0
    for step in range(6):
        x, _, _, aux = batch(step)
        assert tuple(aux["dk"].shape) == (8, gr.CTX, 4) and bool((aux["dk"][:4] == 4).all()) and bool((aux["dq"][:4] == 4).all())
        assert tuple(aux["head_dig"].shape) == (4 * L["ask"], 4) and int(aux["head_dig"].max()) <= 3
        for j, m in enumerate(aux["metas"]):
            assert m["code"] == "4x4" and m["layout"] == layout and gr.abytes_of(m) == 4
            recs, ents, qs = _grid_truth(m)
            xi = x[aux["first"] + j][None]
            off = kc.key_offsets(xi, kc.K)
            on = (off[..., 0] != kc.VOCAB)[0]
            assert set(on.nonzero().flatten().tolist()) == set(recs) | set(ents) | set(qs)
            assert kr.records(xi, off, "4x4")[0].nonzero().flatten().tolist() == sorted(recs)
            assert not kr.records(xi, off)[0].any(), "the 8x2 rule holds no record of a 4x4 row"
            dk = aux["dk"][aux["first"] + j]
            for rec in m["recs"]:
                assert dk[rec["lo"]:rec["hi"]].tolist() == [list(gr.digits(rec["addr"], "4x4"))] * L["slot"]
            n += len(recs)
            hard += int(m.get("hard", False))
    assert n == 6 * 4 * L["comps"] and hard == 6 * 2


def test_the_key_rotary_carries_its_code():
    assert kr.KeyRotary(4, 0, pairs=16, norm=2, rho=2.18).code == "8x2"
    k4 = kr.KeyRotary(4, 0, pairs=16, norm=2, rho=2.18, code="4x4")
    assert k4.code == "4x4" and torch.equal(k4.table, kr.KeyRotary(4, 0, pairs=16, norm=2, rho=2.18).table)   # the phases do not depend on it
    with pytest.raises(AssertionError):
        kr.KeyRotary(4, 0, code="3x3")


def test_the_names_and_the_runner():
    pk = dict(W_BASE, grid_hard="0.5", value_addr="1", ptr_lam="0.1", key_rot="1", key_pairs="16", key_norm="2", key_rho="2.18",
              plant="0.75", plain_doc="1")
    stem = STEM.format(1).replace("_askany_", "_askany_p75d_")
    a96 = run_tag(dict(pk, grid_comps="96", grid_code="4x4", series="code44"))
    a48 = run_tag(dict(pk, grid_comps="48", grid_code="4x4", series="code44"))
    assert a96 == stem + "gc96_ac44_kh50_va1_pt10b4h0_krt16r218_ahdigit1_code44" and a96.count("_ac44_") == 1
    assert a48 == stem + "gc48_ac44_kh50_va1_pt10b4h0_krt16r218_ahdigit1_code44"
    assert run_tag(dict(pk, grid_comps="48", series="plain16")) == stem + "gc48_kh50_va1_pt10b4h0_krt16r218_ahdigit1_plain16"
    assert run_tag(dict(pk, grid_comps="48", grid_code="8x2", series="plain16")) == run_tag(dict(pk, grid_comps="48", series="plain16"))
    runs = br.expand(dict(arms="alhena-both", seeds="1,2", grid="ag", grid_p="0.5", head="digit", head_lam="1", rotary_content="4",
                          grid_hard="0.5", value_addr="1", ptr_lam="0.1", key_rot="1", grid_comps="96", key_pairs="16", key_norm="2",
                          key_rho="2.18", plant="0.75", plain_doc="1", grid_code="4x4", series="code44"))
    assert len(runs) == 2 and all("--grid_code=4x4" in r and "--grid_comps=96" in r for r in runs)
    lab = br.run_label(runs[0])
    assert "the address code 4x4" in lab and "96 records a row" in lab
    assert "grid_code" in br.GRID_KEYS
