# -*- coding: utf-8 -*-
"""0.1.102: three digits of eight (--grid_code=8x3). An address in [0, 512) is three digits of eight, written as three bytes '0' + d and
carried as three streams (the null anchor 8). The anchor levels take twelve of the head's sixteen pairs (three levels of four at
2 pi k / 9: over a level's pairs sum_k cos(w_k d) is 4 at d = 0 and -1/2 at every other difference), so the shared pairs fall to four:
at two content pairs the relative band keeps two (the NTK rule over their 4 dims: wavelengths 6.3 and 2,513 positions). At 8x2 and 4x4
the split is 0.1.101's angle for angle and field for field. On the bed's own 8x3 batches (AG48 and AG96, key-hard rows included) the
streams are three wide, the head targets three digits, the hold's record rule holds every record's coded SEP and nothing else (the 8x2 and
4x4 rules none), and the index entries' dose reads each entry's SEP + 2, the third digit. The key rotation binds all sixteen pairs at the
code; the eight-pair form and the path rotation are refused there. The names carry _ac83 and the runner passes the code. CPU only; the Hub
is never touched."""
import math
import os

import pytest
import torch

from geolip.alephllm.model.attention import rope_half
from geolip.nano_beatrix import anchored_rope as ar
from geolip.nano_beatrix import bed_runner as br
from geolip.nano_beatrix import grid_rows as gr
from geolip.nano_beatrix import key_code as kc
from geolip.nano_beatrix import key_rotary as kr
from geolip.nano_beatrix.names import run_tag
from test_records_hold import _grid_truth
from test_scale import _bed_batch
from test_value_addr import STEM, W_BASE

SRC = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "geolip", "nano_beatrix", "bed_train.py"), encoding="utf-8").read()


def test_the_addresses_digits_and_bytes_by_hand():
    assert gr.n_addr("8x3") == 512 and gr.radix_levels("8x3") == (8, 3) == ar.CODES["8x3"] and ar.code_shape("8x3") == (8, 3, 4)
    for a in range(512):
        d = gr.digits(a, "8x3")
        assert d == (a // 64, (a // 8) % 8, a % 8) and 64 * d[0] + 8 * d[1] + d[2] == a
        assert gr.addr_bytes(a, 2, "8x3") == bytes(48 + x for x in d)
    assert gr.addr_bytes(83, 2, "8x3") == b"123" and gr.digits(511, "8x3") == (7, 7, 7)


def test_the_split_at_the_code():
    assert [ar.shared(c) for c in ("8x2", "4x4", "8x3")] == [8, 8, 4] and ar.shared() == ar.SHARED
    w, kind = ar.frequencies(2, "8x3")
    assert kind.tolist() == [0, 0, 1, 1] + [2] * 4 + [3] * 4 + [4] * 4
    base = 10000.0 * 4.0 ** (4 / 2)                                                 # the NTK rule over the band's 4 dims: 160,000
    assert base == 160000.0 and w[:2].tolist() == [1.0, base ** -0.5] and w[2:4].tolist() == [0.0, 0.0]
    assert [round(2 * math.pi / float(x), 1) for x in w[:2]] == [6.3, 2513.3]
    lev = torch.tensor([2 * math.pi * k / 9 for k in range(1, 5)], dtype=torch.float64)
    assert torch.equal(w[4:8], lev) and torch.equal(w[8:12], lev) and torch.equal(w[12:16], lev)
    for c in (3, 4):
        with pytest.raises(AssertionError):
            ar.frequencies(c, "8x3")                                                # the relative band keeps at least two pairs
    s = ar.spec(2, "8x3")
    assert (s["code"], s["rel_pairs"], s["content_pairs"], s["anchor_freqs"], s["levels"], s["null"]) == ("8x3", 2, 2, 4, 3, 8)
    assert ar.content_of(s) == 2 and ar.code_of(s) == "8x3"
    with pytest.raises(AssertionError):
        ar.content_of(dict(s, rel_pairs=6))                                         # a field that is not its own split is refused


@pytest.mark.parametrize("code,c", [("8x2", 2), ("8x2", 4), ("4x4", 4), ("4x4", 2)])
def test_the_8x2_and_4x4_splits_are_0101s(code, c):
    """0.1.101's split by hand: eight shared pairs, the relative band 8 - c of them under the NTK rule, the levels at 2 pi k / (R + 1)."""
    r, n = gr.CODES[code]
    rel = 8 - c
    dim = 2 * rel
    base = 10000.0 * 4.0 ** (dim / (dim - 2))
    w_want = [base ** (-j / rel) for j in range(rel)] + [0.0] * c
    k_want = [0] * rel + [1] * c
    for lev in range(n):
        w_want += [2 * math.pi * k / (r + 1) for k in range(1, r // 2 + 1)]
        k_want += [2 + lev] * (r // 2)
    w, kind = ar.frequencies(c, code)
    assert w.tolist() == w_want and kind.tolist() == k_want
    s = ar.spec(c, code)
    assert s["rel_pairs"] == rel and s["w"] == w_want and ("code" in s) == (code != "8x2")


@pytest.mark.parametrize("lev", [0, 1, 2])
def test_each_level_is_a_simplex(lev):
    w, kind = ar.frequencies(2, "8x3")
    wl = w[kind == 2 + lev]
    assert len(wl) == 4
    for d in range(-8, 9):
        s = float(torch.cos(wl * d).sum())
        assert abs(s - (4 if d == 0 else -0.5)) < 1e-9, (lev, d, s)


def test_the_streams_turn_each_level_by_its_digit_and_the_null_anchor_by_nothing():
    st = ar.AnchorStreams(2, "8x3")
    assert (st.radix, st.levels) == (8, 3)
    dk = torch.tensor([[[0, 3, 7], [8, 8, 8], [5, 8, 1]]])
    dq = torch.tensor([[[7, 3, 0], [8, 8, 8], [5, 2, 1]]])
    st.set(dk, dq)
    aq, ak = st.angles(1, 3, 0, "cpu")
    for lev in range(3):
        sel = st.kind == 2 + lev
        for p in range(3):
            assert torch.allclose(ak[0, 0, p, sel], (dk[0, p, lev] - 8).float() * st.w[sel])
            assert torch.allclose(aq[0, 0, p, sel], (dq[0, p, lev] - 8).float() * st.w[sel])
        assert torch.all(ak[0, 0, 1, sel] == 0) and torch.all(aq[0, 0, 1, sel] == 0)
    assert torch.equal(aq[..., :4], ak[..., :4])                                   # the shared pairs: the sequence index alone
    with pytest.raises(AssertionError):
        st.set(torch.zeros(1, 3, 2, dtype=torch.long), torch.zeros(1, 3, 2, dtype=torch.long))   # two levels at the 8x3 code
    va = ar.ValueAddress(16, st)
    assert tuple(va.table.shape) == (3, 8, 16)
    with torch.no_grad():
        va.table.copy_(torch.randn(3, 8, 16, generator=torch.Generator().manual_seed(0)))
        y = va.add(torch.zeros(1, 3, 16))
    assert torch.allclose(y[0, 0], va.table[0, 0] + va.table[1, 3] + va.table[2, 7])
    assert torch.equal(y[0, 1], torch.zeros(16))                                  # the null anchor adds nothing
    assert torch.allclose(y[0, 2], va.table[0, 5] + va.table[2, 1])               # a null digit at one level adds nothing there


def test_the_rotated_scores_at_the_code_by_hand():
    """q_m . k_n after the rotation = sum_rel Re[q conj(k) e^{i w_j (m - n)}] + sum_content Re[q conj(k)]
    + sum_l sum_k Re[q_lk conj(k_lk) e^{i w_k (dq_l(m) - dk_l(n))}] (the angles formed in fp32, as the house rotary forms them)."""
    st = ar.AnchorStreams(2, "8x3")
    g = torch.Generator().manual_seed(3)
    n = 6
    dk = torch.randint(0, 9, (1, n, 3), generator=g)
    dq = torch.randint(0, 9, (1, n, 3), generator=g)
    st.set(dk, dq)
    aq, ak = st.angles(1, n, 0, "cpu")
    q = torch.randn(1, 1, n, 2 * ar.PAIRS, generator=g, dtype=torch.float64)
    k = torch.randn(1, 1, n, 2 * ar.PAIRS, generator=g, dtype=torch.float64)
    got = (rope_half(q, aq.cos().double(), aq.sin().double()) @ rope_half(k, ak.cos().double(), ak.sin().double()).transpose(-1, -2))[0, 0]
    w, kind = ar.frequencies(2, "8x3")
    qc = torch.complex(q[0, 0, :, :ar.PAIRS], q[0, 0, :, ar.PAIRS:])
    kc_ = torch.complex(k[0, 0, :, :ar.PAIRS], k[0, 0, :, ar.PAIRS:])
    for m in range(n):
        for t in range(n):
            ang = torch.zeros(ar.PAIRS, dtype=torch.float64)
            ang[kind == 0] = w[kind == 0] * (m - t)
            for lev in range(3):
                sel = kind == 2 + lev
                ang[sel] = w[sel] * float(dq[0, m, lev] - dk[0, t, lev])
            want = float((qc[m] * kc_[t].conj() * torch.exp(1j * ang)).real.sum())
            assert abs(float(got[m, t]) - want) < 1e-3, (m, t, float(got[m, t]), want)


@pytest.mark.parametrize("layout", ["AG48", "AG96"])
def test_the_beds_8x3_batches_streams_head_hold_and_dose(layout):
    batch = _bed_batch(4, 2, layout, "std", GRID_CODE="8x3", GR_RADIX=8, GR_LEVELS=3)
    L = gr.lay(layout)
    n = hard = 0
    for step in range(4):
        x, y, _, aux = batch(step)
        assert tuple(aux["dk"].shape) == (8, gr.CTX, 3) and bool((aux["dk"][:4] == 8).all()) and bool((aux["dq"][:4] == 8).all())
        assert tuple(aux["head_dig"].shape) == (4 * L["ask"], 3) and int(aux["head_dig"].max()) <= 7
        for j, m in enumerate(aux["metas"]):
            assert m["code"] == "8x3" and m["layout"] == layout and gr.abytes_of(m) == 3
            recs, ents, qs = _grid_truth(m)
            xi = x[aux["first"] + j][None]
            off = kc.key_offsets(xi, kc.K)
            on = (off[..., 0] != kc.VOCAB)[0]
            assert set(on.nonzero().flatten().tolist()) == set(recs) | set(ents) | set(qs)
            assert kr.records(xi, off, "8x3")[0].nonzero().flatten().tolist() == sorted(recs)
            assert not kr.records(xi, off)[0].any() and not kr.records(xi, off, "4x4")[0].any(), "the other codes' rules hold none"
            dk = aux["dk"][aux["first"] + j]
            for rec in m["recs"]:
                assert dk[rec["lo"]:rec["hi"]].tolist() == [list(gr.digits(rec["addr"], "8x3"))] * L["slot"]
            n += len(recs)
            hard += int(m.get("hard", False))
        rows, pos = aux["ihead_row"], aux["ihead_pos"] + 2                           # the dose's expression at three levels
        want = torch.tensor([48 + gr.digits(m["recs"][e["rec"]]["addr"], "8x3")[-1] for m in aux["metas"] for e in m["index"]])
        assert len(rows) == 4 * L["ask"] and torch.equal(y[rows, pos], want)
        assert torch.equal(pos, torch.tensor([p[-1] for m in aux["metas"] for p in gr.index_positions(m)]))
    assert n == 4 * 4 * L["comps"] and hard == 4 * 2


def test_the_key_rotary_at_the_code():
    k3 = kr.KeyRotary(4, 0, pairs=16, norm=2, rho=2.18, code="8x3")
    assert k3.code == "8x3" and torch.equal(k3.table, kr.KeyRotary(4, 0, pairs=16, norm=2, rho=2.18).table)   # the phases do not depend on it
    with pytest.raises(AssertionError):
        kr.KeyRotary(4, 0, code="8x3")                                              # eight pairs would turn the first level's anchor pairs
    for c in ("8x2", "4x4"):
        assert kr.KeyRotary(4, 0, code=c).pairs == 8                                # the eight-pair form where eight pairs are shared


def test_the_trainer_guards_the_code():
    assert ("assert not KEY_ROT or KEY_PAIRS == ar.PAIRS or ar.shared(GRID_CODE) == krm.J" in SRC
            and "assert ROTARY != \"anchored\" or ROT_CONTENT <= ar.shared(GRID_CODE) - 2" in SRC)
    assert "assert not PATH_BIND or ar.shared(GRID_CODE) == ar.SHARED" in SRC
    assert "_sh = ar.shared(GRID_CODE)" in SRC and "_rel = _sh - ROT_CONTENT" in SRC and "ar.SHARED - ROT_CONTENT" not in SRC


def test_the_names_and_the_runner():
    pk = dict(W_BASE, grid_hard="0.5", value_addr="1", ptr_lam="0.1", key_rot="1", key_pairs="16", key_norm="2", key_rho="2.18",
              plant="0.75", plain_doc="1")
    stem = STEM.format(1).replace("_askany_", "_askany_p75d_")
    a48 = run_tag(dict(pk, grid_comps="48", grid_code="8x3", rotary_content="2", idx_dose="0.1", series="a48c83"))
    assert a48 == stem.replace("_cp4_", "_") + "gc48_ac83_kh50_va1_pt10b4h0_krt16r218_ahdigit1_id10_a48c83" and a48.count("_ac83_") == 1
    assert run_tag(dict(pk, grid_comps="48", grid_code="4x4", series="code44")) == stem + "gc48_ac44_kh50_va1_pt10b4h0_krt16r218_ahdigit1_code44"
    runs = br.expand(dict(arms="alhena-both", seeds="1,2", grid="ag", grid_p="0.5", head="digit", head_lam="1", rotary_content="2",
                          grid_hard="0.5", value_addr="1", ptr_lam="0.1", key_rot="1", grid_comps="48", key_pairs="16", key_norm="2",
                          key_rho="2.18", plant="0.75", plain_doc="1", grid_code="8x3", idx_dose="0.1", series="a48c83"))
    assert len(runs) == 2 and all("--grid_code=8x3" in r and "--grid_comps=48" in r and "--idx_dose=0.1" in r for r in runs)
    assert "the address code 8x3" in br.run_label(runs[0])
