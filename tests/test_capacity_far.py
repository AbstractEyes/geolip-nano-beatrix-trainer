# -*- coding: utf-8 -*-
"""The far read (0.1.89): the bank's shape keys drawn in numpy rounds are capacity.bank_keys' keys key for key (the standard and the long
digits, the excluded set, word lists that force re-draws) and their phases are key_phases' bit for bit (eight and sixteen pairs, reach 9
and 12); the streamed sums and maxima are the tail conditions' (segments inside and across blocks; E from the base; the base's argmax over
the whole bank); on a saved sixteen-pair core the far read equals the tail read of the same bank (compare_tail finds nothing), the bank's
cache changes no number, the onset and the half point read the grid, and the record is resume-first. CPU only; the Hub is never touched."""
import math

import numpy as np
import pytest
import torch

from geolip.nano_beatrix import anchored_rope as ar
from geolip.nano_beatrix import bed_frames as bf
from geolip.nano_beatrix import capacity as cp
from geolip.nano_beatrix import capacity_far as cf
from geolip.nano_beatrix import key_rotary as kr
from test_capacity import SRC
from test_key_rot import BLOCK, HEAD
from test_norm16 import _saved


def _both(seed, form, n, ex):
    a = cp.bank_keys(np.random.default_rng(seed), form, n, ex)
    rng = np.random.default_rng(seed)
    b = cf.shape_bank(rng, form, n, ex)
    return a, b, rng


def test_the_shape_bank_is_the_loops_key_for_key():
    ex = {bf.key_name(i) for i in range(200)}
    a, b, rng = _both(5, "std", 5000, ex)
    assert [b.key(i) for i in range(len(b))] == a and not set(a) & ex
    assert rng.integers(1 << 30) == (lambda r: (cp.bank_keys(r, "std", 5000, ex), r.integers(1 << 30))[1])(np.random.default_rng(5))
    codes = cf.shape_codes(b.letters, b.wl, b.nums, b.nd)                 # the code is the key's bytes in base 37
    assert [int(c) for c in codes[:50]] == [cf.bytes_code(k) for k in a[:50]] and len(set(codes.tolist())) == 5000
    assert cf.bytes_code(b"key12") is not None and cf.bytes_code(b"Key12") is None and cf.bytes_code(b"k" * 13) is None


def test_re_draws_and_exclusions_follow_the_loop(monkeypatch):
    monkeypatch.setattr(bf, "WORDS", ["a", "bb"])                         # 26 x 1,000 + 676 x 1,000 shapes: duplicates in every round
    ex = {b"a%03d" % i for i in range(300)} | {b"zz999", b"Q123", b"bb0"}
    a, b, _ = _both(11, "std", 20000, ex)
    assert [b.key(i) for i in range(len(b))] == a and len(set(a)) == 20000 and not set(a) & ex
    monkeypatch.setitem(cp.N_FORM, "long", 1000)                          # the long digits through the shape path
    monkeypatch.setattr(bf, "WORDS", ["code", "pin", "number"])
    a, b, _ = _both(12, "long", 3000, {b"pin000001"})
    assert [b.key(i) for i in range(len(b))] == a and all(len(k) in (9, 10, 12) for k in a)


@pytest.mark.parametrize("pairs,k,form", [(16, 9, "std"), (8, 9, "std"), (16, 12, "long")])
def test_the_shape_phases_are_key_phases_bit_for_bit(monkeypatch, pairs, k, form):
    if form == "long":
        monkeypatch.setitem(cp.N_FORM, "long", 1000)
    rot = kr.KeyRotary(BLOCK, HEAD, k=k, pairs=pairs)
    keys = cf.shape_bank(np.random.default_rng(3), form, 4000, set())
    got = cf.shape_phases(rot, keys, chunk=1500)                           # chunks that cut the keys
    want = cp.key_phases(rot, [keys.key(i) for i in range(len(keys))])
    assert got.dtype == want.dtype and torch.equal(got, want)


def _pieces(Q, P, nb, seed):
    g = torch.Generator().manual_seed(seed)
    q, t = torch.randn(Q, 32, generator=g), torch.randn(Q, 32, generator=g)
    k0 = torch.randn(P, 32, generator=g) * 0.7
    bphi = torch.rand(nb, 16, generator=g) * 2 * math.pi
    content = torch.randint(P, (nb,), generator=g)
    st = (q * t).sum(-1) / math.sqrt(2 * ar.PAIRS) + 2.0                  # true scores above most competitors
    return q, t, st, k0, bphi, content


def test_the_streamed_sums_are_the_tail_conditions():
    q, t, st, k0, bphi, content = _pieces(37, 90, 3000, 4)
    ns = (10, 100, 1000, 3000)
    bank = cp.turn(k0[content], bphi)
    want = cp.tail_conditions(q, t, st, bank, ns)
    got = cf.far_scores(q, t, st, k0, bphi, content, ns, block=256)        # blocks of 256: segments inside and across blocks
    for c, tol in (("base", 1e-6), ("M", 1e-6), ("E", 1e-5)):
        assert np.allclose(got[c][0], want[c][0], rtol=tol, atol=0), c
        assert np.allclose(got[c][1], want[c][1], rtol=0, atol=1e-4), c
    d = (q @ bank.T) / math.sqrt(2 * ar.PAIRS) - st[:, None]
    assert torch.allclose(got["best"], d.max(1).values, atol=1e-5)
    assert torch.allclose(d[torch.arange(37), got["arg"]], got["best"], atol=1e-5)
    one = cf.far_scores(q, t, st, k0, bphi, content, ns, block=1 << 16)    # one block: the same numbers (to the product's blocking)
    assert np.allclose(one["base"][0], got["base"][0], rtol=1e-6, atol=0) and np.allclose(one["base"][1], got["base"][1], atol=1e-5)


def test_the_onset_and_the_half_point():
    ns = (10, 100, 1000, 10000)
    assert cf.onset_half(ns, [0, 0, 0, 0]) == dict(onset=None, half=None, look_at_max=0.0)
    r = cf.onset_half(ns, [0, 0.01, 0.3, 0.7])
    assert r["onset"] == 100 and abs(r["half"] - 10 ** 3.5) < 1e-6 and r["look_at_max"] == 0.7
    assert cf.onset_half(ns, [0.6, 0.8, 1, 1])["half"] == 10.0


def test_the_far_read_is_the_tail_read_on_a_saved_core(tmp_path):
    m, st, rot, d = _saved(tmp_path, 16)
    held = [SRC[1:]]
    ns = (30, 100, 1000, 3000)                                             # 3,000 > 1,600: the standard form draws the shape
    tail = cp.tail_core(str(d), held, R=4, ns=ns, log=None)
    cache = {}
    far = cf.far_core(str(d), held, R=4, ns=ns, block=512, cache=cache, log=None)
    assert far["bank_keys"] == "shape" and far["told"]["pairs"] == 16 and far["ns"] == list(ns)
    assert cf.compare_tail(far, tail) == []
    for name in ("base", "M", "E"):
        looks = [far[name][str(n)]["look_error"] for n in ns]
        assert far["far"][name] == cf.onset_half(ns, looks)
    again = cf.far_core(str(d), held, R=4, ns=ns, block=2048, cache=cache, log=None)
    assert len(cache) == 1 and again["host"]["bank_cached"] and not far["host"]["bank_cached"]
    assert cf.compare_tail(again, far, rtol=1e-6, rtol_e=1e-6) == []                  # the cached bank: the same numbers
    assert cf.far_name("abc", 192, 1) == "abc_capfar_r192_s1.json"
    out = cf.run_far(str(d), held, str(tmp_path / "out"), R=4, ns=(30, 3000), log=None)
    assert out == cf.run_far(str(d), held, str(tmp_path / "out"), R=4, ns=(30, 3000), log=None)   # resume-first


def test_the_gate_finds_a_changed_number(tmp_path):
    m, st, rot, d = _saved(tmp_path, 16)
    held = [SRC[1:]]
    tail = cp.tail_core(str(d), held, R=2, ns=(30, 2000), log=None)
    far = cf.far_core(str(d), held, R=2, ns=(30, 2000), log=None)
    assert cf.compare_tail(far, tail) == []
    far["base"]["2000"]["mean"] *= 1.001
    far["M"]["30"]["look_error"] += 1.0 / 48
    bad = cf.compare_tail(far, tail)
    assert ("base", "2000", "mean") in [b[:3] for b in bad] and ("M", "30", "look_error") in [b[:3] for b in bad]
