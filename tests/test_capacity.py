# -*- coding: utf-8 -*-
"""The binding's capacity read (0.1.85): the key dims turn as the head's own rotation turns them; a key's phases from key_phases are the
phases its SEP takes inside a row; the bank's keys are distinct, outside the excluded set and coded at the reach in both key forms; a
planted head whose eight pairs align only for the true key reproduces the moment's mean (the annealed average) within its Monte Carlo error
while the typical value (the median) sits below it; on a saved core the read's capture through the told block equals the full forward's
look capture bit for bit, and the record carries its fields and its run parameters in its name. THE TAIL READ (0.1.86): a pair's
length is one after unit_pairs and its angle kept; tail_conditions' three leaks (base, the competitors' lengths set to the true
record's, the true score set to the median) equal the sums written out by hand; on the saved core the tail's base equals the read's
EMULATED bit for bit, its bands count its look errors and its strongest competitors are classed. CPU only; the Hub is never touched."""
import json
import math

import numpy as np
import torch

from geolip.alephllm.model.attention import rope_half
from geolip.nano_beatrix import anchored_rope as ar
from geolip.nano_beatrix import bed_frames as bf
from geolip.nano_beatrix import capacity as cp
from geolip.nano_beatrix import grid_rows as gr
from geolip.nano_beatrix import key_code as kc
from geolip.nano_beatrix import key_rotary as kr
from geolip.nano_beatrix import rotary_scale as rs
from test_key_code import _save_core
from test_value_addr import _craft

BLOCK, HEAD = 4, 0
SRC = bytes(np.random.default_rng(3).integers(32, 127, size=gr.CTX + 1, dtype=np.uint8))


def test_turn_is_the_heads_rotation():
    g = torch.Generator().manual_seed(1)
    v = torch.randn(5, 2 * ar.PAIRS, generator=g)
    ang = torch.rand(5, ar.PAIRS, generator=g) * 2 * math.pi
    ang[:, kr.J:] = 0.0                                                   # the anchor pairs unturned: only the key pairs read here
    want = rope_half(v, ang.cos(), ang.sin())[:, cp.KD]
    got = cp.turn(v[:, cp.KD], ang[:, :kr.J])
    assert torch.allclose(got, want, atol=1e-6, rtol=0)
    assert torch.allclose(cp.turn(got, -ang[:, :kr.J]), v[:, cp.KD], atol=1e-5, rtol=0)


def test_key_phases_are_the_rows_phases():
    for layout, form, k in (("AG", "std", 9), ("AG48", "long", 12)):
        rot = kr.KeyRotary(BLOCK, HEAD, k=k)
        rows = rs.gauge_rows(1, "capacity-test", 2, [SRC[1:]], layout=layout, key_form=form)
        x = rs.to_x(rows, "cpu")[:, :-1]
        phi, on = rot.phases(x)
        for b, (_, mt) in enumerate(rows):
            seps = [r["v0"] - 1 for r in mt["recs"]]
            assert bool(on[b, seps].all())
            got = cp.key_phases(rot, [r["key"].encode() for r in mt["recs"]])
            assert torch.equal(got, phi[b, seps].float())                  # the same gathers summed in the same order


def test_the_bank_keys():
    rng = np.random.default_rng(5)
    ex = {bf.key_name(i) for i in range(50)}
    std = cp.bank_keys(rng, "std", 5000, ex)                              # the standard form's 16,000 keys cannot give ten banks: its shape
    assert len(set(std)) == 5000 and not set(std) & ex and all(6 <= len(k) <= 9 and k[-3:].isdigit() for k in std)
    cp.key_phases(kr.KeyRotary(BLOCK, HEAD, k=9), std)                    # every one coded at the reach 9 (asserted inside)
    ex = {bf.key_long(i) for i in range(50)}
    lng = cp.bank_keys(rng, "long", 1000, ex)                             # the long form holds ten banks: the form itself
    assert len(set(lng)) == 1000 and not set(lng) & ex and all(9 <= len(k) <= 12 and k[-6:].isdigit() for k in lng)
    assert all(any(k.startswith(w.encode()) for w in bf.WORDS) for k in lng)
    cp.key_phases(kr.KeyRotary(BLOCK, HEAD, k=12), lng)


def _planted(s, Q, N, seed):
    """Q entries and a bank of N competitors on a planted head: every pair of magnitude s when aligned, the true key's phases the
    entry's own, the competitors' phases uniform."""
    g = torch.Generator().manual_seed(seed)
    a = math.sqrt(s * math.sqrt(2 * ar.PAIRS))                           # each pair's |q_j| = |k_j| = a: a^2 / sqrt(hd) = s aligned
    base = torch.cat([torch.full((kr.J,), a), torch.zeros(kr.J)])
    phi_m = torch.rand(Q, kr.J, generator=g) * 2 * math.pi
    q = cp.turn(base.expand(Q, -1), phi_m)
    t = cp.turn(base.expand(Q, -1), phi_m)                               # the true key: the same phases
    st = (q * t).sum(-1) / math.sqrt(2 * ar.PAIRS)
    assert torch.allclose(st, torch.full((Q,), kr.J * s), atol=1e-4)
    return q, st, cp.turn(base.expand(N, -1), torch.rand(N, kr.J, generator=g) * 2 * math.pi)


def test_a_planted_head_meets_the_moment():
    s = 1.5                                                               # small s: the annealed mean is reachable by sampling
    q, st, bank = _planted(s, 64, 10000, 7)
    W, M = cp.emulated(q, st, bank, ns=(1000, 10000))
    for j, n in enumerate((1000, 10000)):
        want = n * float(np.i0(s)) ** kr.J * math.exp(-kr.J * s)          # E[W_N]: the annealed average
        assert abs(W[:, j].mean() / want - 1) < (0.15 if n == 1000 else 0.06), (n, W[:, j].mean(), want)
    assert abs(cp.moment(s, 10000) - want / (1 + want)) < 1e-9
    assert (M[:, 1] >= M[:, 0]).all()                                     # the largest competitor over the larger set
    s = 6.0                                                               # the trained heads' range: rare near-aligned keys dominate the mean
    q, st, bank = _planted(s, 64, 1000, 8)
    W, _ = cp.emulated(q, st, bank, ns=(1000,))
    want = 1000 * float(np.i0(s)) ** kr.J * math.exp(-kr.J * s)
    assert np.median(W[:, 0]) < 0.5 * want                                # the typical sum far below the annealed one


def _saved_core(tmp_path):
    """A crafted core with the value address and the key rotation (the long keys, reach 12) at 48 records a row, saved as a stored core."""
    m, st = _craft(n_ctx=gr.CTX, seed=3)
    va = ar.ValueAddress(m.cfg.d_model, st)
    with torch.no_grad():
        va.table.normal_(std=0.5)
    ar.patch_value(m, va)
    rot = kr.KeyRotary(BLOCK, HEAD, k=12)
    kr.patch(m, rot)
    d = tmp_path / "core"
    sj = _save_core(m, d, va)
    from safetensors.torch import save_file
    save_file({"table": rot.table.detach().cpu().contiguous()}, str(d / "key_rot.safetensors"))
    sj.update(grid_layout="AG48", grid_keys="long", layout=dict(slot=64, comps=48, radix=gr.RADIX, levels=gr.LEVELS))
    sj["key_rot"] = dict(pairs=kr.J, k=12, vocab=kc.VOCAB, block=BLOCK, head=HEAD, seed_tag=kr.seed_tag(12))
    sj["ptr"] = dict(lam=0.1, block=BLOCK, head=HEAD)
    json.dump(sj, open(d / "scale.json", "w", encoding="utf-8"), indent=1)
    return d


def test_the_read_on_a_saved_core(tmp_path):
    d = _saved_core(tmp_path)
    core = rs.Core(str(d), "cpu")
    held = [SRC[1:]]
    rows = cp.read_rows(core, held, 4)
    assert [bool(mt.get("hard")) for _, mt in rows] == [False, False, True, True]
    cap = cp.capture(core, rows, batch=2, log=None)
    assert len(cap["q"]) == 4 * 24 and len(cap["pool_k"]) == 4 * 72          # 24 entries a row; 48 records + 24 entries pooled
    x = rs.to_x(rows, "cpu")[:, :-1]
    core.set_streams([mt for _, mt in rows])
    core.look_cap.on = True
    with torch.no_grad():
        core.readout(core.trunk(x))                                       # the full forward
    full_q = core.look_cap.q[:, HEAD][..., cp.KD].float()
    core.look_cap.on = False
    core.clear()
    seps = [e["a_pos"] - 1 for e in rows[0][1]["index"]]
    assert torch.equal(cap["q"][:24], full_q[0, seps])                    # the capture through the told block = the full forward's
    rec = cp.read_core(str(d), held, R=4, ns=(10, 100), draws=2, bank_n=100, log=None)
    for f in ("core", "version", "rows", "layout", "key_form", "reach", "entries", "pool", "pool_smallest_allowed", "bank", "bank_keys",
              "s_core", "margin", "true_score", "pair_magnitude", "inrow", "emulated", "real", "wall"):
        assert f in rec, f
    assert (rec["layout"], rec["key_form"], rec["reach"], rec["entries"], rec["bank_keys"]) == ("AG48", "long", 12, 96, "form")
    assert set(rec["emulated"]) == {"10", "100"} and set(rec["real"]) == {"10", "100"}
    assert all(0.0 <= v["mean"] <= 1.0 and "moment" in v for v in rec["emulated"].values())
    assert cp.record_name("abc", 192, 1) == "abc_capacity_r192_s1.json"
    out = cp.run(str(d), held, str(tmp_path / "out"), R=4, ns=(10,), draws=1, bank_n=10, log=None)
    again = cp.run(str(d), held, str(tmp_path / "out"), R=4, ns=(10,), draws=1, bank_n=10, log=None)
    assert out == again                                                   # resume-first: the record is read back


def test_pair_lengths_and_unit_pairs():
    g = torch.Generator().manual_seed(5)
    v = torch.randn(7, 2 * cp.J, generator=g)
    v[0, 0] = v[0, cp.J] = 0.0                                            # a zero pair stays zero
    n = cp.pair_norms(cp.unit_pairs(v))
    assert n[0, 0] == 0.0
    n[0, 0] = 1.0
    assert torch.allclose(n, torch.ones_like(n), atol=1e-6)
    ang = lambda x: torch.atan2(x[..., cp.J:], x[..., :cp.J])   # noqa: E731
    assert torch.allclose(ang(cp.unit_pairs(v))[1:], ang(v)[1:], atol=1e-6)                   # the angles kept
    turned = cp.turn(v, torch.rand(7, cp.J, generator=g) * 6.28)
    assert torch.allclose(cp.pair_norms(turned), cp.pair_norms(v), atol=1e-5)                  # a turn keeps every pair's length


def test_the_tail_conditions_by_hand():
    g = torch.Generator().manual_seed(7)
    q, t, bank = (torch.randn(r, 2 * cp.J, generator=g) for r in (6, 6, 50))
    sc = 1.0 / math.sqrt(2 * ar.PAIRS)
    st = (q * t).sum(-1) * sc
    c = cp.tail_conditions(q, t, st, bank, (10, 50))
    qd, td, bd = q.double(), t.double(), bank.double()
    tl = torch.sqrt(td[:, :cp.J] ** 2 + td[:, cp.J:] ** 2)
    bl = torch.sqrt(bd[:, :cp.J] ** 2 + bd[:, cp.J:] ** 2)
    med = float(np.median(st.numpy()))
    for m in range(6):
        for j, n in enumerate((10, 50)):
            s_base = [float((qd[m] * bd[k]).sum()) * sc for k in range(n)]
            s_m = [sc * sum(float(tl[m, p] * (qd[m, p] * bd[k, p] + qd[m, cp.J + p] * bd[k, cp.J + p]) / bl[k, p]) for p in range(cp.J))
                   for k in range(n)]
            want = {"base": sum(math.exp(s - float(st[m])) for s in s_base), "M": sum(math.exp(s - float(st[m])) for s in s_m),
                    "E": sum(math.exp(s - med) for s in s_base)}
            for name, w in want.items():
                assert abs(c[name][0][m, j] - w) <= 1e-4 * w + 1e-12, (name, m, n, c[name][0][m, j], w)


def test_the_tail_on_a_saved_core(tmp_path):
    d = _saved_core(tmp_path)
    held = [SRC[1:]]
    ns = (30, 100, 300)
    rec = cp.read_core(str(d), held, R=4, ns=ns, draws=1, log=None)                 # its bank: max(ns) vectors, as the tail's
    tl = cp.tail_core(str(d), held, R=4, ns=ns, log=None)
    for n in ns:
        a, b = rec["emulated"][str(n)], tl["base"][str(n)]
        for f in ("mean", "median", "p99", "max", "look_error", "argmax_error", "entries", "moment"):
            assert a[f] == b[f], (n, f)                                          # the base is the read's EMULATED, bit for bit
    for f in ("core", "version", "rows", "layout", "key_form", "entries", "pool", "bank", "ns", "s_core", "true_score_median", "base", "M",
              "E", "pool_length_ratio", "true_pair_length", "top_competitor", "wall"):
        assert f in tl, f
    assert tl["s_core"] == rec["s_core"] and (tl["pool"], tl["entries"], tl["bank"]) == (4 * 72, 96, 300)
    pr = tl["pool_length_ratio"]
    assert (pr["all"]["n"], pr["records"]["n"], pr["entries"]["n"]) == (288, 192, 96)
    for name in ("base", "M", "E"):
        for n in ns:
            v = tl[name][str(n)]
            k = round(v["look_error"] * 96)
            assert sum(v["look_by_margin_pct"].values()) == k == sum(v["look_by_true_pct"].values()), (name, n)
    top = tl["top_competitor"]
    assert sum(top["source"].values()) == top["n"] == round(tl["base"]["300"]["look_error"] * 96)
    assert abs(sum(top["bank_share"].values()) - 1.0) < 1e-9
    assert cp.tail_name("abc", 192, 1) == "abc_captail_r192_s1.json"
    out = cp.run_tail(str(d), held, str(tmp_path / "out"), R=4, ns=(30,), log=None)
    assert out == cp.run_tail(str(d), held, str(tmp_path / "out"), R=4, ns=(30,), log=None)   # resume-first
