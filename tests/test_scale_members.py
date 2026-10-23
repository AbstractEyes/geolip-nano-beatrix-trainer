# -*- coding: utf-8 -*-
"""0.1.71: the scale test's members (rotary_scale.py): the codes, the member rows, the stacked real arms against the arm module of
record, the stacked toy members against the night's module, the per-member Adam against torch's Adam, the gate and the floor."""
import math

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from geolip.nano_beatrix import grid_rows as gr
from geolip.nano_beatrix import rotary_scale as rs


def test_codes_are_derangements_of_the_letters():
    t = rs.code_tables(48).numpy()
    assert t.shape == (48, 256)
    for i in range(48):
        assert (t[i, 97:123] != np.arange(97, 123)).all() and sorted(t[i, 97:123]) == list(range(97, 123))
        rest = np.r_[0:97, 123:256]
        assert (t[i, rest] == rest).all()
    assert (rs.code_table(5) == rs.code_table(5)).all() and not (rs.code_table(5) == rs.code_table(6)).all()


def _packed(n=200_000, seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(32, 127, size=n).astype(np.uint8)


def test_member_rows_give_every_owner_three_questions():
    own, never = rs.owned_sets()
    rows = rs.member_rows(np.random.default_rng([1, 2]), _packed(), own, never)
    assert len(rows) == rs.ROWS_LAST
    qt = rs.question_table([m for _, m in rows], own)
    assert (qt["own"] >= 0).all(), "member mode asks only owned records"
    assert (np.bincount(qt["own"], minlength=48) == 3).all()
    for _, m in rows:
        a = set(m["addrs"])
        assert len(a & set(own)) == 18 and len(a & set(never)) == 6 and len(m["questions"]) == 18


def test_owned_row_asks_its_member_once():
    rng = np.random.default_rng(3)
    for addr in (0, 17, 63):
        row, meta, qi = rs.owned_row(rng, bytes(_packed(gr.CTX + 1)), addr)
        asked = [meta["recs"][q["rec"]]["addr"] for q in meta["questions"]]
        assert asked.count(addr) == 1 and asked[qi] == addr and addr in meta["addrs"] and len(meta["questions"]) == gr.N_COMP


def _randomize(mod, seed=0):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in mod.parameters():
            p.add_(0.05 * torch.randn(p.shape, generator=g))


def test_stacked_arms_equal_the_arm_module():
    pytest.importorskip("amoe")
    d, M = 64, 3
    mods = rs.StackedArms.amoe_modules(M, d, "T")
    for a in mods:
        assert float(a.consume[-1].weight.abs().max()) == 0.0 and float(a.consume[-1].bias.abs().max()) == 0.0
    stk = rs.StackedArms(M, d).load_members([{k: v.detach().clone() for k, v in a.state_dict().items()} for a in mods])
    h = torch.randn(M, 5, d)
    assert float(stk.write(h).abs().max()) == 0.0, "exact-inert at birth"
    for a in mods:
        _randomize(a)
    stk = rs.StackedArms(M, d).load_members([{k: v.detach().clone() for k, v in a.state_dict().items()} for a in mods])
    out = stk.write(h)
    for j, a in enumerate(mods):
        ref = a(h[j:j + 1]) - h[j:j + 1]
        assert torch.allclose(out[j:j + 1], ref, atol=1e-6, rtol=1e-5), float((out[j:j + 1] - ref).abs().max())
        sd = {k: v for k, v in stk.member(j).items()}
        for k, v in a.state_dict().items():
            assert torch.equal(sd[k].reshape(v.shape), v), k
    allw = stk.write_all(h[0])
    assert allw.shape == (5, M, d) and torch.allclose(allw[:, 1], mods[1](h[0][None])[0] - h[0], atol=1e-6, rtol=1e-5)
    idx = torch.tensor([2, 0])
    assert torch.allclose(stk.write(h[[2, 0]], idx), out[[2, 0]], atol=1e-6)


class _Member(nn.Module):
    """The night's member (capacity_bench.Member), verbatim."""

    def __init__(self, d=128, hid=128):
        super().__init__()
        self.ln = nn.LayerNorm(d)
        self.w1 = nn.Linear(d, hid)
        self.w2 = nn.Linear(hid, d)
        nn.init.zeros_(self.w2.weight)
        nn.init.zeros_(self.w2.bias)

    def forward(self, h):
        return self.w2(F.gelu(self.w1(self.ln(h))))


def test_stacked_toys_equal_the_nights_member():
    d, M, hid = 32, 3, 48
    stk = rs.StackedToys.build(M, d, "T", hid=hid)
    refs = []
    for j in range(M):
        torch.manual_seed(rs.crc(f"T|{j}"))
        refs.append(_Member(d, hid))
    h = torch.randn(M, 4, d)
    assert float(stk.write(h).abs().max()) == 0.0
    for j, r in enumerate(refs):
        _randomize(r, j)
        with torch.no_grad():
            stk.lnw[j].copy_(r.ln.weight)
            stk.lnb[j].copy_(r.ln.bias)
            stk.w1[j].copy_(r.w1.weight)
            stk.b1[j].copy_(r.w1.bias)
            stk.w2[j].copy_(r.w2.weight)
            stk.b2[j].copy_(r.w2.bias)
    out = stk.write(h)
    for j, r in enumerate(refs):
        assert torch.allclose(out[j], r(h[j]), atol=1e-6, rtol=1e-5)


def test_stacked_adam_is_each_members_own_adam():
    M, steps = 3, 6
    torch.manual_seed(0)
    w = nn.Parameter(torch.randn(M, 4, 5))
    refs = [nn.Parameter(w[j].detach().clone()) for j in range(M)]
    opts = [torch.optim.Adam([p], lr=1e-2, betas=rs.BETAS, eps=rs.EPS, weight_decay=0.0) for p in refs]
    sa = rs.StackedAdam([w], M)
    g = torch.Generator().manual_seed(1)
    for t in range(steps):
        grad = torch.randn(M, 4, 5, generator=g)
        present = torch.tensor([True, t % 2 == 0, t >= 3])
        w.grad = grad * present.view(-1, 1, 1)
        sa.step(present, 1e-2)
        for j in range(M):
            if present[j]:
                refs[j].grad = grad[j].clone()
                opts[j].step()
    for j in range(M):
        assert torch.allclose(w[j], refs[j], atol=1e-6), (j, float((w[j] - refs[j]).abs().max()))
    assert sa.t.tolist() == [6.0, 3.0, 3.0]


def test_the_gate_on_exact_requests():
    dev = torch.device("cpu")
    L9 = rs.level_code(dev)
    own, _ = rs.owned_sets()
    mem = torch.tensor([[a // 8, a % 8] for a in own])
    req = mem[[0, 5, 47]].clone()
    req = torch.cat([req, torch.tensor([[gr.NULL, gr.NULL], [3, 9]])])
    fac = rs.exact_factors(req, mem, L9)
    for U in rs.U_SET:
        for m in rs.M_SET:
            w = rs.norm_w(rs.u_of(fac, U, m), "REST")
            assert abs(float(w[0, 0]) - math.tanh(U)) < 1e-5 and abs(float(w[1, 5]) - math.tanh(U)) < 1e-5
            off = w.clone()
            off[0, 0] = off[1, 5] = 0.0
            if m == 48:
                off[2, 47] = 0.0
                assert abs(float(w[2, 47]) - math.tanh(U)) < 1e-5
            assert float(off.abs().max()) < 1e-5, "a non-owner, a null request or a non-digit reads (near) zero"
            wf = rs.norm_w(rs.u_of(fac, U, m), "FULL")
            assert abs(float(wf[0, 0]) - rs.own_weight("FULL", U, m)) < 1e-5


def test_the_floor():
    s = torch.tensor([0.0, 0.3, 0.5, 0.9, 1.0])
    assert torch.equal(rs.floor_fac(s, 0.0), s)
    f = rs.floor_fac(s, 0.5)
    assert f.tolist()[:3] == [0.0, 0.0, 0.0] and abs(float(f[3]) - 0.8) < 1e-6 and float(f[4]) == 1.0
    assert torch.equal(rs.floor_fac(s, 1.0), torch.zeros_like(s)), "a saturated tau silences every member (no 0 / 0)"
    assert not torch.isnan(rs.floor_fac(torch.ones(3), 1.0)).any()
    hs = torch.rand(7, 2, 8)
    mem = torch.tensor([[1, 2], [7, 0]])
    hf = rs.head_factors(hs, mem, tau=[0.2, 0.4])
    assert torch.allclose(hf[0][:, 1], rs.floor_fac(hs[:, 0, 7], 0.2)) and torch.allclose(hf[1][:, 0], rs.floor_fac(hs[:, 1, 2], 0.4))


def test_lr_schedule_is_the_s_lines():
    assert abs(rs.lr_at(0, 1000) - 1e-3 / 50) < 1e-12 and abs(rs.lr_at(49, 1000) - 1e-3) < 1e-12
    assert abs(rs.lr_at(999, 1000) - 1e-4) < 1e-6
