# -*- coding: utf-8 -*-
"""The room read (0.1.93): with no removal the read's forward is the core's own bit for bit and its bits are the text read's; a removal
zeroes only the named head's named pairs of q and k in every block, after the block's own rotation (the path's turn included), and the
output removal only that head's values - the block's output is the forward with the head's slice zeroed; the pair groups follow the core's
split; the look's attention is the kernel's, the continuation set is its definition and its non-empty rows are the repeat mask, and the
look's mass is the direct sum; the record carries every removal's cost and resumes. CPU only; the Hub is never touched."""
import hashlib
import json
import math

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from geolip.nano_beatrix import anchored_rope as ar
from geolip.nano_beatrix import grid_rows as gr
from geolip.nano_beatrix import path_rotary as prm
from geolip.nano_beatrix import room_read as rr
from geolip.nano_beatrix import rotary_scale as rs
from test_key_code import _save_core
from test_path_rotary import PHEAD, _save_path_core, _text_rows, _units, space_table  # noqa: F401
from test_value_addr import _craft

CTX = 256                                                                                    # short rows: the reads' law at any length


@pytest.fixture()
def short(monkeypatch, space_table):                                                         # noqa: F811
    monkeypatch.setattr(gr, "CTX", CTX)
    monkeypatch.setenv(prm.TABLE_ENV, space_table)                                            # the cores read the made table, pinned
    monkeypatch.setattr(prm, "TABLE_SHA256", hashlib.sha256(open(space_table, "rb").read()).hexdigest())
    return space_table


def _held(path, n=3, seed=3):
    g = np.random.default_rng(seed)
    words = [b"the", b"cat", b"sat", b"on", b"a", b"mat", b"and", b"ran", b"Tom", b"42"]
    with open(path, "w", encoding="utf-8") as f:
        for _ in range(n):
            f.write(json.dumps({"text": b" ".join(words[int(i)] for i in g.integers(0, len(words), size=1500)).decode()}) + "\n")
    return str(path)


def _cores(tmp_path, form="next", seed=2):
    """A path core (the form's) and the same craft without the path, saved as the trainer saves them."""
    m, st = _craft(n_ctx=4096, seed=seed)
    va = ar.ValueAddress(m.cfg.d_model, st)
    with torch.no_grad():
        va.table.normal_(std=0.5)
    ar.patch_value(m, va)
    d0 = tmp_path / "plain"
    _save_core(m, d0, va)
    pr = prm.PathRotary(prm.Units(), form, head=PHEAD)
    prm.patch(m, pr)
    d1 = tmp_path / "path"
    _save_path_core(m, d1, pr, va)
    return str(d0), str(d1)


def _x(held, n_docs=2):
    docs = rs.load_heldout(held)[:n_docs]
    return docs, torch.tensor(np.stack([np.frombuffer(bytes([rs.DOC]) + d[:gr.CTX], dtype=np.uint8) for d in docs]).astype(np.int64))


def test_the_groups_follow_the_cores_split(tmp_path, short):
    d0, _ = _cores(tmp_path)
    g = rr.groups(rs.Core(d0, "cpu", look=None))
    assert g == dict(rel=(0, 1, 2, 3), content=(4, 5, 6, 7), anchors=tuple(range(8, 16)), scores=tuple(range(16)))   # ALHENA-W's split
    m, st = _craft(n_ctx=4096, seed=1, content=2)

    class _C:
        anchor = st
    assert rr.groups(_C()) == dict(rel=(0, 1, 2, 3, 4, 5), content=(6, 7), anchors=tuple(range(8, 16)), scores=tuple(range(16)))
    core = rs.Core(d0, "cpu", look=None)
    names = [p[0] for p in rr.plan(core, "plain")]
    assert names == [f"h{h}_anchors" for h in range(8)] + ["h7_rel", "h7_content", "h7_scores", "h7_out"]
    assert [p[0] for p in rr.plan(core, "path")] == ["h7_anchors", "h7_out"]


def test_no_removal_is_the_cores_forward_and_its_text_read(tmp_path, short):
    d0, d1 = _cores(tmp_path)
    held = _held(tmp_path / "held.jsonl")
    for d in (d0, d1):
        core = rs.Core(d, "cpu", look=None)
        docs, x = _x(held)
        with torch.no_grad():
            exp = core.model(x[:, :-1])[0]
        rm = rr.Removal(core.model)
        assert rm.n == len(core.model.blocks)
        with torch.no_grad():
            got = core.model(x[:, :-1])[0]
        assert torch.equal(got, exp)                                                         # the wrappers pass through untouched
        units = core.path_rot.units if core.path_rot is not None else prm.Units()
        tb = rr.text_bits(core, docs, units, mb=2)
        tr = rs.text_read(d, str(tmp_path / f"text_{len(d)}_{d[-4:]}.json"), held, n_docs=2, mb=2, dev="cpu", log=lambda s: None)
        assert tb["bits"] == tr["bits"] and tb["by_repeat"] == tr["by_repeat"]               # the text read's numbers, exactly
        assert set(tb["by_position"]) >= {"row"}


def _capture(model):
    """An outer capture of every block's rotated q, k (what the kernel reads) and of its v."""
    cap = []
    for b in model.blocks:
        a = b.attn
        rec = {}
        inner_r, inner_s = a._rotate, a._split

        def _rotate(q, k, offset=0, _inner=inner_r, _rec=rec):
            qo, ko = _inner(q, k, offset)
            _rec["q"], _rec["k"] = qo.clone(), ko.clone()
            return qo, ko

        def _split(x, n, _inner=inner_s, _rec=rec):
            q, k, v = _inner(x, n)
            _rec["v"] = v.clone()
            return q, k, v
        a._rotate, a._split = _rotate, _split
        cap.append(rec)
    return cap


def test_a_removal_moves_only_its_head_and_pairs(tmp_path, short):
    _, d1 = _cores(tmp_path)
    held = _held(tmp_path / "held.jsonl")
    core = rs.Core(d1, "cpu", look=None)                                                     # a path core: the removal after the turn
    rm = rr.Removal(core.model)
    cap = _capture(core.model)
    docs, x = _x(held)
    g = rr.groups(core)
    with torch.no_grad():
        core.model(x[:, :-1])
    ref = [dict(r) for r in cap]
    dims = list(g["anchors"]) + [j + 16 for j in g["anchors"]]
    other = [j for j in range(32) if j not in dims]
    assert all(r["q"][:, PHEAD][..., dims].abs().sum() > 0 for r in ref)                     # the path's turned pairs carry signal
    rm.set(PHEAD, g["anchors"])
    with torch.no_grad():
        core.model(x[:, :-1])
    for i, r in enumerate(cap):
        assert torch.equal(r["q"][:, PHEAD][..., dims], torch.zeros_like(r["q"][:, PHEAD][..., dims]))   # zeroed in every block
        assert torch.equal(r["k"][:, PHEAD][..., dims], torch.zeros_like(r["k"][:, PHEAD][..., dims]))
    r0, f0 = cap[0], ref[0]                                                                  # block 1: nothing else moved
    keep = [h for h in range(8) if h != PHEAD]
    assert torch.equal(r0["q"][:, keep], f0["q"][:, keep]) and torch.equal(r0["k"][:, keep], f0["k"][:, keep])
    assert torch.equal(r0["q"][:, PHEAD][..., other], f0["q"][:, PHEAD][..., other]) and torch.equal(r0["v"], f0["v"])
    assert not torch.equal(cap[-1]["q"], ref[-1]["q"])                                       # later blocks read the changed output
    rm.clear()
    with torch.no_grad():
        core.model(x[:, :-1])
    assert all(torch.equal(cap[i]["q"], ref[i]["q"]) and torch.equal(cap[i]["v"], ref[i]["v"]) for i in range(len(cap)))


def test_the_output_removal_is_the_forward_with_the_heads_slice_zeroed(tmp_path, short):
    d0, _ = _cores(tmp_path)
    core = rs.Core(d0, "cpu", look=None)
    a = core.model.blocks[0].attn
    torch.manual_seed(0)
    h_in = torch.randn(2, 64, core.cfg.d_model)
    with torch.no_grad():                                                                    # by hand, before any wrapper
        q, k, v = a._split(h_in, 64)
        q, k = a._guard(q, k)
        q, k = a._rotate(q, k)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y[:, 3] = 0
        exp = a.o(y.transpose(1, 2).reshape(2, 64, -1))
    rm = rr.Removal(core.model)
    rm.set(3, (), out=True)
    with torch.no_grad():
        got = a(h_in)
    assert torch.allclose(got, exp, atol=1e-6, rtol=0)
    rm.set(3, (4, 5), out=False)
    with torch.no_grad():
        got2 = a(h_in)
        q2, k2 = q.clone(), k.clone()
        for j in (4, 5, 20, 21):
            q2[:, 3, :, j] = 0
            k2[:, 3, :, j] = 0
        exp2 = a.o(F.scaled_dot_product_attention(q2, k2, v, is_causal=True).transpose(1, 2).reshape(2, 64, -1))
    assert torch.allclose(got2, exp2, atol=1e-6, rtol=0)


def test_the_look_is_the_kernels_attention_and_the_continuation_set(space_table):           # noqa: F811
    torch.manual_seed(1)
    H, n, hd = 4, 48, 32
    q, k = torch.randn(H, n, hd), torch.randn(H, n, hd)
    eye = torch.eye(n).expand(1, H, n, n)
    ker = F.scaled_dot_product_attention(q[None], k[None], eye, is_causal=True)[0]
    assert torch.allclose(rr.attention(q, k), ker, atol=1e-6, rtol=0)                        # the kernel's own weights
    g = torch.Generator().manual_seed(2)
    ids = torch.randint(0, 6, (n,), generator=g)
    M = rr.continuation(ids)
    for t in range(n):
        for u in range(n):
            assert bool(M[t, u]) == (1 <= u <= t and int(ids[u - 1]) == int(ids[t]))
    x = _text_rows(B=3, n=400)
    units = _units(space_table)
    ids2 = rr.path_ids(units, x)
    rep = prm.repeat_mask(units, x)
    for b in range(3):
        assert np.array_equal(rr.continuation(torch.from_numpy(ids2[b])).any(-1).numpy(), rep[b])   # non-empty exactly at the repeats


def test_the_looks_mass_is_the_direct_sum_and_the_record(tmp_path, short):
    d0, d1 = _cores(tmp_path)
    held = _held(tmp_path / "held.jsonl")
    core = rs.Core(d1, "cpu", look=None)
    docs, x = _x(held, 2)
    units = core.path_rot.units
    lk = rr.look(core, docs, units, rows=2, mb=2)
    cap = _capture(core.model)
    with torch.no_grad():
        core.model(x[:, :-1])
    ids = rr.path_ids(units, x[:, :-1])
    dep = units.depth(x[:, :-1])
    for bi in (0, len(cap) - 1):
        for h in (0, PHEAD):
            tot, cnt, tot1, cnt1 = 0.0, 0, 0.0, 0
            for r in range(2):
                A = rr.attention(cap[bi]["q"][r], cap[bi]["k"][r])[h]
                M = rr.continuation(torch.from_numpy(ids[r]))
                for t in range(A.shape[0]):
                    if M[t].any():
                        s = float(A[t][M[t]].sum())
                        tot, cnt = tot + s, cnt + 1
                        if int(dep[r, t]) == 1:
                            tot1, cnt1 = tot1 + s, cnt1 + 1
            assert cnt == lk["n_repeat"] and abs(lk["blocks"][bi][h]["m"] - tot / cnt) < 1e-6
            assert abs(lk["blocks"][bi][h]["by_length"]["1"] - tot1 / cnt1) < 1e-6
            assert 0.0 <= lk["blocks"][bi][h]["argmax_share"] <= 1.0
    assert 0 < lk["uniform"] < 1 and lk["n_by_length"]["1"] == cnt1
    # the record: every removal of the plan with its cost against the base, the base = the text read's, resume-first
    out = tmp_path / "room" / "plain.json"
    rec = rr.read(d0, str(out), held, "plain", look_rows=2, n_docs=2, mb=2, dev="cpu", log=lambda s: None)
    tr = rs.text_read(d0, str(tmp_path / "text_plain.json"), held, n_docs=2, mb=2, dev="cpu", log=lambda s: None)
    assert rec["base"]["bits"] == tr["bits"] and rec["path"] is None and rec["kind"] == "plain"
    assert list(rec["removed"]) == [p[0] for p in rr.plan(rs.Core(d0, "cpu", look=None), "plain")]
    for name, r in rec["removed"].items():
        assert r["cost"] == r["bits"] - rec["base"]["bits"] and r["bits"] != rec["base"]["bits"], name
    assert rec["removed"]["h7_out"]["out"] and rec["removed"]["h7_out"]["pairs"] == []
    assert len(rec["look"]["blocks"]) == 6 and len(rec["look"]["blocks"][0]) == 8
    assert rr.read(d0, str(out), held, "plain", look_rows=2, n_docs=2, dev="cpu", log=lambda s: None) == rec
    rec1 = rr.read(d1, str(tmp_path / "room" / "path.json"), held, "path", look_rows=0, n_docs=2, mb=2, dev="cpu", log=lambda s: None)
    assert rec1["path"] == dict(form="next", head=PHEAD, k=prm.K) and list(rec1["removed"]) == ["h7_anchors", "h7_out"] and "look" not in rec1
    assert rr.room_name("abc") == "abc_room"
