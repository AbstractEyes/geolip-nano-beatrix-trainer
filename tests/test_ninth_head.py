# -*- coding: utf-8 -*-
"""The path in a head of its own (0.1.94): a ninth softmax head in every anchored block, its output added to the block's. Off, the batches
are 0.1.93's bit for bit; the heads are born from their own generator (the run's untouched; the two rotations' heads equal; the house's
orthogonal statistics); a silent ninth head (its output rows zero) leaves the forward exactly the plain craft's; the head is its written
form (by hand, both rotations, the anchored one with the row's streams too); the path form scores the continuation on all sixteen pairs;
the toggle is the control's rotation exactly; the side file round-trips through rotary_scale.Core, its text read (with the toggle) and the
room read's look (the ninth as each block's head 8); the names and the runner carry the flag. CPU only; the Hub is never touched."""
import ast
import hashlib
import json
import math
import os

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from geolip.alephllm.model.attention import rope_half
from geolip.nano_beatrix import anchored_rope as ar
from geolip.nano_beatrix import bed_runner as br
from geolip.nano_beatrix import grid_rows as gr
from geolip.nano_beatrix import ninth_head as nhm
from geolip.nano_beatrix import path_rotary as prm
from geolip.nano_beatrix import room_read as rr
from geolip.nano_beatrix import rotary_scale as rs
from geolip.nano_beatrix.names import run_tag
from test_key_code import _save_core
from test_key_hard import FP_0174_BATCH, batch_digest
from test_path_rotary import _text_rows, _units, space_table  # noqa: F401
from test_scale import _bed_batch
from test_value_addr import STEM, W_BASE, _craft, _streams

D = 256


def test_off_the_batches_are_0193s_bit_for_bit():
    assert batch_digest(_bed_batch(4)) == FP_0174_BATCH
    src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "geolip", "nano_beatrix", "bed_train.py"), encoding="utf-8").read()
    fns = {n.name: ast.get_source_segment(src, n) for n in ast.parse(src).body if isinstance(n, ast.FunctionDef)}
    for fn in ("cap", "draw_plan", "frame_row", "ag_rows", "grid_aux", "batch"):            # the batch's whole path (test_scale._bed_batch)
        assert "NINTH" not in fns[fn] and "ninth" not in fns[fn] and "nhm" not in fns[fn]


def test_the_birth_is_its_own_draw(space_table):                                             # noqa: F811
    rng = torch.random.get_rng_state()
    a = nhm.NinthHeads(D, 6, "plain", seed=3)
    b = nhm.NinthHeads(D, 6, "path", seed=3, units=_units(space_table), form="next")
    assert torch.equal(torch.random.get_rng_state(), rng)                                    # the run's generator untouched
    for i in range(6):
        assert torch.equal(a.qkv[i].weight, b.qkv[i].weight) and torch.equal(a.o[i].weight, b.o[i].weight)   # equal at birth
    w, o = nhm.birth(D, 3, 1)
    assert torch.equal(a.qkv[0].weight, w) and torch.equal(a.o[0].weight, o) and w.shape == (96, D) and o.shape == (D, 32)
    assert torch.allclose(o.T @ o, torch.eye(32), atol=1e-5)                                 # columns of an orthogonal d x d draw
    assert abs(float((w * w).sum(1).mean()) - 1 / 3) < 0.05                                  # rows of an orthogonal 3d x d draw (norm^2 1/3)
    c = nhm.NinthHeads(D, 6, "plain", seed=4)
    assert not torch.equal(a.qkv[0].weight, c.qkv[0].weight) and not torch.equal(a.qkv[0].weight, a.qkv[1].weight)
    assert nhm.params(a) == 6 * (96 * D + D * 32) == 196608
    assert sorted(a.state_dict()) == sorted([f"qkv.{i}.weight" for i in range(6)] + [f"o.{i}.weight" for i in range(6)])
    assert "table" in nhm.state(b) and "table" not in nhm.state(a)


@pytest.mark.parametrize("rot", nhm.ROTS)
def test_a_silent_ninth_head_is_the_plain_forward(space_table, rot):                        # noqa: F811
    x = _text_rows(B=2, n=200)
    m, st = _craft(n_ctx=256, seed=5)
    with torch.no_grad():
        base = m(x).logits
    m2, st2 = _craft(n_ctx=256, seed=5)
    nh = nhm.NinthHeads(m2.cfg.d_model, 6, rot, seed=5, units=_units(space_table), form="next")
    with torch.no_grad():
        for o in nh.o:
            o.weight.zero_()
    assert nhm.patch(m2, nh) == 6
    assert sorted(m2.state_dict()) == sorted(m.state_dict())                                 # no key added to the model
    with torch.no_grad():
        got = m2(x).logits
    assert torch.equal(got, base)
    nh2 = nhm.NinthHeads(m2.cfg.d_model, 6, rot, seed=5, units=_units(space_table), form="next")
    m3, _ = _craft(n_ctx=256, seed=5)
    nhm.patch(m3, nh2)
    with torch.no_grad():
        assert not torch.equal(m3(x).logits, base)                                          # a born head speaks


def _by_hand(nh, i, x, xb, st):
    B, n, _ = x.shape
    w = nh.qkv[i].weight
    q, k, v = (x @ w.T).view(B, n, 3, 1, 32).permute(2, 0, 3, 1, 4)
    if nh.rot == "path" and not nh.off:
        phi = nh.path.phases(xb)
        pk = phi if nh.form == "same" else torch.cat([torch.zeros_like(phi[:, :1]), phi[:, :-1]], dim=1)
        aq, ak = phi[:, None], pk[:, None]
    else:
        aq, ak = st.angles(B, n, 0, x.device)
    q, k = rope_half(q, aq.cos(), aq.sin()), rope_half(k, ak.cos(), ak.sin())
    s = (q @ k.transpose(-1, -2)) / math.sqrt(32)
    s = s.masked_fill(~torch.ones(n, n, dtype=torch.bool).tril(), float("-inf"))
    y = s.softmax(-1) @ v
    return y.transpose(1, 2).reshape(B, n, 32) @ nh.o[i].weight.T


@pytest.mark.parametrize("rot", nhm.ROTS)
def test_the_head_is_its_written_form(space_table, rot):                                    # noqa: F811
    torch.manual_seed(0)
    xb = _text_rows(B=2, n=96, seed=4)
    x = torch.randn(2, 96, D)
    m, st = _craft(n_ctx=256, seed=1)
    nh = nhm.NinthHeads(D, 6, rot, seed=1, units=_units(space_table), form="next")
    nh.set(xb)
    with torch.no_grad():
        got = nh.head(2, x, st)
        exp = _by_hand(nh, 2, x, xb, st)
    assert torch.allclose(got, exp, atol=2e-5, rtol=0)
    if rot == "plain":                                                                       # with the row's streams: the block's own angles
        dk, dq = _streams(2, 96, seed=3)
        st.set(dk, dq)
        q, k = torch.randn(2, 1, 96, 32), torch.randn(2, 1, 96, 32)
        aq, ak = st.angles(2, 96, 0, q.device)
        rq, rk = nh.rotate(q, k, st)
        assert torch.equal(rq, rope_half(q, aq.cos(), aq.sin())) and torch.equal(rk, rope_half(k, ak.cos(), ak.sin()))
        st.clear()


def test_the_path_form_scores_the_continuation(space_table):                                # noqa: F811
    row = b"go the cat sat on the cab and the cat ran away; the cat sat"
    xb = torch.tensor(list(row), dtype=torch.long)[None]
    n = xb.shape[1]
    nh = nhm.NinthHeads(D, 1, "path", seed=1, units=_units(space_table), form="next")
    nh.set(xb)
    c = torch.randn(32)
    q = k = c.expand(1, 1, n, 32).clone()
    rq, rk = nh.rotate(q, k, None)
    sc = nhm.scores(rq, rk)[0] * math.sqrt(32)                                               # <q(t), k(u)>
    full = float((c * c).sum())
    M = rr.continuation(torch.from_numpy(rr.path_ids(nh.units, xb)[0]))
    assert M.any()
    hit = sc[M]
    assert torch.allclose(hit, torch.full_like(hit, full), atol=1e-4)                       # the continuation meets as unrotated
    rest = sc[(~M) & torch.ones(n, n, dtype=torch.bool).tril()]
    assert float(rest.max()) < full - 1e-3                                                   # every other earlier byte does not


def test_the_toggle_is_the_controls_rotation(space_table):                                  # noqa: F811
    x = _text_rows(B=2, n=200)
    mp, _ = _craft(n_ctx=256, seed=7)
    mc, _ = _craft(n_ctx=256, seed=7)
    hp = nhm.NinthHeads(D, 6, "path", seed=7, units=_units(space_table), form="next")
    hc = nhm.NinthHeads(D, 6, "plain", seed=7)
    nhm.patch(mp, hp)
    nhm.patch(mc, hc)
    with torch.no_grad():
        on = mp(x).logits
        hp.off = True
        off = mp(x).logits
        hp.off = False
        ctl = mc(x).logits
    assert torch.equal(off, ctl) and not torch.equal(on, ctl)


@pytest.fixture()
def short(monkeypatch, space_table):                                                         # noqa: F811
    monkeypatch.setattr(gr, "CTX", 256)
    monkeypatch.setenv(prm.TABLE_ENV, space_table)
    monkeypatch.setattr(prm, "TABLE_SHA256", hashlib.sha256(open(space_table, "rb").read()).hexdigest())
    return space_table


def _held(path, n=3, seed=3):
    g = np.random.default_rng(seed)
    words = [b"the", b"cat", b"sat", b"on", b"a", b"mat", b"and", b"ran", b"Tom", b"42"]
    with open(path, "w", encoding="utf-8") as f:
        for _ in range(n):
            f.write(json.dumps({"text": b" ".join(words[int(i)] for i in g.integers(0, len(words), size=1500)).decode()}) + "\n")
    return str(path)


@pytest.mark.parametrize("rot", nhm.ROTS)
def test_the_side_file_round_trips_through_the_core_and_its_reads(tmp_path, short, rot):
    from safetensors.torch import save_file
    m, st = _craft(n_ctx=4096, seed=2)
    va = ar.ValueAddress(m.cfg.d_model, st)
    with torch.no_grad():
        va.table.normal_(std=0.5)
    ar.patch_value(m, va)
    nh = nhm.NinthHeads(m.cfg.d_model, 6, rot, seed=2, units=(prm.Units() if rot == "path" else None), form="next")
    with torch.no_grad():
        for o in nh.o:
            o.weight.mul_(3.0)                                                               # a trained head's weight (not the birth)
    nhm.patch(m, nh)
    d = tmp_path / "core"
    sj = _save_core(m, d, va)
    save_file(nhm.state(nh), str(d / "extra_head.safetensors"))
    sj["ninth_head"] = nhm.scale_field(nh)
    json.dump(sj, open(d / "scale.json", "w", encoding="utf-8"), indent=1)
    core = rs.Core(str(d), "cpu", look=None)
    assert core.ninth is not None and core.ninth.rot == rot and nhm.digest(core.ninth) == nhm.digest(nh)
    held = _held(tmp_path / "held.jsonl")
    docs = rs.load_heldout(held)[:2]
    xs = torch.tensor(np.stack([np.frombuffer(bytes([rs.DOC]) + dd[:gr.CTX], dtype=np.uint8) for dd in docs]).astype(np.int64))
    with torch.no_grad():
        exp = m(xs[:, :-1]).logits
        got = core.readout(core.trunk(xs[:, :-1]))
        h3 = core.trunk(xs[:, :-1], 0, 3)
        split = core.readout(core.trunk(xs[:, :-1], 3, None, h3))                          # from a given state: the phases set from x
    assert torch.equal(got, exp) and torch.equal(split, exp)                                 # the reader's forward = the trainer's
    rec = rs.text_read(str(d), str(tmp_path / "text.json"), held, n_docs=2, mb=2, dev="cpu", log=lambda s: None)
    with torch.no_grad():
        nll = F.cross_entropy(exp.float().reshape(-1, 256), xs[:, 1:].reshape(-1), reduction="none").numpy()
    assert abs(rec["bits"] - float(nll.sum()) / nll.size / math.log(2)) < 1e-9
    if rot == "path":
        nh.off = True
        with torch.no_grad():
            lo = m(xs[:, :-1]).logits
        nh.off = False
        nll_off = F.cross_entropy(lo.float().reshape(-1, 256), xs[:, 1:].reshape(-1), reduction="none").numpy()
        assert abs(rec["path_off"]["bits"] - float(nll_off.sum()) / nll_off.size / math.log(2)) < 1e-9
        assert rec["path"] == dict(form="next", head="ninth", k=prm.K) and rec["path_off"]["bits"] != rec["bits"]
    else:
        assert rec["path"] is None and "path_off" not in rec and rec["by_repeat"]["n_repeat"] > 0
    lk = rr.look(core, docs, core.ninth.units if rot == "path" else prm.Units(), rows=2, mb=2)
    assert all(len(b) == 9 for b in lk["blocks"]) and core.ninth.capture is False            # the ninth read as each block's head 8
    rm = rr.read(str(d), str(tmp_path / "room.json"), held, "look", look_rows=2, n_docs=2, mb=2, dev="cpu", log=lambda s: None)
    assert rm["removed"] == {} and rm["base"]["bits"] == rec["bits"] and rm["ninth"] == dict(rot=rot, form=core.ninth.form)
    assert [[h["m"] for h in b] for b in rm["look"]["blocks"]] == [[h["m"] for h in b] for b in lk["blocks"]]   # the room look alone
    dg = core.state_digest()
    with torch.no_grad():
        core.ninth.qkv[0].weight[0, 0] += 0.5
    assert core.state_digest() != dg                                                         # a moved weight is a moved core
    os.remove(d / "extra_head.safetensors")
    with pytest.raises(AssertionError):
        rs.Core(str(d), "cpu", look=None)


def test_a_stored_field_is_checked(space_table):                                            # noqa: F811
    f = nhm.scale_field(nhm.NinthHeads(D, 6, "path", seed=1, units=_units(space_table), form="next"))
    nhm.check_field(f)
    nhm.check_field(nhm.scale_field(nhm.NinthHeads(D, 6, "plain", seed=1)))
    for k, v in (("rot", "x"), ("seed_tag", "PATHROT|v1"), ("hd", 16), ("birth_tag", "NINTH|v0")):
        with pytest.raises(AssertionError):
            nhm.check_field(dict(f, **{k: v}))


def test_the_names_and_the_runner():
    pk = dict(W_BASE, grid_hard="0.5", value_addr="1", ptr_lam="0.1", grid_comps="48", key_rot="1", key_pairs="16", key_norm="2",
              key_rho="2.18", plant="0.75", plain_doc="1")
    stem = STEM.format(1).replace("_askany_", "_askany_p75d_") + "gc48_kh50_va1_pt10b4h0_krt16r218_"
    assert run_tag(dict(pk, series="plain16")) == stem + "ahdigit1_plain16"                 # off: 0.1.93's name
    assert run_tag(dict(pk, ninth_head="path", path_form="next", series="pillar1b")) == stem + "nhpn16_ahdigit1_pillar1b"
    assert run_tag(dict(pk, ninth_head="plain", series="pillar1b")) == stem + "nhpl_ahdigit1_pillar1b"
    assert run_tag(dict(pk, ninth_head="path", path_form="same", path_k="12", series="x")) == stem + "nhps16k12_ahdigit1_x"
    runs = br.expand(dict(arms="alhena-both", seeds="1", grid="ag", grid_p="0.5", head="digit", head_lam="1", rotary_content="4",
                          grid_hard="0.5", value_addr="1", ptr_lam="0.1", key_rot="1", grid_comps="48", key_pairs="16", key_norm="2",
                          key_rho="2.18", plant="0.75", plain_doc="1", ninth_head="path,plain", path_form="next", series="pillar1b"))
    assert len(runs) == 2 and {[a for a in r if a.startswith("--ninth_head=")][0] for r in runs} == {"--ninth_head=path", "--ninth_head=plain"}
    labels = [br.run_label(r) for r in runs]
    assert any("a ninth head with the path (next)" in lb for lb in labels) and any("a ninth head on the anchored rotation" in lb for lb in labels)
    assert "ninth_head" in br.GRID_KEYS
