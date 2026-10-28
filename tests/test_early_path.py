# -*- coding: utf-8 -*-
"""The path in the early blocks (0.1.95): the ninth heads' path on a named set of blocks, the others on the block's anchored rotation. The
set's ends are the two forms exactly (every block = 0.1.94's path form, no block = the plain form); the named blocks turn by the path and
the others by the anchored angles, each head its written form; the toggle is the plain form's forward exactly; the set is recorded only
when named (every block keeps 0.1.94's field) and checked when read; the side file round-trips through rotary_scale.Core, its text read
(with the toggle) and the room read's look; the names carry the set and the runner passes it whole. CPU only; the Hub is never touched."""
import json
import math

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from geolip.nano_beatrix import anchored_rope as ar
from geolip.nano_beatrix import bed_runner as br
from geolip.nano_beatrix import grid_rows as gr
from geolip.nano_beatrix import ninth_head as nhm
from geolip.nano_beatrix import path_rotary as prm
from geolip.nano_beatrix import room_read as rr
from geolip.nano_beatrix import rotary_scale as rs
from geolip.nano_beatrix.names import run_tag
from test_key_code import _save_core
from test_ninth_head import _by_hand, _held, short  # noqa: F401
from test_path_rotary import _text_rows, _units, space_table  # noqa: F401
from test_value_addr import STEM, W_BASE, _craft

D = 256


def _patched(space_table, rot, seed=7, **kw):                                                 # noqa: F811
    m, _ = _craft(n_ctx=256, seed=seed)
    nh = nhm.NinthHeads(D, 6, rot, seed=seed, units=(_units(space_table) if rot == "path" else None), form="next", **kw)
    nhm.patch(m, nh)
    return m, nh


def test_the_sets_ends_are_the_two_forms_and_the_toggle_is_the_plain_forward(space_table):   # noqa: F811
    x = _text_rows(B=2, n=200)
    with torch.no_grad():
        every = _patched(space_table, "path")[0](x).logits                                    # 0.1.94's path form
        full = _patched(space_table, "path", path_blocks=range(1, 7))[0](x).logits
        empty = _patched(space_table, "path", path_blocks=())[0](x).logits
        plain = _patched(space_table, "plain")[0](x).logits
        m, nh = _patched(space_table, "path", path_blocks=(1, 2, 3))
        early = m(x).logits
        nh.off = True
        off = m(x).logits
        nh.off = False
        again = m(x).logits
    assert torch.equal(full, every) and torch.equal(empty, plain)
    assert torch.equal(off, plain) and torch.equal(again, early)
    assert not torch.equal(early, every) and not torch.equal(early, plain)


def test_the_named_blocks_turn_by_the_path_the_others_by_the_anchored_angles(space_table):   # noqa: F811
    torch.manual_seed(0)
    xb = _text_rows(B=2, n=96, seed=4)
    _, st = _craft(n_ctx=256, seed=1)
    u = _units(space_table)
    every = nhm.NinthHeads(D, 6, "path", seed=1, units=u, form="next")
    part = nhm.NinthHeads(D, 6, "path", seed=1, units=u, form="next", path_blocks=(3, 1, 2, 2))
    plain = nhm.NinthHeads(D, 6, "plain", seed=1)
    assert part.path_blocks == (1, 2, 3) and every.path_blocks is None
    every.set(xb)
    part.set(xb)
    q, k = torch.randn(2, 1, 96, 32), torch.randn(2, 1, 96, 32)
    for i in range(6):
        gq, gk = part.rotate(q, k, st, i)
        eq, ek = (every if i < 3 else plain).rotate(q, k, st, i)
        assert torch.equal(gq, eq) and torch.equal(gk, ek)
    with pytest.raises(AssertionError):
        part.rotate(q, k, st)                                                                # a set needs the block's index
    x = torch.randn(2, 96, D)
    with torch.no_grad():
        for i in range(6):
            exp = _by_hand(every if i < 3 else plain, i, x, xb, st)
            assert torch.allclose(part.head(i, x, st), exp, atol=2e-5, rtol=0)              # each head its written form


def test_the_set_is_recorded_and_checked(space_table):                                       # noqa: F811
    u = _units(space_table)
    f0 = nhm.scale_field(nhm.NinthHeads(D, 6, "path", seed=1, units=u, form="next"))
    f = nhm.scale_field(nhm.NinthHeads(D, 6, "path", seed=1, units=u, form="next", path_blocks=(2, 1, 3)))
    assert "path_blocks" not in f0 and f == dict(f0, path_blocks=[1, 2, 3])                  # every block: 0.1.94's field
    nhm.check_field(f)
    nhm.check_field(f0)
    for bad in ([3, 1], [1, 1, 2], [0, 1], [1, 7], [1.0, 2.0], (1, 2)):
        with pytest.raises(AssertionError):
            nhm.check_field(dict(f, path_blocks=bad))
    with pytest.raises(AssertionError):
        nhm.check_field(dict(nhm.scale_field(nhm.NinthHeads(D, 6, "plain", seed=1)), path_blocks=[1]))
    for kw in (dict(rot="plain", path_blocks=(1,)), dict(rot="path", path_blocks=(0, 1)), dict(rot="path", path_blocks=(7,))):
        with pytest.raises(AssertionError):
            nhm.NinthHeads(D, 6, kw["rot"], seed=1, units=u, path_blocks=kw["path_blocks"])
    rec = nhm.record(nhm.NinthHeads(D, 6, "path", seed=1, units=u, form="next", path_blocks=(1, 2, 3)))
    assert rec["path_blocks"] == [1, 2, 3] and rec["params"] == 196608                       # the same capacity whatever the set


def test_the_side_file_round_trips_with_the_set(tmp_path, short):                            # noqa: F811
    from safetensors.torch import save_file
    m, st = _craft(n_ctx=4096, seed=2)
    va = ar.ValueAddress(m.cfg.d_model, st)
    with torch.no_grad():
        va.table.normal_(std=0.5)
    ar.patch_value(m, va)
    nh = nhm.NinthHeads(m.cfg.d_model, 6, "path", seed=2, units=prm.Units(), form="next", path_blocks=(1, 2, 3))
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
    assert core.ninth.path_blocks == (1, 2, 3) and nhm.digest(core.ninth) == nhm.digest(nh)
    held = _held(tmp_path / "held.jsonl")
    docs = rs.load_heldout(held)[:2]
    xs = torch.tensor(np.stack([np.frombuffer(bytes([rs.DOC]) + dd[:gr.CTX], dtype=np.uint8) for dd in docs]).astype(np.int64))
    with torch.no_grad():
        exp = m(xs[:, :-1]).logits
        got = core.readout(core.trunk(xs[:, :-1]))
        nh.off = True
        lo = m(xs[:, :-1]).logits
        nh.off = False
    assert torch.equal(got, exp) and not torch.equal(lo, exp)                                 # the reader's forward = the trainer's
    rec = rs.text_read(str(d), str(tmp_path / "text.json"), held, n_docs=2, mb=2, dev="cpu", log=lambda s: None)
    bits = lambda lg: float(F.cross_entropy(lg.float().reshape(-1, 256), xs[:, 1:].reshape(-1), reduction="sum")) / lg.shape[0] / lg.shape[1] / math.log(2)  # noqa: E731
    assert abs(rec["bits"] - bits(exp)) < 1e-6 and abs(rec["path_off"]["bits"] - bits(lo)) < 1e-6
    assert rec["path"] == dict(form="next", head="ninth", k=prm.K, blocks=[1, 2, 3])
    rm = rr.read(str(d), str(tmp_path / "room.json"), held, "look", look_rows=2, n_docs=2, mb=2, dev="cpu", log=lambda s: None)
    assert rm["removed"] == {} and rm["base"]["bits"] == rec["bits"] and rm["ninth"] == dict(rot="path", form="next")
    assert all(len(b) == 9 for b in rm["look"]["blocks"])                                    # every block's ninth read as head 8


def test_the_names_carry_the_set_and_the_runner_passes_it_whole():
    pk = dict(W_BASE, grid_hard="0.5", value_addr="1", ptr_lam="0.1", grid_comps="48", key_rot="1", key_pairs="16", key_norm="2",
              key_rho="2.18", plant="0.75", plain_doc="1")
    stem = STEM.format(1).replace("_askany_", "_askany_p75d_") + "gc48_kh50_va1_pt10b4h0_krt16r218_"
    assert run_tag(dict(pk, ninth_head="path", path_form="next", series="pillar1b")) == stem + "nhpn16_ahdigit1_pillar1b"   # unset: 0.1.94's
    for pb in ("1,2,3", "3,1,2", "1,2,3,3"):
        assert run_tag(dict(pk, ninth_head="path", path_form="next", path_blocks=pb, series="pillar1c")) == stem + "nhpn16b123_ahdigit1_pillar1c"
    assert run_tag(dict(pk, ninth_head="path", path_form="same", path_k="12", path_blocks="2", series="x")) == stem + "nhps16k12b2_ahdigit1_x"
    runs = br.expand(dict(arms="alhena-both", seeds="1,2", grid="ag", grid_p="0.5", head="digit", head_lam="1", rotary_content="4",
                          grid_hard="0.5", value_addr="1", ptr_lam="0.1", key_rot="1", grid_comps="48", key_pairs="16", key_norm="2",
                          key_rho="2.18", plant="0.75", plain_doc="1", ninth_head="path", path_form="next", path_blocks="1,2,3",
                          series="pillar1c"))
    assert len(runs) == 2 and all("--path_blocks=1,2,3" in r for r in runs)                  # one set, not a grid axis
    assert all("a ninth head with the path (next) in blocks 1,2,3" in br.run_label(r) for r in runs)
    assert "path_blocks" not in br.GRID_KEYS
