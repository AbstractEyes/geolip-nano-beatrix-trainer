# -*- coding: utf-8 -*-
"""The slot read (0.1.99): the relabel control on chosen slots of the table (the byte k back). The table with every slot named is P16'
bit for bit and with none P16; a named slot set takes only its slots from P16'; the named blocks turn by the path formula on the mixed
table and the twin follows a change of slots; block_read's relabel mode with slots reads the empty set as the text read and a set as the
forward by hand, names its slots, refuses slots outside the relabel mode, and with every slot named reads as the whole relabel. CPU
only."""
import pytest
import torch

from geolip.alephllm.model.attention import rope_half
from geolip.nano_beatrix import ninth_head as nhm
from geolip.nano_beatrix import path_rotary as prm
from geolip.nano_beatrix import rotary_scale as rs
from test_block_read import _bits, _core, _rows
from test_ninth_head import _held, short  # noqa: F401
from test_path_rotary import _text_rows, _units, space_table  # noqa: F401
from test_value_addr import _craft

D = 256


def test_the_slot_tables():
    full, p = nhm.relabel_table(), nhm.phase_table()
    assert torch.equal(nhm.relabel_table(slots=range(prm.K)), full) and torch.equal(nhm.relabel_table(slots=()), p)
    t = nhm.relabel_table(slots=(0,))
    assert torch.equal(t[:, 0], full[:, 0]) and torch.equal(t[:, 1:], p[:, 1:])
    d = nhm.relabel_table(slots=range(1, prm.K))
    assert torch.equal(d[:, 0], p[:, 0]) and torch.equal(d[:, 1:], full[:, 1:])
    with pytest.raises(AssertionError):
        nhm.relabel_table(slots=(prm.K,))
    assert rs.slot_name((0,)) == "0" and rs.slot_name(range(1, 16)) == "1-15" and rs.slot_name((0, 2, 3, 5)) == "0+2-3+5"


def test_the_named_slots_turn_by_the_mixed_table(space_table):                               # noqa: F811
    xb = _text_rows(B=2, n=96, seed=5)
    _, st = _craft(n_ctx=256, seed=1)
    units = _units(space_table)
    nh = nhm.NinthHeads(D, 6, "path", seed=1, units=units, form="next", path_blocks=(1, 2, 3))
    q, k = torch.randn(2, 1, 96, 32), torch.randn(2, 1, 96, 32)
    nh.set(xb)
    path = [nh.rotate(q, k, st, i) for i in range(6)]
    nh.relabel_blocks = (2,)
    for slots in ((0,), tuple(range(1, prm.K)), None):
        nh.relabel_slots = slots
        nh.set(xb)
        ref = nhm.Path16(units, "next", relabel=True, slots=slots)
        ref.set(xb)
        r = nh.rotate(q, k, st, 1)
        exp = (rope_half(q, ref.phi_q[:, None].cos(), ref.phi_q[:, None].sin()), rope_half(k, ref.phi_k[:, None].cos(), ref.phi_k[:, None].sin()))
        assert torch.equal(r[0], exp[0]) and torch.equal(r[1], exp[1]), slots                    # block 2 on the mixed table
        assert not torch.equal(r[0], path[1][0]), slots
        for i in (0, 2, 3, 4, 5):
            assert torch.equal(nh.rotate(q, k, st, i)[0], path[i][0]), (slots, i)                # every other block as before
    full = nhm.NinthHeads(D, 6, "path", seed=1, units=units, form="next", path_blocks=(1, 2, 3))
    full.relabel_blocks = (2,)
    full.set(xb)
    assert torch.equal(full.rotate(q, k, st, 1)[0], nh.rotate(q, k, st, 1)[0])                    # None = 0.1.98's control


def test_the_block_read_on_slots(tmp_path, short):                                             # noqa: F811
    m, nh, d = _core(tmp_path)
    held = _held(tmp_path / "held.jsonl")
    tr = rs.text_read(str(d), str(tmp_path / "text.json"), held, n_docs=2, mb=2, dev="cpu", log=lambda s: None)
    sets = [(1,), (), (1, 2, 3)]
    kw = dict(sets=sets, n_docs=2, mb=2, dev="cpu", log=lambda s: None, mode="relabel", depth=True)
    rec = rs.block_read(str(d), str(tmp_path / (rs.group_name("x", "relabel", True, (0,)) + ".json")), held, slots=(0,), **kw)
    by = {tuple(r["off"]): r for r in rec["reads"]}
    assert by[()]["bits"] == tr["bits"] and by[()]["by_repeat"] == tr["by_repeat"] and rec["slots"] == [0] and rec["mode"] == "relabel"
    xs = _rows(held)
    for o in ((1,), (1, 2, 3)):
        nh.relabel_blocks, nh.relabel_slots = o, (0,)
        assert abs(by[o]["bits"] - _bits(m, xs)) < 1e-6, o
        nh.relabel_blocks, nh.relabel_slots = (), None
    whole = rs.block_read(str(d), str(tmp_path / "whole.json"), held, **kw)
    every = rs.block_read(str(d), str(tmp_path / "every.json"), held, slots=range(prm.K), **kw)
    for a, b in zip(whole["reads"], every["reads"]):
        assert a["bits"] == b["bits"] and a["by_repeat"] == b["by_repeat"] and a["by_depth"] == b["by_depth"]   # every slot = the whole relabel
    assert "slots" not in whole and every["slots"] == list(range(prm.K))
    assert {tuple(r["off"]): r for r in whole["reads"]}[(1, 2, 3)]["bits"] != by[(1, 2, 3)]["bits"]
    with pytest.raises(AssertionError):
        rs.block_read(str(d), str(tmp_path / "bad.json"), held, sets=sets, n_docs=2, mb=2, dev="cpu", log=lambda s: None, mode="random",
                      slots=(0,))
    assert rs.group_name("abc", "relabel", True, (0,)) == "abc_group_relabel_k0_depth"
    assert rs.group_name("abc", "relabel", True, range(1, 16)) == "abc_group_relabel_k1-15_depth"
    assert rs.group_name("abc", "relabel", True) == "abc_group_relabel_depth"
