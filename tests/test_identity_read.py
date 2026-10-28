# -*- coding: utf-8 -*-
"""The identity read (0.1.97): a read's control that puts chosen path blocks' ninth heads on fresh random phases. The draw is fixed by
the block and the length and gives q and k their own phases; the named path blocks turn by it and the others by the path, an empty set
(the training default) leaves the forward as it was, and the toggle still wins; block_read's random mode reads the empty set as the text
read, a set as the forward with those blocks on random phases, restores the control, and names its mode and record. CPU only."""
import json
import math

import pytest
import torch

from geolip.alephllm.model.attention import rope_half
from geolip.nano_beatrix import ninth_head as nhm
from geolip.nano_beatrix import rotary_scale as rs
from test_block_read import _bits, _core, _rows
from test_ninth_head import _held, short  # noqa: F401
from test_path_rotary import _text_rows, _units, space_table  # noqa: F401
from test_value_addr import _craft

D = 256


def test_the_random_phases_are_fixed_and_their_own():
    a, b = nhm.random_phases(2, 50, 3, "cpu"), nhm.random_phases(2, 50, 3, "cpu")
    c = nhm.random_phases(2, 50, 4, "cpu")
    assert a[0].shape == (2, 1, 50, 16) and torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])     # the same draw every read
    assert not torch.equal(a[0], a[1]) and not torch.equal(a[0], c[0])                             # q and k apart; a block its own
    assert torch.equal(a[0][0], a[0][1]) and float(a[0].min()) >= 0 and float(a[0].max()) < 2 * math.pi


def test_the_named_blocks_turn_by_random_phases_the_others_by_the_path(space_table):         # noqa: F811
    xb = _text_rows(B=2, n=96, seed=5)
    _, st = _craft(n_ctx=256, seed=1)
    nh = nhm.NinthHeads(D, 6, "path", seed=1, units=_units(space_table), form="next", path_blocks=(1, 2, 3))
    nh.set(xb)
    q, k = torch.randn(2, 1, 96, 32), torch.randn(2, 1, 96, 32)
    path = [nh.rotate(q, k, st, i) for i in range(6)]
    nh.random_blocks = (2,)
    rnd = [nh.rotate(q, k, st, i) for i in range(6)]
    aq, ak = nhm.random_phases(2, 96, 2, "cpu")
    exp = (rope_half(q, aq.cos(), aq.sin()), rope_half(k, ak.cos(), ak.sin()))
    assert torch.equal(rnd[1][0], exp[0]) and torch.equal(rnd[1][1], exp[1])                       # block 2: the random draw
    for i in (0, 2, 3, 4, 5):
        assert torch.equal(rnd[i][0], path[i][0]) and torch.equal(rnd[i][1], path[i][1])           # every other block as before
    nh.off = True
    off = nh.rotate(q, k, st, 1)
    nh.off = False
    nh.random_blocks = ()
    plain = nhm.NinthHeads(D, 6, "plain", seed=1).rotate(q, k, st, 1)
    assert torch.equal(off[0], plain[0]) and torch.equal(off[1], plain[1])                        # the toggle still wins
    nh.random_blocks = (4,)                                                                       # not a path block: anchored as before
    assert torch.equal(nh.rotate(q, k, st, 3)[0], path[3][0])
    nh.random_blocks = ()
    again = nh.rotate(q, k, st, 1)
    assert torch.equal(again[0], path[1][0]) and torch.equal(again[1], path[1][1])                 # the empty set: the path again


def test_the_block_read_on_random_phases(tmp_path, short):                                    # noqa: F811
    m, nh, d = _core(tmp_path)
    held = _held(tmp_path / "held.jsonl")
    tr = rs.text_read(str(d), str(tmp_path / "text.json"), held, n_docs=2, mb=2, dev="cpu", log=lambda s: None)
    sets = [(1,), (3,), (), (1, 2, 3)]                                                            # the empty set after others
    rec = rs.block_read(str(d), str(tmp_path / rs.group_name("x", "random")), held, sets=sets, n_docs=2, mb=2, dev="cpu",
                        log=lambda s: None, mode="random")
    by = {tuple(r["off"]): r for r in rec["reads"]}
    assert by[()]["bits"] == tr["bits"] and by[()]["by_repeat"] == tr["by_repeat"] and rec["mode"] == "random"
    xs = _rows(held)
    for o in ((1,), (3,), (1, 2, 3)):
        nh.random_blocks = o
        assert abs(by[o]["bits"] - _bits(m, xs)) < 1e-6, o
        nh.random_blocks = ()
    assert by[(1, 2, 3)]["bits"] != tr["path_off"]["bits"]                                       # random phases are not the anchored switch
    anchored = rs.block_read(str(d), str(tmp_path / "g.json"), held, sets=[(1,)], n_docs=2, mb=2, dev="cpu", log=lambda s: None)
    assert "mode" not in anchored and anchored["reads"][0]["bits"] != by[(1,)]["bits"]
    assert rs.group_name("abc", "random") == "abc_group_random" and rs.group_name("abc") == "abc_group"
    assert json.load(open(tmp_path / "x_group_random", encoding="utf-8"))["mode"] == "random"
    with pytest.raises(AssertionError):
        rs.block_read(str(d), str(tmp_path / "bad.json"), held, n_docs=2, mb=2, dev="cpu", log=lambda s: None, mode="bare")
