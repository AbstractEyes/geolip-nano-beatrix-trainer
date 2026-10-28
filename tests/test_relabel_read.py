# -*- coding: utf-8 -*-
"""The relabel read (0.1.98): a read's control that puts chosen path blocks' ninth heads on the path's phases from P16', a fresh table of
P16's law. The table is fixed and its own; the named path blocks turn by the path formula on P16', every other block as before; equal
paths meet at equal phases under P16' (every exact match kept); P16 itself through the control is the path forward bit for bit; random
phases win over it and the off toggle over both; the depth split's cells sum to the whole and change no other number; block_read's
relabel mode reads the empty set as the text read and a set as the forward by hand, and names its mode and record. CPU only."""
import math

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


def _turn(q, k, pq, pk):
    return (rope_half(q, pq[:, None].cos(), pq[:, None].sin()), rope_half(k, pk[:, None].cos(), pk[:, None].sin()))


def test_the_relabel_table_is_fixed_and_its_own():
    a, b, p = nhm.relabel_table(), nhm.relabel_table(), nhm.phase_table()
    assert a.shape == p.shape == (16, prm.K, prm.VOCAB) and torch.equal(a, b) and not torch.equal(a, p)
    assert float(a.min()) >= 0 and float(a.max()) < 2 * math.pi
    assert not torch.equal(nhm.relabel_table(8), a[:, :8])                                        # another reach: its own draw


def test_the_named_blocks_turn_by_the_path_on_the_fresh_table(space_table):                   # noqa: F811
    xb = _text_rows(B=2, n=96, seed=5)
    _, st = _craft(n_ctx=256, seed=1)
    units = _units(space_table)
    nh = nhm.NinthHeads(D, 6, "path", seed=1, units=units, form="next", path_blocks=(1, 2, 3))
    nh.set(xb)
    q, k = torch.randn(2, 1, 96, 32), torch.randn(2, 1, 96, 32)
    path = [nh.rotate(q, k, st, i) for i in range(6)]
    nh.relabel_blocks = (2,)
    nh.set(xb)
    rel = [nh.rotate(q, k, st, i) for i in range(6)]
    ref = nhm.Path16(units, "next", relabel=True)
    ref.set(xb)
    exp = _turn(q, k, ref.phi_q, ref.phi_k)
    assert torch.equal(rel[1][0], exp[0]) and torch.equal(rel[1][1], exp[1])                       # block 2: the path on P16'
    assert not torch.equal(rel[1][0], path[1][0])
    for i in (0, 2, 3, 4, 5):
        assert torch.equal(rel[i][0], path[i][0]) and torch.equal(rel[i][1], path[i][1])           # every other block as before
    d, xs = units.depth(xb), xb.numpy()                                                              # equal paths, equal phases
    groups = {}
    for b in range(2):
        for t in range(96):
            groups.setdefault(bytes(xs[b, t - min(int(d[b, t]), prm.K) + 1:t + 1].astype("uint8")), []).append((b, t))
    shared = [g for g in groups.values() if len(g) > 1]
    assert len(shared) >= 5 and len(groups) >= 10
    for g in shared:
        assert all(torch.equal(ref.phi_q[b, t], ref.phi_q[g[0][0], g[0][1]]) for b, t in g)
    firsts = [g[0] for g in groups.values()][:10]
    assert len({tuple(ref.phi_q[b, t].tolist()) for b, t in firsts}) == len(firsts)                 # different paths, different phases
    nh.random_blocks = (2,)                                                                         # random phases win
    aq, ak = nhm.random_phases(2, 96, 2, "cpu")
    assert torch.equal(nh.rotate(q, k, st, 1)[0], rope_half(q, aq.cos(), aq.sin()))
    nh.random_blocks = ()
    nh.off = True                                                                                   # the toggle wins over both
    plain = nhm.NinthHeads(D, 6, "plain", seed=1).rotate(q, k, st, 1)
    assert torch.equal(nh.rotate(q, k, st, 1)[0], plain[0])
    nh.off = False
    nh.relabel_blocks = (4,)                                                                        # not a path block: as before
    assert torch.equal(nh.rotate(q, k, st, 3)[0], path[3][0])
    nh.relabel_blocks = ()
    again = nh.rotate(q, k, st, 1)
    assert torch.equal(again[0], path[1][0]) and torch.equal(again[1], path[1][1])                 # the empty set: the path again


def test_p16_itself_through_the_control_is_the_path_bit_for_bit(space_table, monkeypatch):    # noqa: F811
    monkeypatch.setattr(nhm, "relabel_table", nhm.phase_table)
    xb = _text_rows(B=2, n=96, seed=7)
    _, st = _craft(n_ctx=256, seed=1)
    nh = nhm.NinthHeads(D, 6, "path", seed=1, units=_units(space_table), form="next")
    q, k = torch.randn(2, 1, 96, 32), torch.randn(2, 1, 96, 32)
    nh.set(xb)
    path = [nh.rotate(q, k, st, i) for i in range(6)]
    nh.relabel_blocks = (1, 2, 3, 4, 5, 6)
    nh.set(xb)
    for i in range(6):
        r = nh.rotate(q, k, st, i)
        assert torch.equal(r[0], path[i][0]) and torch.equal(r[1], path[i][1])


def test_the_depth_split_sums_to_the_whole_and_changes_nothing_else(tmp_path, short):         # noqa: F811
    m, nh, d = _core(tmp_path)
    held = _held(tmp_path / "held.jsonl")
    core = rs.Core(str(d), "cpu", look=None)
    docs = rs.load_heldout(held)[:2]
    a = rs._held_bits(core, docs, core.ninth.units, 2)
    b = rs._held_bits(core, docs, core.ninth.units, 2, depth=True)
    assert a["bits"] == b["bits"] and a["by_repeat"] == b["by_repeat"] and "by_depth" not in a
    cells, br = b["by_depth"], a["by_repeat"]
    assert tuple(cells) == prm.DepthSplit.BINS
    n = sum(c["n"] for c in cells.values())
    assert n == br["n_repeat"] + br["n_other"] and sum(c["n_repeat"] for c in cells.values()) == br["n_repeat"]
    # the counts exactly; the bits to float32 summation (each cell's nats summed in float32, as the whole's are)
    assert abs(sum(c["bits"] * c["n"] for c in cells.values() if c["n"]) / n - a["bits"]) < 1e-5
    assert abs(sum(c["repeat"] * c["n_repeat"] for c in cells.values() if c["n_repeat"]) / br["n_repeat"] - br["repeat"]) < 1e-5
    assert cells["1"]["n"] > 0 and cells["2"]["n"] > 0


def test_the_block_read_on_the_fresh_table(tmp_path, short):                                   # noqa: F811
    m, nh, d = _core(tmp_path)
    held = _held(tmp_path / "held.jsonl")
    tr = rs.text_read(str(d), str(tmp_path / "text.json"), held, n_docs=2, mb=2, dev="cpu", log=lambda s: None)
    sets = [(1,), (), (1, 2, 3)]                                                                    # the empty set after another
    kw = dict(sets=sets, n_docs=2, mb=2, dev="cpu", log=lambda s: None)
    rec = rs.block_read(str(d), str(tmp_path / (rs.group_name("x", "relabel", True) + ".json")), held, mode="relabel", depth=True, **kw)
    by = {tuple(r["off"]): r for r in rec["reads"]}
    assert by[()]["bits"] == tr["bits"] and by[()]["by_repeat"] == tr["by_repeat"] and rec["mode"] == "relabel"
    assert all("by_depth" in r for r in rec["reads"])
    xs = _rows(held)
    for o in ((1,), (1, 2, 3)):
        nh.relabel_blocks = o
        assert abs(by[o]["bits"] - _bits(m, xs)) < 1e-6, o
        nh.relabel_blocks = ()
    rnd = rs.block_read(str(d), str(tmp_path / "r.json"), held, mode="random", **kw)
    rnd_d = rs.block_read(str(d), str(tmp_path / "rd.json"), held, mode="random", depth=True, **kw)
    for a, b in zip(rnd["reads"], rnd_d["reads"]):
        assert a["off"] == b["off"] and a["bits"] == b["bits"] and a["by_repeat"] == b["by_repeat"]   # the split changes nothing
    assert by[(1, 2, 3)]["bits"] != {tuple(r["off"]): r for r in rnd["reads"]}[(1, 2, 3)]["bits"]
    assert rs.group_name("abc", "relabel", True) == "abc_group_relabel_depth" and rs.group_name("abc", depth=True) == "abc_group_depth"
    assert rs.group_name("abc", "random") == "abc_group_random" and rs.group_name("abc") == "abc_group"
