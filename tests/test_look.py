# -*- coding: utf-8 -*-
"""0.1.80, THE LOOK (the readout tie): the request read off where one attention head looks - the record holding more than half of the
head's attention, both digits from that one record, else no request."""
import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from geolip.nano_beatrix import grid_rows as gr
from geolip.nano_beatrix import rotary_scale as rs


def _meta(frames, addrs, keys=None):
    keys = keys or [f"k{i:03d}" for i in range(len(frames))]
    return {"recs": [dict(lo=lo, frame_end=hi, addr=a, key=k) for (lo, hi), a, k in zip(frames, addrs, keys)]}


def _cap(n, frames_hot, pos, L=40.0, H=2, head=1):
    """q and k (1, H, n, 2) whose head `head` puts its attention at `pos` on the positions of frames_hot (equal scores inside them)."""
    q = torch.zeros(1, H, n, 2)
    k = torch.zeros(1, H, n, 2)
    q[0, head, pos, 0] = L
    for lo, hi in frames_hot:
        k[0, head, lo:hi, 0] = 1.0
    return SimpleNamespace(q=q, k=k, head=head)


def test_the_look_names_the_record_holding_more_than_half_and_both_its_digits():
    frames = [(2, 6), (8, 12), (14, 18)]
    m = _meta(frames, [9, 42, 63])
    cap = _cap(40, [frames[1]], pos=30)
    o = rs.look_of(cap, [m], [0], [30], [1])
    assert o["rec"][0] == 1 and o["mass"][0] > 0.99 and o["true_mass"][0] > 0.99
    assert tuple(o["dig"][0]) == gr.digits(42) == (5, 2)
    assert o["top"][0] == 1 and o["second"][0] < 1e-6


def test_a_split_look_gives_no_request_and_the_largest_mass_form_is_only_a_read():
    frames = [(2, 6), (8, 12), (14, 18)]
    m = _meta(frames, [9, 42, 63])
    cap = _cap(40, [frames[0], frames[2]], pos=30, L=4.0)           # two records share the head's attention (.43 each; the rest elsewhere)
    o = rs.look_of(cap, [m], [0], [30], [0])
    assert o["rec"][0] == -1 and o["mass"][0] == 0.0 and tuple(o["dig"][0]) == (gr.NULL, gr.NULL)
    assert o["top"][0] in (0, 2) and 0.4 < o["top_mass"][0] < 0.45 and 0.4 < o["second"][0] < 0.45
    with pytest.raises(AssertionError):
        rs.look_of(cap, [m], [0], [30], [0], thr=0.4)                # below one half two records could pass: refused


def test_the_look_is_the_causal_softmax_summed_over_each_frame():
    g = torch.Generator().manual_seed(3)
    n, H, hd, head = 64, 4, 8, 2
    q, k = torch.randn(2, H, n, hd, generator=g) * 3, torch.randn(2, H, n, hd, generator=g) * 3
    frames = [(1, 9), (10, 21), (22, 30), (31, 40)]
    metas = [_meta(frames, [3, 17, 33, 60]), _meta(frames, [4, 18, 34, 61])]
    rows, pos, asked = np.array([0, 0, 1, 1, 1]), np.array([45, 63, 41, 50, 63]), np.array([0, 3, 1, 2, 3])
    o = rs.look_of(SimpleNamespace(q=q, k=k, head=head), metas, rows, pos, asked)
    for j, (r, p) in enumerate(zip(rows, pos)):
        s = (q[r, head, p].double() @ k[r, head, :p + 1].double().T) / math.sqrt(hd)
        a = torch.softmax(s, -1)
        fm = np.array([float(a[lo:hi].sum()) for lo, hi in frames])
        assert abs(o["true_mass"][j] - fm[asked[j]]) < 1e-6 and abs(o["top_mass"][j] - fm.max()) < 1e-6
        above = np.nonzero(fm > 0.5)[0]
        assert o["rec"][j] == (above[0] if len(above) == 1 else -1)
        if o["rec"][j] >= 0:
            assert tuple(o["dig"][j]) == gr.digits(metas[r]["recs"][above[0]]["addr"])


def test_a_position_after_the_query_never_reads():
    frames = [(2, 6), (40, 44)]
    m = _meta(frames, [1, 2])
    cap = _cap(64, [frames[1]], pos=20)                              # the hot frame lies after the query: masked
    o = rs.look_of(cap, [m], [0], [20], [1])
    assert o["true_mass"][0] == 0.0 and o["rec"][0] == -1


def test_the_look_sources_and_where_they_join():
    assert rs.FIND_SOURCES["LOOK"] == ("T", "look_i", "exact")
    assert rs.FIND_SOURCES["LOOKF"] == ("F", "look_if", "exact")
    assert rs.FIND_SOURCES["LOOKQF"] == ("F", "look_qf", "exact")
    assert rs.FIND_SOURCES_BY_STREAM == {"look_i": "LOOK", "look_if": "LOOKF", "look_qf": "LOOKQF"}
    assert set(rs.LOOK_STREAMS) == {"look_i", "look_if", "look_qf"} and rs.LOOK_THR == 0.5
    for head, head_at, base, base_mid in ((None, "question", rs.SRC_NOHEAD, rs.SRC_NOHEAD_MID),
                                          (object(), "question", rs.SRC_QHEAD, rs.SRC_QHEAD_MID),
                                          (object(), "both", rs.SRC_FIND, rs.SRC_FIND_MID)):
        off = SimpleNamespace(head=head, head_at=head_at, look=None)
        on = SimpleNamespace(head=head, head_at=head_at, look=(4, 0))
        assert rs.find_sources(off, "last") == base and rs.find_sources(off, "mid") == base_mid     # 0.1.79's sets, unchanged
        assert rs.find_sources(on, "last") == base + ("LOOK", "LOOKF", "LOOKQF")
        assert rs.find_sources(on, "mid") == base_mid + ("LOOKF",)


def test_the_look_reads_name_the_core_and_the_head():
    a, b = rs.look_name("alhena-both_s1_x_pointer", (4, 0)), rs.look_name("alhena-both_s1_x_pointer", (4, 1))
    assert a == "alhena-both_s1_x_pointer_look_b4h0" and a != b
    assert rs.look_name("alhena-both_s1_x_value", (4, 0)) != a


def test_the_look_counts():
    frames = [(0, 4), (4, 8), (8, 12), (12, 16)]
    keys = ["ab17", "cd17", "ef22", "gh35"]
    m = _meta(frames, [1, 2, 3, 4], keys)
    lk = dict(rec=np.array([0, -1, 0, 3, 2]), mass=np.array([.9, 0., .8, .7, .6]), true_mass=np.array([.9, .4, .1, .7, .0]),
              top=np.array([0, 1, 0, 3, 2]), top_mass=np.array([.9, .45, .8, .7, .6]), second=np.array([.05, .44, .1, .2, .35]),
              dig=np.zeros((5, 2), dtype=np.int64))
    ctx = dict(look={"look_if": lk}, asked=np.array([0, 1, 1, 3, 0]), metas=[m], qt=dict(row=np.zeros(5, dtype=np.int64)))
    acc = rs._look_new()
    rs._look_add(acc, ctx, "look_if")
    o = rs._look_out(acc)
    assert o["questions"] == 5 and o["right"] == 2 / 5 and o["no_request"] == 1 / 5 and o["wrong_record"] == 2 / 5
    assert o["wrong_records"] == 2 and o["wrong_same_ending"] == 0.5          # cd17 asked, ab17 named: the same ending; ab17 asked, ef22 named
    assert o["duplicate"] == 2 / 5                                           # record 0 decoded for two questions of the row
    assert o["split"] == 2 / 5 and o["largest_mass_form_right"] == 3 / 5     # questions 1 and 4 split; the largest-mass form right on 0, 1, 3
    assert abs(o["decoded_mass_mean"] - (.9 + .8 + .7 + .6) / 4) < 1e-12
