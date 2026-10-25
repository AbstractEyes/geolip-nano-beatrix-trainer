# -*- coding: utf-8 -*-
"""The find form (0.1.72): the head's reads at the index entries, the free-running index (the fixed point of re-feeding the craft's own
index bytes), the .5 stream decode, and the names of the runs. CPU only; the Hub is never touched."""
import numpy as np

from geolip.alephllm.data.special_tokens import SEP
from geolip.nano_beatrix import bed_runner as br
from geolip.nano_beatrix import grid_rows as gr
from geolip.nano_beatrix.names import run_tag


def _row(seed=5, **kw):
    rng = np.random.default_rng(seed)
    src = bytes(np.random.default_rng(seed + 100).integers(32, 127, size=gr.CTX + 1, dtype=np.uint8))
    return gr.compose_ag(rng, src, **kw)


def test_the_index_head_reads_each_entrys_sep_and_its_records_address():
    row, m = _row()
    pos, dig = gr.index_head_targets(m)
    assert len(pos) == len(m["index"]) == len(m["questions"])
    for p, d, e in zip(pos, dig, m["index"]):
        assert row[p] == SEP                                            # the entry's SEP (an input position)
        assert row[p + 1:p + 3] == gr.addr_bytes(m["recs"][e["rec"]]["addr"])   # its target: the entry's first address byte
        assert d == gr.digits(m["recs"][e["rec"]]["addr"])
    assert [p for p, _ in gr.index_positions(m)] == pos


def test_a_swapped_index_keeps_the_heads_truth():
    for t in range(20):
        row, m = _row(seed=t, swap_index=True)
        _, dig = gr.index_head_targets(m)
        assert dig == [gr.digits(m["recs"][e["rec"]]["addr"]) for e in m["index"]]     # the record's address, never the shown one
        shown = gr.row_request(row, m)
        assert sum(a != b for a, b in zip(shown, dig)) == 2                            # one pair of entries traded


def test_put_index_bytes_and_row_request_round_trip():
    row, m = _row()
    tf = np.frombuffer(row, dtype=np.uint8)[1:].copy()                  # a craft that writes the row's own bytes
    assert gr.put_index_bytes(row, m, tf) == row
    assert gr.row_request(row, m) == [gr.digits(m["recs"][q["rec"]]["addr"]) for q in m["questions"]]
    p1, p2 = gr.index_positions(m)[3]
    tf[p1], tf[p2] = ord("5"), ord("x")
    new = gr.put_index_bytes(row, m, tf)
    assert new[p1 + 1] == ord("5") and new[p2 + 1] == ord("x")
    assert sum(a != b for a, b in zip(new, row)) <= 2
    assert gr.row_request(new, m)[3] == (5, gr.NULL)                    # a byte outside '0'-'7' is the null anchor


def _toy_greedy(ag):
    """A deterministic toy craft: every byte the row's own (teacher-forced), except the index's address bytes, each a function of the
    address bytes written before it (so a wrong early byte changes later ones): the truth, unless the earlier bytes' sum is 3 mod 7."""
    out = []
    for row, m in ag:
        pred = np.frombuffer(row, dtype=np.uint8)[1:].copy()
        earlier = 0
        for e, (p1, p2) in zip(m["index"], gr.index_positions(m)):
            true = gr.addr_bytes(m["recs"][e["rec"]]["addr"])
            for p, tb in ((p1, true[0]), (p2, true[1])):
                pred[p] = (48 + (tb - 48 + 1) % 8) if earlier % 7 == 3 else tb
                earlier += int(row[p + 1])                              # the byte in the row (the input), not the prediction
        out.append(pred)
    return np.stack(out)


def _sequential(row, m):
    """The sequential greedy writing of the toy craft: each address byte from the ones it already wrote."""
    b = bytearray(row)
    for p1, p2 in gr.index_positions(m):
        for p in (p1, p2):
            b[p + 1] = int(_toy_greedy([(bytes(b), m)])[0][p])
    return bytes(b)


def test_the_free_running_fixed_point_is_the_sequential_greedy_writing():
    ag = [_row(seed=s) for s in range(4)]
    fr, passes, conv = gr.free_running(ag, _toy_greedy)
    assert all(conv) and 1 <= passes <= 2 * gr.N_COMP
    for (r, m), (r0, _) in zip(fr, ag):
        assert r == _sequential(r0, m)
        assert gr.put_index_bytes(r, m, _toy_greedy([(r, m)])[0]) == r          # a fixed point
    # whatever the start: a row with scrambled index bytes reaches the same point
    r0, m = ag[0]
    scr = bytearray(r0)
    for p1, p2 in gr.index_positions(m):
        scr[p1 + 1], scr[p2 + 1] = ord("7"), ord("q")
    fr2, _, _ = gr.free_running([(bytes(scr), m)], _toy_greedy)
    assert fr2[0][0] == fr[0][0]


def test_a_pass_cap_stops_the_loop():
    ag = [_row(seed=9)]
    _, passes, conv = gr.free_running(ag, _toy_greedy, max_passes=1)
    assert passes == 1


def test_the_stream_decode_judges_each_digit_by_itself():
    s = np.full((5, 2, 8), 0.01)
    s[0, 0, 3], s[0, 1, 6] = 0.9, 0.51               # one digit above .5 at each level
    s[1, 0, 3], s[1, 0, 4] = 0.9, 0.8                # two above at level 1 -> null there
    s[2, 1, 2] = 0.5                                 # exactly .5 is not above
    s[3] = 0.99                                      # every digit above -> null
    s[4, 0, 0], s[4, 1, 7] = 0.6, 0.7
    assert gr.decode_scores(s) == [(3, 6), (gr.NULL, gr.NULL), (gr.NULL, gr.NULL), (gr.NULL, gr.NULL), (0, 7)]


def test_the_two_pass_sources_and_their_cells():
    from geolip.nano_beatrix import rotary_scale as rs
    for srcs in (rs.SRC_FIND, rs.SRC_QHEAD, rs.SRC_NOHEAD, rs.SRC_FIND_MID, rs.SRC_QHEAD_MID, rs.SRC_NOHEAD_MID):
        assert all(s in rs.FIND_SOURCES for s in srcs) and len(set(srcs)) == len(srcs)
    rest = [c for c in rs._find_cells(rs.SRC_FIND) if c[3] in ("REST", "-")]
    assert len(rest) == 78                                               # 8 sources x U 5 / 8 / 10 x m 16 / 32 / 48, and G2 / G0 at three m
    for s, (row, stream, gate) in rs.FIND_SOURCES.items():
        assert row in ("T", "F") and gate in ("exact", "qhead", "qhead_q999", "ihead", "ihead_q999")
        assert (row == "F") == (stream in ("indexf", "dec_if", "look_if", "look_qf"))   # the free-running row's own index and the
                                                                         # heads read on it (0.1.80: the look's two streams there)

    class _C:                                                            # a stand-in core: only what find_sources reads
        def __init__(self, head, at):
            self.head, self.head_at = head, at
    assert rs.find_sources(_C(object(), "both")) == rs.SRC_FIND and rs.find_sources(_C(object(), "both"), "mid") == rs.SRC_FIND_MID
    assert rs.find_sources(_C(object(), "question")) == rs.SRC_QHEAD and rs.find_sources(_C(None, "question")) == rs.SRC_NOHEAD


def test_a_stream_request_lists_each_rows_questions_in_order():
    from geolip.nano_beatrix import rotary_scale as rs
    rows = [_row(seed=s) for s in (1, 2)]
    owned, _ = rs.owned_sets()
    qt = rs.question_table([m for _, m in rows], owned)
    req = rs._req_lists(qt, 2, qt["dig"])
    for (_, m), r in zip(rows, req):
        assert r == [gr.digits(m["recs"][q["rec"]]["addr"]) for q in m["questions"]]
        dk, dq = gr.streams(m, gr.CTX, r)
        dk0, dq0 = gr.streams(m)
        assert (dk == dk0).all() and (dq == dq0).all()                  # the true digits as a request = the training form


def test_the_find_runs_names():
    base = dict(arm="alhena-both", seed="1", steps="24000", ctx="4096", frames="dyn", sizes="32,128,256,512", plant="1.0",
                grid="ag", grid_p="0.5", head="digit", head_lam="1.0")
    assert run_tag(dict(base, series="scale")).endswith("_gag50_ahdigit1_scale")
    assert run_tag(dict(base, series="scale", head_at="question")) == run_tag(dict(base, series="scale"))
    assert run_tag(dict(base, head_at="both", head_lam_i="1", series="find")).endswith("_gag50_ahdigit1i1_find")
    assert run_tag(dict(base, head_at="both", head_lam_i="0.25", series="find")).endswith("_ahdigit1i0.25_find")
    assert run_tag(dict(base, series="scalerep")).endswith("_gag50_ahdigit1_scalerep")
    lab = br.run_label([f"--{k}={v}" for k, v in dict(base, head_at="both", head_lam_i="1").items()])
    assert "head digit 1.0 + index 1" in lab, lab
    assert "head_at" in br.GRID_KEYS and "head_lam_i" in br.GRID_KEYS
