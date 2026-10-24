# -*- coding: utf-8 -*-
"""The posing screen (0.1.74): the grid rows' address as ONE byte 0x80 + a (--grid_abytes=1), unique in the row, in the frames and the
index alike. The two-digit form is 0.1.73's rows bit for bit (a fingerprint of 0.1.73's rows on fixed draws); the one-byte rows hold
their layout, share every draw and stream with the two-digit form and decode back to the digits; the free-running index, the members'
rows, the names and the runner carry the form. CPU only; the Hub is never touched."""
import hashlib
import json

import numpy as np

from geolip.alephllm.data.special_tokens import CUE, DATA, END, SEP
from geolip.nano_beatrix import bed_runner as br
from geolip.nano_beatrix import grid_rows as gr
from geolip.nano_beatrix import rotary_scale as rs
from geolip.nano_beatrix.names import run_tag

# 0.1.73's rows on the draws of _fingerprint, computed with the 0.1.73 tree (tag v0.1.73): the row bytes, the meta, the streams, the
# segments.
FP_0173 = {"meta": "892d5103d009b2a538ad280300beebf206d9970b8455e4f3ec4f7fc01fd8e05d",
           "rows": "f331967dc4d9ac0f56513b0d628e99d719d937caf79b5981429fd57b94c76f65",
           "segments": "6cfbcc5797bc45fe848b294a2ddc803738b9d2906b1b027c8f976217fb119707",
           "streams": "fa24239e28726767709c193a2c7c4ced68b4c34a2800496059b8f231623079d2"}


def _default(o):
    return o.tolist() if hasattr(o, "tolist") else int(o)


def _fingerprint(**kw):
    """60 rows on fixed draws (random filler of every byte value, drawn and given addresses, every and 18 questions, swapped indexes)."""
    h = {k: hashlib.sha256() for k in FP_0173}
    for t in range(60):
        r = np.random.default_rng([11, t])
        src = bytes(np.random.default_rng([13, t]).integers(0, 256, size=gr.CTX + 1, dtype=np.uint8))
        addrs = None if t % 4 else [int(a) for a in np.random.default_rng([12, t]).choice(gr.N_ADDR, size=gr.N_COMP, replace=False)]
        asked = None if t % 3 else [int(a) for a in np.random.default_rng([14, t]).permutation(gr.N_COMP)[:18]]
        row, m = gr.compose_ag(r, src, gr.CTX, addrs=addrs, asked=asked, swap_index=(t % 5 == 4), **kw)
        h["rows"].update(row)
        h["meta"].update(json.dumps(m, sort_keys=True, default=_default).encode())
        dk, dq = gr.streams(m)
        h["streams"].update(np.ascontiguousarray(dk).tobytes() + np.ascontiguousarray(dq).tobytes())
        h["segments"].update(np.ascontiguousarray(gr.segments(m)).tobytes())
    return {k: v.hexdigest() for k, v in h.items()}


def _row(seed=5, **kw):
    rng = np.random.default_rng(seed)
    src = bytes(np.random.default_rng(seed + 100).integers(32, 127, size=gr.CTX + 1, dtype=np.uint8))
    return gr.compose_ag(rng, src, **kw)


def test_the_two_digit_form_is_0173s_rows_bit_for_bit():
    assert _fingerprint() == FP_0173
    assert _fingerprint(abytes=2) == FP_0173
    _, m = _row()
    assert "abytes" not in m and gr.abytes_of(m) == 2                    # the default meta names no form (0.1.73's keys)


def test_the_one_byte_form_keeps_the_streams_and_changes_the_bytes():
    fp = _fingerprint(abytes=1)
    assert fp["streams"] == FP_0173["streams"]                           # every token's digits as before, on the same draws
    assert fp["rows"] != FP_0173["rows"] and fp["meta"] != FP_0173["meta"]
    gr.selftest(40)                                                      # both forms, the same draws (grid_rows' own self-test)


def test_a_one_byte_row_writes_each_address_as_one_unique_byte():
    for t in range(30):
        swap = t % 5 == 4
        row, m = _row(seed=t, abytes=1, swap_index=swap)
        two, m2 = _row(seed=t, swap_index=swap)
        assert m["abytes"] == 1 and gr.abytes_of(m) == 1
        frame = [row[rec["lo"] + 1] for rec in m["recs"]]
        assert frame == [0x80 + rec["addr"] for rec in m["recs"]] and len(set(frame)) == gr.N_COMP
        for rec, rec2 in zip(m["recs"], m2["recs"]):
            assert row[rec["lo"]] == DATA and row[rec["lo"] + 2] == CUE and row[rec["v0"] - 1] == SEP
            assert row[rec["v0"]:rec["v0"] + gr.VLEN].decode() == rec["value"] and row[rec["v0"] + gr.VLEN] == END
            assert rec["v0"] == rec2["v0"] - 1 and rec["frame_end"] == rec2["frame_end"] - 1   # the frame one byte shorter
            assert (rec["addr"], rec["key"], rec["value"], rec["slot"]) == (rec2["addr"], rec2["key"], rec2["value"], rec2["slot"])
        assert [q["sep"] for q in m["questions"]] == [q["sep"] for q in m2["questions"]]      # the questions where they were
        for e, ps in zip(m["index"], gr.index_positions(m)):
            assert len(ps) == 1 and ps[0] == e["a_pos"] - 1 and row[ps[0]] == SEP              # the head's position: the entry's SEP
            assert row[e["a_pos"]] == 0x80 + m["recs"][e["shown"]]["addr"]
        assert gr.index_head_targets(m) == ([p for (p,) in gr.index_positions(m)],
                                            [gr.digits(m["recs"][e["rec"]]["addr"]) for e in m["index"]])
        assert len(m["index"]) == len(m2["index"]) and m["index_end"] == m2["index_end"] - len(m["index"])


def test_decode_written_reads_both_forms():
    assert gr.decode_written([ord("3"), ord("7")]).tolist() == [3, 7]
    assert gr.decode_written([ord("8"), ord("x")]).tolist() == [gr.NULL, gr.NULL]
    assert gr.decode_written([[ord("0"), ord("9")]], 2).tolist() == [[0, gr.NULL]]          # each digit byte judged alone
    assert gr.decode_written([0x80 + 29], 1).tolist() == [3, 5]
    for b in (0x7F, 0xC0, ord("3"), 0xFF, 0):                                                  # no address byte: null at both levels
        assert gr.decode_written([b], 1).tolist() == [gr.NULL, gr.NULL]
    every = gr.decode_written(np.arange(0x80, 0xC0)[:, None], 1)
    assert every.tolist() == [list(gr.digits(a)) for a in range(gr.N_ADDR)]


def test_a_one_byte_rows_requests_round_trip():
    row, m = _row(abytes=1)
    truth = [gr.digits(m["recs"][q["rec"]]["addr"]) for q in m["questions"]]
    tf = np.frombuffer(row, dtype=np.uint8)[1:].copy()                  # a craft that writes the row's own bytes
    assert gr.put_index_bytes(row, m, tf) == row
    assert gr.row_request(row, m) == truth and gr.written_request(tf, m) == truth
    (p,) = gr.index_positions(m)[3]
    tf[p] = ord("5")                                                     # a digit byte names no address in this form
    new = gr.put_index_bytes(row, m, tf)
    assert new[p + 1] == ord("5") and sum(a != b for a, b in zip(new, row)) == 1
    assert gr.row_request(new, m)[3] == (gr.NULL, gr.NULL) and gr.written_request(tf, m)[3] == (gr.NULL, gr.NULL)
    tf[p] = 0x80 + 63
    assert gr.written_request(tf, m)[3] == (7, 7)


def _toy_greedy(ag):
    """A deterministic toy craft on one-byte rows: every byte the row's own, except each index address byte: the truth, unless the
    earlier address bytes' sum is 3 mod 7 (then the next address's byte), so a wrong early byte changes later ones."""
    out = []
    for row, m in ag:
        pred = np.frombuffer(row, dtype=np.uint8)[1:].copy()
        earlier = 0
        for e, (p,) in zip(m["index"], gr.index_positions(m)):
            a = m["recs"][e["rec"]]["addr"]
            pred[p] = 0x80 + ((a + 1) % gr.N_ADDR if earlier % 7 == 3 else a)
            earlier += int(row[p + 1])                                  # the byte in the row (the input), not the prediction
        out.append(pred)
    return np.stack(out)


def _sequential(row, m):
    b = bytearray(row)
    for (p,) in gr.index_positions(m):
        b[p + 1] = int(_toy_greedy([(bytes(b), m)])[0][p])
    return bytes(b)


def test_the_one_byte_free_running_fixed_point_is_the_sequential_writing():
    ag = [_row(seed=s, abytes=1) for s in range(4)]
    fr, passes, conv = gr.free_running(ag, _toy_greedy)
    assert all(conv) and 1 <= passes <= gr.N_COMP                      # one byte an entry: at most 24 passes
    for (r, m), (r0, _) in zip(fr, ag):
        assert r == _sequential(r0, m)
        assert gr.put_index_bytes(r, m, _toy_greedy([(r, m)])[0]) == r
    r0, m = ag[0]
    scr = bytearray(r0)
    for (p,) in gr.index_positions(m):
        scr[p + 1] = ord("q")
    fr2, _, _ = gr.free_running([(bytes(scr), m)], _toy_greedy)
    assert fr2[0][0] == fr[0][0]                                         # whatever the start


def test_the_members_rows_and_table_carry_the_form():
    own, never = rs.owned_sets()
    packed = np.random.default_rng(0).integers(32, 127, size=200_000).astype(np.uint8)
    rows1 = rs.member_rows(np.random.default_rng([1, 2]), packed, own, never, abytes=1)
    rows2 = rs.member_rows(np.random.default_rng([1, 2]), packed, own, never)
    assert all(gr.abytes_of(m) == 1 for _, m in rows1) and all(gr.abytes_of(m) == 2 for _, m in rows2)
    q1, q2 = rs.question_table([m for _, m in rows1], own), rs.question_table([m for _, m in rows2], own)
    assert q1["ipos"].shape == (len(q1["row"]), 1) and q2["ipos"].shape == (len(q2["row"]), 2)
    k_in_row = np.concatenate([np.arange(len(m["questions"])) for _, m in rows1])
    assert (q2["ipos"][:, 0] - q1["ipos"][:, 0] == k_in_row).all()     # each earlier index entry one byte shorter
    for k in ("row", "addr", "own", "dig", "pos"):
        assert np.array_equal(q1[k], q2[k]), k                           # the same draws: the same questions, owners, answer positions
    for (r, m) in rows1:
        x = np.frombuffer(r, dtype=np.uint8)
        for e, (p,) in zip(m["index"], gr.index_positions(m)):
            assert x[p] == SEP and x[p + 1] == 0x80 + m["recs"][e["shown"]]["addr"]
    row, meta, qi = rs.owned_row(np.random.default_rng(3), bytes(packed[:gr.CTX + 1]), own[0], abytes=1)
    assert gr.abytes_of(meta) == 1 and meta["recs"][meta["questions"][qi]["rec"]]["addr"] == own[0]


def test_the_posing_runs_never_share_a_name():
    base = {"arm": "alhena-both", "seed": "1", "ctx": "4096", "mb": "8", "steps": "24000", "frames": "dyn", "sizes": "32,128,256,512",
            "frame_records": "4", "nest": "0.3", "overlap": "0.0", "ask": "any", "register": "caption", "tags": "mode", "plant": "1.0",
            "grid": "ag", "grid_p": "0.5", "head": "digit", "head_lam": "1"}
    u = run_tag(dict(base, rotary_content="4", grid_abytes="1", series="posing"))
    ua = run_tag(dict(base, seed="3", grid_abytes="1", series="posing"))
    r = run_tag(dict(base, seed="2", rotary_content="4", series="widerep"))
    w = run_tag(dict(base, rotary_content="4", series="wide"))
    stem = "alhena-both_s{}_L24000_ctx4096_frdyn_cap_mode_sz32-128-256-512_r4_n30_o0_askany_gag50_"
    assert u == stem.format(1) + "cp4_ab1_ahdigit1_posing"
    assert ua == stem.format(3) + "ab1_ahdigit1_posing"
    assert r == stem.format(2) + "cp4_ahdigit1_widerep"
    assert run_tag(dict(base, rotary_content="4", grid_abytes="2", series="wide")) == w                  # the default form names nothing
    assert len({u, ua, r, w}) == 4


def test_the_runner_passes_the_form_and_labels_it():
    runs = br.expand(dict(arms="alhena-both", seeds="1,2", grid="ag", grid_p="0.5", head="digit", head_lam="1", rotary_content="4",
                          grid_abytes="1", series="posing"))
    assert len(runs) == 2 and all("--grid_abytes=1" in r for r in runs)
    assert "one-byte address" in br.run_label(runs[0]) and "content pairs 4" in br.run_label(runs[0])
    plain = br.expand(dict(arms="alhena-both", seeds="1", grid="ag", grid_p="0.5", head="digit", head_lam="1", series="find"))
    assert "one-byte" not in br.run_label(plain[0]) and "grid_abytes" in br.GRID_KEYS
