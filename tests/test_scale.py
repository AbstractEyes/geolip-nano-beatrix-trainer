# -*- coding: utf-8 -*-
"""The scale test's cores (0.1.70): the addressed grid rows (grid_rows.py), the anchored multi-position rotary (anchored_rope.py) and the
names of the runs. CPU only; the Hub is never touched."""
import math

import numpy as np
import pytest
import torch

from geolip.alephllm.model.alephlm import AlephLM
from geolip.nano_beatrix import anchored_rope as ar
from geolip.nano_beatrix import bed_runner as br
from geolip.nano_beatrix import grid_rows as gr
from geolip.nano_beatrix.bodies import craft_cfg
from geolip.nano_beatrix.names import run_tag


# ------------------------------------------------------------------ the grid rows
def test_the_grid_rows_hold_their_arithmetic():
    gr.selftest(60)


def test_a_row_carries_24_sorted_addresses_and_asks_each_record_once():
    rng = np.random.default_rng(3)
    src = b"lorem ipsum dolor sit amet " * 200
    row, m = gr.compose_ag(rng, src)
    assert m["addrs"] == sorted(m["addrs"]) and len(set(m["addrs"])) == gr.N_COMP
    assert sorted(q["rec"] for q in m["questions"]) == list(range(gr.N_COMP))
    assert [e["rec"] for e in m["index"]] == [q["rec"] for q in m["questions"]]       # the index lists the questions' records in order
    assert m["quest_end"] <= gr.QUEST_HI and m["index_end"] <= gr.QUEST_LO


def test_a_written_request_reads_the_index_bytes():
    rng = np.random.default_rng(4)
    row, m = gr.compose_ag(rng, b"x" * 5000)
    pred = np.frombuffer(row, dtype=np.uint8)[1:].copy()                            # a craft that writes the row's own bytes
    want = [gr.digits(m["recs"][q["rec"]]["addr"]) for q in m["questions"]]
    assert gr.written_request(pred, m) == want
    p1, _ = gr.index_positions(m)[0]
    pred[p1] = ord("9")                                                              # a byte outside '0'-'7': the null anchor
    assert gr.written_request(pred, m)[0][0] == gr.NULL


def test_the_owned_order_is_fixed():
    a, b = gr.owned_order(), gr.owned_order()
    assert a == b and len(a[0]) == 48 and len(a[1]) == 16 and set(a[0]) | set(a[1]) == set(range(64))


# ------------------------------------------------------------------ the anchored rotation
def test_the_band_frequencies_and_the_simplex():
    w, kind = ar.frequencies()
    rel = w[kind == 0]
    wl = (2 * math.pi / rel).tolist()
    assert [round(x) for x in wl[:3]] == [6, 38, 236] and abs(wl[5] / 54100 - 1) < 0.01         # the NTK x4 band's wavelengths
    assert (w[kind == 1] == 0).all()
    wa = w[kind == 2]
    for d in range(9):
        s = float(torch.cos(wa * d).sum())
        assert abs(s - (4.0 if d == 0 else -0.5)) < 1e-9, (d, s)                                  # the nine digits are a simplex


def _expected_logits(q, k, dq, dk):
    """The written formula: content <q, k> + relative Re[q conj(k) e^{i w (m - n)}] + anchors Re[q conj(k) e^{i w (dq(m) - dk(n))}]."""
    w, kind = ar.frequencies()
    P = ar.PAIRS
    qc = torch.complex(q[..., :P].double(), q[..., P:].double())                    # (B, h, n, P)
    kc = torch.complex(k[..., :P].double(), k[..., P:].double())
    n = q.shape[-2]
    m_idx = torch.arange(n, dtype=torch.float64)
    out = torch.zeros(q.shape[0], q.shape[1], n, n, dtype=torch.float64)
    for j in range(P):
        prod = qc[..., j][..., :, None] * kc[..., j].conj()[..., None, :]            # (B, h, n, n)
        if kind[j] == 0:
            ang = w[j] * (m_idx[:, None] - m_idx[None, :])
        elif kind[j] == 1:
            ang = torch.zeros(n, n, dtype=torch.float64)
        else:
            lev = int(kind[j]) - 2
            ang = w[j] * (dq[:, :, lev].double()[:, None, :, None] - dk[:, :, lev].double()[:, None, None, :])
        out += (prod * torch.exp(1j * ang)).real
    return out


def test_the_rotation_is_the_written_formula():
    torch.manual_seed(0)
    B, h, n = 2, 3, 40
    q, k = torch.randn(B, h, n, 2 * ar.PAIRS), torch.randn(B, h, n, 2 * ar.PAIRS)
    dq, dk = torch.randint(0, 9, (B, n, 2)), torch.randint(0, 9, (B, n, 2))
    st = ar.AnchorStreams()
    st.set(dk, dq)

    class Fake:
        _anchor = st
    qr, kr = ar._anchored_rotate(Fake(), q, k)
    got = (qr.double() @ kr.double().transpose(-1, -2))
    exp = _expected_logits(q, k, dq, dk)
    assert torch.allclose(got, exp, atol=1e-4), float((got - exp).abs().max())
    st.clear()                                                                      # no streams: every token at the null anchor
    qr0, kr0 = ar._anchored_rotate(Fake(), q, k)
    st.set(torch.full((B, n, 2), 8), torch.full((B, n, 2), 8))
    qr8, kr8 = ar._anchored_rotate(Fake(), q, k)
    assert torch.equal(qr0.expand_as(qr8), qr8) and torch.equal(kr0.expand_as(kr8), kr8)


def test_the_patch_keeps_the_state_dict_and_reads_the_streams():
    torch.manual_seed(1)
    m = AlephLM(craft_cfg("castor-both", 256))
    before = {k: v.clone() for k, v in m.state_dict().items()}
    st = ar.AnchorStreams()
    assert ar.patch(m, st) == 6
    after = m.state_dict()
    assert before.keys() == after.keys() and all(torch.equal(before[k], after[k]) for k in before)
    m.eval()
    x = torch.randint(0, 256, (2, 64))
    with torch.no_grad():
        a = m(x)[0]
        st.set(torch.full((2, 64, 2), 8), torch.full((2, 64, 2), 8))
        b = m(x)[0]
        dk = torch.full((2, 64, 2), 8)
        dk[:, 10:20] = torch.tensor([3, 5])
        st.set(dk, dk.clone())
        c = m(x)[0]
        st.clear()
    assert torch.allclose(a, b, atol=1e-6)                                          # the null anchor everywhere = no streams
    assert not torch.allclose(a[:, 10:], c[:, 10:], atol=1e-4)                      # an addressed compartment changes the reads after it
    assert torch.equal(a[:, :10], c[:, :10])                                        # causal: the positions before it are untouched


# ------------------------------------------------------------------ the names
STORED = {"arm": "castor-both", "seed": "1", "ctx": "4096", "mb": "8", "steps": "12000", "frames": "dyn", "sizes": "32,128,256,512",
          "frame_records": "4", "nest": "0.3", "overlap": "0.0", "ask": "any", "register": "caption", "tags": "mode", "plant": "1.0"}


def test_alhena_is_castors_body_at_birth():
    """The third of the triplets: Castor's layout and Castor's init from the same seed (the rotation is the trainer's, no parameter)."""
    torch.manual_seed(1)
    c = AlephLM(craft_cfg("castor-both", 4096))
    torch.manual_seed(1)
    a = AlephLM(craft_cfg("alhena-both", 4096))
    sc, sa = c.state_dict(), a.state_dict()
    assert sc.keys() == sa.keys() and all(torch.equal(sc[k], sa[k]) for k in sc)
    assert craft_cfg("alhena-both", 4096).name == "bed-alhena-both"


def test_the_cores_never_share_a_name_with_the_stored_cell():
    stored = run_tag(STORED)
    assert stored == "castor-both_s1_L12000_ctx4096_frdyn_cap_mode_sz32-128-256-512_r4_n30_o0_askany"
    r = run_tag(dict(STORED, series="scale"))
    r2 = run_tag(dict(STORED, series="scalerep"))
    h = run_tag(dict(STORED, grid="ag", grid_p="0.5", head="digit", head_lam="1", series="scale"))
    c = run_tag(dict(STORED, grid="ag", grid_p="0.5", rotary="anchored", head="digit", head_lam="1", series="scale"))
    al = run_tag(dict(STORED, arm="alhena-both", grid="ag", grid_p="0.5", head="digit", head_lam="1", series="scale"))
    assert r == stored + "_scale" and r2 == stored + "_scalerep"
    assert h == stored + "_gag50_ahdigit1_scale" and c == stored + "_gag50_ranc_ahdigit1_scale"
    assert al == "alhena-both" + h[len("castor-both"):]
    assert len({stored, r, r2, h, c, al}) == 6


def test_the_runner_passes_the_scale_keys_and_labels_them():
    runs = br.expand(dict(arms="castor-both", seeds="1,2", grid="ag", grid_p="0.5", rotary="anchored", head="digit", head_lam="1",
                          series="scale"))
    assert len(runs) == 2 and all("--rotary=anchored" in r and "--series=scale" in r for r in runs)
    lab = br.run_label(runs[0])
    assert "grid ag 0.5" in lab and "rotary anchored" in lab and "head digit 1" in lab and "scale" in lab


@pytest.mark.parametrize("p,mb,n", [(0.5, 8, 4), (0.25, 8, 2), (1.0, 8, 8)])
def test_the_grid_share_is_a_count_of_rows(p, mb, n):
    assert int(round(mb * p)) == n


# ------------------------------------------------------------------ the batch: the plain rows are the plain run's rows
def _bed_batch(n_ag: int, n_hard: int = 0):
    """bed_train's own batch / ag_rows / grid_aux and the framed composer, lifted out of the file (its module reads a command line and a
    corpus at import), bound to the scale test's framed settings at 4,096 x 8 with n_ag grid rows (0.1.75: the last n_hard of them
    key-hard rows)."""
    import ast
    import os
    from geolip.nano_beatrix import bed_frames as bf
    from geolip.alephllm.data.special_tokens import DOC
    src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "geolip", "nano_beatrix", "bed_train.py"), encoding="utf-8").read()
    names = ("cap", "draw_plan", "frame_row", "ag_rows", "grid_aux", "batch")
    keep = [n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert len(keep) == len(names)
    rng = np.random.default_rng(0)
    packed = rng.integers(32, 127, size=400_000, dtype=np.uint8)
    ns = dict(bf=bf, gr=gr, np=np, torch=torch, DOC=DOC, CTX=4096, MB=8, N_AG=n_ag, SEED=1, P_PLANT=1.0, FRAMES="dyn", N_REC=1,
              SIZES=[32, 128, 256, 512], N_REC_FRAME=4, NEST_P=0.3, OVERLAP_P=0.0, REGISTER="caption", TAGS="mode", SIZED=False, SLICED=0,
              ASK="any", FACTS=None, GAUGE_FLOOR={}, N_LOOKUP=2, N_BULK=0, REC_BYTES=bf.capacity(4096, "A", "caption")["record_bytes"],
              BANDS=[(0, 128), (128, 256), (256, 512), (512, 100000)], PLANTED=[0, 0], AG_SALT=gr.crc("AG"), packed=packed,
              GRID_ABYTES=2,                                             # 0.1.74: the grid rows' address form (the default)
              N_HARD=n_hard)                                             # 0.1.75: the batch's key-hard grid rows
    exec(compile(ast.Module(body=keep, type_ignores=[]), "bed_train.py", "exec"), ns)
    return ns["batch"]


def test_the_grid_batch_keeps_the_plain_runs_rows_in_their_places():
    plain, grid = _bed_batch(0), _bed_batch(4)
    for step in (0, 1, 57):
        x0, y0, v0, a0 = plain(step)
        x1, y1, v1, a1 = grid(step)
        assert a0 is None and a1 is not None and x1.shape == x0.shape == (8, 4096)
        assert torch.equal(x0[:4], x1[:4]) and torch.equal(y0[:4], y1[:4]) and torch.equal(v0[:4], v1[:4])   # rows 0-3: the plain run's
        assert (v1[4:] == 0).all() and a1["first"] == 4 and len(a1["metas"]) == 4
        assert (a1["dk"][:4] == gr.NULL).all() and (a1["dq"][:4] == gr.NULL).all()                         # the plain rows: the null anchor
        assert int(a1["head_row"].min()) == 4 and len(a1["head_pos"]) == 4 * gr.N_COMP
        for j, m in enumerate(a1["metas"]):
            row = bytes(x1[4 + j].tolist())
            assert row[:len(gr.HEAD)] == gr.HEAD and row[m["recs"][0]["lo"]] == 0xF9                     # a grid row in its place
