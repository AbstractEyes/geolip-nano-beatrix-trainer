# -*- coding: utf-8 -*-
"""The key in the told head's rotation (0.1.82): at each SEP the key code would code (its last CUE in the 10 positions before it, no SEP
between them) the told head's eight shared pairs turn by the key's phases phi_j(t) = sum_{k=1}^{n(t)} P[j, k, x_{t-k}] mod 2 pi instead
of their anchored angles; P is one seeded uniform draw (no gradient, nothing in any state dict). Off, the batches are 0.1.81's bit for
bit; the phases are their written formula; the same key turns every pair alike and one byte different moves every pair; only the told
head at the coded SEPs turns (every other head, pair and position is the anchored rotation's, value for value; a row with no coded SEP is
the off forward exactly); the pointer term's capture keeps the rotated q and k; the patch keeps the state dict and refuses the wrong order;
the side file round-trips through rotary_scale.Core; the use read; the names and the runner carry the flag. CPU only; the Hub is never
touched."""
import ast
import json
import math
import os

import numpy as np
import pytest
import torch

from geolip.alephllm.data.special_tokens import CUE, SEP
from geolip.alephllm.model.attention import rope_half
from geolip.nano_beatrix import anchored_rope as ar
from geolip.nano_beatrix import bed_runner as br
from geolip.nano_beatrix import grid_rows as gr
from geolip.nano_beatrix import key_code as kc
from geolip.nano_beatrix import key_rotary as kr
from geolip.nano_beatrix import pointer as pt
from geolip.nano_beatrix import rotary_scale as rs
from geolip.nano_beatrix.names import run_tag
from test_key_code import _by_hand, _save_core
from test_key_hard import FP_0174_BATCH, batch_digest
from test_posing import _row
from test_scale import _bed_batch
from test_value_addr import STEM, W_BASE, _craft

BLOCK, HEAD = 4, 0                                                                        # the recipes' told head (scale.json's ptr)


def _rows(seeds=(1, 2)):
    rows = [_row(seed=s) for s in seeds]
    x = torch.tensor(np.stack([np.frombuffer(r, dtype=np.uint8) for r, _ in rows]).astype(np.int64))[:, :-1]
    return x, [m for _, m in rows]


def _set_streams(st, metas):
    dk, dq = zip(*[gr.streams(m) for m in metas])
    st.set(torch.from_numpy(np.stack(dk).astype(np.int64)), torch.from_numpy(np.stack(dq).astype(np.int64)))


def test_off_the_batches_are_0181s_bit_for_bit():
    assert batch_digest(_bed_batch(4)) == FP_0174_BATCH
    src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "geolip", "nano_beatrix", "bed_train.py"), encoding="utf-8").read()
    fns = {n.name: ast.get_source_segment(src, n) for n in ast.parse(src).body if isinstance(n, ast.FunctionDef)}
    for fn in ("cap", "draw_plan", "frame_row", "ag_rows", "grid_aux", "batch"):            # the batch's whole path (test_scale._bed_batch)
        assert "KEY_ROT" not in fns[fn] and "key_rot" not in fns[fn] and "krm" not in fns[fn]


def test_the_phases_are_their_written_formula():
    rot = kr.KeyRotary(BLOCK, HEAD)
    for seed in (3, 8):
        row, _ = _row(seed=seed)
        x = torch.tensor(list(row[:-1]), dtype=torch.long)[None]
        phi, on = rot.phases(x)
        assert tuple(phi.shape) == (1, x.shape[1], kr.J) and phi.dtype == torch.float32 and on.dtype == torch.bool
        hand = {t: key for t, key in _by_hand(row[:-1]).items() if len(key) > 0}
        assert set(torch.nonzero(on[0]).flatten().tolist()) == set(hand)                   # the key code's coded SEPs, exactly
        assert set(hand) and all(6 <= len(k) <= kc.K for k in hand.values())
        for t, key in hand.items():
            s = None
            for k in range(1, len(key) + 1):                                                 # x_{t-1}, x_{t-2}, ... in that order
                term = rot.table[:, k - 1, key[-k]]
                s = term if s is None else s + term
            assert torch.equal(phi[0, t], torch.remainder(s, kr.TWO_PI))                    # the same sums in the same order, to the last bit
        assert float(phi.min()) >= 0.0 and float(phi.max()) < kr.TWO_PI


def test_the_same_key_turns_alike_and_one_byte_moves_every_pair():
    def row(*parts):
        return torch.tensor(list(b"".join(p if isinstance(p, bytes) else bytes([p]) for p in parts)), dtype=torch.long)[None]
    x = row(b"..", CUE, b"cat042", SEP, b"xyz", CUE, b"cat042", SEP, b"..", CUE, b"cat043", SEP, b".", CUE, b"dog042", SEP)
    rot = kr.KeyRotary(BLOCK, HEAD)
    phi, on = rot.phases(x)
    seps = torch.nonzero(on[0]).flatten().tolist()
    assert len(seps) == 4
    a, b, c, d = (phi[0, t] for t in seps)
    assert torch.equal(a, b)                                                                # the same key: every pair alike, bitwise
    for other in (c, d):                                                                    # one byte or the word apart: every pair moves
        diff = torch.remainder(a - other + math.pi, kr.TWO_PI) - math.pi
        assert bool((diff.abs() > 1e-4).all()), diff


def test_the_draw_is_seeded_and_takes_nothing_from_the_runs_generator():
    torch.manual_seed(1)
    a = kr.phase_table()
    after_a = torch.rand(3)
    torch.manual_seed(7)
    torch.rand(5)
    b = kr.phase_table()
    assert torch.equal(a, b) and tuple(a.shape) == (kr.J, kc.K, kc.VOCAB)
    torch.manual_seed(1)
    assert torch.equal(torch.rand(3), after_a)
    rot = kr.KeyRotary(BLOCK, HEAD)
    assert torch.equal(rot.table, a) and not rot.table.requires_grad
    assert not list(rot.parameters()) and "table" not in rot.state_dict()
    assert float(a.min()) >= 0.0 and float(a.max()) < kr.TWO_PI and abs(float(a.mean()) - math.pi) < 0.05     # uniform on [0, 2 pi)


def _told_angles(st, rot, B, H, n):
    """The angles the told block should use: the anchored ones, the told head's pairs 0 .. J - 1 at the coded SEPs replaced by phi."""
    aq, ak = st.angles(B, n, 0, torch.device("cpu"))
    aq = aq.expand(B, H, n, aq.shape[-1]).clone()
    ak = ak.expand(B, H, n, ak.shape[-1]).clone()
    on = rot.on[..., None]
    aq[:, HEAD, :, :kr.J] = torch.where(on, rot.phi, aq[:, HEAD, :, :kr.J])
    ak[:, HEAD, :, :kr.J] = torch.where(on, rot.phi, ak[:, HEAD, :, :kr.J])
    return aq, ak


def test_only_the_told_head_at_the_coded_seps_turns_and_the_capture_keeps_it():
    x, metas = _rows()
    m_off, st_off = _craft(n_ctx=gr.CTX, seed=4)
    cap_off = pt.HeadCapture(m_off.blocks[BLOCK - 1].attn, head=HEAD)
    m_on, st_on = _craft(n_ctx=gr.CTX, seed=4)
    rot = kr.KeyRotary(BLOCK, HEAD)
    kr.patch(m_on, rot)
    cap_on = pt.HeadCapture(m_on.blocks[BLOCK - 1].attn, head=HEAD)                    # the term's capture, after the rotation
    _set_streams(st_off, metas)
    _set_streams(st_on, metas)
    with torch.no_grad():
        lo = m_off(x).logits
        ln = m_on(x).logits
    assert rot.on is not None and int(rot.on.sum()) == sum(gr.N_COMP + len(mt["index"]) + len(mt["questions"]) for mt in metas)
    q0, k0, _ = cap_off.qk()
    q1, k1, _ = cap_on.qk()
    B, H, n, hd = q1.shape
    others = [h for h in range(H) if h != HEAD]
    assert torch.equal(q0[:, others], q1[:, others]) and torch.equal(k0[:, others], k1[:, others])   # every other head, bitwise
    off_pos = ~rot.on
    assert torch.equal(q0[:, HEAD][off_pos], q1[:, HEAD][off_pos])                          # the told head away from the coded SEPs
    aq0, ak0 = st_on.angles(B, n, 0, torch.device("cpu"))
    raw_q = rope_half(q0, aq0.cos(), -aq0.sin())                                            # the off rotation undone: the raw q and k
    raw_k = rope_half(k0, ak0.cos(), -ak0.sin())
    aq, ak = _told_angles(st_on, rot, B, H, n)
    assert torch.allclose(rope_half(raw_q, aq.cos(), aq.sin()), q1, atol=1e-5)              # the captured q and k are the key rotation's
    assert torch.allclose(rope_half(raw_k, ak.cos(), ak.sin()), k1, atol=1e-5)
    anchor = list(range(kr.J, ar.PAIRS)) + list(range(ar.PAIRS + kr.J, 2 * ar.PAIRS))
    assert torch.equal(q0[:, HEAD][..., anchor], q1[:, HEAD][..., anchor])                  # the anchor pairs untouched at every position
    assert not torch.equal(lo, ln)                                                          # the grid rows' forward moves
    st_off.clear()
    st_on.clear()
    t = torch.randint(32, 127, (2, 300))                                                    # text: no coded SEP
    with torch.no_grad():
        assert torch.equal(m_off(t).logits, m_on(t).logits)                                 # the off forward exactly


def test_the_score_is_the_conjunction():
    # two coded SEPs with the same key: the told head's key-pair term = the unrotated term; a byte apart: it is not
    m, st = _craft(n_ctx=256, seed=5)
    rot = kr.KeyRotary(BLOCK, HEAD)
    kr.patch(m, rot)
    x = torch.randint(32, 127, (1, 200))
    for p, key in ((20, b"cat042"), (90, b"cat042"), (150, b"cat043")):
        x[0, p:p + len(key) + 2] = torch.tensor([CUE] + list(key) + [SEP])
    seps = [27, 97, 157]
    rot.set(x)
    assert all(bool(rot.on[0, s]) for s in seps)
    g = torch.Generator().manual_seed(0)
    q = torch.randn(1, 8, 200, 32, generator=g)
    k = torch.randn(1, 8, 200, 32, generator=g)
    a = m.blocks[BLOCK - 1].attn
    qr, kr_ = a._rotate(q, k)
    keyd = kr.KEY_DIMS

    def term(qq, kk, i, j):
        return float((qq[0, HEAD, i, keyd] * kk[0, HEAD, j, keyd]).sum())
    assert abs(term(qr, kr_, 97, 27) - term(q, k, 97, 27)) < 1e-4                       # the same key: as if unrotated
    assert abs(term(qr, kr_, 157, 27) - term(q, k, 157, 27)) > 1e-3                     # one byte apart: turned
    with pytest.raises(AssertionError):
        a._rotate(q[:, :, :100], k[:, :, :100])                                             # the phases of other rows are refused


def test_the_patch_keeps_the_state_dict_and_refuses_the_wrong_order():
    m, st = _craft(n_ctx=256)
    before = {k: v.clone() for k, v in m.state_dict().items()}
    rot = kr.KeyRotary(BLOCK, HEAD)
    kr.patch(m, rot)
    after = m.state_dict()
    assert before.keys() == after.keys() and all(torch.equal(before[k], after[k]) for k in before)
    m2, _ = _craft(n_ctx=256)
    pt.HeadCapture(m2.blocks[BLOCK - 1].attn, head=HEAD)                                   # a capture first: the rotation is refused
    with pytest.raises(AssertionError):
        kr.patch(m2, kr.KeyRotary(BLOCK, HEAD))
    from geolip.alephllm.model.alephlm import AlephLM
    from geolip.nano_beatrix.bodies import craft_cfg
    plain = AlephLM(craft_cfg("alhena-both", 256)).eval()                                   # no anchored rotation: refused
    with pytest.raises(AssertionError):
        kr.patch(plain, kr.KeyRotary(BLOCK, HEAD))


def test_the_side_file_round_trips_through_the_core(tmp_path):
    m, st = _craft(n_ctx=gr.CTX, seed=2)
    va = ar.ValueAddress(m.cfg.d_model, st)
    with torch.no_grad():
        va.table.normal_(std=0.5)
    ar.patch_value(m, va)
    rot = kr.KeyRotary(BLOCK, HEAD)
    kr.patch(m, rot)
    d = tmp_path / "core"
    sj = _save_core(m, d, va)
    from safetensors.torch import save_file
    save_file({"table": rot.table.detach().cpu().contiguous()}, str(d / "key_rot.safetensors"))
    sj["key_rot"] = dict(pairs=kr.J, k=kc.K, vocab=kc.VOCAB, block=BLOCK, head=HEAD, seed_tag=kr.SEED_TAG)
    sj["ptr"] = dict(lam=0.1, block=BLOCK, head=HEAD)
    json.dump(sj, open(d / "scale.json", "w", encoding="utf-8"), indent=1)
    core = rs.Core(str(d), "cpu")
    assert core.key_rot is not None and torch.equal(core.key_rot.table, rot.table)
    assert core.look == (BLOCK, HEAD)                                                       # the look's capture after the rotation
    x, metas = _rows((4,))
    core.set_streams(metas)
    with torch.no_grad():
        got = core.readout(core.trunk(x))
        h3 = core.trunk(x, 0, 3)
        split = core.readout(core.trunk(x, 3, None, h3))                                   # from a given state: the phases set from x
    _set_streams(st, metas)
    with torch.no_grad():
        exp = m(x).logits
    st.clear()
    core.clear()
    assert torch.equal(got, exp) and torch.equal(split, exp)                               # the reader's forward = the trainer's
    dg = core.state_digest()
    assert dg == core.digest
    with torch.no_grad():
        core.key_rot.table[0, 0, 49] += 0.5
    assert core.state_digest() != dg                                                        # a moved phase is a moved core
    os.remove(d / "key_rot.safetensors")
    with pytest.raises(AssertionError):
        rs.Core(str(d), "cpu")
    del sj["key_rot"]
    json.dump(sj, open(d / "scale.json", "w", encoding="utf-8"), indent=1)
    plain = rs.Core(str(d), "cpu")
    assert plain.key_rot is None and plain.state_digest() == rs._digest(plain.model, plain.value_addr.table)   # 0.1.81's digest


def test_the_use_read():
    m, st = _craft(n_ctx=gr.CTX, seed=6)
    rot = kr.KeyRotary(BLOCK, HEAD)
    kr.patch(m, rot)
    cap = pt.HeadCapture(m.blocks[BLOCK - 1].attn, head=HEAD)
    x, metas = _rows()
    _set_streams(st, metas)
    with torch.no_grad():
        m(x)
    q, k, h = cap.qk()
    u = kr.use_read(q, k, h, x, metas, rot.on)
    st.clear()
    assert u["entries"] == sum(len(mt["index"]) for mt in metas)
    assert u["coded_seps"] == int(rot.on.sum())
    for s in ("key_true", "key_other", "key_self", "anchor_true", "anchor_other"):
        assert math.isfinite(u[s])
    assert u["q_key_norm_coded"] > 0 and u["k_key_norm_coded"] > 0 and u["q_key_norm_other"] > 0


def test_the_key_rot_runs_never_share_a_name():
    pk = dict(W_BASE, grid_hard="0.5", value_addr="1", ptr_lam="0.1")
    p = dict(W_BASE, value_addr="1", ptr_lam="0.1")
    krk = run_tag(dict(pk, key_rot="1", series="keyrot"))
    krp = run_tag(dict(p, key_rot="1", series="keyrot"))
    assert krk == STEM.format(1) + "kh50_va1_pt10b4h0_krt_ahdigit1_keyrot"
    assert krp == STEM.format(1) + "va1_pt10b4h0_krt_ahdigit1_keyrot"
    assert run_tag(dict(pk, key_rot="0", series="keypointer")) == run_tag(dict(pk, series="keypointer"))      # off names nothing
    assert len({krk, krp, run_tag(dict(pk, seed="2", key_rot="1", series="keyrot"))}) == 3


def test_the_runner_passes_the_flag_and_labels_it():
    runs = br.expand(dict(arms="alhena-both", seeds="1", grid="ag", grid_p="0.5", head="digit", head_lam="1", rotary_content="4",
                          grid_hard="0.5", value_addr="1", ptr_lam="0.1", key_rot="1", series="keyrot"))
    assert len(runs) == 1 and "--key_rot=1" in runs[0] and "key rotation" in br.run_label(runs[0])
    assert "key_rot" in br.GRID_KEYS
