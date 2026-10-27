# -*- coding: utf-8 -*-
"""The path in one head's rotation (0.1.92): a byte starts a unit under the fusion stage's entropy rule (the library's Fusion.starts, value
for value), the path at t is the bytes from its unit's start through t, and the path head's anchor pairs (8-15) turn by
phi_j(t) = sum_{k=0}^{min(t - s(t) + 1, K) - 1} P[j, k, x_(t-k)] mod 2 pi at every position of every anchored block; "same" turns the query
and the key at t by phi(t), "next" the key at u by phi(u - 1). Off, the batches are 0.1.91's bit for bit; the phases are their written
formula; equal paths turn alike and one byte apart moves every pair; the score is the identity (same) and the continuation (next); only
the path head's anchor pairs turn (every other head, pair and block value for value, the key rotation's head included) and the toggle is
the forward without the path exactly; the draw is seeded; the state dict is kept; the side file round-trips through rotary_scale.Core and
its text read; the repeat mask; the names and the runner carry the flag. The unit rule runs on made tables here (the atlas table itself
when $GEOLIP_ATLAS_TABLE names it). CPU only; the Hub is never touched."""
import ast
import hashlib
import json
import math
import os

import numpy as np
import pytest
import torch

from geolip.alephllm.model.attention import rope_half
from geolip.alephllm.model.fusion import Fusion
from geolip.nano_beatrix import anchored_rope as ar
from geolip.nano_beatrix import bed_runner as br
from geolip.nano_beatrix import grid_rows as gr
from geolip.nano_beatrix import key_rotary as kr
from geolip.nano_beatrix import path_rotary as prm
from geolip.nano_beatrix import pointer as pt
from geolip.nano_beatrix import rotary_scale as rs
from geolip.nano_beatrix.names import run_tag
from test_key_code import _save_core
from test_key_hard import FP_0174_BATCH, batch_digest
from test_key_rot import _rows, _set_streams
from test_scale import _bed_batch
from test_value_addr import STEM, W_BASE, _craft

BLOCK, HEAD, PHEAD = 4, 0, 7                                                                 # the told head; the path head
CELLS = 256 ** 3


def _save_table(path, ent, wit):
    np.savez(path, entropy_x16=ent, witness=wit, thresholds=np.array([1.0, 2.0, 2.5, 3.0, 3.5, 4.0]), witness_floor=np.float64(8))
    return str(path)


@pytest.fixture(scope="module")
def space_table(tmp_path_factory):
    """A made table: a cell is a choice point (4 bits) when its last byte is a space, else 1 bit; every cell witnessed - so a unit starts
    at every byte after a space (and at the forced starts)."""
    c = np.arange(CELLS, dtype=np.int64) % 256
    ent = np.where(c == 32, 64, 16).astype(np.uint8)
    wit = np.full(CELLS, 100, dtype=np.uint16)
    return _save_table(tmp_path_factory.mktemp("atlas") / "space.npz", ent, wit)


@pytest.fixture(scope="module")
def random_table(tmp_path_factory):
    """A made table with every case: entropies 0-80 (in 1/16 bit; the threshold 48 inside), witness counts 0-19 (the floor 8 inside)."""
    g = np.random.default_rng(7)
    ent = g.integers(0, 81, size=CELLS).astype(np.uint8)
    wit = g.integers(0, 20, size=CELLS).astype(np.uint16)
    return _save_table(tmp_path_factory.mktemp("atlas") / "random.npz", ent, wit)


def _text_rows(B=4, n=600, seed=0):
    """Rows of a small alphabet with spaces, newlines and special bytes, so cells and paths repeat."""
    g = np.random.default_rng(seed)
    alpha = np.frombuffer(b"abcde fgh \n", dtype=np.uint8)
    x = alpha[g.integers(0, len(alpha), size=(B, n))].astype(np.int64)
    sp = sorted(prm.SPECIALS)
    x[g.random((B, n)) < 0.02] = sp[0]
    x[:, 0] = sp[0]
    return torch.from_numpy(x)


def _units(table):
    return prm.Units(table, pinned=False)


def test_off_the_batches_are_0191s_bit_for_bit():
    assert batch_digest(_bed_batch(4)) == FP_0174_BATCH
    src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "geolip", "nano_beatrix", "bed_train.py"), encoding="utf-8").read()
    fns = {n.name: ast.get_source_segment(src, n) for n in ast.parse(src).body if isinstance(n, ast.FunctionDef)}
    for fn in ("cap", "draw_plan", "frame_row", "ag_rows", "grid_aux", "batch"):            # the batch's whole path (test_scale._bed_batch)
        assert "PATH" not in fns[fn] and "path_rot" not in fns[fn] and "prm" not in fns[fn] and "REPEAT" not in fns[fn]


def test_the_starts_are_fusion_starts(random_table, space_table):
    for table in (random_table, space_table):
        u = _units(table)
        fu = Fusion(dict(rule="entropy", theta=prm.THETA, table=table, witness_floor=prm.FLOOR), d_model=8)
        for seed in range(4):
            x = _text_rows(B=16, n=512, seed=seed)                                          # 64 rows a table
            assert torch.equal(u.starts(x), fu.starts(x))
        rows, _ = _rows((3, 4))
        assert torch.equal(u.starts(rows), fu.starts(rows))                                   # the grid rows too
    real = os.environ.get(prm.TABLE_ENV)
    if real and os.path.exists(real):                                                         # the atlas table itself, pinned
        u = prm.Units()
        fu = Fusion(dict(rule="entropy", theta=prm.THETA, table=real, witness_floor=prm.FLOOR), d_model=8)
        x = _text_rows(B=64, n=512, seed=9)
        assert torch.equal(u.starts(x), fu.starts(x))


def test_the_depth_and_the_forced_starts(space_table):
    u = _units(space_table)
    row = b"ab cd\nef gh"
    x = torch.tensor(list(row), dtype=torch.long)[None]
    st = u.starts(x)[0].tolist()
    #       a     b     ' '    c     d    \n     e     f    ' '    g     h
    assert st == [True, True, True, True, False, True, True, False, False, True, False]
    assert u.depth(x)[0].tolist() == [1, 1, 1, 1, 2, 1, 1, 2, 3, 1, 2]


def _by_hand(x, table, units, K=prm.K):
    """phi (n, 8) for one row from the written formula: the unit's start by a scan of the starts, the sum over k = 0 .. min(d, K) - 1."""
    st = units.starts(x[None])[0].tolist()
    out, s = [], 0
    for t in range(len(x)):
        if st[t]:
            s = t
        d = t - s + 1
        acc = None
        for k in range(min(d, K)):
            term = table[:, k, int(x[t - k])]
            acc = term if acc is None else acc + term
        out.append(torch.remainder(acc, prm.TWO_PI))
    return torch.stack(out)


def test_the_phases_are_their_written_formula(random_table, space_table):
    for table, K in ((random_table, prm.K), (space_table, prm.K), (space_table, 3)):
        pr = prm.PathRotary(_units(table), "same", k=K)
        x = _text_rows(B=2, n=300, seed=K)
        phi = pr.phases(x)
        assert tuple(phi.shape) == (2, 300, prm.PAIRS) and phi.dtype == torch.float32
        for b in range(2):
            assert torch.equal(phi[b], _by_hand(x[b], pr.table, pr.units, K))              # the same sums in the same order, to the last bit
        assert float(phi.min()) >= 0.0 and float(phi.max()) < prm.TWO_PI
    long_word = b"x " + b"q" * 40 + b" y"                                                     # a path past the reach: its last K bytes only
    x = torch.tensor(list(long_word), dtype=torch.long)[None]
    pr = prm.PathRotary(_units(space_table), "same")
    phi = pr.phases(x)
    assert torch.equal(phi[0, 2 + 30], phi[0, 2 + 39])                                      # 16 q's behind both: the same phases


def test_equal_paths_turn_alike_and_one_byte_moves_every_pair(space_table):
    row = b"go the cat sat on the cab and the cat ran"
    x = torch.tensor(list(row), dtype=torch.long)[None]
    pr = prm.PathRotary(_units(space_table), "same")
    phi = pr.phases(x)[0]
    cats = [i for i in range(len(row)) if row[i:i + 3] == b"cat"]
    cab = row.index(b"cab")
    assert len(cats) == 2
    a, b, c = phi[cats[0] + 2], phi[cats[1] + 2], phi[cab + 2]                               # the paths "cat", "cat", "cab"
    assert torch.equal(a, b)                                                                  # the same path: every pair alike, bitwise
    diff = torch.remainder(a - c + math.pi, prm.TWO_PI) - math.pi
    assert bool((diff.abs() > 1e-4).all()), diff                                             # one byte apart: every pair moves
    assert torch.equal(phi[cats[0] + 1], phi[cab + 1])                                        # "ca" = "ca": the trie's shared node


def _attn_and_rotary(form, table, seed=5, n_ctx=256):
    m, st = _craft(n_ctx=n_ctx, seed=seed)
    krot = kr.KeyRotary(BLOCK, HEAD, pairs=16, norm=2, rho=2.18)
    kr.patch(m, krot)
    pr = prm.PathRotary(_units(table), form, head=PHEAD)
    n = prm.patch(m, pr)
    return m, st, pr, krot, n


def test_the_score_is_the_identity_and_the_continuation(space_table):
    row = b"go the cat sat on the cab and the cat ran away"
    x = torch.tensor(list(row), dtype=torch.long)[None]
    cats = [i + 2 for i in range(len(row)) if row[i:i + 3] == b"cat"]                       # the positions of the path "cat"
    cab = row.index(b"cab") + 2
    g = torch.Generator().manual_seed(0)
    q = torch.randn(1, 8, len(row), 32, generator=g)
    k = torch.randn(1, 8, len(row), 32, generator=g)
    dims = list(range(prm.LO, ar.PAIRS)) + list(range(ar.PAIRS + prm.LO, 2 * ar.PAIRS))

    def term(qq, kk, i, j):
        return float((qq[0, PHEAD, i, dims] * kk[0, PHEAD, j, dims]).sum())
    for form in prm.FORMS:
        m, st, pr, krot, n = _attn_and_rotary(form, space_table)
        a = m.blocks[0].attn
        pr.set(x)
        krot.set(x)
        qr, kr_ = a._rotate(q, k)
        t1, t2 = cats
        if form == "same":
            assert abs(term(qr, kr_, t2, t1) - term(q, k, t2, t1)) < 1e-4                  # equal paths: as if unrotated
            assert abs(term(qr, kr_, t2, cab) - term(q, k, t2, cab)) > 1e-3                # one byte apart: turned
        else:
            assert abs(term(qr, kr_, t2, t1 + 1) - term(q, k, t2, t1 + 1)) < 1e-4          # the key after the earlier path: as if unrotated
            assert abs(term(qr, kr_, t2, t1) - term(q, k, t2, t1)) > 1e-3                  # the earlier path's own key: turned
            assert abs(term(qr, kr_, t2, cab + 1) - term(q, k, t2, cab + 1)) > 1e-3        # after another path: turned
        with pytest.raises(AssertionError):
            a._rotate(q[:, :, :20], k[:, :, :20])                                             # the phases of other rows are refused


def test_only_the_path_heads_anchor_pairs_turn_and_the_toggle_is_exact(space_table):
    x, metas = _rows()
    for form in prm.FORMS:
        m, st, pr, krot, n = _attn_and_rotary(form, space_table, n_ctx=gr.CTX)
        assert n == len(m.blocks)                                                            # every block of the craft is anchored
        caps = [pt.HeadCapture(b.attn, head=HEAD) for b in m.blocks]                         # every block's rotated q and k
        _set_streams(st, metas)
        out = {}
        for off in (True, False):
            pr.off = off
            with torch.no_grad():
                lg = m(x).logits
            out[off] = (lg, [c.qk()[:2] for c in caps])
        pr.off = False
        dims_lo = list(range(prm.LO)) + list(range(ar.PAIRS, ar.PAIRS + prm.LO))
        dims_hi = list(range(prm.LO, ar.PAIRS)) + list(range(ar.PAIRS + prm.LO, 2 * ar.PAIRS))
        others = [h for h in range(8) if h != PHEAD]
        (q0, k0), (q1, k1) = out[True][1][0], out[False][1][0]                                 # block 1: the same input either way
        assert torch.equal(q0[:, others], q1[:, others]) and torch.equal(k0[:, others], k1[:, others])   # every other head, bitwise
        assert torch.equal(q0[:, PHEAD][..., dims_lo], q1[:, PHEAD][..., dims_lo])           # the path head's other pairs, bitwise
        assert torch.equal(k0[:, PHEAD][..., dims_lo], k1[:, PHEAD][..., dims_lo])
        assert not torch.equal(q0[:, PHEAD][..., dims_hi], q1[:, PHEAD][..., dims_hi])       # its anchor pairs turned
        assert not torch.equal(out[True][0], out[False][0])                                   # the forward moves with the path
        # every block's rotation on the same q and k (the told block's key rotation included): only the path head's anchor pairs move,
        # to the input's pairs turned by the path's phases
        g = torch.Generator().manual_seed(11)
        qi = torch.randn(x.shape[0], 8, x.shape[1], 32, generator=g)
        ki = torch.randn(x.shape[0], 8, x.shape[1], 32, generator=g)
        for bi, b in enumerate(m.blocks):
            a = b.attn
            pr.off = True
            qa, ka = a._rotate(qi, ki)
            pr.off = False
            qb, kb = a._rotate(qi, ki)
            assert torch.equal(qa[:, others], qb[:, others]) and torch.equal(ka[:, others], kb[:, others]), bi
            assert torch.equal(qa[:, PHEAD][..., dims_lo], qb[:, PHEAD][..., dims_lo]), bi
            assert torch.equal(ka[:, PHEAD][..., dims_lo], kb[:, PHEAD][..., dims_lo]), bi
            for t_in, got, ph in ((qi, qb, pr.phi_q), (ki, kb, pr.phi_k)):
                a_, b_ = t_in[:, PHEAD][..., prm.LO:ar.PAIRS], t_in[:, PHEAD][..., ar.PAIRS + prm.LO:]
                want = torch.cat([a_ * ph.cos() - b_ * ph.sin(), a_ * ph.sin() + b_ * ph.cos()], dim=-1)
                assert torch.equal(got[:, PHEAD][..., dims_hi], want), bi                    # the written turn, to the last bit
        assert krot.on is not None and bool(krot.on.any())                                    # the told block's key rotation was live
        # the toggle = the forward without the path, exactly: a craft of the same seed with the key rotation alone
        m2, st2 = _craft(n_ctx=gr.CTX, seed=5)
        kr.patch(m2, kr.KeyRotary(BLOCK, HEAD, pairs=16, norm=2, rho=2.18))
        _set_streams(st2, metas)
        with torch.no_grad():
            plain = m2(x).logits
        assert torch.equal(out[True][0], plain)
        # the turned pairs: the raw q and k (the anchored rotation undone) turned by the path's phases
        q0, k0 = out[True][1][0]
        q1, k1 = out[False][1][0]
        B, H, n_, hd = q0.shape
        aq, ak = st.angles(B, n_, 0, torch.device("cpu"))
        raw_q, raw_k = rope_half(q0, aq.cos(), -aq.sin()), rope_half(k0, ak.cos(), -ak.sin())
        phq = pr.phi_q
        phk = pr.phi_k
        if form == "next":
            assert torch.equal(phk[:, 1:], phq[:, :-1]) and not bool(phk[:, 0].any())        # the key at u by phi(u - 1); phi(-1) = 0
        else:
            assert phk is phq
        for raw, got, ph in ((raw_q, q1, phq), (raw_k, k1, phk)):
            a_, b_ = raw[:, PHEAD][..., prm.LO:ar.PAIRS], raw[:, PHEAD][..., ar.PAIRS + prm.LO:]
            want = torch.cat([a_ * ph.cos() - b_ * ph.sin(), a_ * ph.sin() + b_ * ph.cos()], dim=-1)
            assert torch.allclose(got[:, PHEAD][..., dims_hi], want, atol=1e-5)
        st.clear()


def test_the_draw_is_seeded_and_takes_nothing_from_the_runs_generator(space_table):
    torch.manual_seed(1)
    a = prm.phase_table()
    after_a = torch.rand(3)
    torch.manual_seed(7)
    torch.rand(5)
    b = prm.phase_table()
    assert torch.equal(a, b) and tuple(a.shape) == (prm.PAIRS, prm.K, prm.VOCAB)
    torch.manual_seed(1)
    assert torch.equal(torch.rand(3), after_a)
    assert prm.seed_tag() == "PATHROT|v1" and prm.seed_tag(12) == "PATHROT|v1|k12"
    g = torch.Generator().manual_seed(__import__("zlib").crc32(b"PATHROT|v1"))
    assert torch.equal(a, torch.rand(8, 16, 256, generator=g, dtype=torch.float64).mul_(2 * math.pi).float())
    pr = prm.PathRotary(_units(space_table), "next")
    assert torch.equal(pr.table, a) and not pr.table.requires_grad
    assert not list(pr.parameters()) and not pr.state_dict()                                 # buffers only, none persistent
    assert float(a.min()) >= 0.0 and float(a.max()) < prm.TWO_PI and abs(float(a.mean()) - math.pi) < 0.05


def test_the_patch_keeps_the_state_dict(space_table):
    m, st = _craft(n_ctx=256)
    before = {k: v.clone() for k, v in m.state_dict().items()}
    n_par = sum(p.numel() for p in m.parameters())
    prm.patch(m, prm.PathRotary(_units(space_table), "same"))
    after = m.state_dict()
    assert before.keys() == after.keys() and all(torch.equal(before[k], after[k]) for k in before)
    assert sum(p.numel() for p in m.parameters()) == n_par
    from geolip.alephllm.model.alephlm import AlephLM
    from geolip.nano_beatrix.bodies import craft_cfg
    plain = AlephLM(craft_cfg("alhena-both", 256)).eval()                                   # no anchored rotation: refused
    with pytest.raises(AssertionError):
        prm.patch(plain, prm.PathRotary(_units(space_table), "same"))
    with pytest.raises(AssertionError):
        prm.PathRotary(_units(space_table), "both")


def test_the_repeat_mask(space_table):
    u = _units(space_table)
    row = b"go the cat sat on the cat"
    x = torch.tensor(list(row), dtype=torch.long)[None]
    rep = prm.repeat_mask(u, x)[0]
    d = u.depth(x)[0].tolist()
    seen, want = set(), []
    for t in range(len(row)):
        key = row[t - d[t] + 1:t + 1]
        want.append(key in seen)
        seen.add(key)
    assert rep.tolist() == want
    second = row.rindex(b"the cat")
    assert all(rep[second:second + 7])                                                        # "the", " ", "cat" again: every node repeats
    assert not rep[0] and not rep[row.index(b"sat")]
    sp = prm.RepeatSplit()
    nll = np.arange(len(row), dtype=np.float64)[None]
    sp.add(nll, rep[None])
    o = sp.out()
    assert o["n_repeat"] == int(rep.sum()) and o["n_repeat"] + o["n_other"] == len(row)
    assert abs(o["repeat"] - nll[0][rep].mean() / math.log(2)) < 1e-12 and abs(o["other"] - nll[0][~rep].mean() / math.log(2)) < 1e-12


def _save_path_core(m, d, pr, va):
    from safetensors.torch import save_file
    sj = _save_core(m, d, va)
    save_file({"table": pr.table.detach().cpu().contiguous()}, str(d / "path_rot.safetensors"))
    sj["path_rot"] = prm.scale_field(pr)
    json.dump(sj, open(d / "scale.json", "w", encoding="utf-8"), indent=1)
    return sj


@pytest.mark.parametrize("form", prm.FORMS)
def test_the_side_file_round_trips_through_the_core_and_its_text_read(tmp_path, monkeypatch, space_table, form):
    monkeypatch.setenv(prm.TABLE_ENV, space_table)                                            # the core reads the made table, pinned
    monkeypatch.setattr(prm, "TABLE_SHA256", hashlib.sha256(open(space_table, "rb").read()).hexdigest())
    m, st = _craft(n_ctx=gr.CTX, seed=2)
    va = ar.ValueAddress(m.cfg.d_model, st)
    with torch.no_grad():
        va.table.normal_(std=0.5)
    ar.patch_value(m, va)
    pr = prm.PathRotary(prm.Units(), form, head=PHEAD)
    prm.patch(m, pr)
    d = tmp_path / "core"
    sj = _save_path_core(m, d, pr, va)
    core = rs.Core(str(d), "cpu", look=None)
    assert core.path_rot is not None and core.path_rot.form == form and torch.equal(core.path_rot.table, pr.table)
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
    with torch.no_grad():
        core.path_rot.table[0, 0, 99] += 0.5
    assert core.state_digest() != dg                                                        # a moved phase is a moved core
    # the text read: the held-out rows' bits, the repeat split and the path off, against a direct computation
    held = tmp_path / "held.jsonl"
    g = np.random.default_rng(3)
    words = [b"the", b"cat", b"sat", b"on", b"a", b"mat", b"and", b"ran"]
    with open(held, "w", encoding="utf-8") as f:
        for _ in range(3):
            f.write(json.dumps({"text": b" ".join(words[int(i)] for i in g.integers(0, len(words), size=1500)).decode()}) + "\n")
    rec = rs.text_read(str(d), str(tmp_path / "text.json"), str(held), n_docs=2, mb=2, dev="cpu", log=lambda s: None)
    core2 = rs.Core(str(d), "cpu", look=None)
    docs = rs.load_heldout(str(held))[:2]
    xs = torch.tensor(np.stack([np.frombuffer(bytes([rs.DOC]) + dd[:gr.CTX], dtype=np.uint8) for dd in docs]).astype(np.int64))
    with torch.no_grad():
        lg = core2.model(xs[:, :-1])[0].float()
        nll = torch.nn.functional.cross_entropy(lg.reshape(-1, 256), xs[:, 1:].reshape(-1), reduction="none").view(2, -1).numpy()
        core2.path_rot.off = True
        lg_off = core2.model(xs[:, :-1])[0].float()
        core2.path_rot.off = False
    rep = prm.repeat_mask(core2.path_rot.units, xs[:, :-1])
    assert abs(rec["bits"] - float(nll.sum()) / nll.size / math.log(2)) < 1e-9
    assert abs(rec["by_repeat"]["repeat"] - float(nll[rep].mean()) / math.log(2)) < 1e-6 and rec["by_repeat"]["n_repeat"] == int(rep.sum())
    assert rec["by_repeat"]["repeat_share"] > 0.5                                             # eight words: most paths repeat
    nll_off = torch.nn.functional.cross_entropy(lg_off.reshape(-1, 256), xs[:, 1:].reshape(-1), reduction="none").numpy()
    assert abs(rec["path_off"]["bits"] - float(nll_off.sum()) / nll_off.size / math.log(2)) < 1e-9
    assert rec["path"] == dict(form=form, head=PHEAD, k=prm.K) and rec["path_off"]["bits"] != rec["bits"]
    assert rs.text_read(str(d), str(tmp_path / "text.json"), str(held), n_docs=2, dev="cpu", log=lambda s: None) == rec   # resume-first
    os.remove(d / "path_rot.safetensors")
    with pytest.raises(AssertionError):
        rs.Core(str(d), "cpu")
    del sj["path_rot"]
    json.dump(sj, open(d / "scale.json", "w", encoding="utf-8"), indent=1)
    plain = rs.Core(str(d), "cpu")
    assert plain.path_rot is None and plain.state_digest() == rs._digest(plain.model, plain.value_addr.table)   # 0.1.91's digest
    rec2 = rs.text_read(str(d), str(tmp_path / "text_plain.json"), str(held), n_docs=2, dev="cpu", log=lambda s: None)
    assert rec2["path"] is None and "path_off" not in rec2 and rec2["by_repeat"]["n_repeat"] == rec["by_repeat"]["n_repeat"]


def test_a_stored_field_is_checked(space_table):
    pr = prm.PathRotary(_units(space_table), "next")
    f = prm.scale_field(pr)
    prm.check_field(f)
    assert f["seed_tag"] == "PATHROT|v1" and f["lo"] == 8 and f["pairs"] == 8 and f["theta"] == 3.0 and f["floor"] == 8.0
    for bad in (dict(f, seed_tag="PATHROT|v2"), dict(f, theta=2.5), dict(f, table_sha256="0" * 64), dict(f, form="both")):
        with pytest.raises(AssertionError):
            prm.check_field(bad)
    r = prm.record(pr, 6)
    assert r["blocks"] == 6 and r["digest"] == prm.table_digest(prm.phase_table())


def test_the_names_and_the_runner():
    pk = dict(W_BASE, grid_hard="0.5", value_addr="1", ptr_lam="0.1", grid_comps="48", key_rot="1", key_pairs="16", key_norm="2",
              key_rho="2.18", plant="0.75", plain_doc="1")
    stem = STEM.format(1).replace("_askany_", "_askany_p75d_") + "gc48_kh50_va1_pt10b4h0_krt16r218_"
    assert run_tag(dict(pk, series="plain16")) == stem + "ahdigit1_plain16"                 # off: 0.1.91's name
    assert run_tag(dict(pk, path_bind="8", path_form="next", series="pillar1")) == stem + "pbn8h7_ahdigit1_pillar1"
    assert run_tag(dict(pk, path_bind="8", path_form="same", series="pillar1")) == stem + "pbs8h7_ahdigit1_pillar1"
    assert run_tag(dict(pk, path_bind="8", path_form="same", path_head="6", path_k="12", series="x")) == stem + "pbs8h6k12_ahdigit1_x"
    assert run_tag(dict(pk, repeat_read="1", series="plain16")) == stem + "ahdigit1_plain16"   # the read trains nothing: no piece
    runs = br.expand(dict(arms="alhena-both", seeds="1", grid="ag", grid_p="0.5", head="digit", head_lam="1", rotary_content="4",
                          grid_hard="0.5", value_addr="1", ptr_lam="0.1", key_rot="1", grid_comps="48", key_pairs="16", key_norm="2",
                          key_rho="2.18", plant="0.75", plain_doc="1", path_bind="8", path_form="same,next", series="pillar1"))
    assert len(runs) == 2 and {[a for a in r if a.startswith("--path_form=")][0] for r in runs} == {"--path_form=same", "--path_form=next"}
    assert "the path rotation (same) on head 7" in br.run_label(runs[0]) and "the path rotation (next) on head 7" in br.run_label(runs[1])
    assert all(k in br.GRID_KEYS for k in ("path_bind", "path_form", "path_head", "path_k", "repeat_read"))
    assert "path rotation" not in br.run_label([a for a in runs[0] if not a.startswith("--path_")])
