# -*- coding: utf-8 -*-
"""The binding's two forms (0.1.87): sixteen bound pairs (all of the told head's pairs turned by the key's phases at the coded SEPs, the
phases 16 x K x 256 seeded crc32("KEYROT|v1|j16")) and the key's norm (the told head's bound key pairs held at length rho at the coded SEPs
before the turn). The defaults are 0.1.86's draw, seed tags, record field and dims; the sixteen-pair draw is seeded and takes nothing from
the run's generators; only the told head at the coded SEPs moves (every pair there at sixteen, the anchor pairs elsewhere and every other
head value for value; a text row is the off forward exactly); the norm puts every bound key pair at rho there and leaves the query, the
other positions, pairs and heads value for value; the reading core rebuilds either form from scale.json; the capacity read at sixteen pairs
is its sums written out by hand; the names and the runner carry the forms. CPU only; the Hub is never touched."""
import json
import math
import os
import subprocess
import sys

import numpy as np
import pytest
import torch

from geolip.alephllm.model.attention import rope_half
from geolip.nano_beatrix import anchored_rope as ar
from geolip.nano_beatrix import bed_runner as br
from geolip.nano_beatrix import capacity as cp
from geolip.nano_beatrix import grid_rows as gr
from geolip.nano_beatrix import key_code as kc
from geolip.nano_beatrix import key_rotary as kr
from geolip.nano_beatrix import pointer as pt
from geolip.nano_beatrix import rotary_scale as rs
from geolip.nano_beatrix.names import run_tag
from test_capacity import SRC
from test_key_code import _save_core
from test_key_rot import BLOCK, HEAD, _rows, _set_streams
from test_value_addr import STEM, W_BASE, _craft

P9_DIGEST = "e03ed849a52a270be011f9fa0dead7485b549901ffadbe02210ee03fe7789212"      # 0.1.86's P_9 (crc32("KEYROT|v1")), read at v0.1.86
P12_DIGEST = "c32772b78a8ec2879016d2bda6dc1c76be369a4d3d36f7e9a2fc6eeae47d2bc1"     # 0.1.86's P_12 (crc32("KEYROT|v1|k12"))


def test_the_defaults_are_0186s():
    assert kr.table_digest(kr.phase_table()) == P9_DIGEST and kr.table_digest(kr.phase_table(12)) == P12_DIGEST
    assert kr.seed_tag() == "KEYROT|v1" and kr.seed_tag(12) == "KEYROT|v1|k12"
    assert kr.seed_tag(9, 16) == "KEYROT|v1|j16" and kr.seed_tag(12, 16) == "KEYROT|v1|j16|k12"
    rot = kr.KeyRotary(BLOCK, HEAD)
    assert rot.pairs == kr.J == 8 and not rot.norm
    rec = kr.record(rot)
    assert list(rec) == ["form", "pairs", "k", "vocab", "block", "head", "seed_tag", "digest"] and rec["pairs"] == 8
    assert kr.key_dims() == kr.KEY_DIMS == list(range(8)) + list(range(16, 24))
    assert kr.ANCHOR_DIMS == list(range(8, 16)) + list(range(24, 32))
    assert kr.key_dims(16) == list(range(32))
    with pytest.raises(AssertionError):
        kr.KeyRotary(BLOCK, HEAD, pairs=12)


def test_the_sixteen_pair_draw_is_seeded_and_takes_nothing_from_the_runs_generator():
    torch.manual_seed(1)
    a = kr.phase_table(9, 16)
    after = torch.rand(3)
    torch.manual_seed(1)
    torch.rand(0)
    assert torch.equal(torch.rand(3), after)
    assert tuple(a.shape) == (16, kc.K, kc.VOCAB) and torch.equal(a, kr.phase_table(9, 16))
    assert not torch.equal(a[:8], kr.phase_table())                                           # a draw of its own
    assert float(a.min()) >= 0.0 and float(a.max()) < kr.TWO_PI and abs(float(a.mean()) - math.pi) < 0.05
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(kr.__file__))))   # this tree, first in the child (no editable
    code = ("import sys; sys.meta_path[:] = [f for f in sys.meta_path if 'editable' not in type(f).__name__.lower()]; "  # finder)
            f"sys.path.insert(0, {root!r}); from geolip.nano_beatrix import key_rotary as kr; print(kr.__file__); "
            "print(kr.table_digest(kr.phase_table(9, 16)))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.strip().splitlines()
    assert os.path.normcase(os.path.abspath(out[-2])) == os.path.normcase(os.path.abspath(kr.__file__)), out
    assert out[-1] == kr.table_digest(a)                                                      # the same draw in another process
    rot = kr.KeyRotary(BLOCK, HEAD, pairs=16)
    assert torch.equal(rot.table, a) and not list(rot.parameters()) and "table" not in rot.state_dict()
    rec = kr.record(rot)
    assert rec["pairs"] == 16 and rec["seed_tag"] == "KEYROT|v1|j16" and "norm" not in rec


def _forward(pairs=8, norm=False, seed=4):
    """A crafted craft with the key rotation in the given form and the term's capture after it, run on two grid rows."""
    x, metas = _rows()
    m, st = _craft(n_ctx=gr.CTX, seed=seed)
    rot = kr.KeyRotary(BLOCK, HEAD, pairs=pairs, norm=norm)
    kr.patch(m, rot)
    cap = pt.HeadCapture(m.blocks[BLOCK - 1].attn, head=HEAD)
    _set_streams(st, metas)
    with torch.no_grad():
        lg = m(x).logits
    q, k, _ = cap.qk()
    st.clear()
    return m, st, rot, q, k, lg


def test_sixteen_pairs_turn_every_pair_of_the_told_head_at_the_coded_seps_only():
    m8, st8, rot8, q8, k8, lg8 = _forward(8)
    m16, st16, rot16, q16, k16, lg16 = _forward(16)
    assert torch.equal(rot8.on, rot16.on) and int(rot16.on.sum()) > 0
    B, H, n, hd = q16.shape
    others = [h for h in range(H) if h != HEAD]
    assert torch.equal(q8[:, others], q16[:, others]) and torch.equal(k8[:, others], k16[:, others])   # every other head, bitwise
    off = ~rot16.on
    assert torch.equal(q8[:, HEAD][off], q16[:, HEAD][off]) and torch.equal(k8[:, HEAD][off], k16[:, HEAD][off])
    m0, st0 = _craft(n_ctx=gr.CTX, seed=4)                                       # the anchored craft without the key rotation: the raw q, k
    cap0 = pt.HeadCapture(m0.blocks[BLOCK - 1].attn, head=HEAD)
    x, metas = _rows()
    _set_streams(st0, metas)
    with torch.no_grad():
        m0(x)
    q0, k0, _ = cap0.qk()
    a0q, a0k = st0.angles(B, n, 0, torch.device("cpu"))
    raw_q, raw_k = rope_half(q0, a0q.cos(), -a0q.sin()), rope_half(k0, a0k.cos(), -a0k.sin())
    st0.clear()
    rot16.set(x)
    _set_streams(st16, metas)
    aq, ak = st16.angles(B, n, 0, torch.device("cpu"))
    st16.clear()
    aq = aq.expand(B, H, n, aq.shape[-1]).clone()
    ak = ak.expand(B, H, n, ak.shape[-1]).clone()
    on = rot16.on[..., None]
    aq[:, HEAD] = torch.where(on, rot16.phi, aq[:, HEAD])                         # all sixteen pairs at the coded SEPs
    ak[:, HEAD] = torch.where(on, rot16.phi, ak[:, HEAD])
    assert torch.allclose(rope_half(raw_q, aq.cos(), aq.sin()), q16, atol=1e-5)
    assert torch.allclose(rope_half(raw_k, ak.cos(), ak.sin()), k16, atol=1e-5)
    anc = kr.ANCHOR_DIMS                                                         # the anchor pairs moved at the coded SEPs, nowhere else
    assert not torch.equal(q8[:, HEAD][rot16.on][..., anc], q16[:, HEAD][rot16.on][..., anc])
    t = torch.randint(32, 127, (2, 300))                                         # text: no coded SEP - the off forward exactly
    with torch.no_grad():
        assert torch.equal(m8(t).logits, m16(t).logits)


def test_the_norm_holds_the_key_pairs_at_rho_and_nothing_else_moves():
    m8, _, rot8, q8, k8, lg8 = _forward(8)
    mn, _, rotn, qn, kn, lgn = _forward(8, norm=True)
    assert rotn.norm and rotn.rho == kr.RHO == 4.16 and torch.equal(rot8.on, rotn.on)
    assert torch.equal(q8, qn)                                                   # the query untouched everywhere
    H = q8.shape[1]
    others = [h for h in range(H) if h != HEAD]
    assert torch.equal(k8[:, others], kn[:, others])
    on = rotn.on
    assert torch.equal(k8[:, HEAD][~on], kn[:, HEAD][~on])                       # the told head away from the coded SEPs
    anc = kr.ANCHOR_DIMS
    assert torch.equal(k8[:, HEAD][on][..., anc], kn[:, HEAD][on][..., anc])     # the anchor pairs at the coded SEPs
    kd = kr.KEY_DIMS
    lens = cp.pair_norms(kn[:, HEAD][on][..., kd].float())                      # the turn keeps lengths: every bound pair at rho
    assert torch.allclose(lens, torch.full_like(lens, kr.RHO), atol=1e-4)
    ang8 = torch.atan2(k8[:, HEAD][on][..., 16:24], k8[:, HEAD][on][..., :8])
    angn = torch.atan2(kn[:, HEAD][on][..., 16:24], kn[:, HEAD][on][..., :8])
    assert torch.allclose(torch.remainder(ang8 - angn + math.pi, 2 * math.pi) - math.pi, torch.zeros_like(ang8), atol=1e-4)   # angles kept
    assert not torch.equal(lg8, lgn)
    t = torch.randint(32, 127, (2, 300))
    with torch.no_grad():
        assert torch.equal(m8(t).logits, mn(t).logits)                           # text: the off forward exactly
    k = torch.randn(2, 8, 5, 32, requires_grad=True)                             # the norm keeps the gradient path (no detach)
    onm = torch.tensor([[False, True, False, True, False]] * 2)
    out = kr.norm_key(k, HEAD, onm, 2.0)
    out.sum().backward()
    assert k.grad is not None and float(k.grad.abs().sum()) > 0


def _saved(tmp_path, pairs=8, norm=False):
    m, st = _craft(n_ctx=gr.CTX, seed=2)
    va = ar.ValueAddress(m.cfg.d_model, st)
    with torch.no_grad():
        va.table.normal_(std=0.5)
    ar.patch_value(m, va)
    rot = kr.KeyRotary(BLOCK, HEAD, pairs=pairs, norm=norm)
    kr.patch(m, rot)
    d = tmp_path / f"core_{pairs}_{int(norm)}"
    sj = _save_core(m, d, va)
    from safetensors.torch import save_file
    save_file({"table": rot.table.detach().cpu().contiguous()}, str(d / "key_rot.safetensors"))
    sj["key_rot"] = dict(pairs=pairs, k=kc.K, vocab=kc.VOCAB, block=BLOCK, head=HEAD, seed_tag=kr.seed_tag(kc.K, pairs))
    if norm:
        sj["key_rot"].update(norm=1, rho=rot.rho)
    sj["ptr"] = dict(lam=0.1, block=BLOCK, head=HEAD)
    json.dump(sj, open(d / "scale.json", "w", encoding="utf-8"), indent=1)
    return m, st, rot, d


@pytest.mark.parametrize("pairs,norm", [(16, False), (8, True)])
def test_the_reading_core_rebuilds_the_form(tmp_path, pairs, norm):
    m, st, rot, d = _saved(tmp_path, pairs, norm)
    core = rs.Core(str(d), "cpu")
    assert core.key_rot.pairs == pairs and core.key_rot.norm == norm and torch.equal(core.key_rot.table, rot.table)
    x, metas = _rows((4,))
    core.set_streams(metas)
    with torch.no_grad():
        got = core.readout(core.trunk(x))
        h3 = core.trunk(x, 0, 3)
        split = core.readout(core.trunk(x, 3, None, h3))
    _set_streams(st, metas)
    with torch.no_grad():
        exp = m(x).logits
    st.clear()
    core.clear()
    assert torch.equal(got, exp) and torch.equal(split, exp)                     # the reader's forward = the trainer's
    if pairs == 16:                                                              # the eight-pair tag on a sixteen-pair table is refused
        sj = json.load(open(d / "scale.json", encoding="utf-8"))
        sj["key_rot"]["seed_tag"] = kr.SEED_TAG
        json.dump(sj, open(d / "scale.json", "w", encoding="utf-8"), indent=1)
        with pytest.raises(AssertionError):
            rs.Core(str(d), "cpu")


def test_the_capacity_read_at_sixteen_pairs_is_its_sums(tmp_path):
    m, st, rot, d = _saved(tmp_path, 16)
    core = rs.Core(str(d), "cpu")
    held = [SRC[1:]]
    rows = cp.read_rows(core, held, 2)
    cap = cp.capture(core, rows, batch=2, log=None)
    assert cap["q"].shape[-1] == 32 and cap["pool_phi"].shape[-1] == 16
    sc = 1.0 / math.sqrt(32)
    st_ = (cap["q"] * cap["t"]).sum(-1) * sc
    k0 = cp.turn(cap["pool_k"], -cap["pool_phi"])                                # unrotated by its own sixteen phases
    back = cp.turn(k0, cap["pool_phi"])
    assert torch.allclose(back, cap["pool_k"], atol=1e-4)
    bank = k0[:50]
    W, M = cp.emulated(cap["q"], st_, bank, (10, 50))
    d_ = (cap["q"] @ bank.T) * sc - st_[:, None]                                 # by hand: every one of the sixteen pairs
    assert np.allclose(W[:, 1], torch.exp(d_.double()).sum(1).numpy(), rtol=1e-6)
    assert np.allclose(M[:, 0], d_[:, :10].max(1).values.numpy(), atol=1e-6)
    assert cp.moment(7.5, 10 ** 6, 16) < 1e-7 < cp.moment(7.5, 10 ** 6) and cp.moment(7.5, 10 ** 6) == cp.moment(7.5, 10 ** 6, 8)
    rec = cp.tail_core(str(d), held, R=2, ns=(10, 50), log=None)
    assert rec["told"]["pairs"] == 16 and rec["told"]["norm"] == 0 and len(rec["true_pair_length"]["p50"]) == 16
    base = cp.read_core(str(d), held, R=2, ns=(10, 50), draws=1, bank_n=50, log=None)
    for n in ("10", "50"):                                                       # the tail's base = the read's EMULATED, bit for bit
        for f in ("mean", "median", "look_error", "argmax_error", "moment"):
            assert rec["base"][n][f] == base["emulated"][n][f], (n, f)
    assert abs(base["s_core"] * 16 - float((st_ - (cap["q"][:, None, :] * cap["inrow"]).sum(-1).mul(sc).mean(1)).mean())) < 1e-4


def test_the_eight_pair_record_names_nothing_new(tmp_path):
    m, st, rot, d = _saved(tmp_path, 8)
    rec = cp.tail_core(str(d), [SRC[1:]], R=2, ns=(10, 50), log=None)
    assert rec["told"] == dict(block=BLOCK, head=HEAD)                           # 0.1.86's field at the defaults
    _, _, _, dn = _saved(tmp_path, 8, True)
    recn = cp.tail_core(str(dn), [SRC[1:]], R=2, ns=(10, 50), log=None)
    assert recn["told"]["norm"] == 1 and recn["told"]["rho"] == kr.RHO
    lens = torch.tensor(recn["true_pair_length"]["p50"])
    assert torch.allclose(lens, torch.full_like(lens, kr.RHO), atol=1e-3)        # the captured keys carry the norm


def test_the_names_and_the_runner():
    pk = dict(W_BASE, grid_hard="0.5", value_addr="1", ptr_lam="0.1", grid_comps="48", key_rot="1", series="norm16")
    lrk = run_tag(dict(pk, series="largerows"))
    l16 = run_tag(dict(pk, key_pairs="16"))
    lrn = run_tag(dict(pk, key_norm="1"))
    assert lrk == STEM.format(1) + "gc48_kh50_va1_pt10b4h0_krt_ahdigit1_largerows"
    assert l16 == STEM.format(1) + "gc48_kh50_va1_pt10b4h0_krt16_ahdigit1_norm16"
    assert lrn == STEM.format(1) + "gc48_kh50_va1_pt10b4h0_krtn_ahdigit1_norm16"
    assert run_tag(dict(pk, key_pairs="8", key_norm="0")) == run_tag(pk)        # the defaults name nothing
    assert run_tag(dict(pk, key_pairs="16", key_norm="1")).endswith("_krt16n_ahdigit1_norm16")
    runs = br.expand(dict(arms="alhena-both", seeds="1,2", grid="ag", grid_p="0.5", head="digit", head_lam="1", rotary_content="4",
                          grid_hard="0.5", value_addr="1", ptr_lam="0.1", key_rot="1", grid_comps="48", key_pairs="16", series="norm16"))
    assert len(runs) == 2 and all("--key_pairs=16" in r for r in runs) and "over 16 pairs" in br.run_label(runs[0])
    runs = br.expand(dict(arms="alhena-both", seeds="1", grid="ag", grid_p="0.5", head="digit", head_lam="1", rotary_content="4",
                          grid_hard="0.5", value_addr="1", ptr_lam="0.1", key_rot="1", grid_comps="48", key_norm="1", series="norm16"))
    assert "--key_norm=1" in runs[0] and "with the key's norm" in br.run_label(runs[0])
    assert "key_pairs" in br.GRID_KEYS and "key_norm" in br.GRID_KEYS
