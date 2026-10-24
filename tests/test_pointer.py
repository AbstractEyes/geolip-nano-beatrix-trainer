# -*- coding: utf-8 -*-
"""The pointer term (0.1.78): one head of one softmax block told, at each index entry's SEP, its mass on the frame of the record whose
key the entry carries: log m_e = logsumexp over the frame of q(i_e) . k(n) / sqrt(hd) - logsumexp over n <= i_e (pointer.log_mass), the
trainer's term -(1/|E|) sum_e log m_e. Off, the batches are 0.1.77's bit for bit; the mass is the head's causal softmax mass on the span
(a direct softmax agrees); the capture leaves the craft's forward and state dict untouched; the term's gradient reaches the block's q and
k and everything upstream, never the blocks after it, the block's own output projection, the final norm or the output layer; the frame
spans are the records' frames before the entries; the names and the runner carry the term. CPU only; the Hub is never touched."""
import ast
import math
import os

import numpy as np
import pytest
import torch

from geolip.alephllm.model.alephlm import AlephLM
from geolip.alephllm.data.special_tokens import DATA, END
from geolip.nano_beatrix import anchored_rope as ar
from geolip.nano_beatrix import bed_runner as br
from geolip.nano_beatrix import grid_rows as gr
from geolip.nano_beatrix import pointer as pt
from geolip.nano_beatrix.bodies import craft_cfg
from geolip.nano_beatrix.names import run_tag
from test_key_hard import FP_0174_BATCH, batch_digest
from test_posing import _row
from test_scale import _bed_batch
from test_value_addr import STEM, W_BASE, _streams


def test_off_the_batches_are_0177s_bit_for_bit():
    # the term reads the batch, never changes it: the batch the trainer builds is 0.1.77's (= 0.1.74's) to the last byte
    assert batch_digest(_bed_batch(4)) == FP_0174_BATCH
    src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "geolip", "nano_beatrix", "bed_train.py"), encoding="utf-8").read()
    fns = {n.name: ast.get_source_segment(src, n) for n in ast.parse(src).body if isinstance(n, ast.FunctionDef)}
    for fn in ("cap", "draw_plan", "frame_row", "ag_rows", "grid_aux", "batch"):            # the batch's whole path (test_scale._bed_batch)
        assert "PTR_" not in fns[fn] and "ptr" not in fns[fn]


def _direct(q, k, head, rows, pos, lo, hi):
    """The mass by the plain formula: the causal softmax row of the head at each entry's SEP, summed over its span."""
    out = []
    for r, p, a, b in zip(rows.tolist(), pos.tolist(), lo.tolist(), hi.tolist()):
        s = (q[r, head, p].double() @ k[r, head, :p + 1].double().T) / math.sqrt(q.shape[-1])
        out.append(float(torch.softmax(s, -1)[a:b].sum()))
    return torch.tensor(out, dtype=torch.float64)


def test_the_mass_is_the_heads_causal_softmax_mass_on_the_span():
    g = torch.Generator().manual_seed(0)
    B, H, n, hd = 3, 4, 90, 8
    q = torch.randn(B, H, n, hd, generator=g) * 2
    k = torch.randn(B, H, n, hd, generator=g) * 2
    rows = torch.tensor([0, 0, 0, 2, 2])                                                  # grouped by row; row 1 has no entry
    pos = torch.tensor([60, 75, 89, 40, 88])
    lo = torch.tensor([3, 10, 50, 0, 70])
    hi = torch.tensor([9, 30, 89, 1, 88])
    for head in range(H):
        got = pt.log_mass(q, k, head, rows, pos, lo, hi).double().exp()
        assert torch.allclose(got, _direct(q, k, head, rows, pos, lo, hi), rtol=1e-5, atol=1e-7)
    # the span covering the whole causal row is mass 1; a later position never counts (the causal row)
    full = pt.log_mass(q, k, 1, torch.tensor([1]), torch.tensor([50]), torch.tensor([0]), torch.tensor([50]))
    assert abs(float(full.exp()) - float(_direct(q, k, 1, torch.tensor([1]), torch.tensor([50]), torch.tensor([0]), torch.tensor([50]))[0])) < 1e-6
    # malformed entries are refused: rows out of order, a span after its query, an empty span
    with pytest.raises(AssertionError):
        pt.log_mass(q, k, 0, torch.tensor([2, 0]), torch.tensor([60, 60]), torch.tensor([1, 1]), torch.tensor([5, 5]))
    with pytest.raises(AssertionError):
        pt.log_mass(q, k, 0, torch.tensor([0]), torch.tensor([20]), torch.tensor([10]), torch.tensor([30]))
    with pytest.raises(AssertionError):
        pt.log_mass(q, k, 0, torch.tensor([0]), torch.tensor([20]), torch.tensor([5]), torch.tensor([5]))


def test_the_gate_reads_zero_on_the_ces_own_gradient():
    # the control that must fail: a term whose gradient is the CE's own (or its negative) reads novelty 0 (the trainer refuses under .05);
    # an orthogonal one reads 1
    g = torch.Generator().manual_seed(1)
    v = torch.randn(1000, generator=g, dtype=torch.float64)
    w = torch.randn(1000, generator=g, dtype=torch.float64)
    w = w - (w @ v) / (v @ v) * v
    assert pt.novelty(v, v)[0] < 1e-12 and pt.novelty(v, -3 * v)[0] < 1e-12 and pt.novelty(v, -3 * v)[1] < -0.999
    assert abs(pt.novelty(v, w)[0] - 1.0) < 1e-12 and pt.novelty(v, v + 0.01 * w)[0] < 0.05


def _craft(n_ctx=256, seed=1):
    torch.manual_seed(seed)
    m = AlephLM(craft_cfg("alhena-both", n_ctx))
    st = ar.AnchorStreams(4)
    ar.patch(m, st)
    return m, st


def test_the_capture_leaves_the_forward_and_the_state_dict_untouched():
    m, st = _craft()
    m.eval()
    x = torch.randint(0, 256, (2, 64))
    dk, dq = _streams(2, 64)
    st.set(dk, dq)
    before = {k_: v.clone() for k_, v in m.state_dict().items()}
    with torch.no_grad():
        plain = m(x).logits.clone()
    cap = pt.HeadCapture(m.blocks[3].attn)
    with torch.no_grad():
        got = m(x).logits
    assert torch.equal(plain, got)                                                       # fp32, exactly
    assert cap.q is not None and tuple(cap.q.shape) == (2, m.blocks[3].attn.h, 64, 32) and tuple(cap.k.shape) == tuple(cap.q.shape)
    after = m.state_dict()
    assert before.keys() == after.keys() and all(torch.equal(before[k_], after[k_]) for k_ in before)
    cap.clear()
    cap.on = False
    with torch.no_grad():
        m(x)
    assert cap.q is None and cap.k is None                                               # off: nothing kept
    st.clear()


def test_the_terms_gradient_reaches_upstream_only():
    n = 1024                                                                              # a prefix of the row: the head slot and the first frames
    m, st = _craft(n_ctx=n)
    m.train()
    row, meta = _row(seed=3)
    x = torch.tensor(list(row[:n]), dtype=torch.long)[None]
    dk, dq = gr.streams(meta)
    st.set(torch.from_numpy(dk[:n].astype(np.int64))[None], torch.from_numpy(dq[:n].astype(np.int64))[None])
    cap = pt.HeadCapture(m.blocks[3].attn)
    m(x)
    st.clear()
    # the index sits at 3,200+ (outside the prefix): the frames serve as spans before chosen query positions
    rows = torch.tensor([0, 0, 0])
    pos = torch.tensor([meta["recs"][5]["lo"] + 60, meta["recs"][6]["lo"] + 3, n - 1])
    lo = torch.tensor([meta["recs"][1]["lo"], meta["recs"][2]["lo"], meta["recs"][3]["lo"]])
    hi = torch.tensor([meta["recs"][1]["frame_end"], meta["recs"][2]["frame_end"], meta["recs"][3]["frame_end"]])
    term = -pt.log_mass(cap.q, cap.k, 0, rows, pos, lo, hi).mean()
    names, params = zip(*[(nm, p) for nm, p in m.named_parameters() if p.requires_grad])
    grads = dict(zip(names, torch.autograd.grad(term, params, allow_unused=True)))
    nz = {nm for nm, gq in grads.items() if gq is not None and float(gq.abs().max()) > 0}
    for nm in names:
        after = nm.startswith(("blocks.4.", "blocks.5.", "nf.", "head.")) or nm.startswith(("blocks.3.attn.o.", "blocks.3.n2.", "blocks.3.bank."))
        if after:
            assert nm not in nz, f"the term reached {nm}, after the supervised head's q and k"
    assert "blocks.3.attn.qkv.weight" in nz and "blocks.3.n1.weight" in nz                # the block's own q and k
    assert any(nm.startswith("blocks.0.") for nm in nz) and any(nm.startswith("embed.") for nm in nz)   # and upstream


def test_the_spans_are_the_records_frames_before_the_entries():
    for seed in range(6):
        for ab in (2, 1):
            row, meta = _row(seed=seed, abytes=ab)
            pos, _ = gr.index_head_targets(meta)
            spans = gr.index_frame_spans(meta)
            assert len(spans) == len(pos) == len(meta["index"]) == 24
            for e, (a, b), p in zip(meta["index"], spans, pos):
                r = meta["recs"][e["rec"]]
                assert (a, b) == (r["lo"], r["frame_end"]) and b <= gr.INDEX_LO < p
                assert row[a] == DATA and row[b - 1] == END
                assert row[a + 1 + ab + 1:a + 1 + ab + 1 + len(r["key"])] == r["key"].encode()   # the frame holds the entry's key


def test_the_pointer_runs_never_share_a_name():
    p10 = run_tag(dict(W_BASE, value_addr="1", ptr_lam="0.1", series="pointer"))
    p30 = run_tag(dict(W_BASE, value_addr="1", ptr_lam="0.3", series="pointer"))
    v7 = run_tag(dict(W_BASE, seed="7", value_addr="1", series="valuerep"))
    assert p10 == STEM.format(1) + "va1_pt10b4h0_ahdigit1_pointer"
    assert p30 == STEM.format(1) + "va1_pt30b4h0_ahdigit1_pointer"
    assert v7 == STEM.format(7) + "va1_ahdigit1_valuerep"
    assert run_tag(dict(W_BASE, value_addr="1", ptr_lam="0", series="value")) == run_tag(dict(W_BASE, value_addr="1", series="value"))
    assert run_tag(dict(W_BASE, value_addr="1", ptr_lam="0.1", ptr_block="2", ptr_head="5", series="x")).endswith("_va1_pt10b2h5_ahdigit1_x")
    assert len({p10, p30, v7}) == 3


def test_the_runner_passes_the_term_and_labels_it():
    runs = br.expand(dict(arms="alhena-both", seeds="1", grid="ag", grid_p="0.5", head="digit", head_lam="1", rotary_content="4",
                          value_addr="1", ptr_lam="0.1", ptr_block="4", ptr_head="0", series="pointer"))
    assert len(runs) == 1 and "--ptr_lam=0.1" in runs[0] and "--ptr_block=4" in runs[0] and "--ptr_head=0" in runs[0]
    assert "pointer term 0.1 at block 4 head 0" in br.run_label(runs[0]) and "value-path address" in br.run_label(runs[0])
    rep = br.expand(dict(arms="alhena-both", seeds="7", grid="ag", grid_p="0.5", head="digit", head_lam="1", rotary_content="4",
                         value_addr="1", series="valuerep"))
    assert "--ptr_lam" not in " ".join(rep[0]) and "pointer" not in br.run_label(rep[0])
    assert {"ptr_lam", "ptr_block", "ptr_head"} <= set(br.GRID_KEYS)
