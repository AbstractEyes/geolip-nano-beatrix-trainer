# -*- coding: utf-8 -*-
"""The local two-position pointer term (0.1.79): --ptr_pos=both asks the supervised head's mass on the record's frame at each index entry's
SEP and at its first address byte (where the second address digit is written), the term the mean over both; --ptr_stop=1 forms the term's
q and k from the block's attention input detached (pointer.HeadCapture(stop=True)), so the term's gradient reaches the supervised head's
query and key rows of the block's qkv projection and nothing else. The batches stay 0.1.74's bit for bit; the stopped capture leaves the
forward and the state dict untouched and its q and k are the forward's own; the second positions are the entries' first address bytes;
the names and the runner carry the form. CPU only; the Hub is never touched."""
import os
import ast

import numpy as np
import torch

from geolip.nano_beatrix import anchored_rope as ar
from geolip.nano_beatrix import bed_runner as br
from geolip.nano_beatrix import grid_rows as gr
from geolip.nano_beatrix import pointer as pt
from geolip.nano_beatrix.names import run_tag
from test_key_hard import FP_0174_BATCH, batch_digest
from test_pointer import _craft
from test_posing import _row
from test_scale import _bed_batch
from test_value_addr import STEM, W_BASE, _streams


def test_the_batches_are_still_0174s_bit_for_bit():
    assert batch_digest(_bed_batch(4)) == FP_0174_BATCH
    src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "geolip", "nano_beatrix", "bed_train.py"), encoding="utf-8").read()
    fns = {n.name: ast.get_source_segment(src, n) for n in ast.parse(src).body if isinstance(n, ast.FunctionDef)}
    for fn in ("cap", "draw_plan", "frame_row", "ag_rows", "grid_aux", "batch"):
        assert "PTR_" not in fns[fn] and "ptr" not in fns[fn]


def test_the_stopped_capture_keeps_the_forward_and_its_q_and_k_are_the_forwards():
    m, st = _craft()
    m.eval()
    x = torch.randint(0, 256, (2, 64))
    dk, dq = _streams(2, 64)
    st.set(dk, dq)
    attn = m.blocks[3].attn
    before = {k_: v.clone() for k_, v in m.state_dict().items()}
    with torch.no_grad():
        plain = m(x).logits.clone()
    for head in (0, 5):
        ref = pt.HeadCapture(attn)                                       # the forward's own rotated q and k (every head)
        cap = pt.HeadCapture(attn, stop=True, head=head)                 # the block's input, kept
        with torch.no_grad():
            got = m(x).logits
        assert torch.equal(plain, got)                                   # fp32, exactly: both captures leave the forward
        q_ref, k_ref = ref.q[:, head:head + 1].clone(), ref.k[:, head:head + 1].clone()
        ref.on = False
        with torch.no_grad():
            q, k, h = cap.qk()
        assert h == 0 and tuple(q.shape) == (2, 1, 64, 32) and tuple(k.shape) == (2, 1, 64, 32)
        assert torch.allclose(q, q_ref, rtol=1e-5, atol=1e-5) and torch.allclose(k, k_ref, rtol=1e-5, atol=1e-5)
        for c_ in (cap, ref):
            c_.clear()
        del attn._split, attn._rotate                                    # the wrappers off: the patched rotation back
        ar.patch(m, st)
    after = m.state_dict()
    assert before.keys() == after.keys() and all(torch.equal(before[k_], after[k_]) for k_ in before)
    st.clear()


def test_the_stopped_terms_gradient_stays_in_the_heads_query_and_key_rows():
    n = 1024
    m, st = _craft(n_ctx=n)
    m.train()
    row, meta = _row(seed=3)
    x = torch.tensor(list(row[:n]), dtype=torch.long)[None]
    dk, dq = gr.streams(meta)
    st.set(torch.from_numpy(dk[:n].astype(np.int64))[None], torch.from_numpy(dq[:n].astype(np.int64))[None])
    head = 2
    cap = pt.HeadCapture(m.blocks[3].attn, stop=True, head=head)
    m(x)
    rows = torch.tensor([0, 0, 0])
    pos = torch.tensor([meta["recs"][5]["lo"] + 60, meta["recs"][6]["lo"] + 3, n - 1])
    lo = torch.tensor([meta["recs"][1]["lo"], meta["recs"][2]["lo"], meta["recs"][3]["lo"]])
    hi = torch.tensor([meta["recs"][1]["frame_end"], meta["recs"][2]["frame_end"], meta["recs"][3]["frame_end"]])
    q, k, h = cap.qk()
    term = -pt.log_mass(q, k, h, rows, pos, lo, hi).mean()
    st.clear()
    names, params = zip(*[(nm, p) for nm, p in m.named_parameters() if p.requires_grad])
    grads = dict(zip(names, torch.autograd.grad(term, params, allow_unused=True)))
    nz = {nm for nm, gq in grads.items() if gq is not None and float(gq.abs().max()) > 0}
    assert nz == {"blocks.3.attn.qkv.weight"}, nz                         # nothing below the block, nothing else in it
    gw = grads["blocks.3.attn.qkv.weight"]
    d = gw.shape[1]
    hd = d // m.blocks[3].attn.h
    rows_q, rows_k = slice(head * hd, (head + 1) * hd), slice(d + head * hd, d + (head + 1) * hd)
    live = torch.zeros(gw.shape[0], dtype=torch.bool)
    live[rows_q] = True
    live[rows_k] = True
    assert float(gw[live].abs().max()) > 0 and float(gw[~live].abs().max()) == 0.0   # the head's q and k rows alone
    assert float(gw[rows_q].abs().max()) > 0 and float(gw[rows_k].abs().max()) > 0


def test_both_positions_are_the_sep_and_the_first_address_byte_and_the_term_is_their_mean():
    g = torch.Generator().manual_seed(4)
    for seed in range(4):
        row, meta = _row(seed=seed)
        sep, _ = gr.index_head_targets(meta)
        a1 = [e["a_pos"] for e in meta["index"]]
        assert [p + 1 for p in sep] == a1                                 # the trainer's assert (bed_train.ptr_log_mass)
        for e in meta["index"]:
            addr = meta["recs"][e["shown"]]["addr"]                       # the address the entry writes (its record's, unswapped)
            assert e["shown"] == e["rec"]
            assert row[e["a_pos"] - 1] == gr.SEP and row[e["a_pos"]:e["a_pos"] + 2] == gr.addr_bytes(addr, 2)
    # the interleaved call (rows grouped, as the trainer forms it) = the two positions' masses; the term = the mean over 2|E|
    B, H, n, hd = 2, 3, 120, 8
    q = torch.randn(B, H, n, hd, generator=g) * 2
    k = torch.randn(B, H, n, hd, generator=g) * 2
    rows = torch.tensor([0, 0, 1, 1, 1])
    s = torch.tensor([40, 70, 50, 90, 118])
    lo = torch.tensor([2, 20, 0, 60, 100])
    hi = torch.tensor([10, 39, 30, 80, 110])
    both = pt.log_mass(q, k, 1, rows.repeat_interleave(2), torch.stack([s, s + 1], 1).reshape(-1), lo.repeat_interleave(2),
                       hi.repeat_interleave(2)).view(-1, 2)
    at_sep = pt.log_mass(q, k, 1, rows, s, lo, hi)
    at_a1 = pt.log_mass(q, k, 1, rows, s + 1, lo, hi)
    assert torch.allclose(both[:, 0], at_sep, atol=1e-6) and torch.allclose(both[:, 1], at_a1, atol=1e-6)
    assert abs(float(-both.mean()) - float(-(at_sep.sum() + at_a1.sum()) / (2 * len(s)))) < 1e-6


def test_the_local_two_position_runs_never_share_a_name():
    p2 = run_tag(dict(W_BASE, value_addr="1", ptr_lam="0.1", ptr_pos="both", ptr_stop="1", series="pointer2"))
    rep = run_tag(dict(W_BASE, seed="7", value_addr="1", ptr_lam="0.1", series="pointerrep"))
    p = run_tag(dict(W_BASE, value_addr="1", ptr_lam="0.1", series="pointer"))
    assert p2 == STEM.format(1) + "va1_pt10b4h0p2sg_ahdigit1_pointer2"
    assert rep == STEM.format(7) + "va1_pt10b4h0_ahdigit1_pointerrep"
    assert p == STEM.format(1) + "va1_pt10b4h0_ahdigit1_pointer"                 # the defaults: 0.1.78's names
    assert run_tag(dict(W_BASE, value_addr="1", ptr_lam="0.1", ptr_pos="sep", ptr_stop="0", series="pointer")) == p
    assert run_tag(dict(W_BASE, value_addr="1", ptr_lam="0.1", ptr_pos="both", series="x")).endswith("_pt10b4h0p2_ahdigit1_x")
    assert run_tag(dict(W_BASE, value_addr="1", ptr_lam="0.1", ptr_stop="1", series="x")).endswith("_pt10b4h0sg_ahdigit1_x")


def test_the_runner_passes_the_form_and_labels_it():
    runs = br.expand(dict(arms="alhena-both", seeds="1", grid="ag", grid_p="0.5", head="digit", head_lam="1", rotary_content="4",
                          value_addr="1", ptr_lam="0.1", ptr_block="4", ptr_head="0", ptr_pos="both", ptr_stop="1", series="pointer2"))
    assert len(runs) == 1 and "--ptr_pos=both" in runs[0] and "--ptr_stop=1" in runs[0]
    lab = br.run_label(runs[0])
    assert "pointer term 0.1 at block 4 head 0 at both address positions, its gradient kept in the head" in lab
    rep = br.expand(dict(arms="alhena-both", seeds="7", grid="ag", grid_p="0.5", head="digit", head_lam="1", rotary_content="4",
                         value_addr="1", ptr_lam="0.1", series="pointerrep"))
    assert "--ptr_pos" not in " ".join(rep[0]) and "both address" not in br.run_label(rep[0])
    assert {"ptr_pos", "ptr_stop"} <= set(br.GRID_KEYS)
