# -*- coding: utf-8 -*-
"""The value-path address (0.1.76): two level tables of 8 rows (anchored_rope.ValueAddress, born zero) add each token's address digits,
read from the key stream the anchored rotation reads, to its input vector: x_n = embed(t_n) + E_1[dk_1(n)] + E_2[dk_2(n)], a null digit
adding nothing. Off, the batches are 0.1.75's bit for bit; born zero, the patched forward equals the plain one exactly; the tables add
nothing outside the compartments (the head slot, the index, the questions, the plain rows); the patch keeps the state dict; the side
file round-trips through rotary_scale.Core; the names and the runner carry the flag. CPU only; the Hub is never touched."""
import ast
import json
import os

import numpy as np
import pytest
import torch

from geolip.alephllm.model.alephlm import AlephLM
from geolip.nano_beatrix import anchored_rope as ar
from geolip.nano_beatrix import bed_runner as br
from geolip.nano_beatrix import grid_rows as gr
from geolip.nano_beatrix import rotary_scale as rs
from geolip.nano_beatrix.bodies import craft_cfg
from geolip.nano_beatrix.names import run_tag
from test_key_hard import FP_0174_BATCH, batch_digest
from test_posing import _row
from test_scale import _bed_batch

W_BASE = {"arm": "alhena-both", "seed": "1", "ctx": "4096", "mb": "8", "steps": "24000", "frames": "dyn", "sizes": "32,128,256,512",
          "frame_records": "4", "nest": "0.3", "overlap": "0.0", "ask": "any", "register": "caption", "tags": "mode", "plant": "1.0",
          "grid": "ag", "grid_p": "0.5", "head": "digit", "head_lam": "1", "rotary_content": "4"}
STEM = "alhena-both_s{}_L24000_ctx4096_frdyn_cap_mode_sz32-128-256-512_r4_n30_o0_askany_gag50_cp4_"


def _craft(n_ctx=256, seed=1, content=4):
    """A fresh ALHENA-W craft, patched with the anchored rotation at its split, and its streams."""
    torch.manual_seed(seed)
    m = AlephLM(craft_cfg("alhena-both", n_ctx)).eval()
    st = ar.AnchorStreams(content)
    ar.patch(m, st)
    return m, st


def _streams(B, n, seed=0, p_null=0.4):
    """Random key and query streams: digits 0-7, the null anchor on a share of the positions."""
    g = torch.Generator().manual_seed(seed)
    dk = torch.randint(0, gr.RADIX, (B, n, gr.LEVELS), generator=g)
    dq = torch.randint(0, gr.RADIX, (B, n, gr.LEVELS), generator=g)
    dk[torch.rand(B, n, generator=g) < p_null] = gr.NULL
    dq[torch.rand(B, n, generator=g) < p_null] = gr.NULL
    return dk, dq


def test_off_the_batches_are_0175s_bit_for_bit():
    # the flag touches the craft only: the batch the trainer builds is 0.1.75's (= 0.1.74's at no key-hard row) to the last byte
    assert batch_digest(_bed_batch(4)) == FP_0174_BATCH
    src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "geolip", "nano_beatrix", "bed_train.py"), encoding="utf-8").read()
    fns = {n.name: ast.get_source_segment(src, n) for n in ast.parse(src).body if isinstance(n, ast.FunctionDef)}
    for fn in ("cap", "draw_plan", "frame_row", "ag_rows", "grid_aux", "batch"):            # the batch's whole path (test_scale._bed_batch)
        assert "VALUE_ADDR" not in fns[fn] and "value_addr" not in fns[fn]


def test_born_zero_the_forward_is_the_plain_forward_exactly():
    m, st = _craft()
    x = torch.randint(0, 256, (2, 64))
    dk, dq = _streams(2, 64)
    st.set(dk, dq)
    with torch.no_grad():
        plain = m(x).logits.clone()
    va = ar.ValueAddress(m.cfg.d_model, st)
    ar.patch_value(m, va)
    assert float(va.table.detach().abs().max()) == 0.0 and tuple(va.table.shape) == (ar.LEVELS, ar.NULL, m.cfg.d_model)
    with torch.no_grad():
        born = m(x).logits
    assert torch.equal(plain, born)                                                     # fp32, exactly
    st.clear()


def test_the_tables_add_their_written_formula():
    m, st = _craft()
    va = ar.ValueAddress(m.cfg.d_model, st)
    with torch.no_grad():
        va.table.normal_()
    x = torch.randn(2, 48, m.cfg.d_model)
    dk, dq = _streams(2, 48, seed=3)
    st.set(dk, dq)
    got = va.add(x)
    exp = x.clone()
    for b in range(2):
        for n in range(48):
            for lev in range(ar.LEVELS):
                d = int(dk[b, n, lev])
                if d != gr.NULL:
                    exp[b, n] += va.table[lev, d]
    assert torch.allclose(got, exp, atol=1e-6)
    st.clear()
    assert torch.equal(va.add(x), x)                                                     # no streams: nothing added
    st.set(dk[:, :40], dq[:, :40])
    with pytest.raises(AssertionError):
        va.add(x)                                                                        # streams of another length are refused
    st.clear()


def _add_0176(va, x):
    """0.1.76's ValueAddress.add (the indexed gather), verbatim: 0.1.77's embedding form must give the same values."""
    dk = va.streams.dk
    if dk is None:
        return x
    dk = dk.long()
    for lev in range(ar.LEVELS):
        d = dk[..., lev]
        on = (d >= 0) & (d < ar.NULL)
        x = x + va.table[lev][d.clamp(0, ar.NULL - 1)].to(x.dtype) * on[..., None].to(x.dtype)
    return x


def test_the_embedding_form_gives_0176s_values():
    st = ar.AnchorStreams(4)
    va = ar.ValueAddress(32, st)
    with torch.no_grad():
        va.table.normal_()
    for seed in range(5):
        dk, dq = _streams(3, 200, seed=seed)
        dk[0, :7, 0] = torch.tensor([-1, 9, 12, 8, 0, 7, 3])                             # out-of-range digits read the null row
        st.set(dk, dq)
        x = torch.randn(3, 200, 32, requires_grad=True)
        new, old = va.add(x), _add_0176(va, x)
        assert torch.equal(new, old)                                                     # the same values, to the last bit
        g = torch.randn_like(new)
        gn = torch.autograd.grad(new, (x, va.table), g)
        go = torch.autograd.grad(old, (x, va.table), g)
        assert torch.equal(gn[0], go[0]) and torch.allclose(gn[1], go[1], rtol=1e-5, atol=1e-5)   # the table's sums in another order
        st.clear()


def test_the_patch_keeps_the_state_dict():
    m, st = _craft()
    before = {k: v.clone() for k, v in m.state_dict().items()}
    va = ar.ValueAddress(m.cfg.d_model, st)
    ar.patch_value(m, va)
    after = m.state_dict()
    assert before.keys() == after.keys() and all(torch.equal(before[k], after[k]) for k in before)
    assert not any(p is va.table for p in m.parameters())                                # the tables live outside the model


def test_nothing_is_added_outside_the_compartments():
    for seed in range(8):
        for ab in (2, 1):
            row, meta = _row(seed=seed, abytes=ab)
            dk, _ = gr.streams(meta)
            dk = dk.astype(np.int64)
            inside = np.zeros(gr.CTX, dtype=bool)
            for r in meta["recs"]:
                inside[r["lo"]:r["hi"]] = True
                assert (dk[r["lo"]:r["hi"]] == np.array(gr.digits(r["addr"]))).all()   # the compartment's own address on its every token
            assert (dk[~inside] == gr.NULL).all()                                         # the head slot, the index, the questions
            assert (dk[gr.INDEX_LO:gr.QUEST_HI] == gr.NULL).all()
            for q in meta["questions"]:
                assert (dk[q["lo"]:q["end"] + 1] == gr.NULL).all()
            for e in meta["index"]:
                for p in gr.index_positions(meta)[meta["index"].index(e)]:
                    assert (dk[p] == gr.NULL).all()
            # through the tables: a zero input stays zero exactly where the stream is null
            st = ar.AnchorStreams(4)
            va = ar.ValueAddress(16, st)
            with torch.no_grad():
                va.table.normal_()
            st.set(torch.from_numpy(dk)[None], torch.from_numpy(dk)[None])
            out = va.add(torch.zeros(1, gr.CTX, 16))[0]
            assert (out[torch.from_numpy(~inside)] == 0).all() and (out[torch.from_numpy(inside)].abs().sum(-1) > 0).all()
            st.clear()
    st = ar.AnchorStreams(4)
    va = ar.ValueAddress(16, st)
    with torch.no_grad():
        va.table.normal_()
    st.set(torch.full((2, 32, 2), gr.NULL), torch.full((2, 32, 2), gr.NULL))              # a plain row: every token at the null anchor
    x = torch.randn(2, 32, 16)
    assert torch.equal(va.add(x), x)


def test_the_side_file_round_trips_through_the_core(tmp_path):
    from safetensors.torch import save_file
    m, st = _craft(n_ctx=gr.CTX, seed=2)
    va = ar.ValueAddress(m.cfg.d_model, st)
    with torch.no_grad():
        va.table.normal_(std=0.5)
    ar.patch_value(m, va)
    head = torch.nn.Linear(m.cfg.d_model, gr.LEVELS * gr.RADIX)
    d = tmp_path / "core"
    d.mkdir()
    save_file({k: v.detach().cpu().contiguous() for k, v in m.state_dict().items()}, str(d / "model.safetensors"))
    json.dump(m.cfg.to_dict(), open(d / "config.json", "w", encoding="utf-8"), indent=1)
    save_file({k: v.detach().cpu().contiguous() for k, v in head.state_dict().items()}, str(d / "addr_head.safetensors"))
    save_file({"table": va.table.detach().cpu().contiguous()}, str(d / "value_addr.safetensors"))
    sj = dict(grid="ag", grid_p=0.5, n_ag=4, rotary=ar.spec(4), head="digit", head_lam=1.0,
              layout=dict(slot=gr.SLOT, comps=gr.N_COMP, radix=gr.RADIX, levels=gr.LEVELS),
              value_addr=dict(levels=ar.LEVELS, digits=ar.NULL, d=m.cfg.d_model))
    json.dump(sj, open(d / "scale.json", "w", encoding="utf-8"), indent=1)
    core = rs.Core(str(d), "cpu")
    assert core.value_addr is not None and torch.equal(core.value_addr.table, va.table.detach())
    row, meta = _row(seed=4)
    x = torch.tensor(list(row[:-1]), dtype=torch.long)[None]
    core.set_streams([meta])
    with torch.no_grad():
        got = core.readout(core.trunk(x))
    dk, dq = gr.streams(meta)
    st.set(torch.from_numpy(dk.astype(np.int64))[None], torch.from_numpy(dq.astype(np.int64))[None])
    with torch.no_grad():
        exp = m(x).logits
    st.clear()
    core.clear()
    assert torch.equal(got, exp)                                                         # the reader's forward with the tables = the trainer's
    # the digest covers the tables: a moved table is a moved core
    dg = core.state_digest()
    assert dg == core.digest
    with torch.no_grad():
        core.value_addr.table[0, 3, 0] += 1.0
    assert core.state_digest() != dg
    # scale.json names the tables exactly when their file is there
    os.remove(d / "value_addr.safetensors")
    with pytest.raises(AssertionError):
        rs.Core(str(d), "cpu")
    del sj["value_addr"]
    json.dump(sj, open(d / "scale.json", "w", encoding="utf-8"), indent=1)
    plain = rs.Core(str(d), "cpu")
    assert plain.value_addr is None and plain.state_digest() == rs._digest(plain.model)   # a core without them: 0.1.75's digest


def test_the_value_runs_never_share_a_name():
    v = run_tag(dict(W_BASE, value_addr="1", series="value"))
    w7 = run_tag(dict(W_BASE, seed="7", series="widerep"))
    assert v == STEM.format(1) + "va1_ahdigit1_value"
    assert w7 == STEM.format(7) + "ahdigit1_widerep"
    assert run_tag(dict(W_BASE, value_addr="0", series="wide")) == run_tag(dict(W_BASE, series="wide"))     # off names nothing
    assert run_tag(dict(W_BASE, grid_hard="0.5", value_addr="1", series="x")).endswith("_cp4_kh50_va1_ahdigit1_x")
    assert len({v, w7, run_tag(dict(W_BASE, seed="7", value_addr="1", series="value"))}) == 3


def test_the_runner_passes_the_flag_and_labels_it():
    runs = br.expand(dict(arms="alhena-both", seeds="1,2", grid="ag", grid_p="0.5", head="digit", head_lam="1", rotary_content="4",
                          value_addr="1", series="value"))
    assert len(runs) == 2 and all("--value_addr=1" in r for r in runs)
    assert "value-path address" in br.run_label(runs[0]) and "content pairs 4" in br.run_label(runs[0])
    rep = br.expand(dict(arms="alhena-both", seeds="7", grid="ag", grid_p="0.5", head="digit", head_lam="1", rotary_content="4",
                         series="widerep"))
    assert "--value_addr" not in " ".join(rep[0]) and "value-path" not in br.run_label(rep[0])
    assert "value_addr" in br.GRID_KEYS
