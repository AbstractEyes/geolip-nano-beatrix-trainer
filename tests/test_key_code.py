# -*- coding: utf-8 -*-
"""The key code at the end-marker (0.1.81): at each SEP whose last CUE lies in the 10 positions before it (no SEP between them) the input
gains the key's bytes, one table an offset back: e'_t = e_t + sum_{k=1}^{n(t)} T_k[x_{t-k}]; the fixed form's rows are one seeded N(0, 1)
draw (no gradient), the learned form's tables are born zero. Off, the batches are 0.1.80's bit for bit; the code reads exactly the key's
bytes on every row form of the bed (the grid rows' records, index entries and questions, the key-hard rows' too; the frame rows' records
and questions; the rows' heads get none); it is its written formula; born zero the patched forward equals the plain one exactly; the patch
keeps the state dict; the side file round-trips through rotary_scale.Core; the names and the runner carry the flag. CPU only; the Hub is
never touched."""
import ast
import json
import os
import re

import numpy as np
import pytest
import torch

from geolip.alephllm.data.special_tokens import CUE, DOC, MODE, SEP
from geolip.nano_beatrix import anchored_rope as ar
from geolip.nano_beatrix import bed_runner as br
from geolip.nano_beatrix import grid_rows as gr
from geolip.nano_beatrix import key_code as kc
from geolip.nano_beatrix import rotary_scale as rs
from geolip.nano_beatrix.names import run_tag
from test_key_hard import FP_0174_BATCH, batch_digest
from test_posing import _row
from test_scale import _bed_batch
from test_value_addr import STEM, W_BASE, _craft

KEY_FORM = re.compile(rb"^[a-z]{3,6}[0-9]{3}$")                       # bed_frames.key_name: a cue word and three digits


def _by_hand(row) -> dict:
    """Every coded SEP of a row by a plain scan back from it: {t: the key's bytes} for a SEP whose nearest CUE lies at most K + 1 bytes
    back with no SEP between them."""
    row = bytes(row)
    out = {}
    for t, b in enumerate(row):
        if b != SEP:
            continue
        for j in range(1, kc.K + 2):
            if t - j < 0 or row[t - j] == SEP:
                break
            if row[t - j] == CUE:
                out[t] = row[t - j + 1:t]
                break
    return out


def _check_offsets(x: torch.Tensor) -> list:
    """The offsets of rows x (B, n) against the plain scan: the coded positions and every column. -> [{t: key}] per row."""
    off = kc.key_offsets(x)
    assert tuple(off.shape) == (x.shape[0], x.shape[1], kc.K) and off.dtype == torch.long
    keys = []
    for b in range(x.shape[0]):
        hand = {t: k for t, k in _by_hand(x[b].tolist()).items() if len(k) > 0}
        coded = set(torch.nonzero((off[b] != kc.VOCAB).any(-1)).flatten().tolist())
        assert coded == set(hand), (b, sorted(coded ^ set(hand))[:10])
        for t, key in hand.items():
            want = [key[-k] for k in range(1, len(key) + 1)] + [kc.VOCAB] * (kc.K - len(key))     # x_{t-1}, x_{t-2}, ... then null
            assert off[b, t].tolist() == want, (b, t, key)
        keys.append(hand)
    return keys


def test_off_the_batches_are_0180s_bit_for_bit():
    # the flag touches the craft only: the batch the trainer builds is 0.1.80's (= 0.1.74's at no key-hard row) to the last byte
    assert batch_digest(_bed_batch(4)) == FP_0174_BATCH
    src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "geolip", "nano_beatrix", "bed_train.py"), encoding="utf-8").read()
    fns = {n.name: ast.get_source_segment(src, n) for n in ast.parse(src).body if isinstance(n, ast.FunctionDef)}
    for fn in ("cap", "draw_plan", "frame_row", "ag_rows", "grid_aux", "batch"):            # the batch's whole path (test_scale._bed_batch)
        assert "KEY_CODE" not in fns[fn] and "key_code" not in fns[fn] and "kcm" not in fns[fn]


@pytest.mark.parametrize("n_hard", [0, 2])
def test_the_code_reads_the_key_on_every_row_form_of_the_bed(n_hard):
    batch = _bed_batch(4, n_hard)
    for step in (0, 1, 57):
        x, _, _, aux = batch(step)
        keys = _check_offsets(x)
        for b, hand in enumerate(keys):
            row = bytes(x[b].tolist())
            assert all(row[t] == SEP for t in hand)                                         # a code only at a SEP
            assert all(KEY_FORM.match(k) and 6 <= len(k) <= kc.K for k in hand.values()), [k for k in hand.values() if not KEY_FORM.match(k)]
            assert row[0] == DOC and row[1] == MODE and row[3] == SEP and row[5] == SEP     # the row's head: DOC, the tag, SEP, a task byte, SEP
            assert 3 not in hand and 5 not in hand                                          # its SEPs carry no key and get none
            if b < 4:                                                                       # a frame row: every record and the question
                assert row.count(bytes([CUE])) == len(hand)
                continue
            m = aux["metas"][b - 4]
            want = {r["v0"] - 1: r["key"].encode() for r in m["recs"]}                      # the records' SEPs
            want.update({e["a_pos"] - 1: m["recs"][e["rec"]]["key"].encode() for e in m["index"]})        # the index entries' SEPs
            want.update({q["sep"]: m["recs"][q["rec"]]["key"].encode() for q in m["questions"]})          # the questions' SEPs
            assert hand == want and len(hand) == gr.N_COMP + len(m["index"]) + len(m["questions"])
            assert bool(m.get("hard")) == (b >= 8 - n_hard)


def test_the_reach_covers_the_longest_key():
    from geolip.nano_beatrix import bed_frames as bf
    assert max(len(w) for w in bf.WORDS) + 3 == kc.K                                       # a cue word and three digits: 6 to 9 bytes
    assert min(len(w) for w in bf.WORDS) + 3 == 6


def test_the_rule_at_its_edges():
    def row(*parts):
        return torch.tensor(list(b"".join(p if isinstance(p, bytes) else bytes([p]) for p in parts)), dtype=torch.long)[None]
    cases = [
        (row(CUE, b"abc123", SEP, b"xy"), {7: b"abc123"}),                                 # six bytes
        (row(b"..", CUE, b"abcdef123", SEP), {12: b"abcdef123"}),                         # nine bytes: the longest key
        (row(b"..", CUE, b"abcdefg123", SEP), {}),                                        # ten bytes: out of reach
        (row(b"..", CUE, SEP, b"z"), {}),                                                 # no byte between: nothing to add
        (row(CUE, b"ab1", SEP, b"xyz", SEP), {4: b"ab1"}),                                # a second SEP after one CUE: only the first
        (row(SEP, b"ab", SEP), {}),                                                       # no CUE at all
        (row(DOC, MODE, b"c", SEP, b"g", SEP, CUE, b"cat042", SEP), {13: b"cat042"}),     # the head's SEPs: none; the record's: its key
        (row(CUE, CUE, b"dog7", SEP), {6: b"dog7"}),                                      # the nearest CUE wins
    ]
    for x, want in cases:
        off = kc.key_offsets(x)
        got = {t: bytes(int(v) for v in reversed(off[0, t].tolist()) if v != kc.VOCAB)
               for t in torch.nonzero((off[0] != kc.VOCAB).any(-1)).flatten().tolist()}
        assert got == want, (x.tolist(), got, want)
        _check_offsets(x)


def test_the_code_is_its_written_formula():
    torch.manual_seed(0)
    code = kc.KeyCode(8, "learned")
    with torch.no_grad():
        code.table.normal_()
    row, _ = _row(seed=3)
    x = torch.tensor(list(row[:-1]), dtype=torch.long)[None]
    got = code.code(x)[0]
    hand = _by_hand(row[:-1])
    exp = torch.zeros_like(got)
    for t, key in hand.items():
        s = None
        for k in range(1, len(key) + 1):
            term = code.table[k - 1, key[-k]]
            s = term if s is None else s + term
        if s is not None:
            exp[t] = s
    assert torch.equal(got, exp)                                                           # the same sums in the same order, to the last bit
    assert (got[[t for t in range(x.shape[1]) if t not in hand]] == 0).all()
    y = torch.randn(1, x.shape[1], 8)
    assert torch.equal(code.add(x, y), y + got[None])
    with pytest.raises(AssertionError):
        code.add(x[:, :100], y)                                                            # bytes and embedding of other lengths are refused


def test_the_fixed_rows_are_one_seeded_draw_whatever_the_runs_generators():
    torch.manual_seed(1)
    a = kc.fixed_table(16)
    after_a = torch.rand(3)
    torch.manual_seed(7)
    torch.rand(5)                                                                           # the run's generator somewhere else
    b = kc.fixed_table(16)
    assert torch.equal(a, b) and tuple(a.shape) == (kc.K, kc.VOCAB, 16)                     # the same rows whatever the run's generator holds
    torch.manual_seed(1)
    assert torch.equal(torch.rand(3), after_a)                                              # and the draw takes nothing from it
    f = kc.KeyCode(256, "fixed")
    assert torch.equal(f.table, kc.fixed_table(256)) and not f.table.requires_grad
    assert not list(f.parameters()) and "table" not in f.state_dict()                      # a buffer outside every optimizer and state dict
    t = f.table
    assert abs(float(t.mean())) < 0.01 and abs(float(t.std()) - 1.0) < 0.01                 # the byte tables' own init scale


def test_born_zero_the_forward_is_the_plain_forward_exactly_and_one_step_moves_the_key_rows():
    m, st = _craft()
    x = torch.randint(0, 200, (2, 64))
    x[0, 10:18] = torch.tensor([CUE] + list(b"cat042") + [SEP])
    x[1, 30:40] = torch.tensor([CUE] + list(b"horse917") + [SEP])
    with torch.no_grad():
        plain = m(x).logits.clone()
    code = kc.KeyCode(m.cfg.d_model, "learned")
    kc.patch_key(m, code)
    assert float(code.table.detach().abs().max()) == 0.0 and code.table.numel() == kc.K * kc.VOCAB * m.cfg.d_model
    with torch.no_grad():
        born = m(x).logits
    assert torch.equal(plain, born)                                                         # fp32, exactly
    opt = torch.optim.Adam([code.table], lr=1e-3)
    m(x).logits.float().pow(2).mean().backward()
    opt.step()
    moved = {(k, b) for k in range(kc.K) for b in range(kc.VOCAB) if float(code.table[k, b].detach().abs().max()) > 0}
    want = {(k - 1, key[-k]) for key in (b"cat042", b"horse917") for k in range(1, len(key) + 1)}
    assert moved == want                                                                    # only the key bytes' rows at their offsets


def test_the_patch_keeps_the_state_dict_and_composes_with_the_value_tables():
    m, st = _craft(n_ctx=gr.CTX)
    before = {k: v.clone() for k, v in m.state_dict().items()}
    va = ar.ValueAddress(m.cfg.d_model, st)
    with torch.no_grad():
        va.table.normal_()
    ar.patch_value(m, va)
    code = kc.KeyCode(m.cfg.d_model, "fixed")
    kc.patch_key(m, code)
    after = m.state_dict()
    assert before.keys() == after.keys() and all(torch.equal(before[k], after[k]) for k in before)
    assert not any(p is code.table for p in m.parameters())
    row, meta = _row(seed=6)
    x = torch.tensor(list(row[:-1]), dtype=torch.long)[None]
    dk, dq = gr.streams(meta)
    st.set(torch.from_numpy(dk.astype(np.int64))[None], torch.from_numpy(dq.astype(np.int64))[None])
    with torch.no_grad():
        got = m.embed(x)
        inner = type(m.embed).forward(m.embed, x)                                           # the library's embedding, unpatched
        exp = code.add(x, va.add(inner))
    st.clear()
    assert torch.equal(got, exp)                                                            # the value tables first, then the code


def test_the_share_read():
    m, st = _craft(n_ctx=gr.CTX)
    code = kc.KeyCode(m.cfg.d_model, "fixed")
    kc.patch_key(m, code)
    rows = [_row(seed=s) for s in (1, 2)]
    x = torch.tensor(np.stack([np.frombuffer(r, dtype=np.uint8) for r, _ in rows]).astype(np.int64))[:, :-1]
    s = kc.share_read(m, code, x)
    n = sum(gr.N_COMP + len(mt["index"]) + len(mt["questions"]) for _, mt in rows)
    assert s["seps"] == n and 6 <= s["key_len"] <= 9
    assert 0.5 < s["share"] < 1.0 and s["code_norm"] > s["plain_norm"] > 0                   # nine-ish N(0, 1) rows against three
    z = kc.KeyCode(m.cfg.d_model, "learned")
    assert kc.share_read(m, z, x)["share"] == 0.0                                           # born zero: no share


def _save_core(m, d, va=None, code=None):
    from safetensors.torch import save_file
    head = torch.nn.Linear(m.cfg.d_model, gr.LEVELS * gr.RADIX)
    d.mkdir()
    save_file({k: v.detach().cpu().contiguous() for k, v in m.state_dict().items()}, str(d / "model.safetensors"))
    json.dump(m.cfg.to_dict(), open(d / "config.json", "w", encoding="utf-8"), indent=1)
    save_file({k: v.detach().cpu().contiguous() for k, v in head.state_dict().items()}, str(d / "addr_head.safetensors"))
    sj = dict(grid="ag", grid_p=0.5, n_ag=4, rotary=ar.spec(4), head="digit", head_lam=1.0,
              layout=dict(slot=gr.SLOT, comps=gr.N_COMP, radix=gr.RADIX, levels=gr.LEVELS))
    if va is not None:
        save_file({"table": va.table.detach().cpu().contiguous()}, str(d / "value_addr.safetensors"))
        sj["value_addr"] = dict(levels=ar.LEVELS, digits=ar.NULL, d=m.cfg.d_model)
    if code is not None:
        save_file({"table": code.table.detach().cpu().contiguous()}, str(d / "key_code.safetensors"))
        sj["key_code"] = dict(form=code.form, k=kc.K, vocab=kc.VOCAB, d=m.cfg.d_model)
    json.dump(sj, open(d / "scale.json", "w", encoding="utf-8"), indent=1)
    return sj


@pytest.mark.parametrize("form", kc.FORMS)
def test_the_side_file_round_trips_through_the_core(tmp_path, form):
    m, st = _craft(n_ctx=gr.CTX, seed=2)
    va = ar.ValueAddress(m.cfg.d_model, st)
    with torch.no_grad():
        va.table.normal_(std=0.5)
    ar.patch_value(m, va)
    code = kc.KeyCode(m.cfg.d_model, form)
    if form == "learned":
        with torch.no_grad():
            code.table.normal_(std=0.3)                                                     # as grown by training
    kc.patch_key(m, code)
    d = tmp_path / "core"
    sj = _save_core(m, d, va, code)
    core = rs.Core(str(d), "cpu")
    assert core.key_code is not None and core.key_code.form == form and torch.equal(core.key_code.table, code.table.detach())
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
    assert torch.equal(got, exp)                                                            # the reader's forward with the code = the trainer's
    dg = core.state_digest()                                                                # the digest covers the code: a moved code is a moved core
    assert dg == core.digest
    with torch.no_grad():
        core.key_code.table[0, 49, 0] += 1.0
    assert core.state_digest() != dg
    os.remove(d / "key_code.safetensors")                                                   # scale.json names the code exactly when its file is there
    with pytest.raises(AssertionError):
        rs.Core(str(d), "cpu")
    del sj["key_code"]
    json.dump(sj, open(d / "scale.json", "w", encoding="utf-8"), indent=1)
    plain = rs.Core(str(d), "cpu")
    assert plain.key_code is None and plain.state_digest() == rs._digest(plain.model, plain.value_addr.table)   # 0.1.80's digest


def test_the_key_code_runs_never_share_a_name():
    pk = dict(W_BASE, grid_hard="0.5", value_addr="1", ptr_lam="0.1")
    f = run_tag(dict(pk, key_code="fixed", series="keycode"))
    lrn = run_tag(dict(pk, key_code="learned", series="keycode"))
    assert f == STEM.format(1) + "kh50_va1_pt10b4h0_kcf_ahdigit1_keycode"
    assert lrn == STEM.format(1) + "kh50_va1_pt10b4h0_kcl_ahdigit1_keycode"
    assert run_tag(dict(pk, series="keypointer")) == STEM.format(1) + "kh50_va1_pt10b4h0_ahdigit1_keypointer"      # off names nothing
    assert run_tag(dict(pk, key_code="", series="keypointer")) == run_tag(dict(pk, series="keypointer"))
    assert len({f, lrn, run_tag(dict(pk, seed="2", key_code="fixed", series="keycode"))}) == 3


def test_the_runner_passes_the_flag_and_labels_it():
    runs = br.expand(dict(arms="alhena-both", seeds="1", grid="ag", grid_p="0.5", head="digit", head_lam="1", rotary_content="4",
                          grid_hard="0.5", value_addr="1", ptr_lam="0.1", key_code="fixed,learned", series="keycode"))
    assert len(runs) == 2 and {a for r in runs for a in r if a.startswith("--key_code=")} == {"--key_code=fixed", "--key_code=learned"}
    assert "key code fixed" in br.run_label([a for a in runs if "--key_code=fixed" in a][0])
    rep = br.expand(dict(arms="alhena-both", seeds="7", grid="ag", grid_p="0.5", head="digit", head_lam="1", rotary_content="4",
                         grid_hard="0.5", value_addr="1", ptr_lam="0.1", series="keypointerrep"))
    assert "--key_code" not in " ".join(rep[0]) and "key code" not in br.run_label(rep[0])
    assert "key_code" in br.GRID_KEYS
