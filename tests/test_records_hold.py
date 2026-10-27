# -*- coding: utf-8 -*-
"""The hold at the records only (0.1.88): KeyRotary(..., norm=2) holds the told head's bound key pairs at length rho at the coded SEPs that
end a record, found from the bytes at or before each SEP (key_rotary.records): in a grid row a record's CUE follows DATA and the address
bytes, in any other row a record's CUE lies inside an open frame. The rule is the rows' own metadata on every row kind the bed trains on
(the bed's batches, AG48 and AG, standard and key-hard; the bed's framed caption rows, nesting included; the one-byte address form and the
long keys): every record's coded SEP held, no index entry's, no question's. It reads no byte after the SEP (every later byte redrawn, the
mask up to the SEP unchanged). In the forward only the told head's key at the records' coded SEPs moves, every bound pair to length rho
with its angle kept; the query, the index entries' and the questions' keys and every other position, head and block are value for value.
The reading core rebuilds the form from scale.json and the tail read runs on it; the record, the names and the runner carry it; norm 1 is
0.1.87's hold at every coded SEP. CPU only; the Hub is never touched."""
import json
import math

import numpy as np
import pytest
import torch

from geolip.alephllm.data.special_tokens import CUE, DATA, DOC, END, MODE, SEP, THINK
from geolip.nano_beatrix import anchored_rope as ar
from geolip.nano_beatrix import bed_frames as bf
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
from test_key_rot import BLOCK, HEAD
from test_posing import _row
from test_scale import _bed_batch
from test_value_addr import STEM, W_BASE, _craft

RHO_R = 2.18                                                                     # the plan's length: the L16 cores' true records' median


def _hold(x, k=kc.K):
    return kr.records(x, kc.key_offsets(x, k))


def _coded(x, k=kc.K):
    return kc.key_offsets(x, k)[..., 0] != kc.VOCAB


def _grid_truth(m):
    """A grid row's coded SEPs from its meta: the records' (each frame's SEP, before its value), the index entries' (before the address),
    the questions'."""
    return [r["v0"] - 1 for r in m["recs"]], [e["a_pos"] - 1 for e in m["index"]], [q["sep"] for q in m["questions"]]


def _caption_truth(row, m):
    """A framed caption row's coded SEPs from its meta: every frame's own records (written from its open tag on, in its keys' order, each
    CUE key SEP value), then the question's."""
    vlen = bf.REGISTERS[m["register"]][3]
    recs = []
    for f in m["frames"]:
        p = f["start"] + f.get("open", 1)
        for key in f["keys"]:
            kb = key.encode()
            assert row[p] == CUE and row[p + 1:p + 1 + len(kb)] == kb and row[p + 1 + len(kb)] == SEP, (f, key)
            recs.append(p + 1 + len(kb))
            p += 2 + len(kb) + vlen
    kb = m["key"].encode()
    q = m["q0"] + 1 + len(kb)
    assert row[m["q0"]] == CUE and row[q] == SEP
    return recs, [q]


def _check(x, recs, others, k=kc.K):
    """One row's input bytes x (n,) against its truth: the coded SEPs are the records' and the others' exactly, each key's CUE stands at
    c = t - n(t) - 1, and the hold is the records' exactly."""
    off = kc.key_offsets(x[None], k)
    on = (off[..., 0] != kc.VOCAB)[0]
    hold = kr.records(x[None], off)[0]
    assert set(on.nonzero().flatten().tolist()) == set(recs) | set(others) and not set(recs) & set(others)
    t = on.nonzero().flatten()
    c = t - (off[0, t] != kc.VOCAB).sum(-1) - 1
    assert (x[c] == CUE).all()
    assert hold.nonzero().flatten().tolist() == sorted(recs), "the hold is every record's coded SEP and nothing else"
    return len(recs), len(others)


def _bed_rows(n_hard, layout, steps):
    """The bed's own batches (test_scale._bed_batch: the framed caption rows and the grid rows at the bed's settings) with every row's
    meta: the caption rows' metas by replaying batch()'s draws through its own frame_row (each row checked equal to the batch's)."""
    batch = _bed_batch(4, n_hard, layout, "std")
    ns = batch.__globals__                                                       # the bed's functions and settings, as exec'd
    out = []
    for step in steps:
        x, _, _, aux = batch(step)
        rng = np.random.default_rng([ns["SEED"], step])
        pos = rng.integers(0, len(ns["packed"]) - ns["CTX"] - 2, size=ns["MB"])
        for i, p in enumerate(pos[:ns["MB"] - ns["N_AG"]]):
            assert rng.random() < ns["P_PLANT"]
            w, metas = ns["frame_row"](rng, ns["packed"][p:p + 2 * ns["CTX"] + 2].tobytes())
            assert bytes(x[i].tolist()) == w[:-1] and len(metas) == 1      # the batch's own caption row
            out.append(("caption", x[i], metas[0], w))
        for j, m in enumerate(aux["metas"]):
            out.append(("grid", x[aux["first"] + j], m, None))
    return out


@pytest.mark.parametrize("layout", ["AG48", "AG"])
def test_the_rule_is_the_rows_metadata_on_the_beds_batches(layout):
    rows = _bed_rows(2, layout, range(12))
    n = dict(caption=[0, 0], grid=[0, 0], hard=0, nested=0)
    for kind, x, m, w in rows:
        if kind == "grid":
            recs, ents, qs = _grid_truth(m)
            assert len(recs) == gr.lay(layout)["comps"] and len(ents) == len(qs) == gr.lay(layout)["ask"]
            r, o = _check(x, recs, ents + qs)
            n["hard"] += int(m.get("hard", False))
        else:
            recs, qs = _caption_truth(w, m)
            r, o = _check(x, recs, qs)
            n["nested"] += sum(len(f["keys"]) for f in m["frames"] if f["depth"] >= 1)
        n[kind][0] += r
        n[kind][1] += o
    assert n["grid"][0] > 0 and n["caption"][0] > 0 and n["hard"] == 2 * 12 and n["nested"] > 0, n   # every kind, the nested records too


@pytest.mark.parametrize("abytes,key_form", [(1, "std"), (2, "long"), (1, "long")])
def test_the_rule_is_the_metadata_on_the_other_grid_forms(abytes, key_form):
    k = gr.KEY_MAX[key_form]
    for seed in range(6):
        rng = np.random.default_rng(seed)
        keys = gr.hard_keys(rng, "AG48", key_form) if seed % 2 else None
        row, m = gr.compose_ag(rng, bytes(np.random.default_rng(seed + 50).integers(32, 256, size=gr.CTX + 1, dtype=np.uint8)),
                               abytes=abytes, keys=keys, layout="AG48", key_form=key_form)
        x = torch.tensor(np.frombuffer(row, dtype=np.uint8).astype(np.int64))[:-1]
        recs, ents, qs = _grid_truth(m)
        _check(x, recs, ents + qs, k)


def _special_rows(B=24, n=320, seed=0):
    """Rows dense in the grammar's bytes (CUE, SEP, DATA, THINK, END, address digits and bytes, letters), half with a grid head byte."""
    g = torch.Generator().manual_seed(seed)
    alph = torch.tensor([CUE, SEP, DATA, THINK, END, CUE, SEP, DATA] + list(b"01234567") + [0x80, 0x9F, 0xBF] + list(b"gqacx"))
    x = alph[torch.randint(len(alph), (B, n), generator=g)]
    x[::2, 4] = gr.HEAD[4]
    return x


def _redraw(x, t, seed):
    g = torch.Generator().manual_seed(seed)
    y = x.clone()
    alph = _special_rows(1, 64, seed + 1)[0]
    y[:, t + 1:] = alph[torch.randint(len(alph), (y.shape[0], y.shape[1] - t - 1), generator=g)]
    return y


def test_the_rule_reads_no_byte_after_the_sep():
    rows = _bed_rows(2, "AG48", range(2))
    x = torch.stack([r[1] for r in rows])                                        # the bed's caption and grid rows
    base = _hold(x)
    seps = sorted(set(base.nonzero()[:, 1].tolist()))
    cuts = list(range(0, 12)) + seps[::7] + [t - 1 for t in seps[::11]] + [t + 1 for t in seps[::13]]
    for i, t in enumerate(cuts):
        assert torch.equal(_hold(_redraw(x, t, i))[:, :t + 1], base[:, :t + 1]), t
    s = _special_rows()                                                          # dense rows: every branch at every position
    sb = _hold(s)
    assert int(sb[::2].sum()) > 0 and int(sb[1::2].sum()) > 0 and int((_coded(s) & ~sb).sum()) > 0
    for t in range(0, s.shape[1] - 1):
        assert torch.equal(_hold(_redraw(s, t, 1000 + t))[:, :t + 1], sb[:, :t + 1]), t


def test_the_rule_by_hand_on_written_rows():
    def row(*parts, head=b"q"):
        b = bytes([DOC, MODE]) + b"c" + bytes([SEP]) + head + bytes([SEP])     # a head of six bytes (DOC MODE c SEP . SEP)
        for p in parts:
            b += p if isinstance(p, bytes) else bytes([p])
        return torch.tensor(list(b))[None]
    rec = lambda key: bytes([CUE]) + key + bytes([SEP]) + b"abcdef"             # noqa: E731
    g = row(DATA, b"34", rec(b"pin123"), END, b"text ", DATA, rec(b"tag004"), b"12", END, rec(b"pin123"), END, head=b"g")
    held = _hold(g)[0].nonzero().flatten().tolist()
    assert held == [6 + 3 + 1 + 6]                                               # the record only: not the entry (DATA CUE ...), not the question
    c = row(b"ab", THINK, b"xy", DATA, rec(b"key001"), END, rec(b"key002"), END, b" ", rec(b"key001"), END)
    on = _coded(c)[0].nonzero().flatten().tolist()
    held = _hold(c)[0].nonzero().flatten().tolist()
    assert len(on) == 3 and held == on[:2]                                       # the nested and the outer record; the question is free
    assert _hold(row(DATA, rec(b"word9"), END, head=b"g")).sum() == 0            # a grid row's DATA CUE (an index entry) is never held
    assert _hold(row(DATA, b"7", rec(b"word9"), END, head=b"g")).sum() == 0     # one digit is not an address
    assert _hold(row(DATA, bytes([0x85]), rec(b"word9"), END, head=b"g")).sum() == 1   # the one-byte address
    early = torch.tensor([[CUE, ord("a"), SEP, ord("g"), ord("g"), DATA]])       # a SEP before position 4 is never held
    assert _coded(early)[0, 2] and not _hold(early).any()


def _streams_mixed(st, metas, n):
    """The anchored streams of a batch of grid rows (their metas) and other rows (None: the null anchor at every position)."""
    dk, dq = [], []
    for m in metas:
        a, b = gr.streams(m) if m is not None else (np.full((n, gr.LEVELS), gr.NULL, dtype=np.int8),) * 2
        dk.append(a)
        dq.append(b)
    st.set(torch.from_numpy(np.stack(dk).astype(np.int64)), torch.from_numpy(np.stack(dq).astype(np.int64)))


def _mixed_batch():
    """Three rows the bed trains on: an AG48 key-hard grid row and a framed caption row from the bed's batch, and an AG grid row."""
    rows = _bed_rows(2, "AG48", [3])
    cap = next(r for r in rows if r[0] == "caption")
    hard = next(r for r in rows if r[0] == "grid" and r[2].get("hard"))
    ag_row, ag_meta = _row(seed=1)
    x = torch.stack([hard[1], cap[1], torch.tensor(np.frombuffer(ag_row, dtype=np.uint8).astype(np.int64))[:-1]])
    recs = [_grid_truth(hard[2])[0], _caption_truth(cap[3], cap[2])[0], _grid_truth(ag_meta)[0]]
    free = [sum(_grid_truth(hard[2])[1:], []), _caption_truth(cap[3], cap[2])[1], sum(_grid_truth(ag_meta)[1:], [])]
    return x, [hard[2], None, ag_meta], recs, free


def _forward(x, metas, norm, rho=kr.RHO, pairs=16, seed=4):
    m, st = _craft(n_ctx=gr.CTX, seed=seed)
    rot = kr.KeyRotary(BLOCK, HEAD, pairs=pairs, norm=norm, rho=rho)
    kr.patch(m, rot)
    cap = pt.HeadCapture(m.blocks[BLOCK - 1].attn, head=HEAD)
    _streams_mixed(st, metas, x.shape[1])
    with torch.no_grad():
        lg = m(x).logits
    q, k, _ = cap.qk()
    hold = rot.hold
    st.clear()
    return m, rot, q, k, lg, hold


def test_the_hold_moves_only_the_records_keys():
    x, metas, recs, free = _mixed_batch()
    m0, rot0, q0, k0, lg0, h0 = _forward(x, metas, 0)
    mr, rotr, qr, krr, lgr, hr = _forward(x, metas, 2, RHO_R)
    assert h0 is None and rotr.norm == 2 and rotr.rho == RHO_R
    for b in range(3):                                                           # the hold = the records' SEPs on every row kind
        assert hr[b].nonzero().flatten().tolist() == sorted(recs[b]) and all(bool(rotr.on[b, p]) for p in free[b])
    assert torch.equal(q0, qr)                                                   # the query untouched everywhere
    H = q0.shape[1]
    others = [h for h in range(H) if h != HEAD]
    assert torch.equal(k0[:, others], krr[:, others])
    assert torch.equal(k0[:, HEAD][~hr], krr[:, HEAD][~hr])                      # the entries', the questions' and every other position
    lens = cp.pair_norms(krr[:, HEAD][hr].float())                              # every pair of the told head at rho at the records
    assert lens.shape[-1] == 16 and torch.allclose(lens, torch.full_like(lens, RHO_R), atol=1e-4)
    a0 = torch.atan2(k0[:, HEAD][hr][..., 16:], k0[:, HEAD][hr][..., :16])
    ar_ = torch.atan2(krr[:, HEAD][hr][..., 16:], krr[:, HEAD][hr][..., :16])
    assert torch.allclose(torch.remainder(a0 - ar_ + math.pi, 2 * math.pi) - math.pi, torch.zeros_like(a0), atol=1e-4)   # angles kept
    assert not torch.equal(lg0, lgr)
    t = torch.randint(32, 127, (2, 300))                                         # text: no coded SEP - the off forward exactly
    with torch.no_grad():
        assert torch.equal(m0(t).logits, mr(t).logits)


def test_norm_one_is_0187s_hold_and_the_record_names_the_form():
    x, metas, _, _ = _mixed_batch()
    r1 = kr.KeyRotary(BLOCK, HEAD, norm=True)
    r1.set(x)
    assert r1.norm == 1 and torch.equal(r1.hold, r1.on)                         # 0.1.87: every coded SEP
    r2 = kr.KeyRotary(BLOCK, HEAD, pairs=16, norm=2, rho=RHO_R)
    r2.set(x)
    assert torch.equal(r2.hold, _hold(x)) and int(r2.hold.sum()) < int(r2.on.sum())
    r2.clear()
    assert r2.hold is None and r2.phi is None
    rec = kr.record(r2)
    assert rec["norm"] == 2 and rec["rho"] == RHO_R and rec["pairs"] == 16
    assert kr.record(r1)["norm"] == 1 and kr.record(r1)["rho"] == kr.RHO
    assert "norm" not in kr.record(kr.KeyRotary(BLOCK, HEAD, pairs=16))
    for bad in (3, 1.5, -1):
        with pytest.raises(AssertionError):
            kr.KeyRotary(BLOCK, HEAD, norm=bad)


def _saved(tmp_path):
    m, st = _craft(n_ctx=gr.CTX, seed=2)
    va = ar.ValueAddress(m.cfg.d_model, st)
    with torch.no_grad():
        va.table.normal_(std=0.5)
    ar.patch_value(m, va)
    rot = kr.KeyRotary(BLOCK, HEAD, pairs=16, norm=2, rho=RHO_R)
    kr.patch(m, rot)
    d = tmp_path / "core_16_r"
    sj = _save_core(m, d, va)
    from safetensors.torch import save_file
    save_file({"table": rot.table.detach().cpu().contiguous()}, str(d / "key_rot.safetensors"))
    sj["key_rot"] = dict(pairs=16, k=kc.K, vocab=kc.VOCAB, block=BLOCK, head=HEAD, seed_tag=kr.seed_tag(kc.K, 16), norm=2, rho=RHO_R)
    sj["ptr"] = dict(lam=0.1, block=BLOCK, head=HEAD)
    json.dump(sj, open(d / "scale.json", "w", encoding="utf-8"), indent=1)
    return m, st, rot, d


def test_the_reading_core_rebuilds_the_records_hold_and_the_tail_read_runs(tmp_path):
    m, st, rot, d = _saved(tmp_path)
    core = rs.Core(str(d), "cpu")
    assert core.key_rot.pairs == 16 and core.key_rot.norm == 2 and core.key_rot.rho == RHO_R
    x, metas, _, _ = _mixed_batch()
    x, metas = x[[0, 2]], [metas[0], metas[2]]                                   # the grid rows (the core's streams are the grid's)
    core.set_streams(metas)
    with torch.no_grad():
        got = core.readout(core.trunk(x))
        h3 = core.trunk(x, 0, 3)
        split = core.readout(core.trunk(x, 3, None, h3))
    _streams_mixed(st, metas, x.shape[1])
    with torch.no_grad():
        exp = m(x).logits
    st.clear()
    core.clear()
    assert torch.equal(got, exp) and torch.equal(split, exp)                     # the reader's forward = the trainer's
    rec = cp.tail_core(str(d), [SRC[1:]], R=2, ns=(10, 50), log=None)
    assert rec["told"] == dict(block=BLOCK, head=HEAD, pairs=16, norm=2, rho=RHO_R)
    lens = torch.tensor(rec["true_pair_length"]["p50"])
    assert len(lens) == 16 and torch.allclose(lens, torch.full_like(lens, RHO_R), atol=1e-3)   # the captured records carry the hold
    pr = rec["pool_length_ratio"]
    assert np.allclose(pr["records"]["p50"], 1.0, atol=1e-3) and np.allclose(pr["records"]["max"], 1.0, atol=1e-3)
    assert pr["entries"]["n"] > 0 and not np.allclose(pr["entries"]["p50"], 1.0, atol=1e-2)       # the entries keep their own lengths


def test_the_names_and_the_runner():
    pk = dict(W_BASE, grid_hard="0.5", value_addr="1", ptr_lam="0.1", grid_comps="48", key_rot="1", key_pairs="16")
    l16r = run_tag(dict(pk, key_norm="2", key_rho="2.18", series="rec16"))
    assert l16r == STEM.format(1) + "gc48_kh50_va1_pt10b4h0_krt16r218_ahdigit1_rec16"
    assert l16r.count("_krt16r218_") == 1
    assert run_tag(dict(pk, key_norm="2", series="rec16")).endswith("_krt16r_ahdigit1_rec16")       # the default length names nothing
    assert run_tag(dict(pk, key_norm="2", key_rho="4.16", series="rec16")) == run_tag(dict(pk, key_norm="2", series="rec16"))
    assert run_tag(dict(pk, series="norm16")) == STEM.format(1) + "gc48_kh50_va1_pt10b4h0_krt16_ahdigit1_norm16"   # 0.1.87's names
    assert run_tag(dict(pk, key_pairs="8", key_norm="1", series="norm16")) == STEM.format(1) + "gc48_kh50_va1_pt10b4h0_krtn_ahdigit1_norm16"
    runs = br.expand(dict(arms="alhena-both", seeds="1,2", grid="ag", grid_p="0.5", head="digit", head_lam="1", rotary_content="4",
                          grid_hard="0.5", value_addr="1", ptr_lam="0.1", key_rot="1", grid_comps="48", key_pairs="16", key_norm="2",
                          key_rho="2.18", series="rec16"))
    assert len(runs) == 2 and all("--key_norm=2" in r and "--key_rho=2.18" in r for r in runs)
    lab = br.run_label(runs[0])
    assert "over 16 pairs" in lab and "with the records' keys held" in lab and "at length 2.18" in lab
    assert "key_rho" in br.GRID_KEYS
