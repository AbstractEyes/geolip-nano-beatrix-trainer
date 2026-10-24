# -*- coding: utf-8 -*-
"""The key-hard screen (0.1.75): a share of the grid rows whose 24 keys are the factorial of their parts (--grid_hard=KH; grid_rows.hard_keys:
2 words x 2 hundreds digits x 2 tens digits x 3 units digits), so a match on any part of the key fails on them. KH 0 is 0.1.74's rows and
batches bit for bit (fingerprints computed with the 0.1.74 tree, tag v0.1.74); a hard row differs from the standard row on its draw only in
the keys; the batch's last round(N_AG x KH) grid rows are the hard ones; the names and the runner carry the share. CPU only; the Hub is
never touched."""
import hashlib
import json

import numpy as np
import pytest
import torch

from geolip.nano_beatrix import bed_frames as bf
from geolip.nano_beatrix import bed_runner as br
from geolip.nano_beatrix import grid_rows as gr
from geolip.nano_beatrix.names import run_tag
from test_posing import FP_0173, _fingerprint, _row
from test_scale import _bed_batch

# 0.1.74's batch (bed_train's own batch through test_scale._bed_batch: 4 grid rows of 8 at 4,096; steps 0, 1, 57, 1000), computed with the
# 0.1.74 tree (tag v0.1.74) by batch_digest below.
FP_0174_BATCH = "cfb3b89c132ff0a361a038a142db07a1578f48e2d2a846dea11a51010986e070"


def batch_digest(batch, steps=(0, 1, 57, 1000)):
    h = hashlib.sha256()
    for st in steps:
        x, y, v, a = batch(st)
        for t in (x, y, v):
            h.update(np.ascontiguousarray(t.numpy()).tobytes())
        for k in sorted(a):
            val = a[k]
            if isinstance(val, torch.Tensor):
                h.update(k.encode() + np.ascontiguousarray(val.numpy()).tobytes())
            else:
                h.update(k.encode() + json.dumps(val, sort_keys=True, default=lambda o: o.tolist() if hasattr(o, "tolist") else int(o)).encode())
    return h.hexdigest()


def _parts(key: bytes) -> tuple:
    """A key's four parts: the word, the hundreds, tens and units digits (bed_frames.key_name: a word + a three-digit number)."""
    s = key.decode()
    return s[:-3], s[-3], s[-2], s[-1]


def test_kh_0_is_0174s_rows_and_batches_bit_for_bit():
    assert _fingerprint() == FP_0173 and _fingerprint(keys=None) == FP_0173      # 0.1.74's default rows are 0.1.73's
    assert batch_digest(_bed_batch(4)) == FP_0174_BATCH
    assert batch_digest(_bed_batch(4, 0)) == FP_0174_BATCH
    _, m = _row()
    assert "hard" not in m                                                        # the default meta names nothing new


@pytest.mark.parametrize("seed", range(25))
def test_the_hard_keys_are_the_factorial_of_their_parts(seed):
    keys = gr.hard_keys(np.random.default_rng([3, seed]))
    assert len(keys) == gr.N_COMP == 24 and len(set(keys)) == 24
    parts = [_parts(k) for k in keys]
    assert all(p[0] in bf.WORDS for p in parts)
    sizes = [len({p[i] for p in parts}) for i in range(4)]
    assert sizes == [2, 2, 2, 3]
    assert set(parts) == {(a, b, c, d) for a in {p[0] for p in parts} for b in {p[1] for p in parts}
                          for c in {p[2] for p in parts} for d in {p[3] for p in parts}}         # every combination, once
    for p in parts:                                   # a neighbour that differs in exactly one part, for each of the four parts
        for i in range(4):
            assert sum(q[i] != p[i] and all(q[j] == p[j] for j in range(4) if j != i) for q in parts) == sizes[i] - 1


def _matcher_rate(keys, part_idx) -> float:
    """The best a matcher that sees only the parts `part_idx` of the key can do on one row: right on 1 / (the keys sharing what it sees)."""
    parts = [_parts(k) for k in keys]
    seen = [tuple(p[i] for i in part_idx) for p in parts]
    return float(np.mean([1.0 / seen.count(s) for s in seen]))


def test_a_partial_key_match_fails_on_a_hard_row():
    keys = gr.hard_keys(np.random.default_rng(9))
    assert _matcher_rate(keys, (2, 3)) == 0.25                                    # the key's two-digit ending (the posing screen's stall)
    assert _matcher_rate(keys, (1, 2, 3)) == 0.5                                  # the number alone
    assert _matcher_rate(keys, (0, 3)) == 0.25                                    # the word with the units
    assert _matcher_rate(keys, (0, 1, 2)) == pytest.approx(1 / 3)                 # all but the units
    assert _matcher_rate(keys, (0, 1, 2, 3)) == 1.0                               # only the whole key is right on every entry
    std = [r["key"].encode() for r in _row(seed=9)[1]["recs"]]
    assert _matcher_rate(std, (2, 3)) > 0.75                                      # on a standard row the ending mostly suffices


def test_a_hard_row_differs_from_the_standard_row_only_in_its_keys():
    for t in range(12):
        kh = gr.hard_keys(np.random.default_rng([5, t]))
        for ab in (2, 1):
            row0, m0 = _row(seed=t, abytes=ab)
            row1, m1 = _row(seed=t, abytes=ab, keys=kh)
            assert m1["hard"] is True and [r["key"].encode() for r in m1["recs"]] == kh
            for r0, r1 in zip(m0["recs"], m1["recs"]):
                assert (r0["addr"], r0["slot"], r0["value"], r0["lo"]) == (r1["addr"], r1["slot"], r1["value"], r1["lo"])
                assert row1[r1["lo"]:r1["frame_end"]].count(r1["key"].encode()) == 1     # the record's frame names its new key
            assert [q["rec"] for q in m0["questions"]] == [q["rec"] for q in m1["questions"]]
            assert [e["rec"] for e in m0["index"]] == [e["rec"] for e in m1["index"]]
            assert gr.head_targets(m0)[1] == gr.head_targets(m1)[1]                  # the same addresses asked (the keys' lengths move
            assert gr.index_head_targets(m0)[1] == gr.index_head_targets(m1)[1]      # the questions and the entries, not what they name)
            assert len(row0) == len(row1) == gr.CTX + 1
    with pytest.raises(AssertionError):
        _row(keys=[b"code000"] * 24)                                              # a row's keys are distinct


def test_the_batchs_last_grid_rows_are_the_hard_ones():
    std, hard = _bed_batch(4), _bed_batch(4, 2)
    for step in (0, 1, 57):
        x0, y0, v0, a0 = std(step)
        x1, y1, v1, a1 = hard(step)
        assert torch.equal(x0[:6], x1[:6]) and torch.equal(y0[:6], y1[:6]) and torch.equal(v0, v1)   # the plain rows, the first 2 grid rows
        assert [m.get("hard", False) for m in a1["metas"]] == [False, False, True, True]
        assert not torch.equal(x0[6:], x1[6:])
        for m in a1["metas"][2:]:
            keys = [r["key"].encode() for r in m["recs"]]
            assert [len({_parts(k)[i] for k in keys}) for i in range(4)] == [2, 2, 2, 3]
        assert len(a1["head_pos"]) == len(a0["head_pos"]) == 4 * gr.N_COMP
    assert batch_digest(_bed_batch(4, 4)) != FP_0174_BATCH


def test_the_key_hard_runs_never_share_a_name():
    base = {"arm": "alhena-both", "seed": "1", "ctx": "4096", "mb": "8", "steps": "24000", "frames": "dyn", "sizes": "32,128,256,512",
            "frame_records": "4", "nest": "0.3", "overlap": "0.0", "ask": "any", "register": "caption", "tags": "mode", "plant": "1.0",
            "grid": "ag", "grid_p": "0.5", "head": "digit", "head_lam": "1", "rotary_content": "4"}
    stem = "alhena-both_s{}_L24000_ctx4096_frdyn_cap_mode_sz32-128-256-512_r4_n30_o0_askany_gag50_cp4_"
    k = run_tag(dict(base, grid_hard="0.5", series="keys"))
    w5 = run_tag(dict(base, seed="5", series="widerep"))
    assert k == stem.format(1) + "kh50_ahdigit1_keys"
    assert w5 == stem.format(5) + "ahdigit1_widerep"
    assert run_tag(dict(base, grid_hard="0", series="wide")) == run_tag(dict(base, series="wide"))      # none names nothing
    assert run_tag(dict(base, grid_hard="0.25", series="keys")).endswith("_cp4_kh25_ahdigit1_keys")
    assert len({k, w5, run_tag(dict(base, seed="5", grid_hard="0.5", series="keys"))}) == 3


def test_the_runner_passes_the_share_and_labels_it():
    runs = br.expand(dict(arms="alhena-both", seeds="1,2", grid="ag", grid_p="0.5", head="digit", head_lam="1", rotary_content="4",
                          grid_hard="0.5", series="keys"))
    assert len(runs) == 2 and all("--grid_hard=0.5" in r for r in runs)
    assert "key-hard rows 0.5" in br.run_label(runs[0]) and "content pairs 4" in br.run_label(runs[0])
    rep = br.expand(dict(arms="alhena-both", seeds="5", grid="ag", grid_p="0.5", head="digit", head_lam="1", rotary_content="4",
                         series="widerep", hard_read="1"))
    assert "--hard_read=1" in rep[0] and "key-hard" not in br.run_label(rep[0])
    assert {"grid_hard", "hard_read", "hard_batches"} <= set(br.GRID_KEYS)
