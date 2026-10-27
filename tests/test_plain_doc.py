# -*- coding: utf-8 -*-
"""Plain rows opened by DOC (0.1.90): below a share of 1 the trainer's unframed rows are windows of the packed stream at a drawn place;
--plain_doc=1 moves each such row to the first DOC at or after its place, so the row is DOC and then a document from its first byte (the
held-out gauge's row form). The batch draws nothing more: every framed row, grid row, value mask and grid read of a batch is the raw
windows' batch's, and every plain row is the packed stream from its DOC. The names carry "d" after the share; the runner passes the flag
and names it. CPU only; the Hub is never touched."""
import numpy as np
import torch

from geolip.alephllm.data.special_tokens import DOC, MODE
from geolip.nano_beatrix import bed_runner as br
from geolip.nano_beatrix.names import run_tag
from test_scale import _bed_batch
from test_value_addr import STEM, W_BASE

CTX = 4096


def _packed(seed=0, n_docs=300):
    """A toy packed stream as the trainer builds it: documents of 200-6,000 printable bytes, DOC after every document."""
    rng = np.random.default_rng(seed)
    lens = rng.integers(200, 6000, size=n_docs)
    corpus = rng.integers(32, 127, size=int(lens.sum()), dtype=np.uint8)
    offs = np.concatenate([[0], np.cumsum(lens)])
    packed = np.insert(corpus, offs[1:], np.uint8(DOC))
    docpos = np.flatnonzero(packed[:len(packed) - CTX - 1] == DOC)
    return packed, docpos


def _places(step, packed, mb=8):
    """batch()'s first draw: the rows' places in the packed stream."""
    return np.random.default_rng([1, step]).integers(0, len(packed) - CTX - 2, size=mb)


def test_plain_rows_open_at_a_doc_and_every_other_row_is_the_raw_windows_batch():
    packed, docpos = _packed()
    raw = _bed_batch(4, P_PLANT=0.75, packed=packed)
    doc = _bed_batch(4, P_PLANT=0.75, packed=packed, PLAIN_DOC=True, DOCPOS=docpos)
    n_plain = n_framed = 0
    for step in range(12):
        x0, y0, v0, a0 = raw(step)
        x1, y1, v1, a1 = doc(step)
        assert torch.equal(v0, v1)                                                  # the value masks: the plain rows carry none
        assert torch.equal(x0[4:], x1[4:]) and torch.equal(y0[4:], y1[4:])         # the grid rows, row for row
        for k in ("dk", "dq", "head_row", "head_pos", "head_dig", "ihead_row", "ihead_pos", "ihead_dig"):
            assert torch.equal(a0[k], a1[k]), k
        pos = _places(step, packed)
        for i in range(4):
            r0 = np.concatenate([x0[i].numpy(), y0[i, -1:].numpy()]).astype(np.uint8)
            r1 = np.concatenate([x1[i].numpy(), y1[i, -1:].numpy()]).astype(np.uint8)
            if r0[0] == DOC and r0[1] == MODE:                                       # a framed row: the same row
                assert np.array_equal(r0, r1)
                n_framed += 1
                continue
            n_plain += 1
            p = int(pos[i])
            q = int(docpos[min(int(np.searchsorted(docpos, p)), len(docpos) - 1)])
            assert np.array_equal(r0, packed[p:p + CTX + 1])                         # the raw window at the drawn place
            assert np.array_equal(r1, packed[q:q + CTX + 1]) and r1[0] == DOC and r1[1] != MODE   # the row from the first DOC at or after it
            assert q >= p or q == docpos[-1]
            assert not np.any(packed[p:q] == DOC)                                    # no DOC skipped between the place and the row's start
    assert n_plain >= 6 and n_framed >= 24, (n_plain, n_framed)                      # 48 places at a share of .75: both kinds drawn


def test_the_last_doc_serves_a_place_past_every_whole_row():
    packed, docpos = _packed(seed=3, n_docs=40)
    late = int(docpos[-1]) + 1                                                       # no DOC at or after it leaves a whole row
    q = int(docpos[min(int(np.searchsorted(docpos, late)), len(docpos) - 1)])
    assert q == int(docpos[-1]) and len(packed[q:q + CTX + 1]) == CTX + 1


def test_the_names_and_the_runner():
    pk = dict(W_BASE, grid_hard="0.5", value_addr="1", ptr_lam="0.1", grid_comps="48", key_rot="1", key_pairs="16", key_norm="2",
              key_rho="2.18")
    l16rp = run_tag(dict(pk, plant="0.75", plain_doc="1", series="plain16"))
    stem = STEM.format(1).replace("_askany_", "_askany_p75d_")
    assert l16rp == stem + "gc48_kh50_va1_pt10b4h0_krt16r218_ahdigit1_plain16" and l16rp.count("_p75d_") == 1
    assert run_tag(dict(pk, plant="0.75", series="plain16")) == STEM.format(1).replace("_askany_", "_askany_p75_") + \
        "gc48_kh50_va1_pt10b4h0_krt16r218_ahdigit1_plain16"                         # the raw windows: 0.1.89's name
    assert run_tag(dict(pk, series="rec16")) == STEM.format(1) + "gc48_kh50_va1_pt10b4h0_krt16r218_ahdigit1_rec16"   # every row framed
    runs = br.expand(dict(arms="alhena-both", seeds="1,2", grid="ag", grid_p="0.5", head="digit", head_lam="1", rotary_content="4",
                          grid_hard="0.5", value_addr="1", ptr_lam="0.1", key_rot="1", grid_comps="48", key_pairs="16", key_norm="2",
                          key_rho="2.18", plant="0.75", plain_doc="1", series="plain16"))
    assert len(runs) == 2 and all("--plain_doc=1" in r and "--plant=0.75" in r for r in runs)
    assert "plain rows opened by DOC (framed 0.75)" in br.run_label(runs[0])
    assert "plain_doc" in br.GRID_KEYS
    assert "plain rows" not in br.run_label([a for a in runs[0] if not a.startswith("--plain_doc")])
