# -*- coding: utf-8 -*-
"""The larger grid row (0.1.83): the AG48 layout (64 slots of 64 bytes: 48 compartments, the index and the questions of 24 asked records)
and the long keys (a cue word + six digits, 9-12 bytes) with the key's reach 12. The AG layout with the standard keys is 0.1.82's row byte
for byte (a fingerprint of 0.1.82's rows) and the reach 9 is 0.1.82's phases (their digest); the AG48 rows hold every frame, entry and
question in their slots; the key-hard keys at 48 keep a one-part neighbour for every part and four records an ending; key_offsets at 12
codes a 12-byte key's SEP (and 9 does not); the 12-reach phases are their own seeded draw; the members' rows at AG48; the reading core
takes the layout, the keys' form and the reach from scale.json and reads back the trainer's forward; the names and the runner carry the
flags. CPU only; the Hub is never touched."""
import hashlib
import json

import numpy as np
import pytest
import torch

from geolip.alephllm.data.special_tokens import CUE, SEP
from geolip.nano_beatrix import anchored_rope as ar
from geolip.nano_beatrix import bed_frames as bf
from geolip.nano_beatrix import bed_runner as br
from geolip.nano_beatrix import grid_rows as gr
from geolip.nano_beatrix import key_code as kc
from geolip.nano_beatrix import key_rotary as kr
from geolip.nano_beatrix import rotary_scale as rs
from geolip.nano_beatrix.names import run_tag
from test_key_code import _save_core
from test_value_addr import STEM, W_BASE, _craft

BLOCK, HEAD = 4, 0
FP_0182_AG = "25dd2532a35e2f2a2554d8d48b7f7d2ead9538d067129fb6a40f12ccfc2c6aed"        # 0.1.82's rows (lr_fingerprints.py, trainer c8fdd19)
FP_0182_PHASES = "e03ed849a52a270be011f9fa0dead7485b549901ffadbe02210ee03fe7789212"    # 0.1.82's 8 x 9 x 256 phases (the same)
SRC = bytes(np.random.default_rng(3).integers(32, 127, size=gr.CTX + 1, dtype=np.uint8))


def _ag_fingerprint() -> str:
    h = hashlib.sha256()
    for t in range(12):
        for ab in (2, 1):
            asked = None if t % 3 else list(np.random.default_rng([5, t]).permutation(24)[:18])
            keys = gr.hard_keys(np.random.default_rng([21, t])) if t % 4 == 1 else None
            row, m = gr.compose_ag(np.random.default_rng([9, t]), SRC, asked=asked, swap_index=t % 5 == 4, abytes=ab, keys=keys)
            h.update(row)
            h.update(json.dumps(m, sort_keys=True).encode())
    return h.hexdigest()


def _by_hand(row, k: int) -> dict:
    """Every coded SEP of a row by a plain scan back from it at the reach k: {t: the key's bytes}."""
    row = bytes(row)
    out = {}
    for t, b in enumerate(row):
        if b != SEP:
            continue
        for j in range(1, k + 2):
            if t - j < 0 or row[t - j] == SEP:
                break
            if row[t - j] == CUE:
                out[t] = row[t - j + 1:t]
                break
    return {t: key for t, key in out.items() if len(key) > 0}


def test_ag_and_the_standard_keys_are_0182s_rows_and_phases():
    assert _ag_fingerprint() == FP_0182_AG
    assert kr.table_digest(kr.phase_table()) == FP_0182_PHASES and kr.seed_tag() == kr.SEED_TAG == "KEYROT|v1"
    assert kr.table_digest(kr.KeyRotary(BLOCK, HEAD).table) == FP_0182_PHASES
    assert kr.record(kr.KeyRotary(BLOCK, HEAD))["k"] == kc.K == 9


def test_the_layouts_and_what_they_hold():
    a, b = gr.lay("AG"), gr.lay("AG48")
    assert (a["slot"], a["comps"], a["ask"], a["index_lo"], a["quest_lo"]) == (128, 24, 24, 3200, 3584)
    assert (b["slot"], b["comps"], b["ask"], b["index_lo"], b["index_hi"], b["quest_lo"], b["quest_hi"]) == (64, 48, 24, 3136, 3584, 3584, 4096)
    assert gr.fits("AG", "std") and gr.fits("AG48", "std") and gr.fits("AG48", "long") and not gr.fits("AG", "long")
    with pytest.raises(AssertionError):
        gr.compose_ag(np.random.default_rng(1), SRC, key_form="long")                        # AG does not hold the long keys
    assert len(bf.key_long(0)) == 10 and bf.key_long(0) == b"code000000" and max(len(bf.key_long(i)) for i in range(16)) == 12


@pytest.mark.parametrize("key_form", gr.KEY_FORMS)
def test_the_ag48_row(key_form):
    for t in range(20):
        row, m = gr.compose_ag(np.random.default_rng([31, t]), SRC, layout="AG48", key_form=key_form)
        assert m["layout"] == "AG48" and m.get("keys", "std") == key_form and len(m["recs"]) == 48 and len(m["questions"]) == 24
        assert [r["slot"] for r in m["recs"]] == list(range(1, 49)) and m["addrs"] == sorted(m["addrs"]) and len(set(m["addrs"])) == 48
        assert all(r["lo"] == 64 * r["slot"] and r["hi"] == r["lo"] + 64 and r["frame_end"] <= r["hi"] for r in m["recs"])
        assert 3136 < m["index"][0]["lo"] and m["index_end"] <= 3584 and m["questions"][0]["lo"] == 3584 and m["quest_end"] <= 4096
        y = row[1:]
        for qi, q in enumerate(m["questions"]):
            rec = m["recs"][q["rec"]]
            assert row[q["lo"] + 1:q["sep"]].decode() == rec["key"]
            assert bytes(y[p] for p in gr.answer_positions(m)[qi]) == rec["value"].encode() + bytes([gr.END])
        lens = {len(r["key"]) for r in m["recs"]}
        assert lens <= (set(range(6, 10)) if key_form == "std" else set(range(9, 13)))
        dk, dq = gr.streams(m)
        assert all((dk[r["lo"]:r["hi"]] == gr.digits(r["addr"])).all() for r in m["recs"])
        assert (dk[:64] == gr.NULL).all() and (dk[3136:3584] == gr.NULL).all()


@pytest.mark.parametrize("layout,key_form", [("AG48", "std"), ("AG48", "long"), ("AG", "std")])
def test_the_hard_keys(layout, key_form):
    n = gr.lay(layout)["comps"]
    for t in range(20):
        kh = gr.hard_keys(np.random.default_rng([41, t]), layout, key_form)
        assert len(kh) == n == len(set(kh))
        ends = {}
        for k in kh:
            ends[k[-2:]] = ends.get(k[-2:], 0) + 1
        assert set(ends.values()) == {4}                                                     # the trigram's last two bytes on 4 records
        if key_form == "long":
            assert len({k[-6:-3] for k in kh}) == 1                                          # the first three digits one draw for the row
        row, m = gr.compose_ag(np.random.default_rng([7, t]), SRC, keys=kh, layout=layout, key_form=key_form)
        assert m["hard"] and [r["key"].encode() for r in m["recs"]] == kh


def test_key_offsets_at_the_reach_12():
    for t in range(6):
        row, m = gr.compose_ag(np.random.default_rng([51, t]), SRC, layout="AG48", key_form="long")
        x = torch.tensor(list(row[:-1]), dtype=torch.long)[None]
        for k in (9, 12):
            off = kc.key_offsets(x, k)
            hand = _by_hand(row[:-1], k)
            coded = set(torch.nonzero((off[0] != kc.VOCAB).any(-1)).flatten().tolist())
            assert coded == set(hand)
            for p, key in hand.items():
                assert [int(v) for v in off[0, p, :len(key)]] == list(key[::-1]) and (off[0, p, len(key):] == kc.VOCAB).all()
        long12 = [r for r in m["recs"] if len(r["key"]) == 12]
        if long12:                                                                            # a 12-byte key's SEP: coded at 12, not at 9
            sep = long12[0]["v0"] - 1
            assert bool((kc.key_offsets(x, 12)[0, sep] != kc.VOCAB).any()) and not bool((kc.key_offsets(x, 9)[0, sep] != kc.VOCAB).any())
    x = torch.tensor(list(gr.compose_ag(np.random.default_rng(2), SRC)[0][:-1]), dtype=torch.long)[None]
    assert torch.equal(kc.key_offsets(x), kc.key_offsets(x, 9))                            # the default is the reach 9


def test_the_phases_at_the_reach_12():
    p12 = kr.phase_table(12)
    assert tuple(p12.shape) == (kr.J, 12, kc.VOCAB) and torch.equal(p12, kr.phase_table(12))
    assert kr.seed_tag(12) == "KEYROT|v1|k12" and kr.table_digest(p12) != FP_0182_PHASES
    rot = kr.KeyRotary(BLOCK, HEAD, k=12)
    assert rot.k == 12 and torch.equal(rot.table, p12) and not rot.table.requires_grad
    rec = kr.record(rot)
    assert rec["k"] == 12 and rec["seed_tag"] == "KEYROT|v1|k12"
    row, m = gr.compose_ag(np.random.default_rng(61), SRC, layout="AG48", key_form="long")
    x = torch.tensor(list(row[:-1]), dtype=torch.long)[None]
    phi, on = rot.phases(x)
    hand = _by_hand(row[:-1], 12)
    assert set(torch.nonzero(on[0]).flatten().tolist()) == set(hand)
    for t, key in hand.items():
        s = None
        for k in range(1, len(key) + 1):
            term = rot.table[:, k - 1, key[-k]]
            s = term if s is None else s + term
        assert torch.equal(phi[0, t], torch.remainder(s, kr.TWO_PI))


def test_the_members_rows_at_48():
    owned, never = rs.owned_sets()
    rows = rs.member_rows(np.random.default_rng(71), np.frombuffer(SRC * 3, dtype=np.uint8), owned, never, layout="AG48", key_form="long")
    for row, m in rows:
        addrs = [r["addr"] for r in m["recs"]]
        own = [a for a in addrs if a in owned]
        assert len(addrs) == 48 and len(own) == 36 and len([a for a in addrs if a in never]) == 12
        asked = [m["recs"][q["rec"]]["addr"] for q in m["questions"]]
        assert len(asked) == 24 and set(asked) <= set(own)
    for addr in owned[:6]:
        row, m, qi = rs.owned_row(np.random.default_rng([72, addr]), SRC, addr, layout="AG48")
        assert len(m["recs"]) == 48 and len(m["questions"]) == 24 and m["recs"][m["questions"][qi]["rec"]]["addr"] == addr
    ag = rs.member_rows(np.random.default_rng(71), np.frombuffer(SRC * 3, dtype=np.uint8), owned, never)
    assert all(len(m["recs"]) == 24 and len(m["questions"]) == 18 for _, m in ag)       # AG: 0.1.82's 18 + 6, every owned asked


def test_the_core_reads_the_layout_the_keys_and_the_reach(tmp_path):
    m, st = _craft(n_ctx=gr.CTX, seed=3)
    va = ar.ValueAddress(m.cfg.d_model, st)
    with torch.no_grad():
        va.table.normal_(std=0.5)
    ar.patch_value(m, va)
    rot = kr.KeyRotary(BLOCK, HEAD, k=12)
    kr.patch(m, rot)
    d = tmp_path / "core"
    sj = _save_core(m, d, va)
    from safetensors.torch import save_file
    save_file({"table": rot.table.detach().cpu().contiguous()}, str(d / "key_rot.safetensors"))
    sj.update(grid_layout="AG48", grid_keys="long", layout=dict(slot=64, comps=48, radix=gr.RADIX, levels=gr.LEVELS))
    sj["key_rot"] = dict(pairs=kr.J, k=12, vocab=kc.VOCAB, block=BLOCK, head=HEAD, seed_tag=kr.seed_tag(12))
    sj["ptr"] = dict(lam=0.1, block=BLOCK, head=HEAD)
    json.dump(sj, open(d / "scale.json", "w", encoding="utf-8"), indent=1)
    core = rs.Core(str(d), "cpu")
    assert (core.layout, core.key_form, core.key_rot.k) == ("AG48", "long", 12) and torch.equal(core.key_rot.table, rot.table)
    rows = rs.gauge_rows(1, "read", 2, [SRC[1:]], layout=core.layout, key_form=core.key_form)
    assert all(mt["layout"] == "AG48" and mt["keys"] == "long" and len(mt["recs"]) == 48 for _, mt in rows)
    hard = rs.gauge_rows(1, "hard", 1, [SRC[1:]], hard=True, layout=core.layout, key_form=core.key_form)
    assert hard[0][1]["hard"] and len({r["key"][-2:] for r in hard[0][1]["recs"]}) == 12
    x = rs.to_x(rows, "cpu")[:, :-1]
    metas = [mt for _, mt in rows]
    core.set_streams(metas)
    with torch.no_grad():
        got = core.readout(core.trunk(x))
        h3 = core.trunk(x, 0, 3)
        split = core.readout(core.trunk(x, 3, None, h3))
    core.clear()
    dk, dq = zip(*[gr.streams(mt) for mt in metas])
    st.set(torch.from_numpy(np.stack(dk).astype(np.int64)), torch.from_numpy(np.stack(dq).astype(np.int64)))
    with torch.no_grad():
        exp = m(x).logits
    st.clear()
    assert torch.equal(got, exp) and torch.equal(split, exp)                               # the reader's forward = the trainer's
    sj["key_rot"]["k"] = 9                                                                   # a reach that does not match the keys: refused
    json.dump(sj, open(d / "scale.json", "w", encoding="utf-8"), indent=1)
    with pytest.raises(AssertionError):
        rs.Core(str(d), "cpu")


def test_the_trainers_batch_at_the_larger_row():
    from test_scale import _bed_batch
    plain, ag = _bed_batch(4, n_hard=2), _bed_batch(4, n_hard=2, layout="AG48", key_form="long")
    for step in (0, 3):
        x0, y0, v0, a0 = plain(step)
        x1, y1, v1, a1 = ag(step)
        assert torch.equal(x0[:4], x1[:4]) and torch.equal(y0[:4], y1[:4])                   # the plain rows: the same rows in their places
        assert len(a1["metas"]) == 4 and all(m["layout"] == "AG48" and m["keys"] == "long" and len(m["recs"]) == 48 for m in a1["metas"])
        assert [bool(m.get("hard")) for m in a1["metas"]] == [False, False, True, True]
        assert len(a1["head_pos"]) == 4 * 24                                                   # 24 questions a row at both layouts
        for j, m in enumerate(a1["metas"]):
            row = bytes(x1[4 + j].tolist())
            assert row[:len(gr.HEAD)] == gr.HEAD and all(row[r["lo"]] == 0xF9 for r in m["recs"])


def test_the_names_and_the_runner():
    pk = dict(W_BASE, grid_hard="0.5", value_addr="1", ptr_lam="0.1", key_rot="1")
    p = dict(W_BASE, value_addr="1", ptr_lam="0.1", key_rot="1")
    lrk = run_tag(dict(pk, grid_comps="48", series="largerows"))
    lrp = run_tag(dict(p, grid_comps="48", series="largerows"))
    llk = run_tag(dict(pk, grid_comps="48", grid_keys="long", series="largerows"))
    assert lrk == STEM.format(1) + "gc48_kh50_va1_pt10b4h0_krt_ahdigit1_largerows"
    assert lrp == STEM.format(1) + "gc48_va1_pt10b4h0_krt_ahdigit1_largerows"
    assert llk == STEM.format(1) + "gc48_kl_kh50_va1_pt10b4h0_krt_ahdigit1_largerows"
    assert run_tag(dict(pk, grid_comps="24", grid_keys="std", series="keyrot")) == run_tag(dict(pk, series="keyrot"))   # defaults name nothing
    for t in (lrk, lrp, llk):
        assert t.count("_gc48") == 1 and t.count("_kl") == (1 if t == llk else 0)
    runs = br.expand(dict(arms="alhena-both", seeds="1", grid="ag", grid_p="0.5", head="digit", head_lam="1", rotary_content="4",
                          grid_hard="0.5", value_addr="1", ptr_lam="0.1", key_rot="1", grid_comps="48", grid_keys="long", series="largerows"))
    assert len(runs) == 1 and "--grid_comps=48" in runs[0] and "--grid_keys=long" in runs[0]
    lab = br.run_label(runs[0])
    assert "48 records a row" in lab and "long keys" in lab and "grid_comps" in br.GRID_KEYS and "grid_keys" in br.GRID_KEYS
