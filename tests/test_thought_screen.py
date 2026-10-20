# -*- coding: utf-8 -*-
"""THE THOUGHT READS and THE THOUGHT ARMS (0.1.67): the held-by-series split, the prompt cut, the items over the three layouts, the prior, the
arms' plan, repos and control, the marks' arithmetic."""
import copy
import json
import math
import os

import numpy as np
import pytest

from geolip.alephllm.data.special_tokens import CUE, END, SEP
from geolip.nano_beatrix import nano_v1 as V1
from geolip.nano_beatrix import thought_frames as TF
from geolip.nano_beatrix import thought_screen as TS
from test_thought_frames import EXAMPLE, _rows


def _write_pack(folder, n_series=60):
    """A pack of n_series series, three rows each (the example's simple, medium and deep rows under new series and row ids)."""
    os.makedirs(os.path.join(folder, "data", "medium", "photograph"), exist_ok=True)
    with open(os.path.join(folder, "data", "medium", "photograph", "photograph-medium-00000.jsonl"), "w", encoding="utf-8") as fh:
        for s in range(n_series):
            for j, r in enumerate(_rows()):
                r = copy.deepcopy(r)
                r.update(id=f"art-{s:03d}{j}", series=f"s-{s:05d}")
                fh.write(json.dumps(r) + "\n")
    return folder


def test_the_held_split_is_by_series_stable_and_disjoint(tmp_path):
    folder = _write_pack(str(tmp_path))
    train, held = TF.ThoughtPack.from_folder(folder, "train"), TF.ThoughtPack.from_folder(folder, "held")
    assert train.n + held.n == 180 and 3 <= held.n <= 36
    ids_t, ids_h = {r["id"] for r in train.rows}, {r["id"] for r in held.rows}
    assert not (ids_t & ids_h)
    ser_t, ser_h = {r["series"] for r in train.rows}, {r["series"] for r in held.rows}
    assert not (ser_t & ser_h)                                            # a series never splits across the two
    assert all(len([r for r in held.rows if r["series"] == s]) == 3 for s in ser_h)
    assert [r["id"] for r in TF.ThoughtPack.from_folder(folder, "held").rows] == [r["id"] for r in held.rows]   # stable
    marked = dict(EXAMPLE, held={"layout": None, "wordings": False, "values": ["aqua"]}, series="s-99999")
    assert TF.is_held(marked) and TF.is_held(dict(EXAMPLE, held={"layout": "D", "values": []}, series="s-99999"))
    assert not TF.is_held(dict(EXAMPLE, held={"layout": None, "values": []}, series="s-00001")) or TF.is_held(dict(EXAMPLE, held={}, series="s-00001"))
    st = TF.ensure(folder, log=lambda *a: None)                        # shards present: nothing fetched, the counts returned
    assert st["rows"] == 180 and st["train"] == train.n and st["held"] == held.n and set(st["depths"]) == {"simple", "medium", "deep"}


def test_prompt_of_cuts_at_the_last_question():
    rng = np.random.default_rng(3)
    q = EXAMPLE["questions"][0]
    mark = bytes([CUE]) + q["key"].encode() + bytes([SEP])
    for lay in ("A", "B", "C", "D"):
        doc, _ = TF.render(rng, EXAMPLE, lay, ask=0)
        p = TF.prompt_of(doc, q["key"])
        assert p is not None and p.endswith(mark)
        assert doc[len(p):] == q["value"].encode() + bytes([END])
        if lay in ("A", "B", "D"):                                        # the key stands in the spec frame too (and first in D): the cut is the last one
            assert doc.count(mark) >= 2 and doc.find(mark) < len(p) - len(mark)
    assert TF.prompt_of(doc, "never asked") is None


def test_the_items_cover_the_layouts_and_the_needs_and_the_prior_answers(tmp_path):
    folder = _write_pack(str(tmp_path))
    held, train = TF.ThoughtPack.from_folder(folder, "held"), TF.ThoughtPack.from_folder(folder, "train")
    items = TS.items_of(held, 8, 1, 4096)
    live = [it for it in items if not it.get("skip")]
    assert live and {it["layout"] for it in live} == set(TS.LAYOUT_READS)
    assert {it["needs"] for it in live} >= {"record", "chain", "unknown"}
    assert all(it["prompt"].endswith(bytes([CUE]) + it["key"].encode() + bytes([SEP])) and len(it["prompt"]) + len(it["val"]) + 1 <= 4096 for it in live)
    assert len({(it["doc"], it["q"]) for it in live}) * 3 == len(live)   # every question in every layout
    major, n = TS.prior_of(train)
    assert n == train.n and major["look hair"] == "short" and major["wear colour"] == "yellow"
    hd = TS.headline_of({"A": {"record": dict(whole=0.5, said_unknown=0.1), "chain": dict(whole=0.2), "unknown": dict(said_unknown=0.7)},
                         "C": {"chain": dict(whole=0.1)}, "D": {"record": dict(whole=0.4)}}, {"record": 0.3, "chain": 0.05}, {})
    assert hd["record_A"] == 0.5 and hd["unknown_A"] == 0.7 and hd["false_unknown_A"] == 0.1 and hd["chain_C"] == 0.1 and hd["record_D"] == 0.4
    assert hd["prior_record"] == 0.3 and hd["trained_record_A"] is None and hd["chain_D"] is None


def test_the_thought_arms_their_plan_their_repos_and_their_control():
    th, tc = "nano-castor-v1.2-th", "nano-castor-v1.2-tc"
    assert V1.diet_of(th)[-1]["name"] == "thought_continuation" and len(V1.diet_of(th)) == 14
    assert V1.plan_steps(th, V1.TOKENS_PER_STEP) == 61422 + math.ceil(V1.CONT_TOKENS / V1.TOKENS_PER_STEP)
    assert V1.body_of(th) == "nano-castor-v1.2" and V1.start_step(th, V1.TOKENS_PER_STEP) == 61422
    assert V1.framed_of(th) == "p50t" and V1.framed_of(tc) == "p50"
    assert V1.repo_of(th) == V1.repo_of(tc) == V1.REPO_EXPERIMENTS and V1.repo_of("nano-castor-v1.2") == V1.REPO
    init = V1.cont_init(th, V1.TOKENS_PER_STEP, repo=V1.REPO_EXPERIMENTS)
    assert init["repo"] == V1.REPO and init["prefix"] == "nano-castor-v1.2" and init["step"] == 61422
    assert init["payload"] == "resume/boundary_anneal_mix_step61422.pt"
    assert V1.control_of(th) == tc and V1.control_of(tc) is None and V1.control_of("nano-castor-v1.2") == "nano-castor-v1.1"
    assert V1.gauge_init(th, 65237) != V1.gauge_init(tc, 65237) and V1.arm_text(th) == "thoughts"
    for b in ("nano-castor", "nano-pollux-dual"):
        assert f"{b}-v1.2-th" in V1.ALL and f"{b}-v1.2-tc" in V1.ALL and V1.ALL[f"{b}-v1.2-th"]["source"] == f"{b}-v1.2"
    from geolip.nano_beatrix import framed_diet as fd
    assert fd.FORMS["p50t"].get("thought") is not None and "thought" in dict(fd.FORMS["p50t"]["kinds"])


def test_the_marks_arithmetic():
    h = dict(record_A=0.85, chain_A=0.30, chain_C=0.15, unknown_A=0.6, false_unknown_A=0.05, record_D=0.80, chain_D=0.25,
             prior_record=0.40, prior_chain=0.10, trained_record_A=0.95)
    th = dict(val_bpb=1.300, thought=h, bed={"bed/frame/copy_whole": 0.80})
    tc = dict(val_bpb=1.290, thought=dict(h, chain_A=0.12), bed={"bed/frame/copy_whole": 0.84})
    m = V1.thought_marks(th, tc, before=dict(record_A=0.5, chain_A=0.1))
    bars = {k: v for k, v in m.items() if v["bar"] is not None}
    assert len(bars) == 11 and all(v["met"] is True for v in bars.values())
    assert m["the gain over the source: chain A"]["value"] == pytest.approx(0.20)
    assert m["the control's chain read in A (what frame skill alone gives)"]["value"] == 0.12
    worse = V1.thought_marks(dict(th, val_bpb=1.33, thought=dict(h, chain_C=0.28, false_unknown_A=0.2)), tc, None)
    assert worse["TEXT KEPT: the held-out bits within 0.02 of the control's"]["met"] is False
    assert worse["CHAIN: layout A over layout C by 0.10 or more (the thoughts' share)"]["met"] is False
    assert worse["UNKNOWN: false unknowns on record questions at most 0.10"]["met"] is False
    pend = V1.thought_marks(dict(val_bpb=1.3, thought=None, bed=None), None, None)
    assert all(v["met"] is None for v in pend.values())
    t = V1.thought_marks_table(m)
    assert t.count("| MET |") == 11 and "MISSED" not in t
    tab = V1.thought_table({"x": [dict(phase="p", step=1, val_bpb=1.3, thought=h), dict(phase="q", step=2, val_bpb=1.3, thought=None)]}, before=dict(h, step=0))
    assert tab.count("\n") == 3 and "before" in tab and "| x | p |" in tab
