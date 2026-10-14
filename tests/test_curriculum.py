# -*- coding: utf-8 -*-
"""The curriculum's stage loop on the CPU, without training: the block names, the far band's windows, the rules, the walk over records,
the refusal of a record from another start, and the loop end to end with a runner that writes records instead of training."""
import json
import os

import pytest

from geolip.nano_beatrix import bed_curriculum as bc
from geolip.nano_beatrix import bed_frames as bf

BASE = dict(register="caption", frames="dyn", plant=0.75, ask="any", tags="mode", gauge_asks="nested,overlap", log_every=100)
STAGES = [
    dict(name="A", grid=dict(ctx=1024, mb=32, steps=12000, sizes="32,128,256", frame_records=8, nest=0.6, overlap=0.3), rule="formed", max_blocks=2),
    dict(name="B", grid=dict(ctx=4096, mb=8, steps=4000, sizes="32,128,256,512,1024", frame_records=16, nest=0.6, overlap=0.3),
         rule="far_fall", fall=0.1, window=800, band="512", max_blocks=5),
    dict(name="anneal", grid=dict(ctx=4096, mb=8, steps=4000, sizes="32,128,256,512,1024", frame_records=4, nest=0.3, overlap=0.0),
         rule=None, max_blocks=1),
]


def _curve(steps, hi, lo, cut, n=10, every=100):
    """A 0.1.22 curve whose far band reads `hi` bits a value for every step up to `cut` and `lo` after (n values a step)."""
    out, tot = [], 0.0
    for s in range(steps):
        tot += (hi if s + 1 <= cut else lo) * n
        if s % every == 0 or s == steps - 1:
            k = (s + 1) * n
            out.append(dict(step=s + 1, value_bits={"0": 1.0, "512": tot / k}, value_n={"0": k, "512": k}))
    return out


def _record(spec, formed=True, fall=0.0):
    g = {k: str(v) for k, v in spec.items()}
    steps = int(g["steps"])
    cut = 3101 if steps == 4000 else steps - 899
    return dict(arm=g["arms"], seed=int(g["seeds"]), steps=steps, ctx=int(g["ctx"]), micro_batch=int(g["mb"]), init=g.get("init", ""),
                stage=g["stage"], heldout_bpb=1.45, tok_per_s=9e5, train_minutes=2.4,
                formation=dict(formed=formed, failure=not formed, groups_judged=3),
                fetch_by_frame={"data/32/depth0": dict(n=40, full_value=0.9 if formed else 0.0, first_byte=0.9, value_bits=1.0, dist=200)},
                fetch_by_frame_asks={"nested": {"data/32/depth1": dict(n=192, full_value=0.8, first_byte=0.9, value_bits=1.0, dist=300)}},
                canary={k: dict(bits=1.5) for k in ("plain", "tag", "frame")}, curve=_curve(steps, 2.0 + fall, 2.0, cut))


def test_every_block_name_is_new_and_short():
    blocks = bc.plan(STAGES, BASE, "castor-both,pollux-both", "1,2")
    assert len(blocks) == 2 * 2 * (2 + 5 + 1)
    tags = [b["tag"] for b in blocks]
    assert len(set(tags)) == len(tags) and all("_from_" not in t for t in tags)
    assert all(t.endswith(f"_st{b['block']}") for t, b in zip(tags, blocks))
    assert max(len(t) for t in tags) < 140
    lo, hi = bc.bounds(STAGES, BASE)
    assert lo == 12000 * 1024 * 32 + 2 * 4000 * 4096 * 8 and hi == 2 * 12000 * 1024 * 32 + 6 * 4000 * 4096 * 8


def test_stage_names_are_checked():
    with pytest.raises(AssertionError):
        bc.check([dict(name="A1", grid={}, rule=None)], BASE)                 # a block of A11 and of A1's first would collide
    with pytest.raises(AssertionError):
        bc.check([dict(name="A", grid=dict(seeds="1"), rule=None)], BASE)     # the chain sets the seed
    with pytest.raises(AssertionError):
        bc.check([dict(name="A", grid={}, rule="sometimes")], BASE)


def test_band_fall_is_exact_from_the_counts():
    f = bf.band_fall(_curve(4000, 1.75, 1.6, 3101), "512", 800)
    assert f["steps"] == (2301, 3101, 4000) and f["n"] == (8000, 8990)
    assert f["before"] == pytest.approx(1.75) and f["last"] == pytest.approx(1.6) and f["fall"] == pytest.approx(0.15)
    assert bf.band_fall([dict(step=s, value_bits={"512": 2.0}) for s in (1, 2001, 4000)], "512", 800) is None   # no counts: no window
    assert bf.band_fall(_curve(1000, 2.0, 1.0, 500), "512", 800) is None                                      # too short for two windows


def test_the_rules():
    a, b = STAGES[0], STAGES[1]
    assert bc.decide(a, 1, dict(formation=dict(formed=True, groups_judged=3)))["action"] == "advance"
    assert bc.decide(a, 1, dict(formation=dict(formed=False, groups_judged=3)))["action"] == "again"
    d = bc.decide(a, 1, dict(formation=dict(formed=True, groups_judged=0)))     # no group read on enough trials: no verdict, no copy
    assert d["action"] == "again" and "no lookup group" in d["reason"]
    d = bc.decide(a, 2, dict(formation=dict(formed=False, groups_judged=3)))
    assert d["action"] == "advance" and d["mark"] == "not formed after 2 blocks"
    assert bc.decide(b, 1, dict(curve=_curve(4000, 2.15, 2.0, 3101)))["action"] == "again"
    assert bc.decide(b, 1, dict(curve=_curve(4000, 2.05, 2.0, 3101)))["action"] == "advance"
    d = bc.decide(b, 5, dict(curve=_curve(4000, 2.15, 2.0, 3101)))
    assert d["action"] == "advance" and d["mark"] == "still improving after 5 blocks"
    assert bc.decide(b, 1, dict(curve=[]))["mark"] == "no far-band window"
    assert bc.decide(STAGES[2], 1, {})["action"] == "advance"


def test_walk_follows_the_records():
    recs = {}

    def put(block, init, **kw):
        st = next(s for s in STAGES if block.startswith(s["name"]) and block[len(s["name"]):].isdigit())
        spec = bc.block_spec(BASE, st, "castor-both", 1, int(block[len(st["name"]):]), init)
        tag = bc.spec_tag(spec)
        recs[tag] = _record(spec, **kw)
        return tag

    t = put("A1", None, formed=False)
    done, nxt = bc.walk(STAGES, BASE, "castor-both", 1, recs.get)
    assert [b["block"] for b in done] == ["A1"] and nxt["block"] == "A2" and nxt["init"] == t
    t = put("A2", t, formed=False)
    t = put("B1", t, fall=0.15)
    t = put("B2", t, fall=0.05)
    done, nxt = bc.walk(STAGES, BASE, "castor-both", 1, recs.get)
    assert [b["block"] for b in done] == ["A1", "A2", "B1", "B2"] and nxt["block"] == "anneal1" and nxt["init"] == t
    assert done[1]["decision"]["mark"] == "not formed after 2 blocks" and done[2]["decision"]["action"] == "again"
    put("anneal1", t)
    done, nxt = bc.walk(STAGES, BASE, "castor-both", 1, recs.get)
    assert nxt is None and len(done) == 5


def test_walk_refuses_a_record_from_another_start():
    a1 = bc.spec_tag(bc.block_spec(BASE, STAGES[0], "castor-both", 1, 1, None))
    b1 = bc.block_spec(BASE, STAGES[1], "castor-both", 1, 1, "castor-both_s1_somewhere_else")
    recs = {a1: _record(bc.block_spec(BASE, STAGES[0], "castor-both", 1, 1, None), formed=True), bc.spec_tag(b1): _record(b1)}
    with pytest.raises(RuntimeError, match="continued from castor-both_s1_somewhere_else"):
        bc.walk(STAGES, BASE, "castor-both", 1, recs.get)


def test_the_loop_end_to_end_and_its_resume(monkeypatch, tmp_path):
    """The runner replaced by one that writes the block's record: Castor's copy forms at once and its far band settles at once; Pollux
    takes a second A block and three B blocks. A second call runs nothing, a missing record is run again."""
    calls = []
    plan = {("pollux-both", "A1"): dict(formed=False), ("pollux-both", "B1"): dict(fall=0.3), ("pollux-both", "B2"): dict(fall=0.2)}

    def fake_run(spec, out, **kw):
        calls.append((spec["arms"], spec["stage"], spec.get("init")))
        rec = _record(spec, **plan.get((spec["arms"], spec["stage"]), {}))
        os.makedirs(os.path.join(out, "bed"), exist_ok=True)
        json.dump(rec, open(os.path.join(out, "bed", f"bed_{bc.spec_tag(spec)}.json"), "w", encoding="utf-8"))
        return dict(failed=None, stopped=False)
    monkeypatch.setattr(bc.br, "run", fake_run)
    monkeypatch.setattr(bc, "fetch_record", lambda *a, **k: False)
    res = bc.run(STAGES, BASE, "castor-both,pollux-both", "1", str(tmp_path), name="t")
    assert res["done"] and res["failed"] is None and len(res["blocks_run"]) == 3 + 6
    assert [c[1] for c in calls if c[0] == "castor-both"] == ["A1", "B1", "anneal1"]
    assert [c[1] for c in calls if c[0] == "pollux-both"] == ["A1", "A2", "B1", "B2", "B3", "anneal1"]
    assert [c[1] for c in calls[:4]] == ["A1", "A1", "B1", "A2"]                    # rounds: every chain one block before the next round
    assert (tmp_path / "bed" / "CURRICULUM_t.md").exists() and "| pollux-both s1 | A2 |" in res["table"]
    assert json.load(open(tmp_path / "curriculum_t.json", encoding="utf-8"))["done"] is True
    calls.clear()
    assert bc.run(STAGES, BASE, "castor-both,pollux-both", "1", str(tmp_path), name="t")["blocks_run"] == [] and calls == []
    assert bc.report(STAGES, BASE, "castor-both,pollux-both", "1", str(tmp_path)).endswith("Every chain finished its last stage.\n")
    b2 = bc.spec_tag(bc.block_spec(BASE, STAGES[1], "pollux-both", 1, 2, "x"))
    os.remove(tmp_path / "bed" / f"bed_{b2}.json")
    rep = bc.report(STAGES, BASE, "castor-both,pollux-both", "1", str(tmp_path))
    assert "To run next: pollux-both s1 B2." in rep and rep.count("| castor-both s1 |") == 3 and rep.count("| pollux-both s1 |") == 3
    res = bc.run(STAGES, BASE, "castor-both,pollux-both", "1", str(tmp_path), name="t")
    assert [c[1] for c in calls] == ["B2"] and res["done"]


def test_stop_starts_no_block(monkeypatch, tmp_path):
    monkeypatch.setattr(bc.br, "run", lambda *a, **k: pytest.fail("a block started under STOP"))
    monkeypatch.setattr(bc, "fetch_record", lambda *a, **k: False)
    (tmp_path / "STOP").write_text("", encoding="utf-8")
    res = bc.run(STAGES, BASE, "castor-both", "1", str(tmp_path), name="t")
    assert res["stopped"] and not res["done"] and res["next"] == {"castor-both s1": "A1"}


def test_dry_lists_every_block(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(bc, "fetch_record", lambda *a, **k: False)
    res = bc.run(STAGES, BASE, "castor-both", "1,2", str(tmp_path), dry=True)
    text = capsys.readouterr().out
    assert res["next"] == {"castor-both s1": "A1", "castor-both s2": "A1"} and res["blocks_run"] == []
    assert text.count("may run castor-both") == 16 and "_stanneal1" in text
