# -*- coding: utf-8 -*-
"""THE ART THOUGHT DATASET's tooling (0.1.59): the trie and its lists, the seals, the validator's rules on the example and on broken rows,
the series check, the prompt, the checker, the document kept equal to the module's."""
import copy
import json
import os

import pytest

from geolip.nano_beatrix import thought_pack as TP

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_the_trie_and_its_lists():
    ks = TP.keys()
    assert "look hair" in ks and "form composition" in ks and "relation" in ks and "mood evidence" in ks and len(ks) == 1 + sum(len(a) for a in TP.SYSTEMS.values())
    for name, vals in TP.VOCAB.items():
        assert vals and len(vals) == len(set(vals)), name
        for v in vals:
            assert v == v.lower().strip() and len(v.encode("ascii")) <= TP.VAL_MAX, (name, v)
    for k, lst in TP.KEY_LIST.items():
        assert k in ks and lst in TP.VOCAB, k
    assert all(len(k.encode()) <= TP.KEY_MAX for k in ks)
    assert "blonde" in TP.VOCAB["eyes"] and TP.VOCAB["eyes"] is TP.VOCAB["colour"]


def test_the_seals_are_deterministic_and_disjoint_from_the_training_lists():
    a, b = TP.seal(), TP.seal()
    assert a == b and "colour" in a and "eyes" not in a and all(a[n] for n in a)
    for name, vals in a.items():
        assert len(vals) == max(1, round(len(TP.VOCAB[name]) * 0.1)) and set(vals) <= set(TP.VOCAB[name])
    train, held = TP.lists_for(a, "train"), TP.lists_for(a, "held")
    for name in TP.VOCAB:
        assert not (set(train[name]) & set(held[name])) and set(train[name]) | set(held[name]) == set(TP.VOCAB[name])
    assert held["eyes"] == held["colour"] and TP.seal(seed=7) != a


def test_the_example_passes_and_broken_rows_fail_the_right_rule():
    held = TP.seal()
    ex = copy.deepcopy(TP.EXAMPLE)
    assert TP.validate(ex, held) == [], TP.validate(ex, held)
    assert TP.n_records(ex["spec"]) == 14 and ("rel1 rel", "in front of") in TP.records(ex["spec"])
    assert not ({v for _, v in TP.records(ex["spec"])} & {v for vals in held.values() for v in vals})

    def broken(mut):
        r = copy.deepcopy(TP.EXAMPLE)
        mut(r)
        return {rule for rule, _ in TP.validate(r, held)}
    assert 2 in broken(lambda r: r["spec"]["look"].__setitem__("hair", "a very long description of the hair"))
    assert 2 in broken(lambda r: r["spec"]["look"].__setitem__("hair", "Short"))
    assert 2 in broken(lambda r: r["spec"].__setitem__("smell", {"scent": "salt"}))
    assert 2 in broken(lambda r: r["spec"]["relation"].append({"a": "x", "rel": "near", "b": "y"}))
    assert 3 in broken(lambda r: r.__setitem__("thoughts", r["thoughts"][:1]))
    assert 3 in broken(lambda r: r.__setitem__("depth", "deep"))
    assert 1 in broken(lambda r: r["thoughts"].append({"step": "guess", "text": "x"}))
    assert 1 in broken(lambda r: r["wordings"].__setitem__("caption", "as seen in the image, a woman"))
    assert 4 in broken(lambda r: r["questions"].append({"key": "look eyes", "value": "blue", "needs": "record"}))
    assert 4 in broken(lambda r: r["questions"].append({"key": "look hair", "value": "unknown", "needs": "unknown"}))
    assert 4 in broken(lambda r: r["questions"].append({"key": "mood feeling", "value": "furious", "needs": "chain"}))
    assert 5 in broken(lambda r: r.__setitem__("wordings", {}))
    assert 6 in broken(lambda r: r["spec"]["mood"].pop("evidence"))
    assert 6 in broken(lambda r: r["spec"]["mood"].__setitem__("evidence", "shot lens"))
    assert 4 in broken(lambda r: r["questions"].append({"key": "look hair", "value": "short", "needs": "chain"}))
    assert 7 in broken(lambda r: r["spec"]["look"].__setitem__("hair", held["hair"][0]))
    assert 8 in broken(lambda r: r["source"].pop("license"))
    assert 0 in broken(lambda r: r.pop("held"))
    assert set(TP.RULES) == set(range(1, 9))                # the plan's eight rules; a content rule for generated pictures is not the dataset's
    deep = copy.deepcopy(TP.EXAMPLE)
    deep["depth"] = "deep"
    rules = {rule for rule, _ in TP.validate(deep, held)}
    assert rules == {3}                                     # 14 records reach deep; 5 thoughts and 4 questions do not; the wordings and the question kinds do


def test_the_series_check_and_the_file_checker(tmp_path):
    held = TP.seal()
    simple = copy.deepcopy(TP.EXAMPLE)
    simple.update(id="art-000122", depth="simple", rank=1, spec={"person": {"gender": "woman"}, "look": {"hair": "short", "expression": "wry"}},
                  thoughts=[{"step": "see", "text": "A woman with short hair and a wry smile."}],
                  questions=[{"key": "look hair", "value": "short", "needs": "record"}], wordings={"caption": "A woman with short hair and a wry smile."})
    assert TP.validate(simple, held) == []
    assert TP.validate_series([simple, TP.EXAMPLE]) == []
    odd = copy.deepcopy(simple)
    odd["spec"]["look"]["eyes"] = "green"
    assert [r for r, _ in TP.validate_series([odd, TP.EXAMPLE])] == [7]
    assert [r for r, _ in TP.validate_series([simple, simple])] == [7]
    p = tmp_path / "rows.jsonl"
    bad = copy.deepcopy(TP.EXAMPLE)
    bad["id"], bad["series"] = "art-000124", "s-00042"
    bad["spec"]["mood"].pop("evidence")
    with open(p, "w", encoding="utf-8") as fh:
        for r in (simple, TP.EXAMPLE, bad):
            fh.write(json.dumps(r) + "\n")
        fh.write("{not json\n")
    res = TP.check_file(str(p), held)
    assert res["rows"] == 3 and res["ok"] == 2 and res["unparsed"] == 1 and res["by_rule"] == {0: 1, 6: 1} and res["by_depth"] == {"simple": 1, "medium": 2}
    assert any(n == 4 and rule == 0 for n, rule, _ in res["problems"])


def test_the_sampler_draws_valid_nested_specs():
    import numpy as np
    held = TP.seal()
    lists = TP.lists_for(held, "train")
    known = set(TP.keys())
    assert TP.list_of("look eyes") == "colour" and TP.list_of("rel2 rel") == "relation" and TP.list_of("rel1 b") == "thing" and TP.list_of("form palette") is None
    assert TP.sealed_used([("body gesture", "holding"), ("rel1 rel", "holding")], {"gesture": ["holding"], "relation": []}) == [("body gesture", "holding")]
    for seed in range(20):
        rng = np.random.default_rng(seed)
        tri = TP.sample_series(rng, lists)
        for depth in TP.DEPTHS:
            spec = tri[depth]
            lo, hi = TP.SAMPLE_RECORDS[depth]
            assert lo <= TP.n_records(spec) <= hi, (depth, TP.n_records(spec))
            recs = TP.records(spec)
            for k, v in recs:
                assert (k in known or k.startswith("rel")) and len(v.encode()) <= TP.VAL_MAX and v == v.lower(), (k, v)
                lst = TP.KEY_LIST.get(k)
                assert not lst or v in TP.VOCAB[lst], (k, v)
            assert TP.sealed_used(recs, held) == [], (depth, TP.sealed_used(recs, held))
            assert len(spec.get("relation", [])) == TP.SAMPLE_RELATIONS[depth]
        assert set(TP.records(tri["simple"])) <= set(TP.records(tri["medium"])) <= set(TP.records(tri["deep"]))
    row = copy.deepcopy(TP.EXAMPLE)
    rng = np.random.default_rng(3)
    assert TP.sample_spec(rng, "simple", lists) and "deep" in TP.sample_series(rng, TP.lists_for(held, "held"))
    pr = TP.prompt("medium", "painting", held, seed_spec=TP.sample_spec(np.random.default_rng(1), "medium", lists))
    assert "keep every record as given" in pr


def test_the_vocab_files_the_prompt_and_the_doc(tmp_path):
    held = TP.write_vocab(str(tmp_path))
    trie = json.load(open(tmp_path / "vocab" / "trie.json", encoding="utf-8"))
    hj = json.load(open(tmp_path / "vocab" / "held.json", encoding="utf-8"))
    assert trie["keys"] == TP.keys() and trie["counts"]["deep"]["records"] == [14, 30] and hj["held"] == held and trie["version"] == TP.VERSION
    pr = TP.prompt("medium", "photograph", held)
    assert "THE RULES" in pr and json.dumps(TP.EXAMPLE, ensure_ascii=False) in pr and "6-14 records" in pr
    assert "ADULTS ONLY" not in pr and "ADULTS ONLY" not in TP.PROMPT_DOC
    def offered(text, name):
        line = [ln for ln in text.split("\n") if ln.startswith(f"  {name}: ")]
        return set(line[0][len(name) + 4:].split(", ")) if line else set()
    for name in ("setting", "relation", "colour", "feeling"):
        assert offered(pr, name) and not (offered(pr, name) & set(held[name])), name   # the sealed values are not offered to a training row's writer
        assert offered(pr, name) | set(held[name]) == set(TP.VOCAB[name])
    ph = TP.prompt("deep", "painting", held, split="held", series=True, n=2)
    assert "2 series of three rows" in ph and offered(ph, "setting") == set(held["setting"])
    doc = tmp_path / "doc.md"
    TP.write_doc(str(doc))
    assert open(doc, encoding="utf-8").read() == TP.PROMPT_DOC
    shipped = os.path.join(ROOT, "docs", "thought_generation_prompt.md")
    assert os.path.exists(shipped) and open(shipped, encoding="utf-8").read() == TP.PROMPT_DOC, "docs/thought_generation_prompt.md is stale: rerun --doc"
    assert TP.main(["--vocab=" + str(tmp_path / "v2"), "--prompt=1", "--depth=simple", "--domain=design"]) == 0
    assert os.path.basename(__file__) == "test_thought_pack.py"
