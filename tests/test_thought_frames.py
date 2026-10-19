# -*- coding: utf-8 -*-
"""THE ART THOUGHT DATASET in the frame grammar (0.1.62): the chain frame (the depth's letter, the kinds as first records, the text in order
across the slices), the spec frame (the trie, every record recoverable), the four layouts, the window, the loader, the framed diet's kind."""
import copy
import json
import os

import numpy as np
import pytest

from geolip.alephllm.data.special_tokens import CUE, DATA, DOC, END, SEP, THINK
from geolip.nano_beatrix import bed_frames as bf
from geolip.nano_beatrix import frames_data as FD
from geolip.nano_beatrix import caption_pack as cp
from geolip.nano_beatrix import thought_frames as TF
from geolip.nano_beatrix import thought_pack as TP
from test_caption import pack_dir  # noqa: F401  (the module's small caption pack, for the framed stream's caption rows)
from test_carry import corpus  # noqa: F401  (test_caption's fixture needs it)


def _rows():
    simple = copy.deepcopy(TP.EXAMPLE)
    simple.update(id="art-000122", depth="simple", rank=1, spec={"person": {"gender": "woman"}, "look": {"hair": "short", "expression": "wry"}},
                  thoughts=[{"step": "see", "text": "A woman with short hair and a wry smile."}],
                  questions=[{"key": "look hair", "value": "short", "needs": "record"}], wordings={"caption": "A woman with short hair and a wry smile."})
    deep = copy.deepcopy(TP.EXAMPLE)
    deep.update(id="art-000124", depth="deep", rank=3)
    deep["spec"]["form"].update(palette="yellow and teal", contrast="high", medium="photograph", style="realism")
    deep["spec"]["intent"] = {"purpose": "editorial", "for": "a travel feature", "genre": "street"}
    deep["spec"]["story"] = {"moment": "a wait"}
    deep["thoughts"] = deep["thoughts"][:4] + [{"step": kind, "text": f"Thought {kind}: " + "a sentence of some length that fills the run. " * 4}
                                               for kind in ("light", "colour", "story", "compare", "check", "say")]
    deep["wordings"]["critique"] = "The yellow carries the frame; the stall could sit lower."
    deep["questions"] += [{"key": "story moment", "value": "a wait", "needs": "record"}]
    return simple, copy.deepcopy(TP.EXAMPLE), deep


def test_the_chain_frame_carries_the_depth_and_the_thoughts_in_order():
    simple, medium, deep = _rows()
    for row, letters in ((simple, "cd"), (medium, "e"), (deep, "fg")):
        fb, m = TF.chain_frame(np.random.default_rng(1), row["thoughts"], row["depth"])
        assert fb[0] == THINK and chr(fb[1]) in letters and fb[-1] == END and len(fb) == m["size"] + 3 and m["letter"] == chr(fb[1])
        got = TF.chain_text(fb)
        assert [k for k, _ in got] == [t["step"] for t in row["thoughts"]]
        for (k, text), t in zip(got, row["thoughts"]):
            want = t["text"].strip().encode()
            assert text == want[:len(text)] and (text == want or m["cut"] > 0), (k, text[:40], want[:40])
        assert sum(m["slices"]) * 32 <= m["size"]
    long = [{"step": "see", "text": "x" * 3000} for _ in range(3)]
    fb, m = TF.chain_frame(np.random.default_rng(0), long, "deep")
    assert m["size"] == 2048 and m["cut"] > 0 and sum(m["slices"]) == 64
    fb2, m2 = TF.chain_frame(np.random.default_rng(0), long, "deep")
    assert fb == fb2


def test_the_spec_frame_is_the_trie_and_every_record_comes_back():
    simple, medium, deep = _rows()
    for row in (simple, medium, deep):
        recs = [(k.encode(), v.encode()) for k, v in TP.records(row["spec"])]
        fb, kept, dropped = TF.spec_frame(np.random.default_rng(2), recs)
        assert fb[0] == DATA and fb[-1] == END and dropped == 0 and kept == recs
        assert TF.records_in(fb) == recs
        assert (len(fb) - 3) % 32 == 0 or len(fb) == 32
    many = [(b"form " + f"k{i:02d}".encode(), b"v" * 15) for i in range(20)]        # 20 records of 24 bytes: one a slice, sixteen kept
    fb, kept, dropped = TF.spec_frame(np.random.default_rng(0), many)
    assert dropped == 4 and len(kept) == 16 and TF.records_in(fb) == many[:16]


def test_the_four_layouts_and_the_question():
    simple, medium, deep = _rows()
    rng = np.random.default_rng(3)
    a, ma = TF.render(rng, medium, "A", ask=1)
    assert a.startswith(TF.HEAD) and a.count(bytes([THINK])) > 0 and a.count(bytes([DATA])) > 0
    assert a.endswith(bytes([CUE]) + b"rel1 b" + bytes([SEP]) + b"fish stall" + bytes([END])) and ma["asked"] == ("rel1 b", "fish stall", "record")
    assert a.find(bytes([THINK])) < a.find(bytes([DATA])) < a.find(bytes([CUE]) + b"say" + bytes([SEP]) + b"caption\n")
    recs = TF.records_in(a)
    assert set(recs) >= {(k.encode(), v.encode()) for k, v in TP.records(medium["spec"])}
    assert (b"say", b"caption") in recs and (b"say", b"tags") in recs and (b"step", b"see") in recs
    assert TF.chain_text(a) == [(t["step"], t["text"].encode()) for t in medium["thoughts"]]
    b, mb = TF.render(rng, medium, "B", ask=0)
    assert b.count(bytes([THINK])) == 0 and b.find(bytes([CUE]) + b"say") < b.find(bytes([DATA])) and b.endswith(b"short" + bytes([END]))
    c, mc = TF.render(rng, medium, "C", ask=3)
    assert c.count(bytes([THINK])) == 0 and c.count(bytes([DATA])) == 0 and c.endswith(b"unknown" + bytes([END])) and "chain" not in mc
    d, md = TF.render(rng, medium, "D", ask=2)
    assert d[len(TF.HEAD):].startswith(bytes([CUE]) + b"scene layout" + bytes([SEP]) + b"\n") and d.endswith(b"single subject" + bytes([END]))
    assert d.find(bytes([THINK])) < d.find(bytes([DATA]))
    assert TF.chain_text(d)[0] == ("see", medium["thoughts"][0]["text"].encode())
    with pytest.raises(AssertionError):
        TF.render(rng, medium, "E")


def test_the_window_the_pack_and_the_framed_kind(tmp_path, pack_dir):  # noqa: F811
    simple, medium, deep = _rows()
    held_row = copy.deepcopy(medium)
    held_row.update(id="art-000900", series="s-00900")
    held_row["held"] = {"layout": None, "wordings": False, "values": ["aqua"]}
    os.makedirs(tmp_path / "data" / "medium" / "photograph")
    os.makedirs(tmp_path / "held")
    with open(tmp_path / "data" / "medium" / "photograph" / "photograph-medium-00000.jsonl", "w", encoding="utf-8") as fh:
        for r in (simple, medium, deep, held_row):
            fh.write(json.dumps(r) + "\n")
    with open(tmp_path / "held" / "held-00000.jsonl", "w", encoding="utf-8") as fh:
        fh.write(json.dumps(held_row) + "\n")
    pack = TF.ThoughtPack.from_folder(str(tmp_path), "train")
    assert pack.n == 3 and {r["id"] for r in pack.rows} == {"art-000122", "art-000123", "art-000124"}
    assert TF.ThoughtPack.from_folder(str(tmp_path), "held").n == 1
    assert pack.check(TP.seal()) == {}
    w, metas = TF.thought_window(np.random.default_rng(5), pack)
    assert len(w) == FD.ROW_CTX + 1 and len(metas) >= 2 and w.startswith(TF.HEAD) and all(m["layout"] != "D" for m in metas)
    drawn = {m["layout"] for s in range(6) for m in TF.thought_window(np.random.default_rng(s), pack)[1]}
    assert drawn <= {"A", "B", "C"} and len(drawn) >= 2
    w2, _ = TF.thought_window(np.random.default_rng(5), pack)
    assert w == w2
    os.environ[TF.PACK_ENV] = str(tmp_path)
    try:
        assert TF.load().n == 3 and TF.load(split="held").n == 1
        from geolip.nano_beatrix import framed_diet as fd
        from test_framed_diet import FakeInner
        assert "thought" in fd.COUNTED and fd.FORMS["p50t"]["kinds"][-1][0] == "thought" and "thought" not in dict(fd.FORMS["p50"]["kinds"])
        st = fd.FramedStream(FakeInner(), 2, 7, fd.FORMS["p50t"], pack=cp.load(pack_dir))
        kinds = []
        for _ in range(60):
            ids, kind = st.row()
            kinds.append(kind)
            assert ids.shape == (fd.ROW_IDS,) and (kind in ("plain", "note_plain") or ids[-1] == DOC)
            if kind == "thought":
                assert bytes(ids[:len(TF.HEAD)].astype(np.uint8)) == TF.HEAD
        assert kinds.count("thought") >= 3 and st.counts["thought"] == kinds.count("thought")
    finally:
        del os.environ[TF.PACK_ENV]
