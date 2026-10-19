# -*- coding: utf-8 -*-
"""The caption phase (0.1.55): the pack's records (path keys of the symbolic trie) and wordings from rows of every source, the cache and
its loader, the caption frame (flat and nested by system), the document and the window (the shuffle, the question half, the control's
swap), the stream rows beside the mixed arm's, the registered streams, the crafts and their plans, the caption reads with a scripted
model. CPU only; the Hub is never touched."""
import json
import os

import numpy as np
import pytest
import torch

from geolip.alephllm.data.special_tokens import CUE, DATA, DOC, END, SEP
from geolip.nano_beatrix import caption_pack as cp
from geolip.nano_beatrix import caption_screen as cs
from geolip.nano_beatrix import frames_data as fd
from geolip.nano_beatrix import nano_v1

from test_carry import corpus  # noqa: F401  (the module's corpus fixture)

DFF = {"id": "dff:deepfashion_2", "source": "AbstractPhil/qwen-deepfashion-fused", "kind": "photo", "licence": "apache-2.0", "split": "train",
       "tags": "1girl, solo, realistic, shirt, long hair, brown eyes, brown hair, white shirt, pants, looking at viewer",
       "short": "A young woman with long brown hair stands in a doorway, wearing a white t-shirt with a landscape print and dark jeans.",
       "long": "This photograph features a young woman standing in a brightly lit, blue-painted room filled with shelves of ceramic pots.",
       "fused": "11 vases and one person. The primary subject is a person in the middle center of the frame, with long wavy dark brown hair.",
       "intent": "A Latina woman wearing a white t-shirt with a painting on it, in a ceramics and pottery studio.",
       "photo": "tall vertical full-length shot; cool blue-hour light; editorial color fashion shot, whole standing figure framed head to toe",
       "rating": "general", "attributes": {"hair": "dark brown hair", "eye": "light brown eyes", "expression": "with a thoughtful expression",
                                           "makeup": "with winged eyeliner", "jewelry": "wearing no jewelry", "gender": "woman", "age_band": "25-35"},
       "strata": {"clothing": ["emerald green satin dress", "sleeveless dress"], "accessory": ["earrings", "shoes"], "skin": ["light skin tone"],
                  "pose": ["leaning against a doorframe"], "body": ["collarbone"]},
       "scene": {"setting": "pottery studio", "mood": "calm", "layout": "centered"}, "age_flag_pre": {}, "audit": "approved"}
DAN = {"id": "danbooru:2719067", "source": "BootsofLagrangian/danbooru-multitier-captions-202606", "kind": "illustration", "licence": "cc-by-4.0",
       "split": "train", "tags": "mizuho (kancolle), cleavage, white background, closed mouth, green hair, yellow eyes, long hair, smile",
       "short": "C.C. and Mizuho in green hakama and white top sit seiza against white background, holding pizza.",
       "medium": "C.C. from Code Geass and Mizuho sit in seiza against a stark white background. C.C. has long green hair and yellow eyes.",
       "long": "C.C. from Code Geass, depicted with Mizuho, sits in seiza against a stark white background with long green hair.",
       "rating": "sensitive", "age_flag_pre": {"booru": ["1girl"]}, "audit": "approved"}
CC = {"id": "cc12m:166667", "source": "AbstractPhil/conceptual-captions-12m-webdataset-berts", "kind": "web image", "licence": "not recorded",
      "split": "train", "long": "A two-story brick house with a gabled roof and a prominent front door flanked by sidelights.",
      "short": "A two-story brick house with gabled roof, prominent front door, and stone detailing, set against a clear sky.",
      "caption": "CBC-Brown-Covington-KS, the exterior brick color Roof Colors, Brick Colors", "age_flag_pre": {}, "audit": "approved"}
COCO = {"id": "coco:task_1:0", "source": "AbstractPhil/json-coco-format", "kind": "photo", "licence": "cc-by-4.0", "split": "train",
        "caption": "A long restaurant table with rattan rounded back chairs.",
        "structure": "{\"subjects\":[{\"name\":\"restaurant table\",\"attributes\":[\"long\"]},{\"name\":\"chairs\",\"attributes\":[\"rattan\",\"rounded back\"]}],\"actions\":[],\"setting\":\"indoor\"}",
        "age_flag_pre": {}, "audit": "approved"}


def test_records_are_path_keys_of_the_trie_and_the_wordings_are_the_captions():
    r = dict(cp.records_of(DFF))
    assert r[b"look hair"] == b"dark brown" and r[b"look eyes"] == b"light brown" and r[b"look expression"] == b"thoughtful"
    assert r[b"look makeup"] == b"winged eyeliner" and r[b"look jewelry"] == b"none" and r[b"person gender"] == b"woman" and r[b"person age"] == b"25-35"
    assert r[b"wear clothing"] == b"emerald green" and r[b"wear accessory"] == b"earrings" and r[b"scene setting"] == b"pottery studio"
    assert r[b"scene mood"] == b"calm" and b"shot light" in r and all(len(k) + len(v) <= cp.REC_MAX for k, v in r.items())
    assert {cp.system_of(k).decode() for k in r} <= set(cp.SYSTEMS)
    d = dict(cp.records_of(DAN))
    assert d == {b"look hair": b"green", b"look eyes": b"yellow", b"look expression": b"closed mouth"}
    assert cp.records_of(CC) == [] and dict(cp.records_of(COCO)) == {b"thing chairs": b"rattan"}
    assert len(cp.captions_of(DFF)) == 4 and len(cp.captions_of(DAN)) == 3 and len(cp.captions_of(CC)) == 3 and len(cp.captions_of(COCO)) == 1
    assert cp.captions_of(DFF)[0].startswith(b"A young woman") and cp.norm_value("look hair", "Blonde Hair") == b"blonde"
    assert cp.norm_value("look expression", "with a very long and winding expression of some kind") == b"very long"       # the room of a 15-byte key is 12 bytes; cut at a word boundary
    assert cp.norm_value("look eyes", "") is None and cp.norm_value("look hair", "café hair") is None


@pytest.fixture(scope="module")
def pack_dir(tmp_path_factory):
    """A pack from the sample rows: 40 copies of each source with distinct ids, one in eight held; built from files already here."""
    d = tmp_path_factory.mktemp("caption_pack")
    raw = d / "raw"
    raw.mkdir()
    files = {}
    for src, row in (("dff", DFF), ("scf", dict(DFF, id="scf:1")), ("danbooru", DAN), ("cc12m", CC), ("coco", COCO)):
        p = raw / f"{src}-sample.jsonl"
        with open(p, "w", encoding="utf-8") as fh:
            for i in range(40):
                r = dict(row, id=f"{src}:{i}", split="held" if i % 8 == 0 else "train")
                fh.write(json.dumps(r) + "\n")
        files[src] = [str(p)]
    stats = cp.build(str(d), files=files, log=lambda *a: None)
    assert stats["train_docs"] == 5 * 35 and stats["held_docs"] == 5 * 5 and stats["counts"]["cc12m_no_records"] == 40
    return str(d)


def test_the_cache_loads_and_draws_by_kind(pack_dir):
    pk = cp.load(pack_dir)
    assert pk.n == 175 and set(pk.kinds) == {"photo", "cc12m", "danbooru", "coco"}
    i = int(pk.by_kind["photo"][0])
    assert dict(pk.records(i))[b"look hair"] == b"dark brown" and len(pk.captions(i)) == 4 and pk.kind(i) == "photo"
    j = int(pk.by_kind["cc12m"][0])
    assert pk.records(j) == [] and len(pk.captions(j)) == 3
    rng = np.random.default_rng(1)
    draws = [pk.kind(pk.draw(rng)) for _ in range(4000)]
    assert abs(draws.count("photo") / 4000 - 0.45) < 0.04 and abs(draws.count("cc12m") / 4000 - 0.275) < 0.04
    held = cp.load(pack_dir, "held")
    assert held.n == 25 and sum(1 for k in range(held.n) if held.records(k)) == 20
    assert cp.ensure(pack_dir)["train_docs"] == 175                                                   # the cache stands; nothing rebuilt


def _records_in(frame: bytes) -> list:
    """Every CUE key SEP value in a frame's bytes (a value runs to the next special or space run)."""
    out, i = [], 0
    while True:
        i = frame.find(bytes([CUE]), i)
        if i < 0:
            return out
        j = frame.index(bytes([SEP]), i)
        k = j + 1
        while k < len(frame) and frame[k] not in (CUE, END, DATA) and not frame[k:k + 2] == b"  ":
            k += 1
        out.append((frame[i + 1:j], frame[j + 1:k].rstrip(b" ")))
        i = k


def test_the_caption_frame_is_flat_for_one_system_and_nested_by_system_otherwise():
    rng = np.random.default_rng(0)
    one = [(b"look hair", b"green")]
    fb, kept = fd.caption_frame(rng, one)
    assert len(fb) == 32 and fb[0] == DATA and fb[1:2] == b"a" and fb[-1] == END and kept == one and _records_in(fb) == one
    two = one + [(b"look eyes", b"yellow")]                                   # 33 bytes of records: two slices, one system, flat
    fb, kept = fd.caption_frame(rng, two)
    assert len(fb) == 67 and fb[1:2] == b"b" and fb[2] == DATA and fb[3:4] == b"a" and kept == two and _records_in(fb) == two
    recs = cp.records_of(DFF)
    fb, kept = fd.caption_frame(rng, recs)
    systems = []
    for k, _ in recs:
        s = cp.system_of(k)
        if s not in systems:
            systems.append(s)
    assert fb[0] == DATA and fb[1:2] in (b"c", b"d", b"e") and fb[-1] == END and len(fb) in (131, 259, 515)      # a sliced frame: size + 3
    assert dict(_records_in(fb)) == dict(kept) and set(kept) <= set(recs) and len(kept) >= len(systems)
    flat, kept2 = fd.caption_frame(rng, recs, trie=False)
    assert len(flat) in (32, 67, 131) and len(kept2) <= len(recs)


def test_the_document_and_the_window(pack_dir):
    pk = cp.load(pack_dir)
    i = int(pk.by_kind["photo"][0])
    asked = n_small = 0
    seen_orders = set()
    for s in range(120):
        d, m = fd.caption_doc(np.random.default_rng(s), pk, i)
        assert d.startswith(fd.CAPTION_HEAD) and any(c in d for c in pk.captions(i)) and 1 <= len(m["records"]) <= fd.CAPTION["max_records"]
        seen_orders.add(tuple(k for k, _ in m["records"]))
        if m["asked"]:
            asked += 1
            key, val = m["asked"]
            assert d.endswith(bytes([CUE]) + key.encode() + bytes([SEP]) + val.encode() + bytes([END])) and (key, val) in m["records"]
        n_small += len(m["records"]) < len(pk.records(i))
    assert 0.35 < asked / 120 < 0.65 and len(seen_orders) > 20 and n_small > 0                     # the question half; the shuffle; the subset
    j = int(pk.by_kind["cc12m"][0])
    d, m = fd.caption_doc(np.random.default_rng(1), pk, j)
    assert m["records"] == [] and m["asked"] is None and DATA not in d                               # a web image: the plain caption
    d, m = fd.caption_doc(np.random.default_rng(2), pk, j, fd.CAPTION_CTRL, swap_from=i)              # the control: another document's frame
    assert m["records"] and all((k.encode(), v.encode()) in pk.records(i) for k, v in m["records"]) and DATA in d
    row, metas = fd.caption_window(np.random.default_rng(3), pk)
    assert len(row) == fd.ROW_BYTES and len(metas) >= 2 and row[:len(fd.CAPTION_HEAD)] == fd.CAPTION_HEAD
    assert [m["start"] for m in metas] == sorted(m["start"] for m in metas) and all(row[m["start"]] == DOC for m in metas if m["start"] < len(row))


def test_the_caption_stream_keeps_the_text_the_coin_and_the_carry_rows_of_the_mixed_arm(corpus, pack_dir):  # noqa: F811
    packed, pk = fd.load_corpus(corpus), cp.load(pack_dir)
    carries = windows = plain = 0
    for i in range(60):
        m, fm = fd.row_bytes(packed, 7, i, fd.P_FRAMED, fd.MIXED, carry="mixed")
        c, fc = fd.row_bytes(packed, 7, i, fd.P_FRAMED, fd.CAPTION, carry="caption", pack=pk)
        x, fx = fd.row_bytes(packed, 7, i, fd.P_FRAMED, fd.CAPTION_CTRL, carry="caption_ctrl", pack=pk)
        assert fc == fm == fx and len(c) == len(x) == fd.ROW_BYTES
        if not fc:
            plain += 1
            assert c == m == x                                                                       # the same plain row, document start included
            continue
        kind, reg = fd.mixed_draw(np.random.default_rng([7, i, 2]), fd.CAPTION)
        if kind == "carry":
            carries += 1
            assert c == x == m                                                                              # the mixed arm's own carry row (one coin, one register)
        else:
            windows += 1
            assert c.startswith(fd.CAPTION_HEAD) and any(cap[:24] in c for k in range(pk.n) for cap in pk.captions(k))
    assert plain and carries and windows and abs(carries / (carries + windows) - 1 / 3) < 0.25
    os.environ[cp.PACK_ENV] = pack_dir
    os.environ[fd.CORPUS_ENV] = corpus
    gen = fd._rows_caption(7)
    first = next(gen)["ids"]
    assert len(first) == fd.ROW_BYTES and bytes(first.astype(np.uint8)) == fd.row_bytes(packed, 7, 0, fd.P_FRAMED, fd.CAPTION, carry="caption", pack=pk)[0]


def test_the_streams_the_crafts_and_their_plans_are_registered():
    from geolip.alephllm.data import curriculum, streams
    for name, mix in (("nano-frames-caption", fd.CAPTION_MIX), ("nano-frames-caption-ctrl", fd.CAPTION_CTRL_MIX)):
        assert streams.REGISTRY[name]["generator"] == name.replace("-", "_") and curriculum.CURRICULUM_MIXES[mix] == [(name, 1.0)]
    for body in nano_v1.CRAFTS:
        n = f"{body}.1-mx-cp"
        c = nano_v1.ALL[n]
        assert c["source"] == f"{body}.1-mx" and c["cont"] == "cp" and c["tag"] == "v11mxcp" and nano_v1.body_of(n) == f"{body}.1"
        last = nano_v1.diet_of(n)[-1]
        assert last["name"] == "caption_continuation" and last["dataset"] == fd.CAPTION_MIX and last["planned_tokens"] == 1_500_000_000
        assert nano_v1.gauge_args(n, "x", "y")["carry_gauge"] == "1" and nano_v1.version_of([n]) == "v1.1"
    x = nano_v1.ALL["nano-pollux-dual-v1.1-mx-cpx"]
    assert x["cont"] == "cpx" and x["source"] == "nano-pollux-dual-v1.1-mx" and nano_v1.diet_of("nano-pollux-dual-v1.1-mx-cpx")[-1]["dataset"] == fd.CAPTION_CTRL_MIX
    assert nano_v1.ARM_TEXT["cp"] == "caption" and nano_v1.c_arm("nano-castor-v1.1-mx-cp")
    from geolip.nano_beatrix.automodel import ARM_WHAT
    assert "cp" in ARM_WHAT and "cpx" in ARM_WHAT
    assert nano_v1.caption_table({}).count("\n") == 1                                                # the header alone
    h = {"nano-pollux-dual-v1.1-mx-cp": [dict(phase="mixed_continuation", val_bpb=1.0, bed={"bed/frame/copy_whole": 0.5}),
                                         dict(phase="caption_continuation", val_bpb=0.9, bed={"bed/frame/copy_whole": 0.6})],
         "nano-pollux-dual-v1.1-mx-cpx": [dict(phase="caption_control_continuation", val_bpb=0.95, bed={"bed/frame/copy_whole": 0.4})]}
    t = nano_v1.caption_table(h)
    assert "| nano-pollux-dual-v1 | copy | 0.50 | 0.60 | 0.40 | +0.10 | -0.10 |" in t and "| nano-pollux-dual-v1 | val bits a byte | 1.0000 | 0.9000 | 0.9500 | -0.1000 | -0.0500 |" in t
    assert "nano-castor-v1 |" not in t


class _Copier:
    """A model that answers a question from the frame: after the question's SEP it writes the record's value found in the prompt, then END;
    with no frame (the extraction read) it writes END at once. Elsewhere it predicts a space."""

    def __call__(self, x):
        B, T = x.shape
        out = torch.full((B, T, 256), -10.0)
        out[:, :, 32] = 0.0
        for b in range(B):
            row = bytes(int(v) for v in x[b].tolist())
            q = row.rfind(bytes([CUE]))
            if q < 0 or bytes([SEP]) not in row[q:]:
                continue
            s = row.index(bytes([SEP]), q)
            key = row[q + 1:s]
            answer = bytes([END])
            k = row.find(bytes([CUE]) + key + bytes([SEP]))
            if 0 <= k < q:
                v0 = k + len(key) + 2
                v1 = v0
                while v1 < len(row) and row[v1] not in (CUE, END, DATA) and row[v1:v1 + 2] != b"  ":
                    v1 += 1
                answer = row[v0:v1].rstrip(b" ") + bytes([END])
            for t in range(s, T):                         # the prediction for byte t + 1 sits at t
                j = t - s
                nxt = answer[j] if j < len(answer) else END
                out[b, t, :] = -10.0
                out[b, t, nxt] = 10.0
        return (out,)


def test_the_caption_reads_with_a_scripted_model(pack_dir):
    held = cp.load(pack_dir, "held")
    model = _Copier()
    rec = cs.read(lambda x: model(x)[0], torch.device("cpu"), held, n_rows=12, seed=1, ctx=4096, log=lambda s: None)
    assert rec["n"] == 12 and rec["copy"]["whole"] == 1.0 and rec["copy"]["first_byte"] == 1.0 and rec["extraction"]["whole"] == 0.0
    assert rec["copy"]["value_bits"] < 0.01 and rec["extraction"]["value_bits"] > 5
    assert set(rec["copy"]["by_kind"]) <= {"photo", "danbooru", "coco"} and all(v["whole"] == 1.0 for v in rec["copy"]["by_key"].values())
    assert len(rec["copy"]["examples"]) == 12 and rec["copy"]["examples"][0]["right"]
    gens = cs.greedy(lambda x: model(x)[0], [bytes([DOC]) + b"abc" + bytes([CUE]) + b"look hair" + bytes([SEP]) + b"blonde" + b"  " + bytes([CUE]) + b"look hair" + bytes([SEP])],
                     torch.device("cpu"), 10)
    assert gens == [b"blonde"]
