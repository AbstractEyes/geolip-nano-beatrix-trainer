# -*- coding: utf-8 -*-
"""0.1.41: THE NOTE ROW: the deterministic fact extractor (definitional facts about terms, places, quantities and institutions; never a
person), the note row's geometry (the note at the first grid point after the fact, the carries on the grid, the question asking the
key; the self and the planted gauge rows), the note stream (a text without a fact stays plain; registered), the note crafts and their
plans, the gauge arguments, the tag, the scalars and the headline keys. CPU only; the Hub is never touched."""
import os

import numpy as np
import pytest

from geolip.nano_beatrix import bed_frames as bf
from geolip.nano_beatrix import frames_data as fd
from geolip.nano_beatrix import nano_v1
from geolip.nano_beatrix.names import run_tag
from geolip.alephllm.data.special_tokens import CUE, DATA, DOC, END, SEP

WORDS = ("the river carried silt past the old mill while the town slept and the lamps burned low over quiet streets "
         "a teacher wrote numbers on the board and the class copied every line into worn notebooks before the bell").split()
FACTS = (b"Photosynthesis is a process used by plants. The Nile is 6650 km long and the Amazon River is a river in South America. "
         b"The United Nations (UN) was founded in 1945 in San Francisco. The capital of France is Paris. Mount Everest is located in "
         b"Nepal. The Royal Society was founded in 1660. John Smith was born in 1950 and lives at 12 Oak Street. This is a sentence. ")


def text(seed: int, n: int = 400) -> bytes:
    rng = np.random.default_rng(seed)
    return (" ".join(WORDS[int(i)] for i in rng.integers(len(WORDS), size=n)) + ". ").encode()


_SYL = ("Ar", "Bel", "Cor", "Dun", "El", "Fen", "Gal", "Hol", "Ith", "Jor")
_END = ("mont", "ford", "wick", "dale", "burg", "ham", "stead", "shire", "ton", "bury")


def name(i: int) -> str:
    return _SYL[i % 10] + _END[(i // 10) % 10] + ("" if i < 100 else _END[i % 10])


def cluster(j: int) -> bytes:
    """Five distinct facts about made-up places and institutions (one key is stated once in the whole text, as in real prose)."""
    a, b, c, d, e, f, g = (name(7 * j + k) for k in range(7))
    return (f"{a} is a town in the hills. The {b} is {100 + j} km long. The capital of {c} is {d}. {e} is located in {f}. "
            f"{g} was founded in {1500 + j}. ").encode()


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    d = tmp_path_factory.mktemp("note_corpus")
    rng = np.random.default_rng(0)
    docs, offs, total = [], [0], 0
    while total < 2_000_000:
        n = int(rng.integers(80, 1000))
        b = (" ".join(WORDS[int(i)] for i in rng.integers(len(WORDS), size=n)) + ".").encode()
        docs.append(b)
        total += len(b)
        offs.append(total)
    np.frombuffer(b"".join(docs), dtype=np.uint8).tofile(str(d / "bed_corpus.bin"))
    np.save(str(d / "bed_corpus_offsets.npy"), np.asarray(offs, dtype=np.int64))
    old = os.environ.get(fd.CORPUS_ENV)
    os.environ[fd.CORPUS_ENV] = str(d)
    yield str(d)
    if old is None:
        os.environ.pop(fd.CORPUS_ENV, None)
    else:
        os.environ[fd.CORPUS_ENV] = old


def test_the_extractor_reads_definitional_facts_and_never_a_person():
    facts = fd.extract_facts(FACTS)
    d = {k.decode(): (v.decode(), kind) for _, k, v, kind in facts}
    assert d["Photosynthesis"] == ("process", "def") and d["Amazon River"] == ("river", "def")
    assert d["Nile"] == ("6650 km", "qty")                                     # the leading stop word dropped from the term
    assert d["UN"] == ("United Nations", "abbr") and d["France"] == ("Paris", "capital")
    assert d["Mount Everest"] == ("Nepal", "where") and d["Royal Society"] == ("1660", "year")
    assert "John Smith" not in d and "Smith" not in d and all(v != "1950" for v, _ in d.values()) and "This" not in d
    ends = [e for e, _, _, _ in facts]
    assert ends == sorted(ends) and all(0 < e <= len(FACTS) for e in ends) and len({k for _, k, _, _ in facts}) == len(facts)
    assert all(2 <= len(k) <= fd.NOTE["key_max"] and 1 <= len(v) <= fd.NOTE["val_max"] and len(k) + len(v) <= fd.NOTE["rec_max"] for _, k, v, _ in facts)
    assert fd.extract_facts(text(5)) == []                                     # plain lowercase prose holds no fact


def test_the_note_row_places_the_note_on_the_grid_after_the_fact_and_asks_its_key():
    src = b"".join(cluster(j) + text(3 + j, 120) for j in range(30))
    g, nb = fd.NOTE["grid"], bf.frame_bytes(fd.NOTE["size"], fd.NOTE["sliced"])
    seen_kinds, seen_near, seen_far = set(), False, False
    for seed in range(40):
        r = fd.note_row(np.random.default_rng(seed), src)
        assert r is not None
        row, m = r
        assert len(row) == fd.ROW_CTX + 1 and row[0] == DOC and nb == 32
        key, val = m["key"].encode(), m["early_vals"][0].encode()
        assert m["note_pos"] % g == 0 and m["fact_end"] <= m["note_pos"] < m["fact_end"] + g and m["note_pos"] >= fd.NOTE["min_pos"]
        note = row[m["early_start"]:m["early_end"]]
        assert m["early_start"] == m["note_pos"] and len(note) == nb and note[0] == DATA and note[-1] == END and note == m["note_bytes"]
        assert bytes([CUE]) + key + bytes([SEP]) + val in note
        for p_ in m["carries"]:
            assert p_ % g == 0 and row[p_:p_ + nb] == note                        # every carry is the note byte for byte, on the grid
        assert row[m["q0"]] == CUE and row[m["q0"] + 1:m["v0"] - 1] == key and row[m["v0"]:m["v0"] + m["vlen"]] == val and row[m["v0"] + m["vlen"]] == END
        assert m["q0"] >= fd.ROW_CTX + 1 - fd.NOTE["late"] and m["fact_dist"] == m["q0"] - m["fact_end"] and m["dist"] == m["v0"] - (m["carries"][-1] if m["carries"] else m["note_pos"])
        seen_kinds.add(m["fact_kind"])
        seen_near |= m["fact_dist"] < 1024
        seen_far |= m["fact_dist"] >= 2048
        # the self row: the same text, no note, no carries, the same question; the planted row: the note, no carries
        s_row, s_m = fd.note_row(np.random.default_rng(seed), src, notes=False)
        p_row, p_m = fd.note_row(np.random.default_rng(seed), src, notes=True, carries=False)
        assert s_m["key"] == p_m["key"] == m["key"] and s_m["fact_end"] == p_m["fact_end"] == m["fact_end"]
        assert DATA not in s_row[:s_m["q0"]] and s_m["early_start"] is None and s_m["dist"] == s_m["v0"] - s_m["fact_end"] and s_m["n_defs"] == 0
        assert s_row[:m["note_pos"]] == row[:m["note_pos"]] == p_row[:m["note_pos"]]
        assert p_row[:p_m["q0"]].count(DATA) == 1 and p_m["carries"] == [] and p_m["early_end"] - p_m["early_start"] == nb
    assert len(seen_kinds) >= 3 and seen_near and seen_far                      # one data form, both reads
    assert fd.note_row(np.random.default_rng(0), text(9, 6000)) is None        # no fact, no row


def test_the_longest_record_never_overruns_the_window():
    """0.1.42: a key of 20 bytes and a value of 7 (the record room's 27) asked late in the window: 0.1.41 drew the question's offset from
    the bed's synthetic record length and overran the row by up to 8 bytes (Phil's preflight, 2026-10-18 02:2x)."""
    src = (b"Abcdefghijklmnopqrst is located in Uvwxyza. " + text(11, 60)) * 40
    seen_late, built = False, 0
    for seed in range(300):
        r = fd.note_row(np.random.default_rng(seed), src)
        if r is None:                                                          # the one key repeats: its first statement can fall before the first grid point
            continue
        built += 1
        row, m = r
        qlen = 1 + len(m["key"]) + 1 + m["vlen"] + 1
        assert m["key"] == "Abcdefghijklmnopqrst" and m["vlen"] == 7 and len(row) == fd.ROW_CTX + 1
        assert m["q0"] + qlen <= fd.ROW_CTX and row[m["v0"] + m["vlen"]] == END
        seen_late |= m["q0"] + qlen > fd.ROW_CTX + 1 - 24                   # a draw that the old bound would have pushed past the window
    assert seen_late and built >= 200, built


def test_the_note_stream_stays_plain_without_a_fact_and_is_registered(corpus):
    from geolip.alephllm.data import curriculum, streams
    packed = fd.load_corpus(corpus)
    for i in range(12):
        a, fa = fd.row_bytes(packed, 7, i, fd.P_FRAMED, fd.NOTE, "note")
        c, _ = fd.row_bytes(packed, 7, i, 0.0)
        assert a == c and not fa and len(a) == fd.ROW_BYTES                      # the toy words hold no fact: every row is the control's row
    assert streams.REGISTRY["nano-frames-note"]["generator"] == "nano_frames_note"
    assert curriculum.CURRICULUM_MIXES[fd.NOTE_MIX] == [("nano-frames-note", 1.0)]
    first = next(fd.rows(7, fd.P_FRAMED, fd.NOTE, "note"))["ids"]
    assert first.shape == (fd.ROW_BYTES,)


def test_the_note_crafts_their_plans_gauges_tags_and_scalars():
    N1, N2 = nano_v1.CRAFTS_NOTE, nano_v1.CRAFTS_NOTE2
    assert list(N1) == [f"{n}.1-nt" for n in nano_v1.CRAFTS] and list(N2) == [f"{n}.1-cf-nt" for n in nano_v1.CRAFTS]
    assert all(n in nano_v1.ALL for n in list(N1) + list(N2)) and nano_v1.CONT_PHASE["nt"] == ("note_continuation", fd.NOTE_MIX)
    assert len({nano_v1.gauge_init(n, 69052) for n in nano_v1.ALL}) == len(nano_v1.ALL)
    for n in N1:
        ph = nano_v1.diet_of(n)
        assert len(ph) == 14 and ph[-1]["name"] == "note_continuation" and ph[-1]["dataset"] == fd.NOTE_MIX
        spec = nano_v1.cont_init(n, 262144)
        assert spec["payload"] == "resume/boundary_anneal_mix_step61422.pt" and spec["step"] == 61422 and spec["prefix"] == n[:-3]
        assert nano_v1.plan_steps(n, 262144) == 65237
    for n in N2:
        ph = nano_v1.diet_of(n)
        assert [p["name"] for p in ph[13:]] == ["carry_continuation", "note_continuation"] and len(ph) == 15
        spec = nano_v1.cont_init(n, 262144)
        assert spec["payload"] == "resume/boundary_carry_continuation_step65237.pt" and spec["step"] == 65237 and spec["prefix"] == n[:-3]
        assert N2[n]["source"] in nano_v1.CRAFTS_CONT and nano_v1.plan_steps(n, 262144) == 69052
        a = nano_v1.gauge_args(n, nano_v1.gauge_init(n, 69052), "/tmp/x")
        assert a.get("carry_gauge") == "1" and a.get("note_gauge") == "1" and run_tag(a).endswith("_carry_note_gauge")
    a_cf = nano_v1.gauge_args("nano-castor-v1.1-cf", nano_v1.gauge_init("nano-castor-v1.1-cf", 65237), "/tmp/x")
    assert "note_gauge" not in a_cf and run_tag(a_cf).endswith("_carry_gauge")
    rec = {"heldout_bpb": 1.2, "note_self": {"full_value": 0.25, "carries_per_row": 2.0, "rows_with_key": 0.6, "n_scanned": 240,
                                             "by_dist": {"far_ge2048": {"n": 80, "full_value": 0.2, "first_byte": 0.3, "value_bits": 30.0},
                                                         "near_lt1024": {"n": 40, "full_value": None, "first_byte": None, "value_bits": None}},
                                             "dump": {"jsonl": "x"}},
           "note_planted": {"full_value": 0.7, "by_dist": {}}}
    sc = nano_v1.gauge_scalars(rec)
    assert sc["bed/note_self/full_value"] == 0.25 and sc["bed/note_self/far_ge2048_full_value"] == 0.2 and sc["bed/note_self/rows_with_key"] == 0.6
    assert sc["bed/note_planted/full_value"] == 0.7 and "bed/note_self/near_lt1024_full_value" not in sc and "bed/note_self/dump" not in sc
    assert "bed/note_self/full_value" in nano_v1.HEADLINE and "bed/note_planted/far_ge2048_full_value" in nano_v1.HEADLINE
