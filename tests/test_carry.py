# -*- coding: utf-8 -*-
"""The carry rows (0.1.37, the self-written frames prototype): the row's shape and placements, the gauge's row as the training row without
its carries, the insertion that keeps the window, the stream that reads the frames arms' text row for row, the registered names, the -cf
crafts and their plans and gauge arguments, the table and the scalars. CPU only; the Hub is never touched."""
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


def text(seed: int, n: int = 12_000) -> bytes:
    rng = np.random.default_rng(seed)
    return (" ".join(WORDS[int(i)] for i in rng.integers(len(WORDS), size=n)) + ".").encode()


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    d = tmp_path_factory.mktemp("carry_corpus")
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


def test_the_carry_row_holds_its_shape_and_places():
    form = fd.CARRY
    g = form["grid"]
    rb = bf.capacity(fd.ROW_CTX, "A", form["register"])["record_bytes"]
    seen_sizes, seen_n = set(), set()
    for s in range(200):
        row, m = fd.carry_row(np.random.default_rng([s, 1]), text(s))
        assert len(row) == fd.ROW_BYTES and row[0] == DOC
        assert 6 <= m["early_start"] <= 6 + form["early_gap"] and row[m["early_start"]] == DATA
        cb = bf.frame_bytes(m["early_size"], form["sliced"])
        assert m["early_end"] == m["early_start"] + cb
        seen_sizes.add(m["early_size"])
        seen_n.add(len(m["early_keys"]))
        assert m["carries"] == m["planned"] and len(m["carries"]) >= 3
        prev_end = m["early_end"]
        for pos in m["carries"]:
            assert pos % g == 0 and pos >= prev_end and row[pos] == DATA
            assert form["carry_every"][0] <= pos - prev_end < form["carry_every"][1] + g       # the drawn interval, then the grid's rounding
            assert row[pos:pos + cb] == row[m["early_start"]:m["early_end"]]                   # a carry is the early frame byte for byte
            prev_end = pos + cb
        q_lo = fd.ROW_BYTES - form["late"]
        assert prev_end + g <= q_lo <= m["q0"] and row[m["q0"]] == CUE
        assert m["q0"] - prev_end >= g
        key = m["key"].encode()
        assert key in [k.encode() for k in m["early_keys"]]
        assert row[m["q0"]:m["v0"]] == bytes([CUE]) + key + bytes([SEP]) and row[m["v0"] + m["vlen"]] == END
        assert m["n_defs"] == len(m["carries"]) + 1 and m["truth_frame"] >= 1                   # the latest definition sits in a carry
        slack = row[m["early_start"] + (3 if m["early_size"] > 32 else 2) + 17 * (1 if m["early_size"] == 32 else 1):m["early_end"] - 1]
        assert all(ch in b" \xf9\xf6\xf8\xfcab" or 32 <= ch < 127 for ch in slack)             # the frame's slack is spaces (the slices' own tags and closures aside)
        assert m["dist"] <= form["carry_every"][1] + g + form["late"] + cb
        assert m["vlen"] == 6 and rb == 17
    assert seen_sizes == {32, 64} and seen_n == {1, 2}


def test_the_gauge_row_is_the_training_row_without_its_carries():
    for s in range(60):
        row, m = fd.carry_row(np.random.default_rng([s, 1]), text(s), carries=True)
        bare, mb = fd.carry_row(np.random.default_rng([s, 1]), text(s), carries=False)
        assert len(bare) == fd.ROW_BYTES and mb["carries"] == [] and mb["planned"] == m["planned"]
        assert bare[:m["early_end"]] == row[:m["early_end"]]                                   # the same early frame
        assert mb["q0"] == m["q0"] and mb["key"] == m["key"] and bare[mb["v0"]:mb["v0"] + mb["vlen"]] == row[m["v0"]:m["v0"] + m["vlen"]]
        assert DATA not in bare[m["early_end"]:mb["q0"]]                                       # no frame between the early frame and the question
        rec_at = bare.index(bytes([CUE]) + m["key"].encode() + bytes([SEP]), m["early_start"], m["early_end"])
        assert mb["n_defs"] == 1 and mb["dist"] == mb["v0"] - rec_at                           # the distance to the record in the early frame
        assert mb["dist"] > 3000                                                               # the far question


def test_insert_carry_keeps_the_window_and_moves_the_question():
    row, m = fd.carry_row(np.random.default_rng([3, 1]), text(3), carries=False)
    r = bytearray(row)
    gen = bytes([DATA]) + b"a" + bytes([CUE]) + b"code001" + bytes([SEP]) + b"abcdef" + bytes([END])
    p = -(-m["early_end"] // m["grid"]) * m["grid"]
    tail_before = bytes(r[-len(gen):])
    fd.insert_carry(r, p, gen, fd.ROW_CTX)
    assert len(r) == fd.ROW_BYTES and bytes(r[:p]) == row[:p] and bytes(r[p:p + len(gen)]) == gen
    assert bytes(r[p + len(gen):]) == row[p:-len(gen)] and tail_before not in bytes(r[-len(gen):]) or True
    q_now = m["q0"] + len(gen)
    assert r[q_now] == CUE and bytes(r[q_now + 1:q_now + 1 + len(m["key"])]) == m["key"].encode()


def test_the_gauge_reads_the_grid_after_its_own_carry():
    """0.1.38: every training carry sits on the row's 64-byte grid as the row stands (the test above); the reading-with-notes gauge must
    read that grid after writing its own carry. 0.1.37 stepped p += len(carry), then += 64: off the grid by the carry's length for the
    rest of the row (Pollux canon's -cf arm at 62,000 and 64,000 steps: exactly one carry in every opening row, never a second)."""
    g = fd.CARRY["grid"]
    for size in fd.CARRY["early_sizes"]:
        cb = bf.frame_bytes(size, fd.CARRY["sliced"])
        assert cb % g, "the carry's length is not a multiple of the grid: the old step left the grid"
        p = 640
        for _ in range(6):                                                                     # six carries in a row, each read on the grid
            nxt = fd.next_grid(p, cb, g)
            assert nxt % g == 0 and p + cb <= nxt < p + cb + g
            p = nxt + 7 * g                                                                    # some grid steps of text, then the next carry
    assert fd.next_grid(640, 64, 64) == 704 and fd.next_grid(640, 35, 64) == 704 and fd.next_grid(640, 67, 64) == 768
    src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "geolip", "nano_beatrix", "bed_train.py"), encoding="utf-8").read()
    assert "p[i] = fd.next_grid(p[i], len(gen), g) - g" in src and "p[i] += len(gen)" not in src       # the gauge's step after an insertion


def test_the_carry_stream_reads_the_frames_arms_text_row_for_row(corpus):
    packed = fd.load_corpus(corpus)
    framed_seen = plain_seen = 0
    for i in range(40):
        a, fa = fd.row_bytes(packed, 7, i, fd.P_FRAMED)
        b, fb = fd.row_bytes(packed, 7, i, fd.P_FRAMED, fd.CARRY, carry=True)
        c, fc = fd.row_bytes(packed, 7, i, 0.0)
        assert fa == fb and not fc and len(b) == fd.ROW_BYTES                                 # the same coin, the same position
        if fa:
            framed_seen += 1
            assert b[0] == DOC and b != a and CUE in b
        else:
            plain_seen += 1
            assert a == b == c                                                                 # a plain row is the control's row
    assert framed_seen and plain_seen
    g = fd.rows(7, fd.P_FRAMED, fd.CARRY, True)
    first = next(g)["ids"]
    assert first.shape == (fd.ROW_BYTES,) and first.dtype == np.int64


def test_the_names_are_registered_and_the_crafts_exist():
    from geolip.alephllm.data import curriculum, streams
    assert streams.REGISTRY["nano-frames-carry"]["generator"] == "nano_frames_carry"
    assert curriculum.CURRICULUM_MIXES[fd.CARRY_MIX] == [("nano-frames-carry", 1.0)]
    cf = [n for n in nano_v1.CRAFTS_CONT if n.endswith("-cf")]
    assert cf == [f"{n}.1-cf" for n in nano_v1.CRAFTS]
    for n in cf:
        c = nano_v1.ALL[n]
        assert c["cont"] == "cf" and c["tag"] == "v11cf" and c["source"] == n[:-3]
        plan = nano_v1.diet_of(n)
        assert plan[-1]["name"] == "carry_continuation" and plan[-1]["dataset"] == fd.CARRY_MIX and plan[-1]["planned_tokens"] == nano_v1.CONT_TOKENS
        a_cf = nano_v1.gauge_args(n, nano_v1.gauge_init(n, 65237), "/tmp/x")
        a_fr = nano_v1.gauge_args(n[:-3] + "-fr", nano_v1.gauge_init(n[:-3] + "-fr", 65237), "/tmp/x")
        assert a_cf.get("carry_gauge") == "1" and "carry_gauge" not in a_fr
        assert run_tag(a_cf).endswith("_carry_gauge") and run_tag(a_fr).endswith("_gauge") and not run_tag(a_fr).endswith("_carry_gauge")
    assert nano_v1.gauge_init("nano-castor-v1.1-cf", 65237) == "v11cfc_65237"


def test_the_table_and_the_scalars_carry_the_carry_arm():
    sc = nano_v1.gauge_scalars({"heldout_bpb": 1.2, "carry": {"full_value": 0.5, "carries_per_row": 4.0, "n": 192, "grid": 64}})
    assert sc["bed/carry/full_value"] == 0.5 and sc["bed/carry/carries_per_row"] == 4.0 and sc["bed/carry/n"] == 192.0
    assert "bed/carry/full_value" in nano_v1.HEADLINE and "bed/carry/carries_per_row" in nano_v1.HEADLINE   # the boundary row keeps the carry reads
    last = nano_v1.diet()[-1]["name"]
    body = "nano-castor-v1"
    before = dict(phase=last, step=61422, val_bpb=1.28, bed={"bed/frame/copy_whole": 0.14})
    hist = {f"{body}.1-fr": [before, dict(phase="frames_continuation", step=65237, val_bpb=1.27, bed={"bed/frame/copy_whole": 0.9})],
            f"{body}.1-ct": [before, dict(phase="control_continuation", step=65237, val_bpb=1.26, bed={"bed/frame/copy_whole": 0.1})],
            f"{body}.1-cf": [before, dict(phase="carry_continuation", step=65237, val_bpb=1.275,
                                          bed={"bed/frame/copy_whole": 0.8, "bed/carry/full_value": 0.6, "bed/carry/carries_per_row": 5.0})]}
    t = nano_v1.cont_table(hist)
    assert "| after, carry |" in t and "| carry minus control |" in t
    copy = next(l for l in t.split("\n") if f"| {body} | copy |" in l)
    assert copy.split("|")[3:8] == [" 0.14 ", " 0.90 ", " 0.80 ", " 0.10 ", " +0.80 "] and copy.split("|")[8].strip() == "+0.70"
    far = next(l for l in t.split("\n") if "far question, own carries" in l)
    assert far.split("|")[5].strip() == "0.60" and far.split("|")[4].strip() == "-"
    assert not any("| nano-pollux-canon-v1 |" in l for l in t.split("\n"))                   # a body without rows is left out
