# -*- coding: utf-8 -*-
"""The gate screen (0.1.51): the note's parts, the three variants, the gain arithmetic, the random control, the protocol with a scripted
model, the record's shape."""
import numpy as np
import torch

from geolip.alephllm.data.special_tokens import CUE, DATA, DOC, END, SEP
from geolip.nano_beatrix import bed_frames as bf
from geolip.nano_beatrix import frames_data as fd
from geolip.nano_beatrix import gate_screen as gs


def _note(key=b"hair", val=b"golden", size=32):
    state = dict(frames=[], defs={}, facts=None, key_iter=iter(()), fact_iter=None, sized=True, sliced=32)
    rng = np.random.default_rng(0)
    return bf.build_frame(rng, bf.spec("data", size, records=1, keys=[key], values=[val]), 0, bf.Filler(b" " * 4096, 0), "caption", state)


def test_parse_and_withhold_a_note():
    n = _note()
    assert len(n) == 32 and n[0] == DATA and n[-1] == END
    p = gs.parse_note(n)
    assert p["key"] == b"hair" and p["value"] == b"golden"
    w = gs.withhold(n, p)
    assert len(w) == len(n) and w[:p["v0"]] == n[:p["v0"]] and w[p["v0"]:p["v1"]] == b" " * 6 and w[p["v1"]:] == n[p["v1"]:]
    assert gs.parse_note(b"\x05abc") is None                      # no CUE ... SEP record
    assert gs.withhold(b"xyz", None) == b"xyz"


def test_random_note_is_a_one_record_32_byte_frame():
    rng = np.random.default_rng(5)
    n = gs.random_note(rng)
    p = gs.parse_note(n)
    assert len(n) == 32 and p is not None and 6 <= len(p["key"]) <= 9 and len(p["value"]) == 6     # a cue word of 3 to 6 letters + 3 digits
    assert all(c in b"abcdefghijklmnopqrstuvwxyz" for c in p["value"])


def test_the_three_variants_share_the_same_text_bytes():
    text = bytes(range(65, 65 + 26)) * 40                            # letters, 1,040 bytes
    row = bytearray(bytes([DOC]) + text)
    note = _note()
    p = 128
    fd.insert_carry(row, p, note, len(row) - 1)
    v = gs.variants(bytes(row), p, note, 256, control=_note(b"label001", b"qwerty"))
    for name, (r, s) in v.items():
        assert r[s:] == bytes(row[p + 32:p + 32 + 256]), name           # every variant ends with the same horizon bytes
        assert r[:p] == bytes(row[:p]), name                             # and begins with the same head
    assert v["without"][0] == bytes(row[:p]) + bytes(row[p + 32:p + 32 + 256])
    wh = v["withheld"][0]
    parts = gs.parse_note(note)
    assert wh[p + parts["v0"]:p + parts["v1"]] == b" " * 6 and wh[p:p + parts["v0"]] == note[:parts["v0"]]
    assert len(v["control"][0]) == len(v["with"][0])


def test_gains_and_kept_follow_the_two_control_rule():
    bits = {"with": [0.5] * 512, "without": [1.0] * 512, "withheld": [0.75] * 512, "control": [1.0] * 512, "control_withheld": [1.0] * 512}
    g = gs.gains(bits)
    assert g[256]["gain_without"] == 128.0 and abs(g[256]["gain_withheld"] - 64.0) < 1e-9
    assert gs.kept(g[256], 1.0) is True and gs.kept(g[256], 100.0) is False
    assert gs.kept(g[256], 1.0, "control_") is False                   # the control saves nothing
    short = gs.gains({k: v[:100] for k, v in bits.items()})
    assert short[64] is not None and short[128] is None and gs.kept(short[128], 1.0) is None


class _Scripted:
    """A model that opens a frame at every grid point and writes the same note, then predicts spaces; the horizon bytes cost one fixed
    amount, less when the note's value stands in the context (so the gate has something to keep)."""

    def __init__(self, note: bytes, cheap_with_value: float = 2.0, cost: float = 6.0):
        self.note, self.cheap, self.cost = note, cheap_with_value, cost

    def __call__(self, x):
        B, T = x.shape
        out = torch.full((B, T, 256), -10.0)
        xs = x.tolist()
        val = gs.parse_note(self.note)["value"]
        for b in range(B):
            row = bytes(v for v in xs[b])
            for t in range(T):
                pre = row[:t + 1]
                i = pre.rfind(bytes([DATA]))
                if i >= 0 and END not in pre[i:] and len(pre) - i < len(self.note):
                    out[b, t, self.note[len(pre) - i]] = 10.0           # the note byte by byte
                elif (t + 1) % 64 == 0 and (t + 1) >= 128 and i < 0:
                    out[b, t, DATA] = 10.0                               # open at the grid
                else:
                    nxt = row[t + 1] if t + 1 < T else 32
                    seen = val in pre
                    out[b, t, :] = -(self.cheap if seen else self.cost) * 0.0 - 10.0
                    out[b, t, nxt] = (10.0 - self.cheap) if seen else (10.0 - self.cost)
                    out[b, t, 32] = max(out[b, t, 32].item(), 0.0)
        return out


def test_the_protocol_proposes_scores_and_summarizes():
    note = _note()
    text = (b"the quick brown fox jumps over the lazy dog. " * 120)[:1600]
    rows = [bytearray(bytes([DOC]) + text) for _ in range(2)]
    metas = [dict(start=128, q0=len(text) + 1 - 512, fact_end=None, fact_key="hair", fact_val="golden", text=bytes(r)) for r in rows]
    notes, reads = gs.screen(_Scripted(note), rows, metas, torch.device("cpu"), ctx=len(text), horizons=(64, 128, 256), margins=(0.25, 1.0), log=lambda s: None)
    assert notes and all(n["parsed"] and n["key"] == "hair" and n["value"] == "golden" for n in notes)
    assert all(n["pos"] % 64 == 0 for n in notes) and all(r >= 1 for r in reads)
    scored = [n for n in notes if n["gains"] is not None]
    assert scored, "no note had a horizon of text after it"
    g = scored[0]["gains"]["64"]
    assert g["gain_without"] > 0 and "control_gain_without" in g
    rec = gs.summarize(notes, 2, "plain", (64, 128, 256), (0.25, 1.0), ref=(64, 0.25))
    assert rec["notes_a_row"] == len(notes) / 2 and rec["table"]["64|0.25"]["kept"] >= 1
    assert rec["kept_holds_fact_key"] == 1.0 and rec["kept_examples"] and rec["gain_without_kept"]["median"] > 0
    assert set(rec["table"]) == {f"{H}|{m}" for H in (64, 128, 256) for m in (0.25, 1.0)}


def test_a_fact_row_keeps_its_question_inside_the_window():
    """0.1.53: a question row's insertions have only the filler after its answer (bed_train's room), never the plain rows' 512 bytes; the
    model is never handed more than ctx bytes."""
    note = _note()
    ctx = 1600
    text = bytearray((b"the quick brown fox jumps over the lazy dog. " * 120)[:ctx])
    key, val = b"hair", b"golden"
    q = bytes([CUE]) + key + bytes([SEP]) + val + bytes([END])
    q0 = ctx + 1 - 40                                                    # late in the window, as note_row draws it
    row = bytearray(bytes([DOC]) + text)
    row[q0:q0 + len(q)] = q
    v0, vlen = q0 + 1 + len(key) + 1, len(val)
    room = ctx + 1 - (v0 + vlen + 1)
    metas = [dict(start=128, q0=q0, v0=v0, vlen=vlen, fact_end=None, fact_key="hair", fact_val="golden", text=bytes(row[:q0]))]

    class Guard(_Scripted):
        def __call__(self, x):
            assert x.shape[1] <= ctx, x.shape
            return super().__call__(x)

    rows = [row]
    notes, reads = gs.screen(Guard(note), rows, metas, torch.device("cpu"), ctx=ctx, horizons=(64,), margins=(0.25,), log=lambda s: None)
    ins = sum(n["length"] for n in notes)
    assert notes and ins == room and notes[0]["cut"]                    # the one note is cut to the room after the answer
    assert len(rows[0]) == ctx + 1 and bytes(rows[0][q0 + ins:q0 + ins + len(q)]) == q   # the question moved right and stayed whole
