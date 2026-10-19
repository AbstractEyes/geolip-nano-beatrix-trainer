# -*- coding: utf-8 -*-
"""The mixed phase (0.1.49): a framed row is a carry row or a frames row in one of three registers, drawn from the row's third generator so
the text, the coin and the composer's own generator are every other arm's; the registered stream; the -mx crafts, their plans, gauge
arguments and names; the mixed table. CPU only; the Hub is never touched."""
import math

import numpy as np

from geolip.nano_beatrix import bed_frames as bf
from geolip.nano_beatrix import frames_data as fd
from geolip.nano_beatrix import nano_v1
from geolip.nano_beatrix.names import run_tag
from geolip.alephllm.data.special_tokens import CUE, DOC

from test_carry import corpus  # noqa: F401  (the module's corpus fixture)


def test_the_draw_follows_the_form_and_only_its_own_generator():
    rng = np.random.default_rng(3)
    draws = [fd.mixed_draw(rng) for _ in range(6000)]
    carry = sum(k == "carry" for k, _ in draws) / len(draws)
    regs = {r: sum(x == r for _, x in draws) / len(draws) for r in ("caption", "math", "code")}
    assert abs(carry - 2 / 3) < 0.03 and abs(regs["caption"] - 0.5) < 0.03 and abs(regs["math"] - 0.25) < 0.03 and abs(regs["code"] - 0.25) < 0.03
    assert fd.mixed_draw(np.random.default_rng([7, 5, 2])) == fd.mixed_draw(np.random.default_rng([7, 5, 2]))      # deterministic per row
    assert fd.MIXED["carry"] is fd.CARRY and fd.MIXED["frames"] is fd.FORM and sum(w for _, w in fd.MIXED["registers"]) == 1.0 and fd.MIXED["p_doc_start"] == 0.5


def test_the_mixed_stream_keeps_the_text_the_coin_and_the_other_arms_rows(corpus):  # noqa: F811
    packed = fd.load_corpus(corpus)
    seen = {"carry": set(), "frames": set()}
    plain = doc_start = 0
    for i in range(80):
        a, fa = fd.row_bytes(packed, 7, i, fd.P_FRAMED)                               # the frames arm's row
        b, fb = fd.row_bytes(packed, 7, i, fd.P_FRAMED, fd.CARRY, carry=True)        # the carry arm's row
        m, fm = fd.row_bytes(packed, 7, i, fd.P_FRAMED, fd.MIXED, carry="mixed")
        assert fm == fa == fb and len(m) == fd.ROW_BYTES                               # the same coin, the same position
        if not fm:
            plain += 1
            if np.random.default_rng([7, i, 2]).random() < fd.MIXED["p_doc_start"]:                 # 0.1.50: a document-start plain row
                doc_start += 1
                assert m[0] == DOC and m == fd.doc_start_row(packed, int(np.random.default_rng([7, i]).integers(0, len(packed) - 2 * fd.ROW_BYTES - 2)))
                assert m != a or a[0] == DOC
            else:
                assert m == a == b                                                     # a plain row is the control's row
            continue
        kind, reg = fd.mixed_draw(np.random.default_rng([7, i, 2]))
        seen[kind].add(reg)
        assert m[0] == DOC and bf.tag_bytes(reg, "mode") in m[:12] and CUE in m        # the register tag of the drawn register
        if reg == "caption":
            assert m == (b if kind == "carry" else a)                                   # byte for byte the other arm's row
        else:
            assert m != a and m != b
    assert plain and seen["carry"] >= {"caption", "math"} and seen["frames"] >= {"caption"} and (seen["carry"] | seen["frames"]) == {"caption", "math", "code"}
    assert 0 < doc_start < plain                                                       # both plain starts occur
    g = fd.rows(7, fd.P_FRAMED, fd.MIXED, "mixed")
    first = next(g)["ids"]
    assert first.shape == (fd.ROW_BYTES,) and first.dtype == np.int64


def test_a_document_start_row_begins_with_doc_and_the_other_streams_never_change(corpus):  # noqa: F811
    packed = fd.load_corpus(corpus)
    starts = np.flatnonzero(packed == DOC)
    p = int(starts[3]) - 100
    row = fd.doc_start_row(packed, p)
    assert row[0] == DOC and row == packed[int(starts[3]):int(starts[3]) + fd.ROW_BYTES].tobytes() and DOC not in row[1:50]
    assert fd.doc_start_row(packed, int(starts[-1]) + 1) == packed[int(starts[-1]) + 1:int(starts[-1]) + 1 + fd.ROW_BYTES].tobytes() or len(packed) - int(starts[-1]) > fd.ROW_BYTES
    for i in range(30):                                                                 # the frames, control, carry and note streams: no document-start rows
        a, fa = fd.row_bytes(packed, 11, i, fd.P_FRAMED)
        c, _ = fd.row_bytes(packed, 11, i, 0.0)
        b, _ = fd.row_bytes(packed, 11, i, fd.P_FRAMED, fd.CARRY, carry=True)
        if not fa:
            assert a == b == c == packed[int(np.random.default_rng([11, i]).integers(0, len(packed) - 2 * fd.ROW_BYTES - 2)):][:fd.ROW_BYTES].tobytes()


def test_a_math_carry_row_holds_digit_values_and_a_code_frames_row_hex():
    src = b"the lamps burned low over quiet streets while the class copied every line into worn notebooks before the bell " * 120
    row, meta = fd.carry_row(np.random.default_rng(11), src, dict(fd.CARRY, register="math"))
    assert meta["register"] == "math" and all(v.isdigit() for v in meta["early_vals"]) and meta["carries"]
    row2, meta2 = fd.frame_row(np.random.default_rng(11), src, dict(fd.FORM, register="code"))
    assert meta2["register"] == "code" and len(row2) == fd.ROW_CTX + 1 and bf.tag_bytes("code", "mode") in row2[:12]


def test_the_mixed_crafts_their_plans_gauges_tags_and_names():
    from geolip.alephllm.data import curriculum, streams
    assert streams.REGISTRY["nano-frames-mixed"]["generator"] == "nano_frames_mixed"
    assert curriculum.CURRICULUM_MIXES[fd.MIXED_MIX] == [("nano-frames-mixed", 1.0)]
    MX = nano_v1.CRAFTS_MIX
    assert list(MX) == [f"{n}.1-mx" for n in nano_v1.CRAFTS] and all(n in nano_v1.ALL for n in MX)
    assert nano_v1.CONT_PHASE["mx"] == ("mixed_continuation", fd.MIXED_MIX) and nano_v1.CONT_TOKENS_OF == {"mx": 1_500_000_000, "cp": 1_500_000_000, "cpx": 1_500_000_000}
    assert len({nano_v1.gauge_init(n, 67145) for n in nano_v1.ALL}) == len(nano_v1.ALL)
    steps_mx = math.ceil(1_500_000_000 / 262144)
    for n in MX:
        c = nano_v1.ALL[n]
        assert c["cont"] == "mx" and c["tag"] == "v11mx" and c["source"] == n[:-3] and c["source"] in nano_v1.CRAFTS_V11
        ph = nano_v1.diet_of(n)
        assert len(ph) == 14 and ph[-1]["name"] == "mixed_continuation" and ph[-1]["dataset"] == fd.MIXED_MIX and ph[-1]["planned_tokens"] == 1_500_000_000
        spec = nano_v1.cont_init(n, 262144)
        assert spec["payload"] == "resume/boundary_anneal_mix_step61422.pt" and spec["step"] == 61422 and spec["prefix"] == n[:-3]
        assert nano_v1.plan_steps(n, 262144) == 61422 + steps_mx == 67145 and nano_v1.start_step(n) == 61422
        a = nano_v1.gauge_args(n, nano_v1.gauge_init(n, 67145), "/tmp/x")
        assert a.get("carry_gauge") == "1" and "note_gauge" not in a and run_tag(a).endswith("_carry_gauge")
        assert nano_v1.arm_text(n) == "mixed" and nano_v1.body_of(n) == n[:-3] and nano_v1.version_of([n]) == "v1.1"
    # the other arms' phases are untouched
    assert nano_v1.diet_of("nano-castor-v1.1-cf")[-1]["planned_tokens"] == nano_v1.CONT_TOKENS and nano_v1.plan_steps("nano-castor-v1.1-cf", 262144) == 65237
    assert nano_v1.plan_steps("nano-pollux-dual-v1.1-cf-nt", 262144) == 69052


def test_the_mixed_table_reads_near_against_frames_and_far_against_carry():
    last = nano_v1.diet()[-1]["name"]
    body = "nano-pollux-dual-v1"
    before = dict(phase=last, step=61422, val_bpb=1.28, bed={"bed/frame/copy_whole": 0.14})
    hist = {f"{body}.1-mx": [before, dict(phase="mixed_continuation", step=67145, val_bpb=1.255,
                                          bed={"bed/frame/copy_whole": 0.85, "bed/carry/full_value": 0.5, "bed/carry/carries_per_row": 4.0})],
            f"{body}.1-fr": [before, dict(phase="frames_continuation", step=65237, val_bpb=1.27, bed={"bed/frame/copy_whole": 0.9})],
            f"{body}.1-cf": [before, dict(phase="carry_continuation", step=65237, val_bpb=1.262,
                                          bed={"bed/frame/copy_whole": 0.4, "bed/carry/full_value": 0.6, "bed/carry/carries_per_row": 5.0})],
            f"{body}.1-ct": [before, dict(phase="control_continuation", step=65237, val_bpb=1.26, bed={"bed/frame/copy_whole": 0.1})]}
    t = nano_v1.mixed_table(hist)
    assert "| after mixed |" in t and "| minus the frames arm |" in t and "| minus the carry arm |" in t
    copy = next(l for l in t.split("\n") if f"| {body} | copy |" in l)
    assert copy.split("|")[3:8] == [" 0.14 ", " 0.85 ", " 0.90 ", " 0.40 ", " 0.10 "] and copy.split("|")[8].strip() == "-0.05" and copy.split("|")[9].strip() == "+0.45"
    far = next(l for l in t.split("\n") if "far question, own carries" in l)
    assert far.split("|")[4].strip() == "0.50" and far.split("|")[6].strip() == "0.60" and far.split("|")[9].strip() == "-0.10"
    assert not any("| nano-castor-v1 |" in l for l in t.split("\n"))


def test_the_second_seed_mixed_craft_shares_everything_but_its_data_seed():
    from geolip.alephllm.presets import TrainConfig
    MX2 = nano_v1.CRAFTS_MIX2
    assert list(MX2) == ["nano-pollux-dual-v1.1-mx2"] and all(n in nano_v1.ALL for n in MX2)
    n2, n1 = "nano-pollux-dual-v1.1-mx2", "nano-pollux-dual-v1.1-mx"
    c, c1 = nano_v1.ALL[n2], nano_v1.ALL[n1]
    assert c["cont"] == "mx" and c["tag"] == "v11mx2" and c["source"] == "nano-pollux-dual-v1.1" and c["arm"] == c1["arm"] and c["code"] == c1["code"]
    assert nano_v1.train_of(n2) == {**nano_v1.V11_TRAIN, "seed": 2} and "seed" not in nano_v1.train_of(n1)
    assert TrainConfig().seed == 1337 and TrainConfig(**nano_v1.train_of(n2)).seed == 2 and TrainConfig(**nano_v1.train_of(n1)).seed == 1337
    assert nano_v1.diet_of(n2) == nano_v1.diet_of(n1)
    assert nano_v1.plan_steps(n2, 262144) == 67145 and nano_v1.start_step(n2) == 61422
    assert nano_v1.gauge_init(n2, 67145) == "v11mx2pd_67145" and nano_v1.gauge_init(n1, 67145) == "v11mxpd_67145"
    assert len({nano_v1.gauge_init(n, 67145) for n in nano_v1.ALL}) == len(nano_v1.ALL)
    assert nano_v1.arm_text(n2) == "mixed" and nano_v1.body_of(n2) == "nano-pollux-dual-v1.1" and nano_v1.version_of([n2]) == "v1.1"
    assert nano_v1.gauge_args(n2, "v11mx2pd_67145", "/tmp/x").get("carry_gauge") == "1"
    # the mixed table reads the second seed beside the body's other arms, labeled by its craft name; the bodies' own rows unchanged
    row = dict(phase="mixed_continuation", step=67145, tokens=0, val_bpb=1.0, bed={"bed/frame/copy_whole": 0.8})
    t = nano_v1.mixed_table({n2: [row], n1: [dict(row, bed={"bed/frame/copy_whole": 0.81})]})
    assert "| nano-pollux-dual-v1 | copy | - | 0.81 |" in t and "| nano-pollux-dual-v1.1-mx2 | copy | - | 0.80 |" in t
    assert t.index("| nano-pollux-dual-v1 | copy") < t.index("| nano-pollux-dual-v1.1-mx2 | copy")
