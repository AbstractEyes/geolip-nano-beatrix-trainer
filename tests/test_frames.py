# -*- coding: utf-8 -*-
"""The frame composer on the CPU, no model: the row arithmetic, the registry tags, the three sources, the fit."""
import numpy as np
import pytest

from geolip.alephllm.data.special_tokens import CUE, DATA, DOC, END, ESC, MODE, SEP, THINK

from geolip.nano_beatrix import bed_frames as bf


@pytest.fixture(scope="module")
def src():
    rng = np.random.default_rng(7)
    return bytes(rng.integers(97, 123, size=20000).astype(np.uint8)) + bytes([DOC]) + b"the next document begins here. " * 400


def test_capacity_grows_with_the_window():
    c1, c4 = bf.capacity(1024, "A", "caption"), bf.capacity(4096, "A", "caption")
    assert c1["lookup_max"] == 4 and c1["bulk_max"] >= 30
    assert c4["bulk_max"] > 5 * c1["bulk_max"]


@pytest.mark.parametrize("ctx", [1024, 4096])
@pytest.mark.parametrize("scheme", ["mode", "esc"])
@pytest.mark.parametrize("register", list(bf.REGISTERS))
@pytest.mark.parametrize("nl,nb,ask", [(2, 0, "lookup"), (4, 8, "bulk"), (1, 10_000, "bulk"), (3, 5, "weights")])
def test_compose_layout_a(src, ctx, scheme, register, nl, nb, ask):
    rng = np.random.default_rng(11)
    cap = bf.capacity(ctx, "A", register)
    facts = bf.fact_table(256, register) if ask == "weights" else None
    row, m = bf.compose(rng, src, ctx, "A", scheme, register, min(nl, cap["lookup_max"]), min(nb, cap["bulk_max"]), facts, ask)
    comps = m["comps"]
    assert len(row) == ctx + 1
    assert row[0] == DOC and row[1] == (MODE if scheme == "mode" else ESC)
    assert row[comps["lookup"][0]] == DATA and row[comps["lookup"][1] - 1] == END
    assert row[comps["think"][0]] == THINK and row[comps["think"][1] - 1] == END
    assert row[m["q0"]] == CUE and row[m["v0"] - 1] == SEP and row[m["v0"] + m["vlen"]] == END
    assert not (set(row[comps["think"][0] + 1:comps["think"][1] - 1]) & bf.SPECIALS)
    if ask == "weights":
        assert m["key"].encode() not in row[:m["q0"]] and m["dist"] is None
    else:
        assert m["source"] == ask and m["dist"] > 0


def test_layout_a_frames_and_bits_by_frame(src):
    """Layout A's compartments are frames too (the trainer's bits by frame and fetch by frame read both modes alike)."""
    rng = np.random.default_rng(17)
    metas = []
    for ask in ("lookup", "bulk"):
        row, m = bf.compose(rng, src, 1024, "A", "mode", "caption", 2, 4, None, ask)
        assert [f["kind"] for f in m["frames"]] == ["data", "think", "data"]
        for f in m["frames"]:
            assert row[f["start"]] == bf.KIND_TAG[f["kind"]] and row[f["end"] - 1] == END
        assert m["truth_frame"] == (0 if ask == "lookup" else 2) and m["truth_depth"] == 0 and m["n_defs"] == 1
        metas.append(m)
    res = bf.finish_frame_bits(bf.bits_by_frame(np.ones((2, 1024)), metas))
    assert sum(v["n"] for v in res["by_kind_depth"].values()) + res["outside"]["n"] == 2 * 1024
    assert "think/512/body" in res["by_kind_size_segment"] and "data/128/open" in res["by_kind_size_segment"]
    assert bf.size_bucket(73) == 128 and bf.size_bucket(32) == 32 and bf.size_bucket(5000) == 4096
    row, m = bf.compose(rng, src, 1024, "A", "mode", "caption", 2, 0, bf.fact_table(64, "caption"), "weights")
    assert m["truth_frame"] is None and len(m["frames"]) == 2


def test_fact_table_is_seed_stable():
    assert bf.fact_table(64, "math") == bf.fact_table(64, "math")
    assert bf.fact_table(64, "math") != bf.fact_table(64, "caption")


def test_bands():
    assert bf.bands_for(1024) == (0, 8, 81, 401, 1024)
    assert bf.bands_for(4096) == (0, 8, 81, 401, 1024, 2048, 4096)


def _check_dynamic(row, m, ctx):
    assert len(row) == ctx + 1 and row[0] == DOC
    for f in m["frames"]:
        assert row[f["start"]] == bf.KIND_TAG[f["kind"]] and row[f["end"] - 1] == END and f["end"] - f["start"] == f["size"]
        if f["depth"] >= 1:
            assert any(p["depth"] == f["depth"] - 1 and p["start"] < f["start"] and f["end"] <= p["end"] - 1 for p in m["frames"])
    assert row[m["q0"]] == CUE and row[m["v0"] - 1] == SEP and row[m["v0"] + m["vlen"]] == END
    assert int((bf.frame_masks(m, ctx)["seg"] == bf.SEG_ID["open"]).sum()) == sum(f.get("open", 1) for f in m["frames"])


@pytest.mark.parametrize("ctx", [1024, 4096])
@pytest.mark.parametrize("scheme", ["mode", "esc"])
@pytest.mark.parametrize("ask", ["any", "nested", "first", "last"])
def test_dynamic_frames(src, ctx, scheme, ask):
    rng = np.random.default_rng(3)
    specs = [bf.spec("data", 32, records=1), bf.spec("think", 256, children=[bf.spec("data", 64, records=2)]), bf.spec("data", 128, records=4)]
    row, m = bf.compose_dynamic(rng, src, ctx, specs, "caption", scheme, ask)
    _check_dynamic(row, m, ctx)
    assert m["source"] == "frame" and m["dist"] > 0
    if ask == "nested":
        assert m["truth_depth"] == 1


def test_dynamic_weights_source(src):
    rng = np.random.default_rng(5)
    facts = bf.fact_table(128, "math")
    row, m = bf.compose_dynamic(rng, src, 1024, [bf.spec("data", 128, records=3), bf.spec("data", 256, records=6)], "math", "mode", "weights", facts)
    _check_dynamic(row, m, 1024)
    assert m["source"] == "weights" and m["dist"] is None and m["key"].encode() not in row[:m["q0"]]


def test_overlap_truth_is_the_latest(src):
    rng = np.random.default_rng(9)
    row, m = bf.compose_dynamic(rng, src, 1024, bf.overlap_specs(rng, "caption", 128, 128, 3, "conflict"), "caption", "mode", "overlap")
    _check_dynamic(row, m, 1024)
    assert m["n_defs"] == 2 and m["truth_frame"] == 1
    k = m["key"].encode()
    i1 = row.index(bytes([CUE]) + k + bytes([SEP]))
    assert row[i1 + len(k) + 2:i1 + len(k) + 2 + m["vlen"]] != row[m["v0"]:m["v0"] + m["vlen"]]



def test_uniform_bits_per_register():
    assert abs(bf.uniform_bits("caption") - 4.70) < 0.01
    assert abs(bf.uniform_bits("math") - 3.32) < 0.01
    assert abs(bf.uniform_bits("code") - 4.0) < 1e-9


@pytest.mark.parametrize("register", ["caption", "math", "code"])
def test_canary_rows_shapes_and_grammar(src, register):
    docs = [src[i:i + 1200] for i in range(0, 4800, 1200)]
    rows, meta = bf.canary_rows(docs, 1024, register, "mode", seed=1)
    assert set(rows) == {"plain", "tag", "frame"} and all(len(r) == 1025 for rs in rows.values() for r in rs)
    plain, tag, frame = rows["plain"][0], rows["tag"][0], rows["frame"][0]
    p = 1 + meta["prefix_bytes"]
    assert plain[0] == DOC and plain[1:] == docs[0][:1024]
    assert tag[:p] == bytes([DOC]) + bf.tag_bytes(register, "mode") + bytes([SEP]) + b"q" + bytes([SEP]) and tag[p:] == docs[0][:1025 - p]
    f = frame[p:p + 32]
    assert f[0] == DATA and f[-1] == END and f.count(bytes([CUE])) == 1 and meta["frame_records"] == 1 and meta["frame_bytes"] == 32
    assert frame[p + 32:] == docs[0][:1025 - p - 32]
    assert rows["frame"][1][p:p + 32] == f                      # one frame, shared by every row
    assert bf.canary_rows(docs, 1024, register, "mode", seed=1)[0]["frame"][0] == frame      # seed-stable


def test_formation_rule():
    u = bf.uniform_bits("math")
    g = lambda *vals: {f"data/{s}/depth0": dict(full_value=v, n=35) for s, v in zip((32, 128, 256, 512), vals)}
    formed = [dict(step=s, value_bits={"0": (u if s < 6000 else 0.3), "w": None}) for s in range(500, 12001, 500)]
    r = bf.formation(formed, "math", g(0.9, 0.5, 0.4, 0.3))
    assert r["formed"] and not r["failure"] and r["first_step_under"] == 6000 and abs(r["near_band_value_bits_at_end"] - 0.3) < 1e-9
    stuck = [dict(step=s, value_bits={"0": u - 0.05, "w": None}) for s in range(500, 12001, 500)]
    r = bf.formation(stuck, "math", g(0.0, 0.0, 0.0, 0.0))                     # never formed: at the rate, nothing read
    assert r["failure"] and not r["formed"] and r["first_step_under"] is None and r["at_the_rate"] and r["groups_judged"] == 4
    misaligned = [dict(step=s, value_bits={"0": (u if s < 4000 else 2.26), "w": None}) for s in range(500, 12001, 500)]
    r = bf.formation(misaligned, "math", g(0.0, 0.03, 0.0, 0.03))             # the bits fell, the copy is wrong everywhere: a failure
    assert r["failure"] and not r["at_the_rate"] and r["first_step_under"] == 4000
    r = bf.formation(stuck, "math", g(0.1, 0.0, 0.0, 0.0))                     # one group reads above the floor: no failure call
    assert not r["failure"] and r["at_the_rate"]
    r = bf.formation(stuck, "math", {"data/32/depth0": dict(full_value=0.0, n=4)})   # too few trials: no verdict
    assert not r["failure"] and r["groups_judged"] == 0
    r = bf.formation([], "math", None)                                            # nothing logged, nothing gauged: no verdict
    assert r["near_band_value_bits_at_end"] is None and not r["failure"] and r["first_step_under"] is None


def test_plain_collapse_flag():
    ok = {"0-64": 3.1, "64-256": 1.9, "256-1024": 1.6, "1024-2048": 1.5, "2048-4096": 1.5, "4096+": None}
    r = bf.plain_collapse(ok)
    assert not r["flag"] and r["worst_window"] == "64-256" and r["windows_over"] == []
    bad = dict(ok, **{"1024-2048": 4.3, "2048-4096": 3.4})
    r = bf.plain_collapse(bad)
    assert r["flag"] and r["worst_window"] == "1024-2048" and r["worst_bits"] == 4.3 and r["windows_over"] == ["1024-2048", "2048-4096"]
    assert not bf.plain_collapse({"0-64": 9.0, "64-256": 2.4})["flag"]          # the head never counts


def test_bits_by_frame_sums_to_the_window(src):
    rng = np.random.default_rng(13)
    specs = [bf.spec("data", 32, records=1), bf.spec("think", 256, children=[bf.spec("data", 64, records=2)])]
    metas = [bf.compose_dynamic(rng, src, 1024, specs, "caption", "mode", "any")[1] for _ in range(4)]
    res = bf.finish_frame_bits(bf.bits_by_frame(np.ones((4, 1024)), metas))
    cells = list(res["by_kind_size_segment"].values()) + list(res["by_kind_depth"].values()) + [res["outside"]]
    assert all(abs(c["bits"] - 1 / np.log(2.0)) < 1e-9 for c in cells)
    assert sum(v["n"] for v in res["by_kind_depth"].values()) + res["outside"]["n"] == 4 * 1024
    assert "data/64/open" in res["by_kind_size_segment"] and "data/depth2" in res["by_kind_depth"]


# ------------------------------------------------------------------ 0.1.21: the sized and the sliced frames
def _h(b) -> str:
    import hashlib
    import json
    return hashlib.sha256(b if isinstance(b, (bytes, bytearray)) else json.dumps(b, sort_keys=True, default=str).encode()).hexdigest()[:16]


def _specs(ctx):
    sp = [bf.spec("data", 32, records=1), bf.spec("think", 256, children=[bf.spec("data", 64, records=2)]), bf.spec("data", 128, records=4)]
    return sp + ([bf.spec("think", 1024, children=[bf.spec("data", 256, records=4)]), bf.spec("data", 512, records=8)] if ctx == 4096 else [])


def test_default_rows_are_the_0_1_20_rows(src):
    """With the new options off the composer writes the 0.1.20 rows and metadata byte for byte (digests taken from 0.1.20, d3d5e33)."""
    row, m = bf.compose_dynamic(np.random.default_rng([1024, 4, 7, 3]), src, 1024, _specs(1024), "caption", "mode", "any")
    assert (_h(row), _h(m)) == ("67d5fd58463afc86", "33d37b1906620933")
    row, m = bf.compose_dynamic(np.random.default_rng([4096, 3, 4, 6]), src, 4096, _specs(4096), "code", "esc", "nested")
    assert (_h(row), _h(m)) == ("0d76932736fbac2a", "9029a04f08a7054f")
    rng = np.random.default_rng([1024, 99])
    row, m = bf.compose_dynamic(rng, src, 1024, bf.overlap_specs(rng, "caption", 128, 128, 3, "conflict"), "caption", "mode", "overlap")
    assert (_h(row), _h(m)) == ("3481016c9ade88a4", "4c0a592302ced436")
    rows, meta = bf.canary_rows([src[i:i + 1200] for i in range(0, 4800, 1200)], 1024, "caption", "mode", seed=1)
    assert (_h(b"".join(r for rs in rows.values() for r in rs)), _h(meta)) == ("300670fc34fda72b", "93a5772f138938af")


@pytest.mark.parametrize("ctx", [1024, 4096])
@pytest.mark.parametrize("ask", ["any", "nested", "first", "last"])
def test_sized_open_tags_name_the_size(src, ctx, ask):
    row, m = bf.compose_dynamic(np.random.default_rng(21), src, ctx, _specs(ctx), "caption", "mode", ask, sized=True)
    _check_dynamic(row, m, ctx)
    assert m["sized"] is True and "sliced" not in m
    for f in m["frames"]:
        assert row[f["start"]:f["start"] + 2] == bytes([bf.KIND_TAG[f["kind"]]]) + bf.SIZE_LETTER[f["size"]]    # the letter names the size
        assert f["generic"] == f["size"] and f["open"] == 2 and row[f["start"] + f["size"] - 1] == END              # END where it says
        if f["n_records"]:
            assert row[f["start"] + 2] == CUE                                                                       # records after the tag
    assert int((bf.frame_masks(m, ctx)["seg"] == bf.SEG_ID["open"]).sum()) == 2 * len(m["frames"])


@pytest.mark.parametrize("ctx", [1024, 4096])
@pytest.mark.parametrize("ask", ["any", "nested", "first", "last"])
def test_sliced_frames_are_their_slices_and_an_end(src, ctx, ask):
    row, m = bf.compose_dynamic(np.random.default_rng(23), src, ctx, _specs(ctx), "caption", "mode", ask, sized=True, sliced=32)
    _check_dynamic(row, m, ctx)
    assert m["sliced"] == 32 and bf._check_sliced(row, m) == (2 if ctx == 1024 else 4)       # every frame above 32 bytes is sliced
    for i, f in enumerate(m["frames"]):
        if not f.get("sliced"):
            continue
        assert row[f["start"]:f["start"] + 2] == bytes([bf.KIND_TAG[f["kind"]]]) + bf.SIZE_LETTER[f["generic"]]
        assert f["size"] == f["generic"] + 3 and row[f["start"] + 2 + f["generic"]] == END    # the slices, then the frame's own END
        kids = sorted((g for g in m["frames"] if g.get("slice_of") == i), key=lambda g: g["start"])
        own = [g for g in kids if g["ldepth"] == f["depth"]]
        filled = [g for g in own if g["n_records"]]
        assert own[:len(filled)] == filled and sum(g["n_records"] for g in own) == f["n_records"]    # records fill the first slices
        run = [g for g in kids if g["ldepth"] == f["depth"] + 1]                                    # a nested lookup frame: a run
        if run:
            assert len(run) == run[0]["unit"] // 32 and all(b["start"] - a["start"] == 32 for a, b in zip(run, run[1:]))
    t = m["frames"][m["truth_frame"]]
    if t.get("slice_of") is not None:                                     # the fetch groups a sliced record by the frame it belongs to
        assert bf.frame_group(t) == f"{t['kind']}/{t['unit']}/depth{t['ldepth']}"
    if ask == "nested":
        assert m["truth_depth"] == 1 and bf.frame_group(t).endswith("/depth1")
    if ask in ("first", "last"):
        tops = [j for j, f in enumerate(m["frames"]) if f["depth"] == 0]
        top = m["frames"][tops[0] if ask == "first" else tops[-1]]
        assert top["start"] < m["v0"] - m["dist"] < top["end"]           # the truth lies in that top-level frame


def test_sliced_nested_child_is_a_run_of_lookup_slices(src):
    row, m = bf.compose_dynamic(np.random.default_rng(29), src, 4096, [bf.spec("think", 1024, children=[bf.spec("data", 256, records=4)])],
                                "caption", "mode", "nested", sized=True, sliced=32)
    parent = m["frames"][0]
    run = sorted((g for g in m["frames"] if g.get("slice_of") == 0 and g["ldepth"] == 1), key=lambda g: g["start"])
    think = [g for g in m["frames"] if g.get("slice_of") == 0 and g["ldepth"] == 0]
    assert len(run) == 8 and all(g["kind"] == "data" and g["unit"] == 256 for g in run) and (run[0]["start"] - parent["start"] - 2) % 32 == 0
    assert all(b["start"] - a["start"] == 32 for a, b in zip(run, run[1:])) and [g["n_records"] for g in run] == [1, 1, 1, 1, 0, 0, 0, 0]
    assert len(think) == 24 and all(g["kind"] == "think" and g["n_records"] == 0 for g in think)
    assert m["truth_depth"] == 1 and bf.frame_group(m["frames"][m["truth_frame"]]) == "data/256/depth1"


def test_capacity_and_length_by_form():
    rec = bf.capacity(1024, "A", "caption")["record_bytes"]
    sizes = (32, 64, 128, 256)
    assert [bf.frame_capacity(s, rec) for s in sizes] == [(s - 2) // rec for s in sizes]
    assert [bf.frame_capacity(s, rec, sized=True) for s in sizes] == [(s - 3) // rec for s in sizes]
    assert [bf.frame_capacity(s, rec, True, 32) for s in (32, 64, 128, 1024)] == [1, 2, 4, 32]
    assert bf.frame_bytes(1024) == 1024 and bf.frame_bytes(1024, 32) == 1027 and bf.frame_bytes(32, 32) == 32
    assert bf.open_tag("data", 1024) == bytes([DATA]) and bf.open_tag("think", 1024, True) == bytes([THINK]) + b"f"
    assert bf.frame_group(dict(kind="data", size=121, depth=0)) == "data/128/depth0"                # the 0.1.20 group, unchanged


def test_sliced_needs_sized(src):
    with pytest.raises(AssertionError):
        bf.compose_dynamic(np.random.default_rng(1), src, 1024, [bf.spec("data", 128, records=2)], sliced=32)


def test_sized_canary_frame(src):
    rows, meta = bf.canary_rows([src[i:i + 1200] for i in range(0, 4800, 1200)], 1024, "caption", "mode", seed=1, sized=True)
    p = 1 + meta["prefix_bytes"]
    f = rows["frame"][0][p:p + 32]
    assert f[:2] == bytes([DATA]) + b"a" and f[2] == CUE and f[-1] == END and meta["sized"] is True and meta["frame_bytes"] == 32
