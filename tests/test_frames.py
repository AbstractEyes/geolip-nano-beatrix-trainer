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
    assert "think/320/body" in res["by_kind_size_segment"] and "data/73/open" in res["by_kind_size_segment"]
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
    assert int((bf.frame_masks(m, ctx)["seg"] == bf.SEG_ID["open"]).sum()) == len(m["frames"])


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


def test_bits_by_frame_sums_to_the_window(src):
    rng = np.random.default_rng(13)
    specs = [bf.spec("data", 32, records=1), bf.spec("think", 256, children=[bf.spec("data", 64, records=2)])]
    metas = [bf.compose_dynamic(rng, src, 1024, specs, "caption", "mode", "any")[1] for _ in range(4)]
    res = bf.finish_frame_bits(bf.bits_by_frame(np.ones((4, 1024)), metas))
    cells = list(res["by_kind_size_segment"].values()) + list(res["by_kind_depth"].values()) + [res["outside"]]
    assert all(abs(c["bits"] - 1 / np.log(2.0)) < 1e-9 for c in cells)
    assert sum(v["n"] for v in res["by_kind_depth"].values()) + res["outside"]["n"] == 4 * 1024
    assert "data/64/open" in res["by_kind_size_segment"] and "data/depth2" in res["by_kind_depth"]
