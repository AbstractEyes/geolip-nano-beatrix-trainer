# -*- coding: utf-8 -*-
"""The nano v1 crafts (nano_v1.py, bodies.py, the gauge-only read): the births, the diet, the preset, the stream in a child process,
the gauge record's TensorBoard map. CPU only; the Hub is never touched (the stream tests read the library's local synthetic corpus)."""
import json
import math
import os

import pytest
import torch

from geolip.nano_beatrix import bodies, nano_v1
from geolip.nano_beatrix.names import run_tag

FIX = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")
BODIES = json.load(open(os.path.join(FIX, "nano_v1_bodies.json"), encoding="utf-8"))


def _norm(d):
    return json.loads(json.dumps(d))


def test_the_crafts_are_the_strongest_seeds_of_record():
    assert {n: (c["arm"], c["seed"]) for n, c in nano_v1.CRAFTS.items()} == {
        "nano-pollux-dual-canon-v1": ("altdc-both", 1), "nano-pollux-canon-v1": ("polluxc-both", 2),
        "nano-pollux-dual-v1": ("altd-both", 1), "nano-castor-v1": ("castor-both", 3)}
    assert len({c["code"] for c in nano_v1.CRAFTS.values()}) == 4


@pytest.mark.parametrize("arm", sorted(BODIES))
def test_each_body_is_the_config_its_bed_run_stored(arm):
    stored = _norm(BODIES[arm]["config"])
    mine = _norm(bodies.craft_cfg(arm, 1024).to_dict())
    from geolip.alephllm.presets import AlephLMConfig
    default = _norm(AlephLMConfig().to_dict())
    assert all(mine[k] == default[k] for k in mine if k not in stored)       # a newer library key holds its default (the flag off)
    assert {k: v for k, v in mine.items() if k in stored} == stored


@pytest.mark.parametrize("name", list(nano_v1.CRAFTS))
def test_the_birth_is_the_bed_birth_carried_to_4096(name):
    c = nano_v1.CRAFTS[name]
    a, b = nano_v1.birth_state(c["arm"], c["seed"]), nano_v1.birth_state(c["arm"], c["seed"])
    assert set(a) == set(b) and all(torch.equal(a[k], b[k]) for k in a)      # twice the same, bit for bit
    pos = a["embed.pos"]
    assert pos.shape == (1, 4096, 256) and bool((pos[:, 1024:] == 0).all()) and float(pos[:, :1024].abs().sum()) > 0
    rec = BODIES[c["arm"]]
    # the bed's own construction: the seed, then the craft at 1,024 (bed_train.Craft without --init)
    from geolip.alephllm.model.alephlm import AlephLM
    torch.manual_seed(c["seed"])
    m = AlephLM(bodies.craft_cfg(c["arm"], 1024))
    assert m.param_count() == rec["params_1024"]
    for k, v in m.state_dict().items():
        assert torch.equal(a[k][:, :1024] if k == "embed.pos" else a[k], v), k
    # the start loads strictly into the craft at 4,096 (the trainer's weights-only start) with the record's parameter count
    m4 = AlephLM(bodies.craft_cfg(c["arm"], 4096))
    m4.load_state_dict(a, strict=True)
    assert m4.param_count() == rec["params_4096"]


def test_the_birth_file_is_written_once_and_checked(tmp_path):
    p = nano_v1.write_birth("nano-castor-v1", str(tmp_path))
    t0 = os.path.getmtime(p)
    assert nano_v1.write_birth("nano-castor-v1", str(tmp_path)) == p and os.path.getmtime(p) == t0
    from safetensors.torch import save_file
    sd = nano_v1.birth_state("castor-both", 2)                                # another seed under the craft's name: refused
    save_file(sd, p)
    with pytest.raises(AssertionError):
        nano_v1.write_birth("nano-castor-v1", str(tmp_path))


def test_the_diet_is_the_2s_diet():
    ph = nano_v1.diet()
    assert [p["name"] for p in ph] == (["warmup_wikitext", "fineweb_main"] + [f"curriculum_s{i}" for i in range(9)]
                                        + ["anneal_nochat", "anneal_mix"])
    assert [p["dataset"] for p in ph][:2] == ["wikitext-103", "fineweb-edu"] and ph[-1]["dataset"] == "anneal-mix"
    assert sum(p["planned_tokens"] for p in ph) == 16_100_000_000
    steps = 0
    for p in ph:
        steps += math.ceil(p["planned_tokens"] / 262_144)
    assert steps == 61_422                                                     # the 2s twins' last boundary (manifest.set_cursor)
    assert all(p["status"] == "planned" for p in ph)


def test_the_preset_is_the_2s_recipe(tmp_path):
    p = nano_v1.preset("nano-pollux-dual-canon-v1", str(tmp_path / "b.safetensors"))
    t, m = p.train, p.model
    assert (t.muon_lr, t.adam_lr, t.warmup_steps, t.grad_clip, t.governor, t.head_addr_frozen, t.phase_seed_offset, t.precision,
            t.loss_form, t.seed) == (2e-2, 3e-4, 200, 1.0, "", False, True, "bf16", "ce", 1337)
    assert t.micro_batch * t.grad_accum * m.context == 262_144 and m.context == 4096
    assert m.name == "nano-pollux-dual-canon-v1" and m.canon == "AC" and m.hub_dual and m.hub_layers == (0, 2, 4)
    assert p.prefix == m.name and p.hf_repo == nano_v1.REPO and p.data_scale == 1.0
    assert p.init_from == {"file": str(tmp_path / "b.safetensors"), "step": 0}
    with pytest.raises(AssertionError):
        nano_v1.preset("nano-castor-v1", "x", micro_batch=16, grad_accum=2)   # a step that is not the 2s step


def test_the_child_stream_is_the_library_stream_exactly():
    from geolip.alephllm.data.streams import build_stream
    from geolip.alephllm.data.tokenizers import build_tokenizer
    tok = build_tokenizer("byte-trigram")
    plain = build_stream("synthetic", tok, 64, 3, seed=11)
    ps = nano_v1.ProcessStream("synthetic", "byte-trigram", 64, 3, seed=11, depth=2)
    try:
        got = [ps.next_batch() for _ in range(4)]
        want = [plain.next_batch() for _ in range(4)]
        assert all(torch.equal(g, w) for g, w in zip(got, want)) and got[0].dtype == torch.int64
        st = ps.state_dict()                                                   # the state after the 4th batch handed out
        assert ps.rows_consumed == st["rows_consumed"] == plain.rows_consumed
    finally:
        ps.close()
    again = build_stream("synthetic", tok, 64, 3, seed=11)
    again.load_state_dict(st)
    nxt = [plain.next_batch() for _ in range(2)]
    assert all(torch.equal(again.next_batch(), w) for w in nxt)               # a checkpoint resumes where the trainer stood
    ps2 = nano_v1.ProcessStream("synthetic", "byte-trigram", 64, 3, seed=11, state=st, depth=2)
    try:
        assert all(torch.equal(ps2.next_batch(), w) for w in nxt)              # and so does a child started from it
    finally:
        ps2.close()


def test_a_child_failure_raises_in_the_trainer():
    ps = nano_v1.ProcessStream("no-such-corpus", "byte-trigram", 64, 3, seed=1, depth=2, timeout=120)
    try:
        with pytest.raises(RuntimeError, match="no-such-corpus"):
            ps.next_batch()
    finally:
        ps.close()


def test_the_gauge_record_maps_to_tensorboard():
    rec = json.load(open(os.path.join(FIX, "castor_s3_anneal1_record.json"), encoding="utf-8"))
    sc = nano_v1.gauge_scalars(rec)
    assert sc["bed/text/heldout_bits"] == pytest.approx(rec["heldout_bpb"])
    assert sc["bed/frame/copy_whole"] == pytest.approx(0.8385, abs=5e-5)       # the splat tables' copy for this record
    assert sc["bed/frame/nested_whole"] == pytest.approx(0.7969, abs=5e-5)
    assert sc["bed/frame/redefined_key_whole"] == pytest.approx(0.5625, abs=5e-5)
    assert sc["bed/scaling/frame_1024_whole"] == pytest.approx(rec["fetch_by_frame"]["data/1024/depth0"]["full_value"])
    assert {f"bed/scaling/frame_{s}_whole" for s in (32, 128, 256, 512, 1024)} <= set(sc)
    assert {f"bed/window/planted_{d}_bits" for d in (64, 256, 512, 896)} <= set(sc)
    assert any(k.startswith("bed/window/position_") for k in sc) and any(k.startswith("bed/canary/") for k in sc)
    assert all(isinstance(v, float) and math.isfinite(v) for v in sc.values())


def test_a_gauge_read_has_its_own_tag():
    a = nano_v1.gauge_args("nano-castor-v1", "v1c_4000", "/out")
    tag = run_tag(a)
    assert tag.startswith("castor-both_s1_ctx4096_frdyn_cap_mode_sized_sl32_sz32-128-256-512-1024_r4_n30_o0_askany_p75")
    assert tag.endswith("_from_v1c_4000_gauge") and "_L" not in tag
    b = dict(a, gauge_only="0")
    assert not run_tag(b).endswith("_gauge")


def test_a_failed_gauge_is_skipped_and_leaves_no_weights(tmp_path):
    import sys

    class T:                                                                   # the trainer as the gauge sees it
        step = 7
        raw_model = torch.nn.Linear(2, 2)
    rec = nano_v1.bed_gauge(T(), "nano-castor-v1", str(tmp_path), gauge_cmd=[sys.executable, "-c", "import sys; sys.exit(3)"],
                            gauge_gb=1.0)
    assert rec is None                                                         # logged and skipped: the train goes on
    assert not (tmp_path / "bed" / "v1c_7").exists() and (tmp_path / "bed" / "gauge_logs" / "v1c_7.log").exists()


def test_the_card_while_training_names_every_craft():
    card = nano_v1.card_in_training()
    assert card.startswith("---\nlicense: cc-by-nc-sa-4.0") and all(f"`{n}`" in card for n in nano_v1.CRAFTS)
    assert "61,422 steps" in card and "trust_remote_code=True" in card


def test_phase_index_reads_without_activating():
    from geolip.alephllm.train.manifest import RunManifest

    class T:
        manifest = RunManifest.fresh("x", {}, nano_v1.diet())
    t = T()
    assert nano_v1.phase_index(t) == 0 and t.manifest.phases[0]["status"] == "planned"
    t.manifest.phases[0]["status"] = "done"
    t.manifest.phases[1]["status"] = "active"
    assert nano_v1.phase_index(t) == 1
    for p in t.manifest.phases:
        p["status"] = "done"
    assert nano_v1.phase_index(t) is None
