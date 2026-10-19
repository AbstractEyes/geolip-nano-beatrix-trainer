# -*- coding: utf-8 -*-
"""The micro tier (0.1.57): the sizes scale the nano body by one rule and the nano stays bit for bit; the counts; the crafts; the size
flows to the births, the presets, the gauge child and the run tag; the scan's preset, projection and table (the scan itself needs a card
and the stream: the notebook)."""
import os

import pytest
import torch
from safetensors.torch import load_file

from geolip.nano_beatrix import micro, nano_v1
from geolip.nano_beatrix.bodies import MILLIONS, NANO_ONLY, SIZES, craft_cfg, size_fields
from geolip.nano_beatrix.names import run_tag

MICRO = ["micro26-pollux-dual-v1", "micro45-pollux-dual-v1", "micro66-pollux-dual-v1"]


@pytest.mark.parametrize("size", sorted(SIZES))
def test_the_sizes_scale_every_address_with_the_width(size):
    s, cfg = SIZES[size], craft_cfg("altd-both", 64, size=size)
    d, L = s["d_model"], s["n_layers"]
    assert (cfg.d_model, cfg.n_layers, cfg.n_heads) == (d, L, s["n_heads"])
    assert (cfg.hub_K, cfg.hub_D, cfg.head_K, cfg.head_D, cfg.bank_ff, cfg.bank_experts) == (d // 8, d // 16, d // 4, d // 16, d, 7)
    assert cfg.hub_layers == tuple(range(0, L, 2)) and cfg.hub_dual and cfg.attn_rotary and cfg.bank_byte_table == "add"
    assert cfg.n_heads == (8 if size == "nano" else d // 64) and d % cfg.n_heads == 0
    assert craft_cfg("castor-both", 64, size=size).hub_layers == () and craft_cfg("hubd-both", 64, size=size).hub_layers == tuple(range(L))


def test_the_nano_is_the_bed_body_bit_for_bit_and_the_size_rides_in_args():
    assert craft_cfg("altd-both", 1024).to_dict() == craft_cfg("altd-both", 1024, size="nano").to_dict()
    assert size_fields("nano") == dict(SIZES["nano"], hub_K=32, hub_D=16, bank_ff=256, head_K=64, head_D=16)
    assert craft_cfg("altd-both", 64, {"size": "micro45"}).d_model == 512 and craft_cfg("altd-both", 64, {"size": "micro45"}, size="micro26").d_model == 384
    with pytest.raises(AssertionError):
        craft_cfg("altd-both", 64, size="giga")
    for attn in NANO_ONLY:                               # the six-block layouts and the bigger address are the nano's
        craft_cfg(f"{attn}-both", 64)
        with pytest.raises(AssertionError):
            craft_cfg(f"{attn}-both", 64, size="micro26")


def test_the_counts_at_the_v1_context():
    from geolip.alephllm.model.alephlm import AlephLM
    for size, m in MILLIONS.items():
        n = AlephLM(craft_cfg("altd-both", nano_v1.CTX, size=size)).param_count()
        assert round(n / 1e6, 1) == m, (size, n)
    assert AlephLM(craft_cfg("altd-both", nano_v1.CTX)).param_count() == 9_267_281


def test_the_micro_crafts_and_their_versions():
    for n in MICRO:
        c, c11 = nano_v1.ALL[n], nano_v1.ALL[f"{n}.1"]
        assert c["arm"] == "altd-both" and c["seed"] == 1 and c["size"] == n.split("-")[0] and c["code"] == f"m{n[5:7]}pd"
        assert c11["train"] == nano_v1.V11_TRAIN and c11["tag"] == "v11" and nano_v1.size_of(n) == nano_v1.size_of(f"{n}.1") == c["size"]
        assert nano_v1.version_of([n]) == "v1" and nano_v1.version_of([f"{n}.1"]) == "v1.1"
        assert nano_v1.body_of(f"{n}.1") == f"{n}.1" and nano_v1.arm_text(n) is None
        assert nano_v1.plan_steps(n, nano_v1.TOKENS_PER_STEP) == 61_422          # the 2s diet is their plan
    codes = [c["code"] + c.get("tag", "v1") for c in nano_v1.ALL.values()]
    assert len(codes) == len(set(codes)), "a gauge folder name would collide"
    assert nano_v1.size_of("nano-pollux-dual-v1.1-mx-cp2") == "nano"
    assert list(micro.SCAN_NAMES) == ["nano-pollux-dual-v1.1"] + [f"{n}.1" for n in MICRO]


def test_the_size_reaches_the_gauge_child_and_the_tag():
    a = nano_v1.gauge_args("micro26-pollux-dual-v1.1", "v11m26pd_100", "out")
    b = nano_v1.gauge_args("nano-pollux-dual-v1.1", "v11pd_100", "out")
    assert a["size"] == "micro26" and "size" not in b
    ta, tb = run_tag(a), run_tag(b)
    assert "_micro26" in ta and "_micro26" not in tb and ta.replace("_micro26", "").replace("m26pd", "pd") == tb


def test_the_birth_of_a_micro_craft_is_the_bed_build_padded(tmp_path):
    name = "micro26-pollux-dual-v1"
    path = nano_v1.write_birth(name, str(tmp_path))
    sd = load_file(path)
    pos = sd["embed.pos"]
    assert pos.shape[1:] == (nano_v1.CTX, 384) and torch.count_nonzero(pos[:, nano_v1.BIRTH_CTX:]) == 0 and torch.count_nonzero(pos[:, :nano_v1.BIRTH_CTX]) > 0
    from geolip.alephllm.model.alephlm import AlephLM
    m = AlephLM(craft_cfg("altd-both", nano_v1.CTX, size="micro26"))        # the state dict carries the buffers too (33,024 beyond the parameters)
    assert sum(v.numel() for v in sd.values()) == sum(v.numel() for v in m.state_dict().values()) == 25_820_865 and m.param_count() == 25_787_841
    from safetensors import safe_open
    with safe_open(path, "pt") as f:
        assert f.metadata()["size"] == "micro26" and f.metadata()["arm"] == "altd-both"
    assert nano_v1.write_birth(name, str(tmp_path)) == path                     # a present file is checked, not rewritten


def test_the_scan_preset_projection_and_table():
    p = micro.scan_preset("micro45-pollux-dual-v1.1", "birth.safetensors", 16, 4, 300)
    assert (p.model.d_model, p.model.n_layers, p.model.name) == (512, 8, "micro45-pollux-dual-v1.1-scan")
    assert len(p.curriculum) == 1 and p.curriculum[0]["dataset"] == "fineweb-edu" and p.curriculum[0]["planned_tokens"] == 300 * 262_144
    assert p.train.eval_every >= 10 ** 9 and p.train.ckpt_every >= 10 ** 9 and p.train.decay_params == nano_v1.READOUT and p.init_from == {"file": "birth.safetensors", "step": 0}
    assert micro.scan_preset("micro45-pollux-dual-v1", "b", 16, 4, 10).train.decay_wd == 0.0
    pr = micro.projection(610_000.0, 9_267_281, 16_100_000_000)
    assert round(pr["day_B"], 1) == 52.7 and round(pr["week_B"]) == 369 and round(pr["bytes_per_param_week"]) == 39_810 and round(pr["diet_hours"], 1) == 7.3
    assert micro.projection(0.0, 1)["diet_hours"] == float("inf") and micro.diet_bytes() == 16_100_000_000
    res = dict(crafts=[dict(name="x", params=9_267_281, day_B=52.7, week_B=369.0, bytes_per_param_week=39_818.0, diet_hours=7.3, diet4x_hours=29.3,
                            stream=dict(micro_batch=16, grad_accum=4, bytes_s=610_000.0, steps_s=2.33, peak_gb=7.5))])
    t = micro.scan_table(res)
    assert t.count("\n") == 2 and "| x | 9.3 M | 16 x 4 | 610 | 2.33 | 7.5 | 52.7 | 369 | 39,818 | 7.3 | 29.3 |" in t
    assert os.path.basename(__file__) == "test_micro.py"
