# -*- coding: utf-8 -*-
"""The last-write screen (lastfix_screen.py): every arm starts from its v1 craft's start, the fix arms that promise it equal the plain
craft at birth, the presets carry the recipe and the slice of the diet, the gauge reads what it says, the drift-memory estimator reads
known signals. CPU only; the Hub is never touched."""
import numpy as np
import pytest
import torch

from geolip.alephllm.model.alephlm import AlephLM
from geolip.nano_beatrix import lastfix_screen as LS, nano_v1


def test_the_arms_are_the_bare_crafts_and_the_four_forms():
    from geolip.alephllm.model.lastfix import MODES
    assert len({a["name"] for a in LS.ARMS}) == len(LS.ARMS) and all(a["mode"] in MODES for a in LS.ARMS)
    castor = [a for a in LS.ARMS if a["craft"] == "nano-castor-v1"]
    assert {a["mode"] for a in castor} == set(MODES) and {a["seed"] for a in castor if not a["mode"]} == {3, 4}
    assert all(a["seed"] == nano_v1.CRAFTS[a["craft"]]["seed"] for a in LS.ARMS if not a["name"].endswith("s4"))
    assert all(a["seed"] == 4 and a["craft"] == "nano-castor-v1" for a in LS.ARMS if a["name"].endswith("s4"))


@pytest.mark.parametrize("name", ["c-register", "c-temperature", "c-gain", "c-normwrite"])
def test_a_fix_arm_starts_from_the_v1_start(name):
    arm = LS.BY_NAME[name]
    sd = LS.arm_state(arm)
    v1 = nano_v1.birth_state("castor-both", 3)
    assert all(torch.equal(sd[k], v1[k]) for k in v1)
    extra = set(sd) - set(v1)
    assert extra and all(k.startswith("blocks.5.fix.") for k in extra)
    again = LS.arm_state(arm)
    assert set(again) == set(sd) and all(torch.equal(again[k], sd[k]) for k in sd)          # twice the same, bit for bit


def test_gain_register_and_temperature_are_the_plain_craft_at_birth():
    torch.manual_seed(0)
    x = torch.randint(32, 127, (2, 128))
    out = {}
    for name in ("c-bare-s3", "c-register", "c-temperature", "c-gain"):
        m = AlephLM(LS.arm_cfg(LS.BY_NAME[name]))
        m.load_state_dict(LS.arm_state(LS.BY_NAME[name]), strict=True)
        m.eval()
        with torch.no_grad():
            out[name] = m(x)[0]
    assert all(torch.equal(out[n], out["c-bare-s3"]) for n in out)


def test_the_start_file_is_written_once_and_checked(tmp_path):
    arm = LS.BY_NAME["c-gain"]
    p = LS.write_arm_birth(arm, str(tmp_path))
    assert LS.write_arm_birth(arm, str(tmp_path)) == p
    from safetensors.torch import save_file
    save_file(LS.arm_state(LS.BY_NAME["c-bare-s4"]), p)                        # another start under the arm's name: refused
    with pytest.raises(AssertionError):
        LS.write_arm_birth(arm, str(tmp_path))


def test_the_preset_is_the_2s_recipe_on_the_start_of_the_diet():
    p = LS.arm_preset(LS.BY_NAME["pdc-register"], "b.safetensors")
    t, m = p.train, p.model
    assert (t.muon_lr, t.adam_lr, t.warmup_steps, t.grad_clip, t.governor, t.phase_seed_offset, t.eval_every) == (2e-2, 3e-4, 200, 1.0, "",
                                                                                                              True, 500)
    assert t.micro_batch * t.grad_accum * m.context == 262_144 and m.bank_last_fix == "register" and m.canon == "AC"
    assert p.prefix == "v1_lastfix/pdc-register" and p.hf_repo == LS.REPO and p.init_from == {"file": "b.safetensors", "step": 0}
    ph = p.curriculum
    assert [(x["name"], x["dataset"], x["planned_tokens"]) for x in ph] == [("warmup_wikitext", "wikitext-103", 300_000_000),
                                                                           ("fineweb_main", "fineweb-edu", 1_000_000_000)]
    assert sum(-(-x["planned_tokens"] // 262_144) for x in ph) == 4960
    assert ph == nano_v1.diet()[:1] + [dict(nano_v1.diet()[1], planned_tokens=1_000_000_000)]   # the v1 diet's own phases and seeds


def test_the_gauge_reads_a_planted_direction():
    m = AlephLM(LS.arm_cfg(LS.BY_NAME["c-bare-s3"]))
    m.load_state_dict(LS.arm_state(LS.BY_NAME["c-bare-s3"]), strict=True)
    torch.manual_seed(1)
    docs = [[0] + torch.randint(32, 127, (511,)).tolist() for _ in range(4)]
    g0 = LS.lastwrite_gauge(m, docs)
    v = torch.zeros(256)
    v[7] = 1.0
    m.blocks[-1].bank.register_forward_hook(lambda mod, i, o: o + 500.0 * v.to(o.dtype))   # the last bank writes one big fixed direction
    g1 = LS.lastwrite_gauge(m, docs)
    assert 0 <= g0["share_u"]["5"] < 0.5 and g1["share_u"]["5"] > 0.99 and g1["bank"]["share_own_mean"] > 0.99
    assert abs(float(g1["_u"][7].abs()) - 1.0) < 1e-3 and g1["rank"]["4"] == g0["rank"]["4"]   # the blocks before it read as before
    sc = LS.gauge_scalars(g1)
    assert sc["lastwrite/share_last"] > 0.99 and all(np.isfinite(list(sc.values())))


def test_the_drift_memory_estimator_reads_known_signals():
    g = np.random.default_rng(0)
    white = LS._analyze([g.standard_normal(2047) for _ in range(24)])
    walk = LS._analyze([np.cumsum(g.standard_normal(2047)) for _ in range(24)])
    assert white["H_median"] == pytest.approx(0.5, abs=0.05) and walk["H_median"] == pytest.approx(1.5, abs=0.06)
    assert white["H_ci95"][0] <= white["H_median"] <= white["H_ci95"][1] and len(white["octave_slopes"]) == len(LS.SCALES) - 2


def test_the_table_compares_each_arm_with_bare_at_the_same_step():
    def rec(mode, bpb, share):
        g = {"step": 4960, "val_bpb": bpb, "share_u": {"5": share, "nf": share}, "rank": {str(i): 10.0 for i in range(6)},
             "bank": {"share_own_mean": share, "size": 100.0}}
        return {"arm": {"mode": mode}, "gauges": [g], "exit_memory": {"exit": {"H_median": 0.6, "H_ci95": [0.55, 0.65]}}}
    t = LS.table({"c-bare-s3": rec("", 1.5, 0.95), "c-register": rec("register", 1.5004, 0.6)})
    assert "| c-register | register | 4,960 | 1.5004 | +0.0004 | 0.600 |" in t and "| c-bare-s3 | none | 4,960 | 1.5000 | +0.0000 |" in t


@pytest.mark.parametrize("name,wd", [("c-head-adam", 0.0), ("c-head-adamw0.1", 0.1), ("c-head-adamw1.2", 1.2)])
def test_an_output_layer_arm_is_the_plain_start_with_its_readout_on_adamw(name, wd):
    from geolip.alephllm.train.optim import Muon, build_optimizers
    arm = LS.BY_NAME[name]
    sd, bare = LS.arm_state(arm), LS.arm_state(LS.BY_NAME["c-bare-s3"])
    assert set(sd) == set(bare) and all(torch.equal(sd[k], bare[k]) for k in sd)            # the plain v1 start, bit for bit
    p = LS.arm_preset(arm, "b.safetensors")
    t = p.train
    assert t.decay_params == LS.HEAD_PARAMS and t.decay_wd == wd and p.model.bank_last_fix == ""
    assert (t.muon_lr, t.adam_lr, t.warmup_steps, t.grad_clip, t.eval_every, p.prefix) == (2e-2, 3e-4, 200, 1.0, 500, f"v1_lastfix/{name}")
    m = AlephLM(p.model)
    opts = build_optimizers(m, t.muon_lr, t.muon_momentum, t.adam_lr, decay_prefixes=t.decay_params, decay_wd=t.decay_wd)
    named = dict(m.named_parameters())
    readout = {id(named["head.w_h.weight"]), id(named["head.w_s.weight"])}
    g1 = opts[1].param_groups[1]
    assert {id(x) for x in g1["params"]} == readout and g1["weight_decay"] == wd and opts[1].param_groups[0]["weight_decay"] == 0.0
    assert isinstance(opts[0], Muon) and not readout & {id(x) for x in opts[0].param_groups[0]["params"]}
    assert LS.arm_preset(LS.BY_NAME["c-bare-s3"], "b.safetensors").train.decay_params == ()     # every other arm: the 2s split


def test_the_gauge_reads_the_output_layer():
    m = AlephLM(LS.arm_cfg(LS.BY_NAME["c-bare-s3"]))
    m.load_state_dict(LS.arm_state(LS.BY_NAME["c-bare-s3"]), strict=True)
    with torch.no_grad():
        m.head.w_s.weight.zero_()                                    # the readout alone: the head is linear in the post-norm state
    torch.manual_seed(1)
    docs = [[0] + torch.randint(32, 127, (511,)).tolist() for _ in range(4)]
    h0 = LS.lastwrite_gauge(m, docs)["head"]
    assert all(np.isfinite(v) for v in h0.values()) and 0 < h0["room"] <= 1 and h0["loudness"] == pytest.approx(h0["room"] * h0["gain_content"])
    with torch.no_grad():
        m.head.w_h.weight.mul_(2.0)                                  # a readout twice as large reads every direction twice as loudly
    g1 = LS.lastwrite_gauge(m, docs)
    h1 = g1["head"]
    assert h1["w_h_fro"] == pytest.approx(2 * h0["w_h_fro"], rel=1e-5) and h1["w_h_top_sv"] == pytest.approx(2 * h0["w_h_top_sv"], rel=1e-4)
    assert all(h1[k] == pytest.approx(2 * h0[k], rel=1e-3) for k in ("gain_content", "gain_random", "gain_u"))
    sc = LS.gauge_scalars(g1)
    assert sc["lastwrite/head_w_h_fro"] == pytest.approx(h1["w_h_fro"]) and "lastwrite/head_loudness" in sc


def test_the_table_labels_the_output_layer_arms():
    def rec(arm, head):
        g = {"step": 4960, "val_bpb": 1.5, "share_u": {"5": 0.9, "nf": 0.9}, "rank": {str(i): 10.0 for i in range(6)},
             "bank": {"share_own_mean": 0.9, "size": 100.0}}
        if head:
            g["head"] = head
        return {"arm": arm, "gauges": [g]}
    t = LS.table({"c-bare-s3": rec({"mode": ""}, None),
                  "c-head-adamw0.1": rec(dict(LS.BY_NAME["c-head-adamw0.1"]), {"w_h_fro": 30.0, "loudness": 1.5})})
    assert "| c-head-adamw0.1 | head AdamW 0.1 | 4,960 |" in t and t.rstrip().endswith("| 30.0 / 1.50 |") and "| - | - |" in t


def test_the_carry_over_arms_start_from_their_plain_runs_and_compare_with_them():
    for name, plain in (("pdc-head-adamw0.1", "pdc-bare"), ("pdc-head-adamw1.2", "pdc-bare"), ("c-head-adamw0.1-s4", "c-bare-s4"),
                        ("c-head-adamw1.2-s4", "c-bare-s4")):
        arm = LS.BY_NAME[name]
        assert LS.PLAIN[(arm["craft"], arm["seed"])] == plain
        sd, ref = LS.arm_state(arm), LS.arm_state(LS.BY_NAME[plain])
        assert set(sd) == set(ref) and all(torch.equal(sd[k], ref[k]) for k in sd)
        t = LS.arm_preset(arm, "b.safetensors").train
        assert t.decay_params == LS.HEAD_PARAMS and t.decay_wd == arm["head_wd"]

    def rec(arm, bpb):
        g = {"step": 4960, "val_bpb": bpb, "share_u": {"5": 0.9, "nf": 0.9}, "rank": {str(i): 10.0 for i in range(6)},
             "bank": {"share_own_mean": 0.9, "size": 100.0}}
        return {"arm": dict(arm), "gauges": [g]}
    recs = {n: rec(LS.BY_NAME[n], b) for n, b in (("c-bare-s3", 1.50), ("c-bare-s4", 1.60), ("pdc-bare", 1.40),
                                                  ("c-head-adamw0.1-s4", 1.61), ("pdc-head-adamw0.1", 1.39))}
    t = LS.table(recs)
    assert "| c-head-adamw0.1-s4 | head AdamW 0.1 | 4,960 | 1.6100 | +0.0100 |" in t and "| pdc-head-adamw0.1 | head AdamW 0.1 | 4,960 | 1.3900 | -0.0100 |" in t
