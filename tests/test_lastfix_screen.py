# -*- coding: utf-8 -*-
"""The last-write screen (lastfix_screen.py): every arm starts from its v1 craft's start, the fix arms that promise it equal the plain
craft at birth, the presets carry the recipe and the slice of the diet, the gauge reads what it says, the drift-memory estimator reads
known signals. CPU only; the Hub is never touched."""
import math

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
    assert "| c-head-adamw0.1 | head AdamW 0.1 | 4,960 |" in t and t.rstrip().endswith("| 30.0 / 1.50 | - | - |") and "| - | - | - | - |" in t


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


# ------------------------------------------------------------------ the head's dial arms (0.1.34)
HEAD_NEW = {"dial": {"head.w_t", "head.a_bar"}, "dialraw": {"head.w_t", "head.a_bar"}, "dialplus": {"head.w_t", "head.a_bar"},
            "pergain": {"head.w_g.weight", "head.a_bar"}}


def test_the_head_arms_are_the_seven_at_the_v1_1_recipe():
    assert LS.HEAD_ARMS == ["c-dial-s3", "c-dial-s4", "c-dialfrz-s3", "c-dialraw-s3", "c-dialplus-s3", "c-pergain-s3", "pdc-dial-s1"]
    for n in LS.HEAD_ARMS:
        a = LS.BY_NAME[n]
        ref = LS.BY_NAME[a["ref"]]
        assert a["head_wd"] == 0.1 and a["mode"] == "" and ref["head_wd"] == 0.1 and not ref.get("head_mode")
        assert (ref["craft"], ref["seed"]) == (a["craft"], a["seed"])
        assert LS.head_decay_params(a) == LS.HEAD_PARAMS                    # a dial's own weights stay on pure Adam


@pytest.mark.parametrize("name", LS.HEAD_ARMS)
def test_a_head_arm_starts_from_its_reference_runs_start_with_the_dial_born_silent(name):
    arm = LS.BY_NAME[name]
    sd, ref = LS.arm_state(arm), LS.arm_state(LS.BY_NAME[arm["ref"]])
    new = set(sd) - set(ref)
    assert new == HEAD_NEW[arm["head_mode"]] and all(int(torch.count_nonzero(sd[k])) == 0 for k in new)
    assert set(ref) <= set(sd) and all(torch.equal(sd[k], ref[k]) for k in ref)
    p = LS.arm_preset(arm, "b.safetensors")
    t = p.train
    assert p.model.head_mode == arm["head_mode"] and p.model.bank_last_fix == ""
    assert t.decay_params == LS.head_decay_params(arm) and t.decay_wd == 0.1 and t.head_addr_frozen is bool(arm.get("addr_frozen", False))
    assert LS.arm_preset(LS.BY_NAME["c-head-adamw0.1"], "b.safetensors").train.head_addr_frozen is False


def test_a_head_arm_is_the_plain_craft_at_birth():
    torch.manual_seed(0)
    x = torch.randint(32, 127, (2, 128))
    ref = AlephLM(LS.arm_cfg(LS.BY_NAME["c-head-adamw0.1"]))
    ref.load_state_dict(LS.arm_state(LS.BY_NAME["c-head-adamw0.1"]), strict=True)
    ref.eval()
    with torch.no_grad():
        want = ref(x)[0]
        h = ref.nf(ref._trunk(ref.embed(x), x))
    for name in ("c-dial-s3", "c-dialplus-s3", "c-pergain-s3", "c-dialfrz-s3", "c-dialraw-s3"):
        m = AlephLM(LS.arm_cfg(LS.BY_NAME[name]))
        m.load_state_dict(LS.arm_state(LS.BY_NAME[name]), strict=True)
        m.eval()
        with torch.no_grad():
            assert torch.equal(m(x)[0], want), name                      # the plain craft at birth, bit for bit
            m.head.train()
            tr = m.head(h)                                              # a training forward moves nothing at birth (the running mean stays 0)
            m.head.eval()
            assert torch.equal(tr, m.head(h)) and float(m.head.a_bar.abs().max()) == 0.0, name


def test_the_gauge_reads_the_head_toggle_and_the_dial():
    torch.manual_seed(1)
    docs = [[0] + torch.randint(32, 127, (511,)).tolist() for _ in range(4)]
    plain = AlephLM(LS.arm_cfg(LS.BY_NAME["c-head-adamw0.1"]))
    plain.load_state_dict(LS.arm_state(LS.BY_NAME["c-head-adamw0.1"]), strict=True)
    h0 = LS.lastwrite_gauge(plain, docs)["head"]
    assert h0["toggle_bits"] == 0.0 and h0["bpb_docs"] == h0["bpb_docs_head_off"] > 0 and "dial_log_std" not in h0   # the carrying read at zero
    assert "toggle_perbyte_bits" not in h0
    m = AlephLM(LS.arm_cfg(LS.BY_NAME["c-dial-s3"]))
    m.load_state_dict(LS.arm_state(LS.BY_NAME["c-dial-s3"]), strict=True)
    h1 = LS.lastwrite_gauge(m, docs)["head"]
    assert h1["toggle_bits"] == 0.0 and h1["dial_log_std"] == 0.0 and h1["dial_sat"] == 0.0 and h1["dial_w_norm"] == 0.0    # the dial at 1
    assert h1["toggle_perbyte_bits"] == 0.0 and h1["toggle_global_bits"] == 0.0 and h1["dial_log_hold"] == 0.0
    with torch.no_grad():
        m.head.w_t.normal_()
    g = LS.lastwrite_gauge(m, docs)
    h2 = g["head"]
    assert h2["toggle_bits"] != 0.0 and h2["dial_log_std"] > 0 and 0 <= h2["dial_sat"] <= 1 and h2["dial_w_norm"] > 0
    assert all(np.isfinite(h2[k]) for k in ("dial_corr_entropy", "dial_corr_share_u", "addr_erank", "addr_usage_ppl", "addr_drift",
                                             "dial_log_mean", "dial_center"))
    assert abs(h2["dial_corr_entropy"]) <= 1 + 1e-6 and 1 <= h2["addr_erank"] <= 16 + 1e-6
    assert h2["toggle_perbyte_bits"] + h2["toggle_global_bits"] == pytest.approx(h2["toggle_bits"], abs=1e-9)
    assert np.isfinite(h2["dial_corr_entropy_dialed"]) and h2["toggle_perbyte_bits"] != 0.0 and m.head.dial_hold is None
    sc = LS.gauge_scalars(g)
    assert sc["lastwrite/head_toggle_bits"] == pytest.approx(h2["toggle_bits"]) and "lastwrite/head_dial_log_std" in sc


def test_the_table_compares_a_head_arm_with_its_reference_run():
    def rec(arm, bpb, head=None):
        g = {"step": 4960, "val_bpb": bpb, "share_u": {"5": 0.9, "nf": 0.3}, "rank": {str(i): 10.0 for i in range(6)},
             "bank": {"share_own_mean": 0.9, "size": 100.0}}
        if head:
            g["head"] = head
        return {"arm": dict(arm), "gauges": [g]}
    recs = {"c-bare-s3": rec(LS.BY_NAME["c-bare-s3"], 1.50),
            "c-head-adamw0.1": rec(LS.BY_NAME["c-head-adamw0.1"], 1.52, {"w_h_fro": 14.0, "loudness": 1.4, "toggle_bits": 0.0003}),
            "c-dial-s3": rec(LS.BY_NAME["c-dial-s3"], 1.51, {"w_h_fro": 13.0, "loudness": 1.3, "toggle_bits": 0.0123, "dial_log_std": 0.25,
                                                             "dial_sat": 0.05, "dial_corr_entropy": -0.4})}
    t = LS.table(recs)
    assert "| c-dial-s3 | head dial + AdamW 0.1 | 4,960 | 1.5100 | -0.0100 |" in t          # against c-head-adamw0.1, not c-bare-s3
    assert "| 13.0 / 1.30 | +0.0123 | 0.250 / 0.05 / -0.40 |" in t and "| 14.0 / 1.40 | +0.0003 | - |" in t
    recs["c-dialfrz-s3"] = rec(LS.BY_NAME["c-dialfrz-s3"], 1.53, {"w_h_fro": 13.0, "loudness": 1.3, "toggle_bits": 0.0123,
                                                                      "toggle_perbyte_bits": 0.0041})
    t = LS.table(recs)
    assert "| c-dialfrz-s3 | head dial (frozen address) + AdamW 0.1 | 4,960 | 1.5300 | +0.0100 |" in t and "| +0.0123 (+0.0041) |" in t
    recs["pdc-dial-s1"] = rec(LS.BY_NAME["pdc-dial-s1"], 1.35)               # its reference run not in recs: no comparison
    assert "| pdc-dial-s1 | head dial + AdamW 0.1 | 4,960 | 1.3500 | - |" in LS.table(recs)


def test_the_dial_reads_confidence_before_the_dial():
    m = AlephLM(LS.arm_cfg(LS.BY_NAME["c-dial-s3"]))
    m.load_state_dict(LS.arm_state(LS.BY_NAME["c-dial-s3"]), strict=True)
    m.eval()
    g = torch.Generator().manual_seed(2)
    with torch.no_grad():
        m.head.w_t.copy_(torch.randn(m.head.w_t.shape, generator=g) * 3)
        Hs = torch.randn(512, m.head.proj.in_features, generator=g)
        pu = torch.randn(Hs.shape[1], generator=g)
        pu = pu / pu.norm()
        out = LS._dial_reads(m.head, Hs, m.head(Hs).float(), pu)
        lt = m.head.dial_log(m.head.addr.signed(m.head.proj(Hs))).float().mean(-1)
        p = torch.softmax(m.head.readout(Hs).float(), -1)
        H = -(p * p.clamp_min(1e-12).log()).sum(-1) / math.log(2)
    a, b = lt - lt.mean(), -H - (-H).mean()
    assert out["dial_corr_entropy"] == pytest.approx(float((a * b).sum() / (a.norm() * b.norm())), abs=1e-5)
    assert out["dial_corr_entropy"] != pytest.approx(out["dial_corr_entropy_dialed"], abs=1e-6)


def test_the_toggle_splits_a_per_byte_gain():
    torch.manual_seed(3)
    docs = [[0] + torch.randint(32, 127, (255,)).tolist() for _ in range(4)]
    m = AlephLM(LS.arm_cfg(LS.BY_NAME["c-pergain-s3"]))
    m.load_state_dict(LS.arm_state(LS.BY_NAME["c-pergain-s3"]), strict=True)
    with torch.no_grad():
        m.head.w_g.weight.normal_()
    t = LS.head_toggle(m, docs)
    assert t["toggle_perbyte_bits"] + t["toggle_global_bits"] == pytest.approx(t["toggle_bits"], abs=1e-9)
    assert t["toggle_bits"] != 0.0 and np.isfinite(t["dial_log_hold"]) and m.head.dial_hold is None and m.training


def test_the_bed_takes_the_head_form_and_names_it():
    from geolip.nano_beatrix.bodies import craft_cfg
    from geolip.nano_beatrix.bed_runner import GRID_KEYS
    from geolip.nano_beatrix.names import run_tag
    assert craft_cfg("castor-both", 1024).head_mode == "" and craft_cfg("castor-both", 1024, {"head_mode": "dial"}).head_mode == "dial"
    assert "head_mode" in GRID_KEYS
    base = {"arm": "castor-both", "seed": "1", "steps": "4000", "wd": "0.1"}
    assert run_tag(base) == "castor-both_s1_L4000_wd0.1" and run_tag({**base, "head_mode": "dial"}) == "castor-both_s1_L4000_wd0.1_hddial"
