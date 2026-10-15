# -*- coding: utf-8 -*-
"""The bed's optimizer (bed_optim): pure Adam by default, AdamW on a decay set; the run names that carry it; the re-run of the bed's
deciding tests (bed_rerun): every part's runs named apart from the pure-Adam runs and from the other parts, every continuation starting
from a craft its own part trains, the pure-Adam specs reproducing the earlier notebooks' names."""
import os

import pytest
import torch

from geolip.alephllm.model.alephlm import AlephLM
from geolip.nano_beatrix import bed_curriculum as bc
from geolip.nano_beatrix import bed_optim, bed_rerun
from geolip.nano_beatrix import bed_runner as br
from geolip.nano_beatrix.bodies import craft_cfg
from geolip.nano_beatrix.names import run_tag


def _model(arm="castor-both"):
    torch.manual_seed(1)
    return AlephLM(craft_cfg(arm, 1024))


# ------------------------------------------------------------------ the optimizer
@pytest.mark.parametrize("arm", ["castor-both", "altdc-both", "pollux-both"])
def test_the_matrix_set_reaches_the_stream_matrices_only(arm):
    m = _model(arm)
    names = set(bed_optim.decay_names(m, "mat"))
    for n, p in m.named_parameters():
        never = n.startswith("embed.") or n.endswith((".codebook", ".byte_amp")) or p.ndim < 2
        assert (n in names) == (not never), n
    assert {"head.w_h.weight", "head.w_s.weight", "head.proj.weight"} <= names
    assert any(".bank.w_in" in n for n in names) and any(".attn." in n for n in names)
    assert not any(n.endswith((".bias", ".gamma", ".gates")) or ".n1." in n or n.startswith("nf.") for n in names)
    if arm == "altdc-both":
        assert any(".canon_a." in n for n in names)


def test_the_head_set_is_the_two_readout_matrices():
    assert bed_optim.decay_names(_model(), "head") == ["head.w_h.weight", "head.w_s.weight"]


def test_no_decay_is_torch_adam_as_before():
    m = _model()
    params = [p for p in m.parameters() if p.requires_grad]
    opt = bed_optim.build(m, params, 1e-3)
    assert type(opt) is torch.optim.Adam and len(opt.param_groups) == 1
    g = opt.param_groups[0]
    assert g["betas"] == (0.9, 0.95) and g["eps"] == 1e-8 and g["weight_decay"] == 0 and g["lr"] == 1e-3


def test_adamw_puts_the_set_at_the_decay_and_the_rest_at_zero():
    m = _model()
    params = [p for p in m.parameters() if p.requires_grad]
    opt = bed_optim.build(m, params, 1e-3, wd=0.1)
    assert type(opt) is torch.optim.AdamW and [g["weight_decay"] for g in opt.param_groups] == [0.0, 0.1]
    ids = {id(p) for p in opt.param_groups[1]["params"]}
    names = {n for n, p in m.named_parameters() if id(p) in ids}
    assert names == set(bed_optim.decay_names(m, "mat"))
    assert sum(len(g["params"]) for g in opt.param_groups) == len(params)


def test_adamw_moves_only_the_decayed_weights_off_adam_and_by_exactly_the_decay():
    """Two copies from the same weights and gradients, one step: the zero-decay group equals Adam bit for bit; a decayed weight differs
    from Adam's by lr x wd x its weight before the step (decoupled decay)."""
    lr, wd = 1e-3, 0.1
    a, b = _model(), _model()
    pa = [p for p in a.parameters() if p.requires_grad]
    pb = [p for p in b.parameters() if p.requires_grad]
    before = {n: p.detach().clone() for n, p in b.named_parameters()}
    oa, ob = bed_optim.build(a, pa, lr), bed_optim.build(b, pb, lr, wd=wd)
    torch.manual_seed(7)
    x = torch.randint(0, 256, (2, 64))
    for m in (a, b):
        m.train()
    for m, o in ((a, oa), (b, ob)):
        torch.manual_seed(11)
        out = m(x)[0]
        out.float().logsumexp(-1).mean().backward()
        o.step()
    dec = set(bed_optim.decay_names(b, "mat"))
    for (n, p), (_, q) in zip(a.named_parameters(), b.named_parameters()):
        if n in dec:
            assert torch.allclose(p - q, lr * wd * before[n], atol=1e-7, rtol=1e-5), n
        else:
            assert torch.equal(p, q), n


def test_census_counts_every_parameter_once():
    m = _model("altdc-both")
    c = bed_optim.census(m, "mat")
    assert c["decay_params"] + c["other_params"] == sum(p.numel() for p in m.parameters() if p.requires_grad)
    assert c["decay_tensors"] == len(bed_optim.decay_names(m, "mat"))


# ------------------------------------------------------------------ the names
def test_the_tag_carries_the_optimizer_and_only_when_set():
    base = {"arm": "castor-both", "seed": "3", "steps": "12000", "frames": "dyn", "plant": "0.75", "frame_records": "8"}
    plain = run_tag(base)
    assert plain == "castor-both_s3_L12000_frdyn_cap_mode_sz32-128-256_r8_n30_o0_askany_p75"
    assert run_tag(dict(base, wd="0")) == plain and run_tag(dict(base, wd="0.0", wd_set="head")) == plain
    assert run_tag(dict(base, wd="0.1")) == "castor-both_s3_L12000_wd0.1_frdyn_cap_mode_sz32-128-256_r8_n30_o0_askany_p75"
    assert run_tag(dict(base, wd="1.2", wd_set="head")) == "castor-both_s3_L12000_wd1.2head_frdyn_cap_mode_sz32-128-256_r8_n30_o0_askany_p75"
    k = dict(base, hub_K="64", hub_D="32", wd="0.1")
    assert run_tag(k) == "castor-both_s3_L12000_K64D32c1_wd0.1_frdyn_cap_mode_sz32-128-256_r8_n30_o0_askany_p75"


def test_the_runner_takes_the_optimizer_as_grid_keys_and_labels_it():
    assert "wd" in br.GRID_KEYS and "wd_set" in br.GRID_KEYS
    runs = br.expand({"arms": "castor-both", "seeds": "1", "wd": "0,0.1"})
    assert len(runs) == 2
    assert "AdamW 0.1" in br.run_label(["--arm=castor-both", "--seed=1", "--wd=0.1"])
    assert "AdamW" not in br.run_label(["--arm=castor-both", "--seed=1"])
    assert "AdamW 0.1 head" in br.run_label(["--arm=castor-both", "--seed=1", "--wd=0.1", "--wd_set=head"])


# ------------------------------------------------------------------ the re-run
def _all_tags(ps: dict) -> dict:
    """{part: set of every record name the part may write}."""
    out = {}
    for name, steps in ps.items():
        tags = set()
        for st in steps:
            if st["kind"] == "grid":
                tags |= set(bed_rerun.tags_of(st["spec"]))
            else:
                tags |= {b["tag"] for b in bc.plan(st["stages"], st["base"], st["arms"], st["seeds"])}
        out[name] = tags
    return out


@pytest.mark.parametrize("over", [None, bed_rerun.smoke_over])
def test_every_part_is_named_apart_from_adam_and_from_the_others(over):
    new, old = _all_tags(bed_rerun.parts(0.1, over=over)), _all_tags(bed_rerun.parts(0.0, over=over))
    assert list(new) == list(bed_rerun.ORDER)
    every_old = set().union(*old.values())
    for part, tags in new.items():
        assert tags and all("_wd0.1" in t for t in tags), part
        assert not tags & every_old, part
        for other, ot in new.items():
            assert other == part or not tags & ot, (part, other, sorted(tags & ot)[:3])
        assert len(tags) == len(old[part]), part                        # one AdamW run for every pure-Adam run of the part


@pytest.mark.parametrize("over", [None, bed_rerun.smoke_over])
def test_every_continuation_starts_from_a_craft_its_own_part_trains_first(over):
    for part, steps in bed_rerun.parts(0.1, over=over).items():
        trained = set()
        for st in steps:
            if st["kind"] != "grid":
                continue
            for tag, kv in bed_rerun.runs_of(st):
                for key in ("init", "pair_init"):
                    if kv.get(key):
                        assert kv[key] in trained, (part, st["name"], kv[key])
            trained |= set(bed_rerun.tags_of(st["spec"]))


def test_the_pure_adam_specs_are_the_earlier_notebooks_names():
    ps = bed_rerun.parts(0.0)
    form = {t for st in ps["frames-form"] for t in bed_rerun.tags_of(st["spec"])}
    # the fourth pass's records on the Hub (2026-10-13), three of its 24
    assert "castor-both_s3_L12000_frdyn_cap_mode_sized_sl32_sz32-128-256_r8_n30_o0_askany_p75" in form
    assert "pollux-both_s4_L12000_frdyn_cap_mode_sz32-128-256_r8_n30_o0_askany_p75" in form
    assert any(t.startswith("castor-both_s3_L4000_ctx4096_frdyn_cap_mode_sized_sl32_sz32-128-256-512-1024_r8") for t in form)
    cap = {t for st in ps["frames-capacity"] for t in bed_rerun.tags_of(st["spec"])}
    assert "castor-both_s2_L12000_ctx4096_frdyn_cap_mode_sz32-128-256-512-1024_r4_n30_o0_askany_p75" in cap   # test_names' record
    reg = {t for st in ps["frames-registers"] for t in bed_rerun.tags_of(st["spec"])}
    # the first pass's records on the Hub (every row framed, so no plant piece), three of its 12
    assert {"castor-both_s1_L12000_ctx4096_frdyn_mat_mode_sz32-128-256-512_r4_n30_o0_askany",
            "castor-both_s1_L12000_ctx4096_frdyn_cod_mode_sz32-128-256-512_r4_n30_o0_askany",
            "pollux-both_s2_L12000_ctx4096_frdyn_cap_mode_sz32-128-256-512_r4_n30_o0_askany"} <= reg
    tw = ps["bodies-twins"][0]
    a1 = {b["tag"] for b in bc.plan(tw["stages"], tw["base"], tw["arms"], tw["seeds"]) if b["block"] == "A1"}
    assert "castor-both_s1_L12000_frdyn_cap_mode_sized_sl32_sz32-128-256_r8_n60_o30_askany_p75_stA1" in a1


def test_the_pairs_continue_from_their_own_first_stage_and_their_solo_controls_are_bodies_twins_blocks():
    import re
    ps = bed_rerun.parts(0.1)
    tw = ps["bodies-twins"][0]
    solo = {b["tag"] for b in bc.plan(tw["stages"], tw["base"], tw["arms"], tw["seeds"])}
    for st in ps["pair"]:
        for tag in bed_rerun.tags_of(st["spec"]):
            plain = re.sub(r"_pair-[a-z]+-[a-z]+-[tsm][va](w[0-9.e-]+)?(f[0-9]+)?", "", tag)
            assert plain in solo, (st["name"], plain)


def test_price_is_tokens_over_rate():
    rate = lambda arm, ctx, mb: 1e6                                     # noqa: E731
    lo, hi = bed_rerun.price("frames-registers", rate)
    assert lo == hi and abs(lo - 12 * 12000 * 8 * 4096 / 1e6 / 3600) < 1e-9
    lo, hi = bed_rerun.price("bodies-twins", rate)
    a, b = bc.bounds(bed_rerun.STAGES, {})
    assert abs(lo - 8 * a / 1e6 / 3600) < 1e-9 and abs(hi - 8 * b / 1e6 / 3600) < 1e-9
    assert set(bed_rerun.shapes("bodies-splat")) == {(arm, c, m) for arm in bed_rerun.BODIES_SPLAT.split(",") for c, m in ((1024, 32), (4096, 8))}


def test_the_read_of_an_empty_folder_is_dashes(tmp_path):
    md = bed_rerun.read("frames-form", str(tmp_path), repo="")
    assert md.count("| castor-both s3") >= 3 and "-> " not in md.replace("Adam -> AdamW", "")
    md = bed_rerun.read("bodies-twins", str(tmp_path), repo="")
    assert "#### the twins and the Canon bodies" in md


def test_the_read_judges_formation_by_one_rule_on_both_sides():
    g = lambda v, n=10: {"full_value": v, "n": n, "first_byte": 0.0, "value_bits": 0.0}       # noqa: E731
    old = {"formation": {"formed": True}, "fetch_by_frame": {"data/32/1": g(0.40)}}               # a record before 0.1.19: no groups_judged
    assert bed_rerun._reads(old)["formed"] is True
    assert bed_rerun._reads({"formation": {"formed": True}, "fetch_by_frame": {"data/32/1": g(0.03), "data/128/1": g(0.05)}})["formed"] is False
    assert bed_rerun._reads({"fetch_by_frame": {"data/32/1": g(0.90, n=7)}})["formed"] is False   # too few trials to judge
    assert bed_rerun._reads({"fetch_by_frame": {"data/32/1": g(0.02), "data/128/1": g(0.30)}})["formed"] is True
