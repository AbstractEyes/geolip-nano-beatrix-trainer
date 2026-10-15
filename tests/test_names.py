# -*- coding: utf-8 -*-
from geolip.nano_beatrix.names import init_suffix


def test_own_prefix_stripped():
    assert init_suffix("hubr-free_s1_L12000_dense4_K64D32c2", "hubr-free", 1) == "L12000_dense4_K64D32c2"


def test_foreign_craft_kept_whole():
    assert init_suffix("hubd-free_s2_L12000_dense8", "hubr-free", 1) == "hubd-free_s2_L12000_dense8"


def test_two_continuations_never_share_a_name():
    base = "hubr-free_s1_L12000_dense8_K64D32c2_ctx4096_from_"
    a = base + init_suffix("hubr-free_s1_L12000_dense4_K64D32c2", "hubr-free", 1)
    b = base + init_suffix("hubr-free_s1_L12000_dense8_K64D32c2_ctx4096", "hubr-free", 1)
    assert a != b and a.endswith("_from_L12000_dense4_K64D32c2") and b.endswith("_from_L12000_dense8_K64D32c2_ctx4096")


def test_the_frame_tag_carries_the_diet_when_not_all_framed():
    # the tag expression lives in bed_train (a script); the rule it encodes: plant 1.0 leaves the name as it was, any other diet marks it
    mark = lambda plant: f"_p{int(round(plant * 100))}" if plant != 1.0 else ""
    assert mark(1.0) == "" and mark(0.75) == "_p75" and mark(0.5) == "_p50" and mark(0.25) == "_p25"


def test_run_tag_names_the_head_form():
    from geolip.nano_beatrix.names import run_tag
    base = {"arm": "castor-both", "seed": "2", "steps": "4000"}
    assert run_tag(base) == "castor-both_s2_L4000" and run_tag({**base, "head_mode": "pergain"}) == "castor-both_s2_L4000_hdpergain"


def test_run_tag_matches_the_record_names():
    from geolip.nano_beatrix.names import run_tag
    # verbatim record names from the Hub and the local bed (2026-10-13): the pass-2 capacity run, the night's r8 run, today's smoke
    cap = {"out": "/content/out", "register": "caption", "ctx": "4096", "mb": "8", "steps": "12000", "frames": "dyn", "plant": "0.75", "nest": "0.3",
           "overlap": "0.0", "ask": "any", "tags": "mode", "log_every": "100", "sizes": "32,128,256,512,1024", "arm": "castor-both", "seed": "2", "frame_records": "4"}
    assert run_tag(cap) == "castor-both_s2_L12000_ctx4096_frdyn_cap_mode_sz32-128-256-512-1024_r4_n30_o0_askany_p75"
    assert run_tag({"arm": "pollux-both", "seed": "1", "steps": "12000", "frames": "dyn", "plant": "0.75", "frame_records": "8"}) == \
        "pollux-both_s1_L12000_frdyn_cap_mode_sz32-128-256_r8_n30_o0_askany_p75"
    assert run_tag({"arm": "castor-both", "seed": "1", "smoke": "1", "frames": "dyn", "register": "math", "sizes": "32,128", "plant": "0.75", "ctx": "1024"}) == \
        "castor-both_s1_smoke_frdyn_mat_mode_sz32-128_r4_n30_o0_askany_p75"
    assert run_tag({"arm": "smr-fixed", "seed": "1"}) == "smr-fixed_s1_L3000"          # the default 3,000 steps is above the bare-name line
    assert run_tag({"arm": "smr-fixed", "seed": "1", "steps": "2000"}) == "smr-fixed_s1"   # the short grid keeps its bare names
    assert run_tag({"arm": "smr-fixed", "seed": "2", "steps": "3000", "ctx": "4096"}) == "smr-fixed_s2_L3000_ctx4096"
    assert run_tag({"arm": "castor-both", "seed": "1", "steps": "12000", "ctx": "4096", "init": "castor-both_s1_L12000_frdyn_cap_mode_sz32-128-256_r4_n30_o0_askany_p75"}) == \
        "castor-both_s1_L12000_ctx4096_from_L12000_frdyn_cap_mode_sz32-128-256_r4_n30_o0_askany_p75"


def test_run_tag_names_the_sized_and_sliced_forms():
    from geolip.nano_beatrix.names import run_tag
    base = {"arm": "castor-both", "seed": "1", "steps": "12000", "frames": "dyn", "plant": "0.75", "frame_records": "8"}
    assert run_tag(base) == "castor-both_s1_L12000_frdyn_cap_mode_sz32-128-256_r8_n30_o0_askany_p75"
    assert run_tag(dict(base, sized="1")) == "castor-both_s1_L12000_frdyn_cap_mode_sized_sz32-128-256_r8_n30_o0_askany_p75"
    assert run_tag(dict(base, sized="1", sliced="32")) == "castor-both_s1_L12000_frdyn_cap_mode_sized_sl32_sz32-128-256_r8_n30_o0_askany_p75"
    assert run_tag(dict(base, sized="0", sliced="0")) == run_tag(base)


def test_a_stage_name_replaces_the_starting_craft_in_the_tag():
    from geolip.nano_beatrix.names import run_tag
    a = {"arm": "castor-both", "seed": "3", "steps": "12000", "frames": "dyn", "plant": "0.75", "frame_records": "8"}
    b1 = dict(a, steps="4000", ctx="4096", sizes="32,128,256,512,1024", frame_records="16", init=run_tag(a))
    assert run_tag(b1).endswith("_from_L12000_frdyn_cap_mode_sz32-128-256_r8_n30_o0_askany_p75")   # the plain continuation, as before
    s1 = run_tag(dict(b1, stage="B1"))
    s2 = run_tag(dict(b1, init=s1, stage="B2"))
    assert s1.endswith("_r16_n30_o0_askany_p75_stB1") and s2.endswith("_stB2") and "_from_" not in s1 + s2
    assert len(s2) == len(s1) < 100                                          # a chain's tags stay short, block after block


def test_run_tag_names_the_rate_and_the_warmup_when_moved():
    from geolip.nano_beatrix.names import run_tag
    base = {"arm": "pollux-both", "seed": "1", "steps": "12000", "frames": "dyn", "plant": "0.75", "frame_records": "8"}
    assert run_tag(dict(base, lr="1e-3", warmup="100")) == run_tag(base) == "pollux-both_s1_L12000_frdyn_cap_mode_sz32-128-256_r8_n30_o0_askany_p75"
    assert run_tag(dict(base, lr="5e-4")) == "pollux-both_s1_L12000_lr0.0005_frdyn_cap_mode_sz32-128-256_r8_n30_o0_askany_p75"
    assert run_tag(dict(base, lr="0.002", warmup="1000")) == "pollux-both_s1_L12000_lr0.002_wu1000_frdyn_cap_mode_sz32-128-256_r8_n30_o0_askany_p75"
    assert run_tag(dict(base, lr="2e-3")) == run_tag(dict(base, lr="0.002"))                 # one rate, one name, however it is written
    assert run_tag(dict(base, smoke="1", warmup="5")) == run_tag(dict(base, smoke="1"))      # the smoke's own default warmup


def test_run_tag_names_a_pair_run_and_its_partner():
    from geolip.nano_beatrix.names import run_tag
    a = {"arm": "castor-both", "seed": "1", "steps": "12000", "frames": "dyn", "plant": "0.75", "frame_records": "8", "stage": "A1"}
    t = run_tag(dict(a, pair="pollux-both"))
    s = run_tag(dict(a, arm="pollux-both", pair="castor-both", pair_role="s"))
    assert t == "castor-both_s1_L12000_frdyn_cap_mode_sz32-128-256_r8_n30_o0_askany_p75_pair-pollux-both-tv_stA1"
    assert s == "pollux-both_s1_L12000_frdyn_cap_mode_sz32-128-256_r8_n30_o0_askany_p75_pair-castor-both-sv_stA1"
    m = run_tag(dict(a, pair="pollux-both", xc="mutual", xc_pos="all", xc_w="0.5", xc_from="2000"))
    assert m.endswith("_pair-pollux-both-maw0.5f2000_stA1")
    z = run_tag(dict(a, pair="pollux-both", xc_w="0"))
    assert z.endswith("_pair-pollux-both-tvw0_stA1")                        # the zero-weight pair (the parity run) has its own name
    assert len({t, s, m, z, run_tag(a)}) == 5
