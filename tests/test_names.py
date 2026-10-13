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
