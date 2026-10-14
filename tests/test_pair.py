# -*- coding: utf-8 -*-
"""The pair run's pieces that need no model: the debiased CKA, the solo tag of a pair tag, the partner's arguments."""
import torch

from geolip.nano_beatrix import bed_pair
from geolip.nano_beatrix.names import partner_args, run_tag


def test_cka_reads_a_rotation_and_a_scale_as_the_same_and_noise_as_unrelated():
    g = torch.Generator().manual_seed(0)
    X = torch.randn(400, 16, generator=g)
    Q, _ = torch.linalg.qr(torch.randn(16, 16, generator=g))
    assert abs(bed_pair.cka_debiased(X, X @ Q) - 1.0) < 1e-9
    assert abs(bed_pair.cka_debiased(X, 3.0 * X @ Q) - 1.0) < 1e-9
    assert abs(bed_pair.cka_debiased(X, torch.randn(400, 16, generator=g))) < 0.05      # the debiased estimator sits near 0 on noise
    Y = torch.cat([X, torch.randn(400, 16, generator=g)], 1)                              # half the features shared
    assert 0.3 < bed_pair.cka_debiased(X, Y) < 0.9


def test_solo_tag_strips_every_form_of_the_pair_piece():
    t = "castor-both_s1_L12000_frdyn_cap_mode_sized_sl32_sz32-128-256_r8_n60_o30_askany_p75_pair-pollux-both-tv_stA1"
    solo = "castor-both_s1_L12000_frdyn_cap_mode_sized_sl32_sz32-128-256_r8_n60_o30_askany_p75_stA1"
    assert bed_pair.solo_tag(t) == solo
    for piece in ("-sv_", "-ma_", "-maw0.5f2000_", "-tvw0_", "-saw2_"):
        assert bed_pair.solo_tag(t.replace("-tv_", piece)) == solo
    assert bed_pair.solo_tag(solo) == solo


def test_partner_args_swap_the_crafts_and_their_starting_crafts():
    a = {"arm": "castor-both", "seed": "1", "steps": "4000", "pair": "pollux-both", "init": "C_A1", "pair_init": "P_A1", "stage": "B1"}
    p = partner_args(a)
    assert (p["arm"], p["pair"], p["init"], p["pair_init"], p["pair_role"]) == ("pollux-both", "castor-both", "P_A1", "C_A1", "s")
    assert partner_args(dict(a, xc="mutual"))["pair_role"] == "m"
    assert run_tag(a).startswith("castor-both_s1_L4000") and run_tag(a).endswith("_pair-pollux-both-tv_stB1")
    assert run_tag(p).startswith("pollux-both_s1_L4000") and run_tag(p).endswith("_pair-castor-both-sv_stB1")
    b = {k: v for k, v in a.items() if k not in ("init", "pair_init")}                   # a pair from birth: no starting crafts
    assert "init" not in partner_args(b) and "pair_init" not in partner_args(b)
