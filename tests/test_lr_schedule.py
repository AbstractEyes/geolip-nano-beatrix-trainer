# -*- coding: utf-8 -*-
"""0.1.100: the bed's schedule (bed_optim.lr_schedule). The defaults are the stored cosine to .1 over the whole run, bit for bit at every
step; a higher floor and a flat share follow their formulas; the run's name carries _lf / _fl only when not the defaults."""
import math

import pytest

from geolip.nano_beatrix import bed_optim as bo
from geolip.nano_beatrix import names


def _stored(step, lr, warm, steps):
    # bed_train.lr_at through 0.1.99, verbatim
    if step < warm:
        return lr * (step + 1) / warm
    t = (step - warm) / max(1, steps - warm)
    return lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * t)))


def test_the_defaults_are_the_stored_schedule_bit_for_bit():
    for steps, warm in ((24000, 100), (48000, 100), (3000, 100), (20, 5)):
        for step in range(steps):
            assert bo.lr_schedule(step, 1e-3, warm, steps) == _stored(step, 1e-3, warm, steps), (steps, warm, step)
            assert bo.lr_schedule(step, 1e-3, warm, steps, 0.1, 0.0) == _stored(step, 1e-3, warm, steps), (steps, warm, step)


def test_a_higher_floor_and_a_flat_share_follow_their_formulas():
    lr, warm, steps = 1e-3, 100, 24000
    e = warm + 0.8 * (steps - warm)                      # 19,220
    assert e == 19220.0
    for step in (0, 50, 99, 100, 5000, 12000, 19219, 19220, 19221, 20000, 22000, 23999):
        floor = (lr * (step + 1) / warm if step < warm else
                 lr * (0.5 + 0.5 * 0.5 * (1 + math.cos(math.pi * (step - warm) / (steps - warm)))))
        assert abs(bo.lr_schedule(step, lr, warm, steps, floor=0.5) - floor) <= 1e-15, step
        flat = (lr * (step + 1) / warm if step < warm else lr if step < e else
                lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (step - e) / (steps - e)))))
        assert abs(bo.lr_schedule(step, lr, warm, steps, flat=0.8) - flat) <= 1e-15, step
    assert bo.lr_schedule(19219, lr, warm, steps, flat=0.8) == lr and bo.lr_schedule(19220, lr, warm, steps, flat=0.8) == lr
    assert bo.lr_schedule(19221, lr, warm, steps, flat=0.8) < lr
    assert abs(bo.lr_schedule(steps - 1, lr, warm, steps, floor=0.5) - 0.5 * lr) < 1e-8
    assert abs(bo.lr_schedule(steps - 1, lr, warm, steps, flat=0.8) - 0.1 * lr) < 1e-8
    with pytest.raises(AssertionError):
        bo.lr_schedule(500, lr, warm, steps, floor=0.0)
    with pytest.raises(AssertionError):
        bo.lr_schedule(500, lr, warm, steps, flat=1.0)


def test_the_names_carry_the_schedule_only_when_not_the_defaults():
    base = {"arm": "alhena-both", "seed": "4", "steps": "24000", "ctx": "4096"}
    t0 = names.run_tag(dict(base))
    assert t0.count("_L24000") == 1 and "_lf" not in t0 and "_fl" not in t0
    assert names.run_tag(dict(base, lr_floor="0.1", lr_flat="0")) == t0
    t1, t2 = names.run_tag(dict(base, lr_floor="0.5")), names.run_tag(dict(base, lr_flat="0.8"))
    assert t1 == t0.replace("_L24000", "_L24000_lf0.5") and t2 == t0.replace("_L24000", "_L24000_fl0.8")
    assert names.run_tag(dict(base, lr_floor="0.5", lr_flat="0.8")) == t0.replace("_L24000", "_L24000_lf0.5_fl0.8")
