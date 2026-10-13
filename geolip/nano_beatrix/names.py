# -*- coding: utf-8 -*-
"""names.py - the pieces of a run's tag that must never collide."""


def init_suffix(init, arm, seed):
    """The starting craft's tag as it enters a continued run's tag: this run's own arm and seed are stripped from its front when
    they match, the rest stays. Two continuations of one recipe from different crafts then never share a name (a 12,000-step
    continuation from a 1,024-context craft and one from a 4,096-context craft once did, and the second overwrote the first)."""
    p = f"{arm}_s{seed}_"
    return init[len(p):] if init.startswith(p) else init
