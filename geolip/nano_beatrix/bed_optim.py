# -*- coding: utf-8 -*-
"""bed_optim.py - the bed's optimizer. The default is pure Adam (lr from the run, betas 0.9 / 0.95, eps 1e-8, no decay): every run before
0.1.31 trained that way and still does. With a weight decay the run trains with AdamW instead: a named set of weights takes decoupled
decay (p <- p (1 - lr wd) each step, on the run's own schedule) and every other parameter takes the same update at decay 0, which is
Adam's update exactly.

The decay sets:
  mat   every weight of two or more dimensions that maps the stream: the attention projections, the bank's transforms and experts, the
        Canon kernels, the output layer's readout, its projection and its aleph read. Never the byte and position tables (embed.*), the
        address books (*.codebook) or the byte-class amplitude tables (*.byte_amp): their size is part of their geometry, and none of them
        was ever trained under a decay.
  head  the output layer's two readout matrices only (head.w_h.weight, head.w_s.weight).
Norm gains, biases, gates and every other one-dimensional parameter never decay."""
from __future__ import annotations

import torch

DECAY_SETS = ("mat", "head")
BETAS, EPS = (0.9, 0.95), 1e-8
_NEVER = ("embed.",)                         # name prefixes outside every decay set
_NEVER_TAIL = (".codebook", ".byte_amp")     # name endings outside every decay set
HEAD = ("head.w_h.weight", "head.w_s.weight")


def decays(name: str, p: torch.Tensor, wd_set: str = "mat") -> bool:
    """Does the decay set reach this parameter?"""
    assert wd_set in DECAY_SETS, f"the decay set is one of {DECAY_SETS}: {wd_set!r}"
    if wd_set == "head":
        return name in HEAD
    return p.ndim >= 2 and not name.startswith(_NEVER) and not name.endswith(_NEVER_TAIL)


def decay_names(model: torch.nn.Module, wd_set: str = "mat") -> list:
    """The names of the trainable parameters the decay set reaches, in the model's order."""
    return [n for n, p in model.named_parameters() if p.requires_grad and decays(n, p, wd_set)]


def build(model: torch.nn.Module, params: list, lr: float, wd: float = 0.0, wd_set: str = "mat") -> torch.optim.Optimizer:
    """The run's optimizer over `params` (the model's trainable parameters, in order). wd 0: torch's Adam, the bed's optimizer of every
    earlier run. wd > 0: AdamW with two groups, the decay set at wd and the rest at 0."""
    assert wd >= 0, f"the weight decay is not negative: {wd}"
    if not wd:
        return torch.optim.Adam(params, lr=lr, betas=BETAS, eps=EPS)
    names = {id(p): n for n, p in model.named_parameters()}
    dec = {id(p) for p in params if decays(names[id(p)], p, wd_set)}
    assert dec, f"the decay set {wd_set!r} reaches no parameter of this model"
    return torch.optim.AdamW([{"params": [p for p in params if id(p) not in dec], "weight_decay": 0.0},
                              {"params": [p for p in params if id(p) in dec], "weight_decay": float(wd)}],
                             lr=lr, betas=BETAS, eps=EPS)


def census(model: torch.nn.Module, wd_set: str = "mat") -> dict:
    """How much of the model the decay set reaches: tensors and parameters in and out of it, by top-level part."""
    out = {"set": wd_set, "decay_tensors": 0, "decay_params": 0, "other_tensors": 0, "other_params": 0, "by_part": {}}
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        k = "decay" if decays(n, p, wd_set) else "other"
        out[f"{k}_tensors"] += 1
        out[f"{k}_params"] += p.numel()
        part = out["by_part"].setdefault(n.split(".")[0], {"decay": 0, "other": 0})
        part[k] += p.numel()
    return out
