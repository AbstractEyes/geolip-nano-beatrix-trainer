# -*- coding: utf-8 -*-
"""pointer.py - THE POINTER TERM (0.1.78): one head of one softmax block told, at each index entry's SEP, where the record whose key the
entry carries is. For an entry e (its SEP an input position i_e; T(e) = [lo, hi) the input positions of its record's frame
DATA a1 a2 CUE key SEP value END) and the block's head h with rotated query and key q, k (head dim hd):
    log m_e = logsumexp_{n in T(e)} q(i_e) . k(n) / sqrt(hd) - logsumexp_{n <= i_e} q(i_e) . k(n) / sqrt(hd)
m_e is the head's causal softmax mass on the frame: the quantity the band read measures (find_band_read: the softmax over the positions
up to the SEP, summed over the true record's frame). The trainer's term is -(1 / |E|) sum_e log m_e, a cross-entropy on the attention
row itself. Nothing here is a parameter: the capture wraps the block's rotation as an instance attribute, so the block's output, the
craft's state dict and every other forward are unchanged.
0.1.79: the same mass at any query position (the trainer asks it at the entry's first address byte too, where the second address digit
is written), and the capture's stopped form (HeadCapture(stop=True)): the term's q and k from the block's attention input detached, so
its gradient stays in the supervised head's own query and key projections."""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


class HeadCapture:
    """Keep the rotated q and k of one softmax block's latest forward (the block's _rotate wrapped as an instance attribute: no
    parameter, no state-dict key, the block's output unchanged). The tensors are the ones the block's kernel reads, so the term's
    gradient flows back through the same q and k. on: capture while True; clear() drops the references after a step.
    0.1.79, stop=True: keep the block's attention input instead (its _split wrapped the same way); qk() then forms the supervised head's
    query and key from that input detached, through the block's own rotation - the forward's values (within float rounding), and a
    gradient that reaches only that head's query and key rows of the block's qkv projection."""

    def __init__(self, attn: torch.nn.Module, stop: bool = False, head: int = 0) -> None:
        self.attn, self.q, self.k, self.x, self.on = attn, None, None, None, True
        self.stop, self.head = bool(stop), int(head)
        if self.stop:
            assert not getattr(attn, "qk_norm", ""), "the stopped form forms one head's q and k without the block's qk-norm guard"
            inner_split = attn._split

            def _split(x, n):
                if self.on:
                    self.x = x
                return inner_split(x, n)
            attn._split = _split
        else:
            inner = attn._rotate

            def _rotate(q, k, offset: int = 0):
                q, k = inner(q, k, offset)
                if self.on:
                    self.q, self.k = q, k
                return q, k
            attn._rotate = _rotate

    def ready(self) -> bool:
        return (self.x if self.stop else self.q) is not None

    def qk(self) -> tuple:
        """(q, k, the supervised head's index in them) for the term: the forward's own rotated q and k (every head), or with stop=True
        the head's query and key, (B, 1, n, hd), from the kept input detached through the block's rotation (fp32)."""
        if not self.stop:
            return self.q, self.k, self.head
        a, h = self.attn, self.head
        w = a.qkv.weight
        x = self.x.detach().to(w.dtype)
        B, n, d = x.shape
        hd = d // a.h
        q = F.linear(x, w[h * hd:(h + 1) * hd]).view(B, n, 1, hd).transpose(1, 2)
        k = F.linear(x, w[d + h * hd:d + (h + 1) * hd]).view(B, n, 1, hd).transpose(1, 2)
        q, k = a._rotate(q, k)
        return q, k, 0

    def clear(self) -> None:
        self.q = self.k = self.x = None


def novelty(v_ce: torch.Tensor, v_term: torch.Tensor) -> tuple:
    """The collinearity gate's reading for the term (the bed's rule: refused under .05): 1 - |cos| between the flattened gradient of the
    CE and the term's over the shared trunk (fp64). -> (novelty, cos). A term whose gradient is the CE's own reads 0."""
    cos = float((v_ce @ v_term) / (v_ce.norm() * v_term.norm()).clamp(min=1e-300))
    return 1.0 - abs(cos), cos


def log_mass(q: torch.Tensor, k: torch.Tensor, head: int, rows, pos, lo, hi) -> torch.Tensor:
    """log m_e for each entry e: the head's causal softmax mass at query position pos[e] of batch row rows[e] on the key positions
    [lo[e], hi[e]). q, k: (B, H, n, hd), rotated (any dtype; the scores are formed in fp32). rows, pos, lo, hi: (E,) integer tensors (any
    device), the entries grouped by row in non-decreasing row order (grid_aux's order), every span before its query (hi <= pos).
    -> (E,) fp32 on q's device, differentiable in q and k."""
    rows, pos, lo, hi = (torch.as_tensor(t).long().cpu() for t in (rows, pos, lo, hi))
    assert rows.dim() == 1 and rows.shape == pos.shape == lo.shape == hi.shape and rows.numel() > 0, "one row, SEP and span per entry"
    assert bool((rows[1:] >= rows[:-1]).all()), "the entries come grouped by row (grid_aux's order)"
    assert bool((lo < hi).all()) and bool((hi <= pos).all()), "every target span is non-empty and lies before its query"
    hd, n = q.shape[-1], k.shape[2]
    assert int(pos.max()) < n, "a query position past the forward's length"
    dev = q.device
    t = torch.arange(n, device=dev)
    out = []
    uniq, counts = torch.unique_consecutive(rows, return_counts=True)
    i0 = 0
    for r, c in zip(uniq.tolist(), counts.tolist()):
        p_, lo_, hi_ = (x[i0:i0 + c].to(dev) for x in (pos, lo, hi))
        s = (q[r, head, p_].float() @ k[r, head].float().transpose(0, 1)) / math.sqrt(hd)       # (c, n)
        s = s.masked_fill(t[None, :] > p_[:, None], float("-inf"))                             # the causal row
        inside = (t[None, :] >= lo_[:, None]) & (t[None, :] < hi_[:, None])
        out.append(torch.logsumexp(s.masked_fill(~inside, float("-inf")), -1) - torch.logsumexp(s, -1))
        i0 += c
    return torch.cat(out)
