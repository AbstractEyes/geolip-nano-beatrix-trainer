# -*- coding: utf-8 -*-
"""anchored_rope.py - THE ANCHORED MULTI-POSITION ROTARY (0.1.70) for the bed's softmax blocks: the head's rotate-half pairs split by
job. For a head of 32 dims (16 pairs; pair j = dims j and j + 16):
    pairs 0-5    the relative band on the sequence index: w_j = (10000 * 4^(12/10))^(-j/6), j = 0..5 (NTK-aware scaling x4 over the
                 band's own 12 dims, base * s^(dim / (dim - 2)); wavelengths 6.3 / 38 / 236 / 1,443 / 8,840 / 54,100 positions)
    pairs 6-7    content: unrotated
    pairs 8-11   anchor level 1: w_k = 2 pi k / 9, k = 1..4, on the first address digit
    pairs 12-15  anchor level 2: the same frequencies on the second digit
A token's address digits arrive as two streams (grid_rows.streams): the key stream turns k, the query stream turns q, so on the anchor
pairs q_m . k_n depends on the digit differences only:
    q_m . k_n = sum_content <q_j, k_j> + sum_rel Re[q_j conj(k_j) e^{i w_j (m - n)}]
                + sum_{l=1,2} sum_{k=1..4} Re[q_lk conj(k_lk) e^{i w_k (dq_l(m) - dk_l(n))}]
Over the four anchor frequencies the nine digits (0-7 and the null anchor 8) form a simplex: sum_k cos(w_k d) is 4 at d = 0 and -1/2 at
every other difference. The null anchor turns by angle 0, so a row without streams (every token at the null anchor) turns the relative
band alone. Nothing here is a parameter: a patched craft's state dict is the house body's, key for key."""
from __future__ import annotations

import math
import types

import torch

from geolip.alephllm.model.attention import CausalSDPA, rope_half

REL_PAIRS, CONTENT_PAIRS, ANCHOR_FREQS, LEVELS = 6, 2, 4, 2
PAIRS = REL_PAIRS + CONTENT_PAIRS + LEVELS * ANCHOR_FREQS
NTK_SCALE, BASE = 4.0, 10000.0
NULL = 8


def frequencies() -> tuple:
    """-> (w (16,) float64, kind (16,) long): kind 0 = the sequence index, 1 = content (w 0), 2 + l = anchor level l."""
    dim = 2 * REL_PAIRS
    base = BASE * NTK_SCALE ** (dim / (dim - 2))
    w = [base ** (-j / REL_PAIRS) for j in range(REL_PAIRS)] + [0.0] * CONTENT_PAIRS
    kind = [0] * REL_PAIRS + [1] * CONTENT_PAIRS
    for lev in range(LEVELS):
        w += [2 * math.pi * k / 9 for k in range(1, ANCHOR_FREQS + 1)]
        kind += [2 + lev] * ANCHOR_FREQS
    return torch.tensor(w, dtype=torch.float64), torch.tensor(kind, dtype=torch.long)


class AnchorStreams:
    """The address streams every patched block reads during one forward: set() before the forward, clear() after (None = every token at
    the null anchor). The angles are formed in fp32 and cast once per block, as the house rotary forms its tables."""

    def __init__(self) -> None:
        w, kind = frequencies()
        self.w, self.kind = w.float(), kind
        self.dk = self.dq = None

    def set(self, dk=None, dq=None) -> None:
        """dk, dq: (B, n, 2) integer tensors of digits 0-8 on the craft's device, or both None."""
        assert (dk is None) == (dq is None), "the key and the query streams come together"
        if dk is not None:
            assert dk.shape == dq.shape and dk.dim() == 3 and dk.shape[-1] == LEVELS, (tuple(dk.shape), tuple(dq.shape))
        self.dk, self.dq = dk, dq

    def clear(self) -> None:
        self.dk = self.dq = None

    def angles(self, B: int, n: int, offset: int, device) -> tuple:
        """-> (angles of q, angles of k): (n, 16) each without streams, (B, 1, n, 16) each with them."""
        w, kind = self.w.to(device), self.kind.to(device)
        pos = torch.arange(offset, offset + n, device=device, dtype=torch.float32)
        base = pos[:, None] * torch.where(kind == 0, w, torch.zeros_like(w))[None, :]
        if self.dk is None:
            return base, base
        assert tuple(self.dk.shape[:2]) == (B, n), f"the streams are {tuple(self.dk.shape[:2])}, the forward is {(B, n)}"
        aq = base.expand(B, n, PAIRS).clone()
        ak = aq.clone()
        for lev in range(LEVELS):
            sel = kind == 2 + lev
            aq[..., sel] = (self.dq[..., lev:lev + 1].float() - NULL) * w[sel]
            ak[..., sel] = (self.dk[..., lev:lev + 1].float() - NULL) * w[sel]
        return aq[:, None], ak[:, None]


def _anchored_rotate(self, q, k, offset: int = 0):
    """CausalSDPA._rotate for a patched block: the split by job above, on the streams of self._anchor."""
    B, n = q.shape[0], q.shape[-2]
    aq, ak = self._anchor.angles(B, n, offset, q.device)
    return (rope_half(q, aq.cos().to(q.dtype), aq.sin().to(q.dtype)),
            rope_half(k, ak.cos().to(k.dtype), ak.sin().to(k.dtype)))


def patch(model: torch.nn.Module, streams: AnchorStreams) -> int:
    """Turn every softmax block of the craft (each CausalSDPA with the rotary on, head dim 32) to the anchored rotation reading `streams`.
    -> the number of blocks turned. The parameters are untouched."""
    n = 0
    for b in model.blocks:
        a = b.attn
        if isinstance(a, CausalSDPA):
            hd = a.qkv.weight.shape[1] // a.h
            assert a.rotary, "the anchored rotation replaces a rotary body's rotation (attn_rotary on)"
            assert hd == 2 * PAIRS, f"the anchored split is written for a head of {2 * PAIRS} dims, got {hd}"
            a._anchor = streams
            a._rotate = types.MethodType(_anchored_rotate, a)
            n += 1
    assert n, "no softmax block to turn"
    return n


def spec() -> dict:
    """The rotation as a record field (the weights folder's scale.json and the run's record)."""
    w, kind = frequencies()
    return dict(form="anchored", pairs=PAIRS, rel_pairs=REL_PAIRS, content_pairs=CONTENT_PAIRS, anchor_freqs=ANCHOR_FREQS, levels=LEVELS,
                ntk_scale=NTK_SCALE, base=BASE, null=NULL, w=[float(x) for x in w], kind=[int(x) for x in kind])
