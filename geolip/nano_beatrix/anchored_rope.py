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
band alone. Nothing here is a parameter: a patched craft's state dict is the house body's, key for key.

0.1.73: THE SPLIT AS A PARAMETER (the wide-band screen). The eight pairs before the anchor levels are shared between the relative band and
the content pairs: with C content pairs (content_pairs, default 2 = the form above, bit for bit) the relative band has R = 8 - C pairs,
w_j = (10000 * 4^(2R / (2R - 2)))^(-j / R), j = 0..R-1 (the same NTK-aware rule over the band's own 2R dims), and pairs R..7 are content.
ALHENA-W, C = 4: pairs 0-3 relative (wavelengths 6.3 / 100 / 1,583 / 25,100 positions), pairs 4-7 content, pairs 8-15 the anchor levels.

0.1.91: THE ADDRESS CODE (grid_rows.CODES). The eight anchor pairs carry L levels of R / 2 pairs each (L R / 2 = 8), level l's pairs at
w_k = 2 pi k / (R + 1), k = 1 .. R / 2, turned by (digit - R) w_k: over a level's pairs sum_k cos(w_k d) is R / 2 at d = 0 and -1/2 at
every other difference, so the R + 1 symbols (the digits and the null anchor R) form a simplex at every code. "8x2" is the form above
bit for bit (two levels of four pairs at 2 pi k / 9); "4x4" four levels of two pairs at 2 pi k / 5 (the split before the anchors
unchanged).

0.1.102: THE SHARED PAIRS AT THE CODE. A code's anchor levels take L R / 2 of the sixteen pairs; the pairs before them, which the relative
band and the content share, are the rest: shared(code) = 16 - L R / 2 (SHARED = 8 at 8x2 and 4x4, every angle and field as 0.1.101's).
"8x3" (R 8, L 3: 512 addresses) turns three levels of four pairs at 2 pi k / 9 on pairs 4-15 and shares four: at C = 2 content pairs the
relative band keeps two (the NTK rule over their 4 dims: wavelengths 6.3 and 2,513 positions), pairs 2-3 the content."""
from __future__ import annotations

import math
import types

import torch
import torch.nn.functional as F

from geolip.alephllm.model.attention import CausalSDPA, rope_half

REL_PAIRS, CONTENT_PAIRS, ANCHOR_FREQS, LEVELS = 6, 2, 4, 2
SHARED = REL_PAIRS + CONTENT_PAIRS                      # the pairs the relative band and the content share (0.1.73: the split's room)
PAIRS = SHARED + LEVELS * ANCHOR_FREQS
NTK_SCALE, BASE = 4.0, 10000.0
NULL = 8
CODES = {"8x2": (8, 2), "4x4": (4, 4), "8x3": (8, 3)}  # 0.1.91: the address codes (radix, levels), grid_rows.CODES; 0.1.102: 8x3
CODE = "8x2"                                            # the default: LEVELS levels of ANCHOR_FREQS pairs, the null anchor NULL


def code_shape(code: str = CODE) -> tuple:
    """0.1.91: a code's (radix, levels, pairs a level); its anchor pairs fill the pairs after the shared ones (0.1.102: shared(code) of
    them - eight at 8x2 and 4x4, four at 8x3)."""
    assert code in CODES, f"the address code {code!r}: one of {sorted(CODES)}"
    r, n = CODES[code]
    assert r % 2 == 0 and PAIRS - n * (r // 2) >= 3, (code, r, n)
    return r, n, r // 2


def shared(code: str = CODE) -> int:
    """0.1.102: the pairs before a code's anchor levels, which the relative band and the content share: 16 - levels x radix / 2."""
    r, n, f = code_shape(code)
    return PAIRS - n * f


def _check(content_pairs: int, code: str = CODE) -> int:
    c = int(content_pairs)
    s = shared(code)
    assert 1 <= c <= s - 2, f"content_pairs {content_pairs!r}: 1 .. {s - 2} at the {code} code (the relative band keeps at least two pairs)"
    return c


def frequencies(content_pairs: int = CONTENT_PAIRS, code: str = CODE) -> tuple:
    """-> (w (16,) float64, kind (16,) long): kind 0 = the sequence index, 1 = content (w 0), 2 + l = anchor level l. 0.1.91: the anchor
    levels at the code (level l's R / 2 pairs at 2 pi k / (R + 1)); 0.1.102: the relative band and the content in the code's shared pairs."""
    c = _check(content_pairs, code)
    r, n, f = code_shape(code)
    rel = shared(code) - c
    dim = 2 * rel
    base = BASE * NTK_SCALE ** (dim / (dim - 2))
    w = [base ** (-j / rel) for j in range(rel)] + [0.0] * c
    kind = [0] * rel + [1] * c
    for lev in range(n):
        w += [2 * math.pi * k / (r + 1) for k in range(1, f + 1)]
        kind += [2 + lev] * f
    return torch.tensor(w, dtype=torch.float64), torch.tensor(kind, dtype=torch.long)


class AnchorStreams:
    """The address streams every patched block reads during one forward: set() before the forward, clear() after (None = every token at
    the null anchor). The angles are formed in fp32 and cast once per block, as the house rotary forms its tables. content_pairs: the
    split (0.1.73; default 2 = ALHENA)."""

    def __init__(self, content_pairs: int = CONTENT_PAIRS, code: str = CODE) -> None:
        self.content_pairs = _check(content_pairs, code)
        self.code = code                                  # 0.1.91: the address code (its radix is the null anchor, its levels the streams)
        self.radix, self.levels, _ = code_shape(code)
        w, kind = frequencies(self.content_pairs, code)
        self.w, self.kind = w.float(), kind
        self.dk = self.dq = None

    def set(self, dk=None, dq=None) -> None:
        """dk, dq: (B, n, 2) integer tensors of digits 0-8 on the craft's device, or both None. 0.1.91: (B, n, levels) of digits 0 .. radix
        at the streams' code."""
        assert (dk is None) == (dq is None), "the key and the query streams come together"
        if dk is not None:
            assert dk.shape == dq.shape and dk.dim() == 3 and dk.shape[-1] == self.levels, (tuple(dk.shape), tuple(dq.shape))
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
        for lev in range(self.levels):
            sel = kind == 2 + lev
            aq[..., sel] = (self.dq[..., lev:lev + 1].float() - self.radix) * w[sel]
            ak[..., sel] = (self.dk[..., lev:lev + 1].float() - self.radix) * w[sel]
        return aq[:, None], ak[:, None]


def _anchored_rotate(self, q, k, offset: int = 0):
    """CausalSDPA._rotate for a patched block: the split by job above, on the streams of self._anchor."""
    B, n = q.shape[0], q.shape[-2]
    aq, ak = self._anchor.angles(B, n, offset, q.device)
    return (rope_half(q, aq.cos().to(q.dtype), aq.sin().to(q.dtype)),
            rope_half(k, ak.cos().to(k.dtype), ak.sin().to(k.dtype)))


def patch(model: torch.nn.Module, streams: AnchorStreams) -> int:
    """Turn every softmax block of the craft (each CausalSDPA with the rotary on, head dim 32) to the anchored rotation reading `streams`
    (its split is the streams'). -> the number of blocks turned. The parameters are untouched."""
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


def spec(content_pairs: int = CONTENT_PAIRS, code: str = CODE) -> dict:
    """The rotation as a record field (the weights folder's scale.json and the run's record). The default split's field is 0.1.72's, key
    for key. 0.1.91: a code other than 8x2 names itself (the field's levels, anchor_freqs and null are the code's; 0.1.102: its rel_pairs
    the code's shared pairs less the content)."""
    c = _check(content_pairs, code)
    r, n, f = code_shape(code)
    w, kind = frequencies(c, code)
    out = dict(form="anchored", pairs=PAIRS, rel_pairs=shared(code) - c, content_pairs=c, anchor_freqs=f, levels=n,
               ntk_scale=NTK_SCALE, base=BASE, null=r, w=[float(x) for x in w], kind=[int(x) for x in kind])
    if code != CODE:
        out["code"] = code
    return out


def code_of(rotary_field) -> str:
    """0.1.91: the address code a recorded rotation field carries (a field without one is the 8x2 code's)."""
    return rotary_field.get("code", CODE)


def content_of(rotary_field) -> int:
    """The content pair count a recorded rotation field carries (0.1.73; a field without one is ALHENA's 2), checked against spec()
    (0.1.91: at the field's code)."""
    assert isinstance(rotary_field, dict) and rotary_field.get("form") == "anchored", rotary_field
    c = int(rotary_field.get("content_pairs", CONTENT_PAIRS))
    assert rotary_field == spec(c, code_of(rotary_field)), "the recorded rotation is not this module's anchored rotation at its split"
    return c


class ValueAddress(torch.nn.Module):
    """0.1.76: THE ADDRESS ON THE VALUE PATH. Two level tables (LEVELS x 8 rows x d, born zero) add each token's address digits, read from
    the key stream of the same AnchorStreams the rotation reads, to its input vector:
        x_n = embed(t_n) + E_1[dk_1(n)] + E_2[dk_2(n)],     a digit outside 0-7 (the null anchor) adds nothing
    The anchor pairs put a record's address in the attention scores only; with the tables a head whose look lands on any token of the
    record's compartment returns the address in its value too. On the grid rows the key stream carries an address only inside the
    compartments (the index entries, the questions' answer spans and the plain rows are null there), so the tables never see an answer.
    Born zero, the patched forward equals the unpatched one until the first step moves them. 0.1.91: at the streams' code (levels x
    radix rows: 2 x 8 at 8x2, 4 x 4 at 4x4; 0.1.102: 3 x 8 at 8x3)."""

    def __init__(self, d_model: int, streams: AnchorStreams) -> None:
        super().__init__()
        self.table = torch.nn.Parameter(torch.zeros(streams.levels, streams.radix, d_model))
        self.streams = streams                            # the rotation's streams (a plain attribute: not a submodule, not in the state dict)

    def add(self, x: torch.Tensor) -> torch.Tensor:
        """x: the embedding's output (B, n, d) for the positions 0 .. n - 1 of the streams' rows. 0.1.77: each level is an embedding
        lookup into its 8 rows and a zero null row (padding_idx: the null row takes no gradient), the same values as 0.1.76's gather; the
        gradient is the embedding backward, which spreads the ~24,000 null positions of a batch over the card instead of summing them into
        one row in series (0.1.76's indexed gather cost a third of the step on the card)."""
        dk = self.streams.dk
        if dk is None:
            return x
        assert tuple(dk.shape[:2]) == tuple(x.shape[:2]), f"the streams are {tuple(dk.shape[:2])}, the embedding is {tuple(x.shape[:2])}"
        dk = dk.long()
        null_row = self.table.new_zeros(1, self.table.shape[-1])
        nl, rx = self.table.shape[0], self.table.shape[1]
        for lev in range(nl):
            d = dk[..., lev]
            d = torch.where((d >= 0) & (d < rx), d, torch.full_like(d, rx))            # any digit outside 0-7 reads the null row
            x = x + F.embedding(d, torch.cat([self.table[lev], null_row]), padding_idx=rx).to(x.dtype)
        return x


def patch_value(model: torch.nn.Module, va: ValueAddress) -> None:
    """Route the craft's embedding through `va` (0.1.76): every forward that embeds tokens (AlephLM.forward, rotary_scale.Core.trunk) adds
    the tables. The model's parameters and state dict are untouched; the tables live in `va` and their own side file."""
    emb = model.embed
    inner = emb.forward

    def forward(idx, *args, **kwargs):
        return va.add(inner(idx, *args, **kwargs))
    emb.forward = forward                                 # an instance attribute: no submodule registered, the state dict unchanged
