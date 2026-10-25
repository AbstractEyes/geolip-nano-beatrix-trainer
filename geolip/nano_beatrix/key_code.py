# -*- coding: utf-8 -*-
"""key_code.py - THE KEY CODE AT THE END-MARKER (0.1.81). The byte-trigram input e_t = E0[x_t] + E1[x_{t-1}] + E2[x_{t-2}] + P[t] hands a
SEP the last two bytes of the key before it, so on rows whose keys share their endings a head comparing one SEP with another ties every
record whose key ends the same way. The key code adds the whole key at the SEP, one table an offset back:
    e'_t = e_t + sum_{k=1}^{n(t)} T_k[x_{t-k}]     at a SEP whose last CUE c(t) lies in the K + 1 positions before it, with no SEP
                                                   between them; n(t) = t - c(t) - 1, the key's length
    e'_t = e_t                                     everywhere else (the text, the other specials, a SEP with no CUE in reach)
so the inner product of two SEPs' codes adds over the offsets where their keys' bytes agree. Two forms:
    fixed    T_k[b] ~ N(0, I_d), drawn once from a generator seeded crc32("KEYCODE|v1") (the byte tables' own init scale); a buffer, no
             gradient, the same rows for every seed
    learned  T_k born zero (weight-zero: the patched forward equals the plain one until the first step moves them)
The code reads only the bytes between a CUE and its SEP, never a value or an address: on every row form of the bed those bytes are the
key (the grid rows' records, index entries and questions; the frame rows' records and questions), at most K = 9 bytes (the longest cue
word and three digits, bed_frames.key_name); the rows' heads (DOC, the tag, SEP, a task byte, SEP) carry no CUE and get none. Patched into
the embedding's forward as the value-path tables are (anchored_rope.patch_value): the model's parameters and state dict are untouched."""
from __future__ import annotations

import hashlib
import zlib

import torch
import torch.nn.functional as F

from geolip.alephllm.data.special_tokens import CUE, SEP

K = 9                                   # the longest key in bytes: the longest cue word (6 letters) and three digits
VOCAB = 256                             # the byte rows; row VOCAB of each lookup is the null row (padding_idx: no gradient)
SEED_TAG = "KEYCODE|v1"
FORMS = ("fixed", "learned")


def fixed_table(d: int) -> torch.Tensor:
    """The fixed form's rows (K, VOCAB, d): N(0, 1) from a CPU generator seeded crc32(SEED_TAG), the same draw in every process."""
    g = torch.Generator().manual_seed(zlib.crc32(SEED_TAG.encode()))
    return torch.randn(K, VOCAB, d, generator=g)


def key_offsets(idx: torch.Tensor) -> torch.Tensor:
    """idx (B, n) bytes -> (B, n, K) long: at each coded SEP (its last CUE in the K + 1 positions before it, no SEP between them) column
    k - 1 holds x_{t-k} for k = 1 .. n(t) and VOCAB (the null row) past the key; every other position VOCAB in every column."""
    assert idx.dim() == 2, tuple(idx.shape)
    n = idx.shape[1]
    pad = F.pad(idx.long(), (K + 1, 0), value=-1)                         # position t of idx is position t + K + 1 of pad
    back = torch.stack([pad[:, K + 1 - j:K + 1 - j + n] for j in range(K + 2)], dim=-1)   # back[..., j] = x_{t-j}, j = 0 .. K + 1
    js = torch.arange(1, K + 2, device=idx.device)
    cue_at = torch.where(back[..., 1:] == CUE, js, torch.full_like(js, K + 2)).min(-1).values   # the nearest CUE's distance (K + 2: none)
    klen = cue_at - 1                                                      # the key's length n(t) where a CUE is in reach
    ks = torch.arange(1, K + 1, device=idx.device)
    in_key = ks <= klen[..., None]                                         # (B, n, K): offset k inside the key
    sep_in_key = ((back[..., 1:K + 1] == SEP) & in_key).any(-1)
    on = (back[..., 0] == SEP) & (cue_at <= K + 1) & ~sep_in_key
    take = on[..., None] & in_key
    return torch.where(take, back[..., 1:K + 1], torch.full_like(back[..., 1:K + 1], VOCAB))


class KeyCode(torch.nn.Module):
    """The key code's K tables (K, VOCAB, d): the fixed form's seeded rows (a buffer) or the learned form's born-zero parameter."""

    def __init__(self, d_model: int, form: str) -> None:
        super().__init__()
        assert form in FORMS, f"the key code's form {form!r}: one of {FORMS}"
        self.form = form
        if form == "fixed":
            self.register_buffer("table", fixed_table(d_model), persistent=False)
        else:
            self.table = torch.nn.Parameter(torch.zeros(K, VOCAB, d_model))

    def code(self, idx: torch.Tensor, off: torch.Tensor | None = None) -> torch.Tensor:
        """The code at every position of idx (B, n) -> (B, n, d) in the table's dtype (zero wherever nothing is coded): one embedding
        lookup an offset into that offset's rows and a zero null row."""
        off = key_offsets(idx) if off is None else off
        null_row = self.table.new_zeros(1, self.table.shape[-1])
        c = None
        for k in range(K):
            e = F.embedding(off[..., k], torch.cat([self.table[k], null_row]), padding_idx=VOCAB)
            c = e if c is None else c + e
        return c

    def add(self, idx: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """x: the embedding's output (B, n, d) for the positions 0 .. n - 1 of the rows idx (B, n)."""
        assert tuple(idx.shape) == tuple(x.shape[:2]), f"the bytes are {tuple(idx.shape)}, the embedding is {tuple(x.shape[:2])}"
        return x + self.code(idx).to(x.dtype)


def patch_key(model: torch.nn.Module, kc: KeyCode) -> None:
    """Route the craft's embedding through `kc` (after any earlier patch, e.g. the value-path tables): every forward that embeds a row from
    its first byte (AlephLM.forward, rotary_scale.Core.trunk) adds the code. The model's parameters and state dict are untouched; the
    tables live in `kc` and their own side file."""
    emb = model.embed
    inner = emb.forward

    def forward(idx, *args, **kwargs):
        return kc.add(idx, inner(idx, *args, **kwargs))
    emb.forward = forward                                 # an instance attribute: no submodule registered, the state dict unchanged


def table_digest(t: torch.Tensor) -> str:
    return hashlib.sha256(t.detach().float().cpu().contiguous().numpy().tobytes()).hexdigest()


def record(kc: KeyCode) -> dict:
    """The tables as a record field: the form, the size, how far they reach (the Frobenius norm of each offset's rows), the digest."""
    t = kc.table.detach().float().cpu()
    out = dict(form=kc.form, k=K, vocab=VOCAB, d=int(t.shape[-1]), params=(int(t.numel()) if kc.form == "learned" else 0),
               norm=[float(t[k].norm()) for k in range(K)], digest=table_digest(t))
    if kc.form == "fixed":
        out["seed_tag"] = SEED_TAG
    return out


@torch.no_grad()
def share_read(model: torch.nn.Module, kc: KeyCode, idx: torch.Tensor) -> dict:
    """At every coded SEP of the rows idx (B, n): the code's norm, the norm of the input it joins (the patched embedding's output, the code
    in it), their ratio, and the keys' mean length. The anchored streams (when the craft reads them) are the caller's to set."""
    off = key_offsets(idx)
    on = (off != VOCAB).any(-1)
    c = kc.code(idx, off).float()[on]
    e = model.embed(idx).float()[on]
    cn, en = c.norm(dim=-1), e.norm(dim=-1)
    return dict(seps=int(on.sum()), key_len=float((off != VOCAB).sum(-1)[on].float().mean()), code_norm=float(cn.mean()),
                input_norm=float(en.mean()), share=float((cn / en.clamp(min=1e-12)).mean()),
                plain_norm=float((e - c).norm(dim=-1).mean()))
