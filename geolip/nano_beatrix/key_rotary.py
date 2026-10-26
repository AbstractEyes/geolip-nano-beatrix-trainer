# -*- coding: utf-8 -*-
"""key_rotary.py - THE KEY IN THE TOLD HEAD'S ROTATION (0.1.82). The anchored rotation (0.1.70, 0.1.73) puts a record's address in the
attention scores: the anchor pairs turn by the address digits, so on them q_m . k_n depends on the digits' differences only. This module
puts the KEY into one head's scores the same way. At every coded SEP (key_code's rule: its last CUE in the K + 1 positions before it, no
SEP between them; n(t) = t - c(t) - 1 the key's length) each of the told head's J = 8 shared pairs (the pairs before the anchor levels:
ALHENA-W's four relative and four content pairs) turns by
    phi_j(t) = sum_{k=1}^{n(t)} P[j, k, x_{t-k}]   (mod 2 pi)
in place of its anchored angle, for the query and the key alike; P (J, K, VOCAB) is uniform on [0, 2 pi), drawn once from a CPU generator
seeded crc32("KEYROT|v1"). Between two coded SEPs m and n the told head's eight pairs then score
    sum_j Re[q_j(m) conj(k_j(n)) e^{i (phi_j(m) - phi_j(n))}]
so two SEPs with the same key turn every pair alike (the term as if unrotated), and one byte different at any offset turns every pair by
an unrelated angle: a binding of the whole key (the angle addition of FHRR), where the key code's sum of rows is a bundle. Every other
position, pair, head and block keeps the anchored angles; nothing is added to any input; no parameter. A forward's phases are set from
its rows' bytes before the blocks run (the craft's embedding forward, patched as key_code patches it; rotary_scale.Core.trunk sets them
itself), and the told block's rotation reads them. The rotation is installed before any capture of the block, so pointer.HeadCapture
keeps the rotated q and k the kernel reads. 0.1.83: the reach K is a parameter (KeyRotary(..., k)): the long keys of the larger grid row
read K = 12 with phases drawn from crc32("KEYROT|v1|k12"); K = 9 is 0.1.82's rotation and draw exactly."""
from __future__ import annotations

import hashlib
import math
import types
import zlib

import torch

try:
    from . import anchored_rope as ar
    from . import key_code as kcm
except ImportError:                                      # run as a file: the folder on the path
    import anchored_rope as ar  # noqa: E402
    import key_code as kcm  # noqa: E402
from geolip.alephllm.data.special_tokens import SEP  # noqa: E402
from geolip.alephllm.model.attention import rope_half  # noqa: E402

J = ar.SHARED                           # the told head's key pairs: the eight pairs before the anchor levels (pairs 0 .. 7)
SEED_TAG = "KEYROT|v1"
TWO_PI = 2 * math.pi
assert J == 8, "the key rotation is written for the anchored split's eight shared pairs"


def seed_tag(k: int = kcm.K) -> str:
    """0.1.83: the phases' seed tag at the reach K: 0.1.82's crc32("KEYROT|v1") at K = 9, crc32("KEYROT|v1|k<K>") at any other reach."""
    return SEED_TAG if int(k) == kcm.K else f"{SEED_TAG}|k{int(k)}"


def phase_table(k: int = kcm.K) -> torch.Tensor:
    """(J, K, VOCAB) float32 phases uniform on [0, 2 pi) from a CPU generator seeded crc32(seed_tag(K)): the same draw in every process
    (0.1.83: the reach K a parameter; K = 9 is 0.1.82's draw exactly)."""
    g = torch.Generator().manual_seed(zlib.crc32(seed_tag(k).encode()))
    return torch.rand(J, int(k), kcm.VOCAB, generator=g, dtype=torch.float64).mul_(TWO_PI).float()


class KeyRotary(torch.nn.Module):
    """The phases P (a buffer, not in any state dict) and the current forward's key angles: set(idx) before the blocks run, read by the
    told block's rotation. block is numbered from 1, as scale.json's ptr. k (0.1.83): the reach - a SEP is coded when its CUE lies in the
    k + 1 positions before it (9: 0.1.82's; 12: the long keys of the larger grid row)."""

    def __init__(self, block: int, head: int, k: int = kcm.K) -> None:
        super().__init__()
        self.block, self.head, self.k = int(block), int(head), int(k)
        self.register_buffer("table", phase_table(self.k), persistent=False)
        self.phi = self.on = None

    def phases(self, idx: torch.Tensor, off: torch.Tensor | None = None) -> tuple:
        """idx (B, n) bytes -> (phi (B, n, J) float32 in [0, 2 pi), on (B, n) bool: the coded SEPs). The phases of a key are the same
        float values wherever the key stands (the same gathers summed in the same order); past a key's length the null column adds 0."""
        off = kcm.key_offsets(idx, self.k) if off is None else off
        t = torch.cat([self.table, self.table.new_zeros(J, self.k, 1)], dim=-1)           # (J, K, VOCAB + 1)
        phi = None
        for k in range(self.k):
            p = t[:, k][:, off[..., k]]                                                     # (J, B, n)
            phi = p if phi is None else phi + p
        phi = torch.remainder(phi, TWO_PI).permute(1, 2, 0).contiguous()
        return phi, off[..., 0] != kcm.VOCAB

    def set(self, idx: torch.Tensor) -> None:
        self.phi, self.on = self.phases(idx)

    def clear(self) -> None:
        self.phi = self.on = None


def _keyrot_rotate(self, q, k, offset: int = 0):
    """The told block's rotation: anchored_rope._anchored_rotate's angles, with the told head's pairs 0 .. J - 1 turned by the key's phases
    at the coded SEPs instead. The other heads' angles are the anchored ones, value for value."""
    kr = self._keyrot
    B, H, n = q.shape[0], q.shape[1], q.shape[-2]
    assert offset == 0, "the key rotation is written for a whole-row forward (no cache)"
    assert kr.phi is not None and tuple(kr.phi.shape[:2]) == (B, n), \
        f"the key phases are {None if kr.phi is None else tuple(kr.phi.shape[:2])}, the forward is {(B, n)}: set them from the rows first"
    aq, ak = self._anchor.angles(B, n, offset, q.device)
    if aq.dim() == 2:                                     # no streams: (n, 16) for every row
        aq, ak = aq[None, None], ak[None, None]
    aq = aq.expand(B, H, n, aq.shape[-1]).clone()
    ak = ak.expand(B, H, n, ak.shape[-1]).clone()
    on = kr.on.to(q.device)[..., None]
    phi = kr.phi.to(device=q.device, dtype=aq.dtype)
    h = kr.head
    aq[:, h, :, :J] = torch.where(on, phi, aq[:, h, :, :J])
    ak[:, h, :, :J] = torch.where(on, phi, ak[:, h, :, :J])
    return (rope_half(q, aq.cos().to(q.dtype), aq.sin().to(q.dtype)),
            rope_half(k, ak.cos().to(k.dtype), ak.sin().to(k.dtype)))


def patch(model: torch.nn.Module, kr: KeyRotary) -> None:
    """Install the key rotation on the told block (after the anchored patch, before any capture of that block) and set the phases from
    every embedded row (the craft's embedding forward, after any earlier patch of it). The parameters and the state dict are untouched."""
    assert 1 <= kr.block <= len(model.blocks), f"the told block {kr.block}: 1 .. {len(model.blocks)}"
    a = model.blocks[kr.block - 1].attn
    assert getattr(a, "_anchor", None) is not None, "the key rotation turns pairs of the anchored rotation (anchored_rope.patch first)"
    assert getattr(a._rotate, "__func__", None) is ar._anchored_rotate, \
        "the told block's rotation is not the anchored one as patched (install the key rotation before any capture of the block)"
    hd = a.qkv.weight.shape[1] // a.h
    assert hd == 2 * ar.PAIRS and 0 <= kr.head < a.h, (hd, kr.head, a.h)
    a._keyrot = kr
    a._rotate = types.MethodType(_keyrot_rotate, a)
    emb = model.embed
    inner = emb.forward

    def forward(idx, *args, **kwargs):
        kr.set(idx)
        return inner(idx, *args, **kwargs)
    emb.forward = forward                                 # an instance attribute: no submodule registered, the state dict unchanged


def table_digest(t: torch.Tensor) -> str:
    return hashlib.sha256(t.detach().float().cpu().contiguous().numpy().tobytes()).hexdigest()


def record(kr: KeyRotary) -> dict:
    """The rotation as a record field: the form, the pairs, the table's size and digest, the told head (0.1.83: the reach and its seed
    tag; at K = 9 the field is 0.1.82's)."""
    return dict(form="phase", pairs=J, k=kr.k, vocab=kcm.VOCAB, block=kr.block, head=kr.head, seed_tag=seed_tag(kr.k),
                digest=table_digest(kr.table))


KEY_DIMS = list(range(J)) + list(range(ar.PAIRS, ar.PAIRS + J))                                   # pairs 0 .. J - 1 (rotate-half layout)
ANCHOR_DIMS = list(range(J, ar.PAIRS)) + list(range(ar.PAIRS + J, 2 * ar.PAIRS))                 # the anchor levels' pairs


@torch.no_grad()
def use_read(q: torch.Tensor, k: torch.Tensor, h: int, idx: torch.Tensor, metas: list, on: torch.Tensor) -> dict:
    """The told head's use of the key pairs on one forward's grid rows: q, k its rotated query and key (B, H, n, hd), h the head, idx the
    rows' input bytes (B, n), metas their grid metas (grid_rows), on the coded SEPs (B, n). At each index entry's SEP (where the pointer term
    trains the head) the score toward its true record's SEP, toward the row's other records' SEPs (their mean) and toward itself, split into
    the key pairs and the anchor pairs, in logit units (q . k / sqrt(hd)); the key pairs' magnitude (the norm over their dims) at the coded
    SEPs and at every other position, for the query and the key."""
    hd = q.shape[-1]
    sc = 1.0 / math.sqrt(hd)
    qh, kh = q[:, h].float(), k[:, h].float()                                       # (B, n, hd)
    kd, ad = torch.tensor(KEY_DIMS, device=q.device), torch.tensor(ANCHOR_DIMS, device=q.device)
    parts = {s: [] for s in ("key_true", "key_other", "key_self", "anchor_true", "anchor_other")}
    for b, m in enumerate(metas):
        row = idx[b]
        rec_sep = []
        for r in m["recs"]:                                                         # each record's SEP: the first SEP of its frame
            span = row[r["lo"]:r["frame_end"]]
            hit = (span == SEP).nonzero()
            assert len(hit), "a record frame without its SEP"
            rec_sep.append(r["lo"] + int(hit[0]))
        for e in m["index"]:
            i = int(e["a_pos"]) - 1                                                 # the entry's SEP (its first address byte follows)
            assert int(row[i]) == SEP and bool(on[b, i]), "an index entry's SEP is a coded SEP"
            t = rec_sep[e["rec"]]
            others = [p for j, p in enumerate(rec_sep) if j != e["rec"]]

            def score(ps, dims):
                p = torch.tensor(ps, device=q.device)
                return float((qh[b, i, dims] * kh[b, p][..., dims]).sum(-1).mean()) * sc
            parts["key_true"].append(score([t], kd))
            parts["key_other"].append(score(others, kd))
            parts["key_self"].append(score([i], kd))
            parts["anchor_true"].append(score([t], ad))
            parts["anchor_other"].append(score(others, ad))
    out = {s: float(sum(v) / max(1, len(v))) for s, v in parts.items()}
    out["entries"] = len(parts["key_true"])
    qk_, kk_ = qh[..., kd].norm(dim=-1), kh[..., kd].norm(dim=-1)                 # (B, n)
    out.update(q_key_norm_coded=float(qk_[on].mean()), q_key_norm_other=float(qk_[~on].mean()),
               k_key_norm_coded=float(kk_[on].mean()), k_key_norm_other=float(kk_[~on].mean()), coded_seps=int(on.sum()))
    return out
