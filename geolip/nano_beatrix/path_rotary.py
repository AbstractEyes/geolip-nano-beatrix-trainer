# -*- coding: utf-8 -*-
"""path_rotary.py - THE PATH IN ONE HEAD'S ROTATION (0.1.92): a byte-resolution trie feature sequence bound into the attention scores.
The key rotation (key_rotary.py) binds a record's whole key into the told head's pairs at the coded SEPs; this module binds every byte's
UNIT PATH into one head's pairs at every position. A byte STARTS A UNIT under the fusion stage's entropy rule (the library's
model/fusion.py, Fusion.starts at rule "entropy"): the atlas cell of the three bytes before it has next-byte entropy at or above THETA
bits or fewer than FLOOR witnesses, or it is one of the row's first three bytes, or it or the byte before it is a special byte or a
newline. s(t) is the start of the unit holding t, and the path at t is the bytes x_(s(t)) .. x_t (a node of the trie of the units). The
path head's eight anchor pairs (pairs 8-15, the pairs the address digits turn) turn by
    phi_j(t) = sum_{k=0}^{min(t - s(t) + 1, K) - 1} P[j, k, x_(t-k)]   (mod 2 pi),   j = 0 .. 7, K = 16 (the path's last K bytes)
in place of their installed angles, at every position of every row; P (8, K, 256) is uniform on [0, 2 pi), drawn once from a CPU
generator seeded crc32("PATHROT|v1"). Two forms:
    same  the query and the key at t both turn by phi(t): on those pairs two positions with equal paths score as unrotated (the node's
          identity), two paths one byte apart by an unrelated angle on every pair;
    next  the query at t turns by phi(t), the key at u by phi(u - 1) (phi(-1) = 0): the query meets unrotated the positions right after
          an earlier occurrence of its own path, whose byte is the one that followed that path before (a continuation lookup).
Every other head, pair and block keeps the installed rotation value for value (the told head's key rotation included); nothing is added
to any input; no parameter. off = True returns the installed rotation exactly (the forward without the path). A forward's phases are set
from its rows' bytes before the blocks run (the craft's embedding forward, patched as key_rotary patches it; rotary_scale.Core.trunk sets
them itself), and every patched block's rotation reads them.

The unit rule's table is the atlas's (v3_atlas_entropy_table.npz: 256^3 cells of next-byte entropy in 1/16 bit and witness counts), read
as the library reads it (fusion.load_code), from a given path, $GEOLIP_ATLAS_TABLE, or the rotary repo's pillar/ copy fetched tokenless,
and checked against its sha256 either way. repeat_mask() marks the positions whose path occurred earlier in the same row (the positions
a continuation lookup can match): the held-out text's split by it is the repeat read."""
from __future__ import annotations

import hashlib
import math
import os
import zlib

import numpy as np
import torch

try:
    from . import anchored_rope as ar
except ImportError:                                      # run as a file: the folder on the path
    import anchored_rope as ar  # noqa: E402
from geolip.alephllm.data.special_tokens import SPECIALS  # noqa: E402

LO = ar.SHARED                          # the first bound pair: the anchor pairs 8 .. 15 (the pairs the address digits turn)
PAIRS = ar.PAIRS - ar.SHARED            # eight bound pairs
K = 16                                  # the path's reach: its last 16 bytes (the entropy rule's units: 14 bytes at q99.9 on the bed's text)
VOCAB = 256
SEED_TAG = "PATHROT|v1"
TWO_PI = 2 * math.pi
THETA, FLOOR = 3.0, 8.0                 # the entropy rule's threshold (bits) and the witness floor
FORMS = ("same", "next")
NEWLINE = 10
UNWITNESSED = 255                       # fusion.load_code's code for a cell under the witness floor
TABLE_REPO = "AbstractPhil/geolip-rotary-experiments"
TABLE_FILE = "pillar/v3_atlas_entropy_table.npz"
TABLE_SHA256 = "a6c47b1e7f74e02e2afd91056f6f654f40c745b76ab7183a5be8f7fb673fd15a"
TABLE_ENV = "GEOLIP_ATLAS_TABLE"
assert PAIRS == 8 and LO == 8, "the path rotation is written for the anchored split's eight anchor pairs"


def table_file(path: str | None = None, pinned: bool = True) -> str:
    """The atlas table's local path: path, else $GEOLIP_ATLAS_TABLE, else the rotary repo's copy (huggingface_hub's cache, fetched without a
    token); its sha256 is checked against TABLE_SHA256 either way (pinned = False: a given table unchecked, for the tests' made tables)."""
    p = path or os.environ.get(TABLE_ENV) or ""
    if not p:
        from huggingface_hub import hf_hub_download
        p = hf_hub_download(TABLE_REPO, TABLE_FILE, token=False)
    if pinned:
        with open(p, "rb") as f:
            h = hashlib.sha256(f.read()).hexdigest()
        assert h == TABLE_SHA256, f"the atlas table at {p} is not the pinned one (sha256 {h})"
    return p


def hit_table(path: str, theta: float = THETA, floor: float = FLOOR) -> torch.Tensor:
    """(256^3,) bool: the cells where the entropy rule starts a unit - code >= 16 theta or unwitnessed, the code read as fusion.load_code
    reads it (entropy_x16 clipped to 254, 255 under the witness floor)."""
    from geolip.alephllm.model.fusion import load_code
    code = load_code(path, floor)
    tc = int(min(254, round(theta * 16)))
    return torch.from_numpy((code >= tc) | (code == UNWITNESSED))


class Units(torch.nn.Module):
    """The entropy rule's unit starts (fusion.Fusion.starts at rule "entropy", theta and the witness floor): buffers only, never in a
    state dict. table: the atlas table's path (table_file's order when None); pinned: its sha256 checked (table_file)."""

    def __init__(self, table: str | None = None, theta: float = THETA, floor: float = FLOOR, pinned: bool = True) -> None:
        super().__init__()
        self.path = table_file(table, pinned)
        self.theta, self.floor = float(theta), float(floor)
        self.register_buffer("hit", hit_table(self.path, self.theta, self.floor), persistent=False)
        special = torch.zeros(VOCAB, dtype=torch.bool)
        special[list(SPECIALS)] = True
        self.register_buffer("special", special, persistent=False)

    def starts(self, x: torch.Tensor) -> torch.Tensor:
        """x (B, n) bytes -> (B, n) bool: True where a unit starts (Fusion.starts at rule "entropy", value for value)."""
        x = x.long()
        B, n = x.shape
        st = torch.zeros(B, n, dtype=torch.bool, device=x.device)
        st[:, :min(3, n)] = True
        if n > 3:
            st[:, 3:] |= self.hit[x[:, :-3] * 65536 + x[:, 1:-2] * 256 + x[:, 2:-1]]
        sp, nl = self.special[x], x == NEWLINE
        f = sp | nl
        f[:, 1:] |= sp[:, :-1] | nl[:, :-1]
        return st | f

    def depth(self, x: torch.Tensor) -> torch.Tensor:
        """x (B, n) -> (B, n) long: the path's length t - s(t) + 1 (1 at a unit's first byte)."""
        st = self.starts(x)
        pos = torch.arange(x.shape[1], device=x.device)
        s = torch.cummax(torch.where(st, pos, torch.zeros_like(pos)), dim=1).values
        return pos - s + 1


def seed_tag(k: int = K) -> str:
    """The phases' seed tag: crc32("PATHROT|v1") at the reach 16, "PATHROT|v1|k<K>" at any other."""
    return SEED_TAG if int(k) == K else f"{SEED_TAG}|k{int(k)}"


def phase_table(k: int = K) -> torch.Tensor:
    """(8, K, VOCAB) float32 phases uniform on [0, 2 pi) from a CPU generator seeded crc32(seed_tag(K)): the same draw in every process."""
    g = torch.Generator().manual_seed(zlib.crc32(seed_tag(k).encode()))
    return torch.rand(PAIRS, int(k), VOCAB, generator=g, dtype=torch.float64).mul_(TWO_PI).float()


class PathRotary(torch.nn.Module):
    """The phases P (a buffer, not in any state dict), the unit rule, and the current forward's angles: set(idx) before the blocks run,
    read by every patched block's rotation. form: "same" or "next"; head: the path head (every patched block's); k: the reach; off: the
    installed rotation instead (a read's toggle)."""

    def __init__(self, units: Units, form: str, head: int = 7, k: int = K) -> None:
        super().__init__()
        assert form in FORMS, f"form {form!r}: one of {FORMS}"
        assert int(k) >= 1, k
        self.units, self.form, self.head, self.k = units, form, int(head), int(k)
        self.register_buffer("table", phase_table(self.k), persistent=False)
        self.phi_q = self.phi_k = None
        self.off = False

    def phases(self, x: torch.Tensor) -> torch.Tensor:
        """x (B, n) bytes -> phi (B, n, pairs) float32 in [0, 2 pi): eight pairs here (0.1.94: sixteen in the ninth head, ninth_head.Path16).
        A path's phases are the same float values wherever it stands (the same gathers summed in the same order; past the path's length
        the null column adds 0)."""
        x = x.long()
        B, n = x.shape
        d = self.units.depth(x)
        t = torch.cat([self.table, self.table.new_zeros(self.table.shape[0], self.k, 1)], dim=-1)   # (pairs, K, VOCAB + 1)
        phi = None
        for k in range(self.k):
            xs = torch.full_like(x, VOCAB)
            if k < n:
                xs[:, k:] = x[:, :n - k]                                                         # x_(t-k)
            off = torch.where(d > k, xs, torch.full_like(xs, VOCAB))
            p = t[:, k][:, off]                                                                  # (pairs, B, n)
            phi = p if phi is None else phi + p
        return torch.remainder(phi, TWO_PI).permute(1, 2, 0).contiguous()

    def set(self, x: torch.Tensor) -> None:
        phi = self.phases(x)
        self.phi_q = phi
        self.phi_k = phi if self.form == "same" else torch.cat([torch.zeros_like(phi[:, :1]), phi[:, :-1]], dim=1)

    def clear(self) -> None:
        self.phi_q = self.phi_k = None


def turn(t_pre: torch.Tensor, t_out: torch.Tensor, h: int, phi: torch.Tensor) -> torch.Tensor:
    """t_out (B, H, n, 32) with head h's pairs LO .. LO + 7 replaced by t_pre's turned by phi (B, n, 8) (rotate-half: pair j = dims j and
    j + 16; the cosines and sines formed in fp32 and cast once, as the anchored rotation forms them); every other value t_out's."""
    c, s = phi.cos().to(t_pre.dtype), phi.sin().to(t_pre.dtype)
    x, y = t_pre[:, h], t_out[:, h]
    a, b = x[..., LO:ar.PAIRS], x[..., ar.PAIRS + LO:]
    yh = torch.cat([y[..., :LO], a * c - b * s, y[..., ar.PAIRS:ar.PAIRS + LO], a * s + b * c], dim=-1)
    return torch.cat([t_out[:, :h], yh[:, None], t_out[:, h + 1:]], dim=1)


def patch(model: torch.nn.Module, pr: PathRotary) -> int:
    """Install the path rotation on every anchored block (after the anchored patch and the key rotation, before any capture of a block)
    and set the phases from every embedded row (the craft's embedding forward, after any earlier patch of it). -> the number of blocks
    turned. The parameters and the state dict are untouched."""
    n = 0
    for b in model.blocks:
        a = b.attn
        if getattr(a, "_anchor", None) is None:
            continue
        hd = a.qkv.weight.shape[1] // a.h
        assert hd == 2 * ar.PAIRS and 0 <= pr.head < a.h, (hd, pr.head, a.h)
        inner = a._rotate

        def _rotate(q, k, offset: int = 0, _inner=inner, _h=a.h):
            qo, ko = _inner(q, k, offset)
            if pr.off:
                return qo, ko
            B, H, n_ = q.shape[0], q.shape[1], q.shape[-2]
            assert offset == 0, "the path rotation is written for a whole-row forward (no cache)"
            assert H == _h, f"the path rotation turns one head of the block's {_h}: a forward of {H} heads (a stopped capture) is not supported"
            assert pr.phi_q is not None and tuple(pr.phi_q.shape[:2]) == (B, n_), \
                f"the path phases are {None if pr.phi_q is None else tuple(pr.phi_q.shape[:2])}, the forward is {(B, n_)}: set them from the rows first"
            return turn(q, qo, pr.head, pr.phi_q.to(q.device)), turn(k, ko, pr.head, pr.phi_k.to(k.device))
        a._rotate = _rotate
        n += 1
    assert n, "no anchored block to turn (anchored_rope.patch first)"
    emb = model.embed
    inner_f = emb.forward

    def forward(idx, *args, **kwargs):
        pr.set(idx)
        return inner_f(idx, *args, **kwargs)
    emb.forward = forward                                 # an instance attribute: no submodule registered, the state dict unchanged
    return n


def table_digest(t: torch.Tensor) -> str:
    return hashlib.sha256(t.detach().float().cpu().contiguous().numpy().tobytes()).hexdigest()


def scale_field(pr: PathRotary) -> dict:
    """The rotation as scale.json's path_rot field (rotary_scale.Core installs it from this and path_rot.safetensors)."""
    return dict(form=pr.form, head=pr.head, lo=LO, pairs=PAIRS, k=pr.k, vocab=VOCAB, seed_tag=seed_tag(pr.k), rule="entropy",
                theta=pr.units.theta, floor=pr.units.floor, table=TABLE_FILE, table_sha256=TABLE_SHA256)


def record(pr: PathRotary, blocks: int) -> dict:
    """The rotation as a record field: scale_field's, the blocks turned and the phases' digest."""
    return dict(scale_field(pr), blocks=int(blocks), digest=table_digest(pr.table))


def check_field(f: dict) -> None:
    """A stored path_rot field is this module's rotation (the draw, the rule and the table pinned)."""
    assert f.get("form") in FORMS and f.get("lo") == LO and f.get("pairs") == PAIRS and f.get("vocab") == VOCAB, f
    assert f.get("seed_tag") == seed_tag(int(f["k"])) and f.get("rule") == "entropy", f
    assert float(f.get("theta")) == THETA and float(f.get("floor")) == FLOOR and f.get("table_sha256") == TABLE_SHA256, f


def repeat_mask(units: Units, x: torch.Tensor) -> np.ndarray:
    """x (B, n) bytes -> (B, n) bool numpy: the positions whose path (the bytes from the unit's start through the position, whole) occurred
    at an earlier position of the same row - the positions a continuation lookup can match."""
    d = units.depth(x).cpu().numpy()
    xb = x.detach().cpu().numpy().astype(np.uint8)
    out = np.zeros(d.shape, dtype=bool)
    for b in range(xb.shape[0]):
        row, seen, db = xb[b].tobytes(), set(), d[b]
        for t in range(xb.shape[1]):
            key = row[t - int(db[t]) + 1:t + 1]
            if key in seen:
                out[b, t] = True
            else:
                seen.add(key)
    return out


class RepeatSplit:
    """The held-out bits split by repeat_mask: add(nll (B, n) nats, rep (B, n) bool) per batch; out() -> bits a byte at the repeat
    positions and at the others, their counts and the repeat share."""

    def __init__(self) -> None:
        self.s = [0.0, 0.0]
        self.n = [0, 0]

    def add(self, nll: np.ndarray, rep: np.ndarray) -> None:
        assert nll.shape == rep.shape, (nll.shape, rep.shape)
        self.s[0] += float(nll[rep].sum())
        self.n[0] += int(rep.sum())
        self.s[1] += float(nll[~rep].sum())
        self.n[1] += int((~rep).sum())

    def out(self) -> dict:
        ln2 = math.log(2.0)
        r, o = (self.s[i] / self.n[i] / ln2 if self.n[i] else None for i in (0, 1))
        return dict(repeat=r, other=o, n_repeat=self.n[0], n_other=self.n[1],
                    repeat_share=self.n[0] / max(1, self.n[0] + self.n[1]))
