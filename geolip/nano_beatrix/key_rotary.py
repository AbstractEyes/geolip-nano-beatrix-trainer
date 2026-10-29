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
read K = 12 with phases drawn from crc32("KEYROT|v1|k12"); K = 9 is 0.1.82's rotation and draw exactly.

0.1.87: two forms of the binding (the capacity read's next step). pairs = 16 binds all sixteen of the told head's pairs at the coded SEPs
(the anchor levels' pairs too: there they turn by the key's phases instead of the address digits'; everywhere else they keep the anchored
angles), with phases (16, K, VOCAB) drawn from crc32("KEYROT|v1|j16"). norm = True holds the key's bound pairs at one length rho at the
coded SEPs before the turn, k_j <- rho k_j / |k_j| (the query, the other pairs, positions, heads and blocks untouched): every competitor's
pair lengths equal every true record's by construction. The defaults (pairs 8, no norm) are 0.1.86's rotation and draw exactly.

0.1.88: the hold at the records only (norm = 2). A record's coded SEP is found from the bytes at or before it, never after (the SEP's own
position predicts the next byte): in a grid row (its head DOC MODE c SEP g SEP) a record is DATA a1 a2 CUE key SEP value END and an index
entry CUE key SEP a1 a2, so a record's SEP is one whose CUE follows DATA and the address bytes; in any other row the records stand inside
frames (DATA / THINK ... END) and the question after them, so a record's SEP is one whose CUE lies inside an open frame. The index
entries' and the questions' keys keep the lengths training gives them. norm = 1 is 0.1.87's hold at every coded SEP exactly.

0.1.103: THE ENTRY FORM (entry = True). Each index entry's address positions after its SEP - the positions that write its second .. last
address digit (key_code.entry_offsets: a coded SEP followed by the code's digit bytes up to the position) - are coded as the SEP: the told
head's bound pairs there turn by the entry key's phases, value for value the SEP's, for the query and the key alike. The told head can then
look at the record from every address position, as from the SEP. The hold stays the records' SEPs' (norm 2) or every coded SEP's (norm 1);
entry = False is 0.1.102's rotation exactly."""
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
from geolip.alephllm.data.special_tokens import CUE, DATA, END, SEP, THINK  # noqa: E402
from geolip.alephllm.model.attention import rope_half  # noqa: E402

J = ar.SHARED                           # the told head's key pairs: the eight pairs before the anchor levels (pairs 0 .. 7)
SEED_TAG = "KEYROT|v1"
TWO_PI = 2 * math.pi
assert J == 8, "the key rotation is written for the anchored split's eight shared pairs"
PAIR_FORMS = (J, ar.PAIRS)              # 0.1.87: the bound pairs - the eight shared pairs (0.1.82), or all sixteen of the told head
RHO = 4.16                              # 0.1.87: the norm's length (the 48-record models' median key-pair length at the true record)
NORMS = (0, 1, 2)                       # 0.1.88: the hold - none, every coded SEP (0.1.87), the records' coded SEPs only
GRID_TASK = ord("g")                    # a grid row's head: DOC MODE c SEP g SEP (grid_rows.HEAD); the task byte at position 4
ADDR_DIGITS = (ord("0"), ord("7"))      # the two-digit address bytes (grid_rows.addr_bytes); the one-byte form 0x80-0xBF
ADDR_BYTE = (0x80, 0xBF)


def records(idx: torch.Tensor, off: torch.Tensor, code: str = ar.CODE) -> torch.Tensor:
    """0.1.88: idx (B, n) bytes, off its key offsets (key_code.key_offsets) -> (B, n) bool: the coded SEPs that end a record, from the
    bytes at or before each SEP only. The key's CUE stands at c = t - n(t) - 1. A grid row (task byte 'g' at position 4): x[c - 3] = DATA
    and x[c - 2], x[c - 1] address digits, or the one-byte form x[c - 2] = DATA and x[c - 1] in 0x80-0xBF. Any other row: the CUE inside
    an open frame - the DATA and THINK bytes before c outnumber the END bytes before c. The row's kind is read at position 4, so a SEP
    before it is never held (every row's head is six bytes; no row codes a SEP there). 0.1.91: at the address code (anchored_rope.CODES)
    a grid record is x[c - L - 1] = DATA and x[c - L] .. x[c - 1] digit bytes '0' .. '0' + R - 1 (4x4: DATA and four digits '0'-'3';
    0.1.102, 8x3: DATA and three digits '0'-'7'); the one-byte form is the 8x2 code's only."""
    B, n = idx.shape
    x = idx.long()
    pos = torch.arange(n, device=idx.device)
    on = (off[..., 0] != kcm.VOCAB) & (pos >= 4)                                                       # the row's kind known from 4 on
    c = pos - (off != kcm.VOCAB).sum(-1) - 1                                                           # (B, n): the key's CUE
    c = torch.where(on, c, torch.zeros_like(c)).clamp_min(0)

    def at(back: int) -> torch.Tensor:                                                                # x[c - back], -1 before the row
        p = c - back
        return torch.where(p >= 0, torch.gather(x, 1, p.clamp_min(0)), torch.full_like(p, -1))
    if code == ar.CODE:                                                                              # 0.1.88's rule, bit for bit
        digit = lambda v: (v >= ADDR_DIGITS[0]) & (v <= ADDR_DIGITS[1])                              # noqa: E731
        grid_rec = ((at(3) == DATA) & digit(at(2)) & digit(at(1))) | ((at(2) == DATA) & (at(1) >= ADDR_BYTE[0]) & (at(1) <= ADDR_BYTE[1]))
    else:                                                                                            # 0.1.91: DATA and the code's L digits
        rx, nl, _ = ar.code_shape(code)
        grid_rec = at(nl + 1) == DATA
        for i in range(1, nl + 1):
            v = at(i)
            grid_rec = grid_rec & (v >= ADDR_DIGITS[0]) & (v <= ADDR_DIGITS[0] + rx - 1)
    depth = torch.cumsum(((x == DATA) | (x == THINK)).long() - (x == END).long(), dim=1)               # through each position
    before = torch.where(c >= 1, torch.gather(depth, 1, (c - 1).clamp_min(0)), torch.zeros_like(c))   # the bytes before the CUE
    grid = (x[:, 4] == GRID_TASK)[:, None] if n > 4 else torch.zeros(B, 1, dtype=torch.bool, device=idx.device)
    return on & torch.where(grid, grid_rec, before >= 1)                                             # (tests: x[c] is the CUE at every coded SEP)


def seed_tag(k: int = kcm.K, pairs: int = J) -> str:
    """0.1.83: the phases' seed tag at the reach K: 0.1.82's crc32("KEYROT|v1") at K = 9, crc32("KEYROT|v1|k<K>") at any other reach;
    0.1.87: "|j16" after "KEYROT|v1" at sixteen bound pairs (eight pairs name nothing)."""
    base = SEED_TAG if int(pairs) == J else f"{SEED_TAG}|j{int(pairs)}"
    return base if int(k) == kcm.K else f"{base}|k{int(k)}"


def phase_table(k: int = kcm.K, pairs: int = J) -> torch.Tensor:
    """(pairs, K, VOCAB) float32 phases uniform on [0, 2 pi) from a CPU generator seeded crc32(seed_tag(K, pairs)): the same draw in every
    process (0.1.83: the reach K a parameter; K = 9 is 0.1.82's draw exactly; 0.1.87: the pairs a parameter, 8 the same draw)."""
    assert int(pairs) in PAIR_FORMS, f"pairs {pairs!r}: one of {PAIR_FORMS}"
    g = torch.Generator().manual_seed(zlib.crc32(seed_tag(k, pairs).encode()))
    return torch.rand(int(pairs), int(k), kcm.VOCAB, generator=g, dtype=torch.float64).mul_(TWO_PI).float()


def key_dims(pairs: int = J) -> list:
    """The bound pairs' dims in the head, rotate-half: pair j = dims j and j + PAIRS (first the pairs' first components, then their
    second); at sixteen pairs every dim of the head in order."""
    return list(range(int(pairs))) + list(range(ar.PAIRS, ar.PAIRS + int(pairs)))


class KeyRotary(torch.nn.Module):
    """The phases P (a buffer, not in any state dict) and the current forward's key angles: set(idx) before the blocks run, read by the
    told block's rotation. block is numbered from 1, as scale.json's ptr. k (0.1.83): the reach - a SEP is coded when its CUE lies in the
    k + 1 positions before it (9: 0.1.82's; 12: the long keys of the larger grid row). pairs (0.1.87): the bound pairs, 8 or 16; norm
    (0.1.87): the key's bound pairs held at length rho at the coded SEPs; 0.1.88: norm 0 (none), 1 (every coded SEP, 0.1.87's True), 2 (the
    records' coded SEPs only; records()). hold: the current forward's held positions (None without a norm). code (0.1.91): the grid rows'
    address code, which the records' rule reads (the hold at norm 2). 0.1.102: at a code whose shared pairs are not eight (8x3: four) the
    eight-pair form would turn anchor pairs, so the rotation binds all sixteen there. 0.1.103: entry (the entry form above) codes each index
    entry's address positions as its SEP."""

    def __init__(self, block: int, head: int, k: int = kcm.K, pairs: int = J, norm: int = 0, rho: float = RHO,
                 code: str = ar.CODE, entry: bool = False) -> None:
        super().__init__()
        assert int(pairs) in PAIR_FORMS, f"pairs {pairs!r}: one of {PAIR_FORMS}"
        assert int(norm) in NORMS and int(norm) == norm, f"norm {norm!r}: one of {NORMS}"
        ar.code_shape(code)
        assert int(pairs) == ar.PAIRS or ar.shared(code) == J, \
            f"the {code} code shares {ar.shared(code)} pairs: the key rotation binds all {ar.PAIRS} there (pairs {ar.PAIRS})"
        self.block, self.head, self.k, self.pairs, self.code = int(block), int(head), int(k), int(pairs), code
        self.entry = bool(entry)                          # 0.1.103: the entries' address positions coded as their SEPs
        self.radix, self.levels, _ = ar.code_shape(code)
        self.norm, self.rho = int(norm), float(rho)
        assert self.rho > 0, rho
        self.register_buffer("table", phase_table(self.k, self.pairs), persistent=False)
        self.phi = self.on = self.hold = None

    def phases(self, idx: torch.Tensor, off: torch.Tensor | None = None) -> tuple:
        """idx (B, n) bytes -> (phi (B, n, pairs) float32 in [0, 2 pi), on (B, n) bool: the coded SEPs). The phases of a key are the same
        float values wherever the key stands (the same gathers summed in the same order); past a key's length the null column adds 0."""
        off = kcm.key_offsets(idx, self.k) if off is None else off
        t = torch.cat([self.table, self.table.new_zeros(self.pairs, self.k, 1)], dim=-1)  # (pairs, K, VOCAB + 1)
        phi = None
        for k in range(self.k):
            p = t[:, k][:, off[..., k]]                                                     # (J, B, n)
            phi = p if phi is None else phi + p
        phi = torch.remainder(phi, TWO_PI).permute(1, 2, 0).contiguous()
        return phi, off[..., 0] != kcm.VOCAB

    def set(self, idx: torch.Tensor) -> None:
        off = kcm.key_offsets(idx, self.k)
        self.phi, self.on = self.phases(idx, kcm.entry_offsets(idx, off, self.radix, self.levels) if self.entry else off)
        sep_on = off[..., 0] != kcm.VOCAB                 # the coded SEPs (= self.on without the entry form)
        self.hold = {0: None, 1: sep_on, 2: records(idx, off, self.code) if self.norm == 2 else None}[self.norm]   # 0.1.88: where the norm holds

    def clear(self) -> None:
        self.phi = self.on = self.hold = None


def norm_key(k: torch.Tensor, h: int, on: torch.Tensor, rho: float, pairs: int = J) -> torch.Tensor:
    """0.1.87: the told head's bound key pairs at the coded SEPs held at length rho, k_j <- rho k_j / |k_j| (rotate-half: pair j = dims j
    and j + PAIRS; the length in float32); every other position, pair and head value for value (a factor of exactly one)."""
    kh = k[:, h]                                                                            # (B, n, hd)
    a, b = kh[..., :pairs], kh[..., ar.PAIRS:ar.PAIRS + pairs]
    m = torch.sqrt(a.float() ** 2 + b.float() ** 2).clamp_min(1e-6)
    f = torch.where(on[..., None], rho / m, torch.ones_like(m)).to(kh.dtype)
    kh = torch.cat([a * f, kh[..., pairs:ar.PAIRS], b * f, kh[..., ar.PAIRS + pairs:]], dim=-1)
    return torch.cat([k[:, :h], kh[:, None], k[:, h + 1:]], dim=1)


def _keyrot_rotate(self, q, k, offset: int = 0):
    """The told block's rotation: anchored_rope._anchored_rotate's angles, with the told head's pairs 0 .. pairs - 1 turned by the key's
    phases at the coded SEPs instead (0.1.87: and, with the norm, the key's bound pairs there at length rho before the turn). The other
    heads' angles are the anchored ones, value for value."""
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
    h, P = kr.head, kr.pairs
    aq[:, h, :, :P] = torch.where(on, phi, aq[:, h, :, :P])
    ak[:, h, :, :P] = torch.where(on, phi, ak[:, h, :, :P])
    if kr.norm:                                           # 0.1.87: the key's bound pairs at one length (the turn keeps lengths); 0.1.88: where
        k = norm_key(k, h, kr.hold.to(k.device), kr.rho, P)   # the hold says (every coded SEP, or the records' only)
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
    tag; at K = 9 the field is 0.1.82's; 0.1.87: the bound pairs, and the norm and its length when on - off and at eight pairs the field
    is 0.1.86's; 0.1.88: the norm's code, 1 every coded SEP as 0.1.87 wrote it, 2 the records' only)."""
    out = dict(form="phase", pairs=kr.pairs, k=kr.k, vocab=kcm.VOCAB, block=kr.block, head=kr.head, seed_tag=seed_tag(kr.k, kr.pairs),
               digest=table_digest(kr.table))
    if kr.norm:
        out.update(norm=kr.norm, rho=kr.rho)
    if kr.entry:                                          # 0.1.103: the entries' address positions coded as their SEPs (off: 0.1.102's field)
        out["entry"] = True
    return out


KEY_DIMS = key_dims(J)                                                                             # pairs 0 .. J - 1 (rotate-half layout)
ANCHOR_DIMS = list(range(J, ar.PAIRS)) + list(range(ar.PAIRS + J, 2 * ar.PAIRS))                 # the anchor levels' pairs


@torch.no_grad()
def use_read(q: torch.Tensor, k: torch.Tensor, h: int, idx: torch.Tensor, metas: list, on: torch.Tensor, pairs: int = J) -> dict:
    """The told head's use of the key pairs on one forward's grid rows: q, k its rotated query and key (B, H, n, hd), h the head, idx the
    rows' input bytes (B, n), metas their grid metas (grid_rows), on the coded SEPs (B, n). At each index entry's SEP (where the pointer term
    trains the head) the score toward its true record's SEP, toward the row's other records' SEPs (their mean) and toward itself, split into
    the key pairs and the anchor pairs, in logit units (q . k / sqrt(hd)); the key pairs' magnitude (the norm over their dims) at the coded
    SEPs and at every other position, for the query and the key. 0.1.87: pairs = the bound pairs (at sixteen, every pair is a key pair and
    the anchor terms are 0)."""
    hd = q.shape[-1]
    sc = 1.0 / math.sqrt(hd)
    qh, kh = q[:, h].float(), k[:, h].float()                                       # (B, n, hd)
    kdl = key_dims(pairs)
    adl = [d for d in range(2 * ar.PAIRS) if d not in kdl]
    kd, ad = torch.tensor(kdl, device=q.device), torch.tensor(adl, device=q.device, dtype=torch.long)
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
