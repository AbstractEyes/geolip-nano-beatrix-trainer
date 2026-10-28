# -*- coding: utf-8 -*-
"""ninth_head.py - 0.1.94, THE PATH IN A HEAD OF ITS OWN: a ninth softmax head in every anchored block, beside the block's eight, its own
query, key, value and output rows, its output added to the block's attention output. In each block b, reading the block's attention
input x (B, n, d):

    [q9, k9, v9] = W9_b x            (W9_b: 3 * 32 x d),       y9(t) = sum_{u <= t} softmax_u(<rot_q q9(t), rot_k k9(u)> / sqrt(32)) v9(u),
    the block's attention output += O9_b y9                     (O9_b: d x 32).

Born as the block's heads are born (the house's qkv and output weights are orthogonal draws): W9_b is the first 96 rows of a fresh
orthogonal 3d x d draw and O9_b the first 32 columns of a fresh orthogonal d x d draw, from a CPU generator seeded
crc32("NINTH|v1|s<seed>|b<block>") - nothing is drawn from the run's generators, so the eight heads, the rest of the model and the data
order start as the run without it; the two rotations' heads are equal at birth.

The rotation (rot):
    "path"  all sixteen pairs (pair j = dims j and j + 16) of q9(t) turned by phi(t) and of k9(u) by phi(u - 1) (the byte-after form;
            "same" turns k9(u) by phi(u)), phi_j(t) = sum_{k=0}^{min(t - s(t) + 1, K) - 1} P16[j, k, x_(t-k)] (mod 2 pi), j = 0 .. 15,
            P16 (16, K, 256) uniform on [0, 2 pi) from crc32("PATHROT|v1|j16") (path_rotary's units and phases at sixteen pairs);
    "plain" the block's anchored rotation (AnchorStreams.angles: the angles every head of the block gets) - the capacity control.
off (a read's toggle, the path rotation only): the anchored rotation instead, the weights unchanged.

0.1.95, THE PATH IN THE EARLY BLOCKS: path_blocks (the path rotation only) names the blocks, numbered from 1, whose ninth head carries the
path; the others' ninth heads take the block's anchored rotation (the plain form's). None (the default) = every block, 0.1.94's form; the
full set gives the same forward, the empty set the plain form's. The heads' birth and capacity are the same whatever the set; off puts
every path block on the anchored rotation (the plain form's forward on the same weights).

The heads live outside the model's module tree (their forward is a closure over the block's attention): the model's state dict stays the
house body's; the trainer adds their parameters to its optimizer and ships them beside the weights (extra_head.safetensors, scale.json's
ninth_head); rotary_scale.Core installs them from there."""
from __future__ import annotations

import hashlib
import math
import zlib

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from . import anchored_rope as ar
    from . import path_rotary as prm
except ImportError:                                      # run as a file: the folder on the path
    import anchored_rope as ar  # noqa: E402
    import path_rotary as prm  # noqa: E402
from geolip.alephllm.model.attention import rope_half  # noqa: E402

HD = 2 * ar.PAIRS                       # the head's 32 dims
PAIRS = ar.PAIRS                        # sixteen pairs, all bound in the path form
ROTS = ("path", "plain")
SEED_TAG = "PATHROT|v1|j16"
BIRTH_TAG = "NINTH|v1"


def seed_tag(k: int = prm.K) -> str:
    """The sixteen-pair phases' seed tag: crc32("PATHROT|v1|j16") at the reach 16, "...|k<K>" at any other."""
    return SEED_TAG if int(k) == prm.K else f"{SEED_TAG}|k{int(k)}"


def phase_table(k: int = prm.K) -> torch.Tensor:
    """(16, K, 256) float32 phases uniform on [0, 2 pi) from a CPU generator seeded crc32(seed_tag(K)): the same draw in every process."""
    g = torch.Generator().manual_seed(zlib.crc32(seed_tag(k).encode()))
    return torch.rand(PAIRS, int(k), prm.VOCAB, generator=g, dtype=torch.float64).mul_(prm.TWO_PI).float()


class Path16(prm.PathRotary):
    """path_rotary's phases at sixteen pairs (the ninth head's): the same units, reach and forms, the table P16."""

    def __init__(self, units: prm.Units, form: str, k: int = prm.K) -> None:
        super().__init__(units, form, head=-1, k=k)
        self.register_buffer("table", phase_table(self.k), persistent=False)


def birth(d: int, seed: int, block: int) -> tuple:
    """(W9 (96, d), O9 (d, 32)) for block `block` (numbered from 1) of a run of seed `seed`: rows of a fresh orthogonal 3d x d draw and
    columns of a fresh orthogonal d x d draw, from their own generator (the run's generators untouched)."""
    g = torch.Generator().manual_seed(zlib.crc32(f"{BIRTH_TAG}|s{int(seed)}|b{int(block)}".encode()))
    w = torch.empty(3 * d, d)
    nn.init.orthogonal_(w, generator=g)
    o = torch.empty(d, d)
    nn.init.orthogonal_(o, generator=g)
    return w[:3 * HD].clone(), o[:, :HD].clone()


class NinthHeads(nn.Module):
    """The ninth heads of a craft's anchored blocks: qkv[i] (Linear d -> 96) and o[i] (Linear 32 -> d), born by birth(); rot "path" or
    "plain"; the path's phases set from the rows before the blocks run (set()); off: the path head on the anchored rotation (a read);
    path_blocks (0.1.95, the path only): the blocks (from 1) whose head carries the path, None = every block."""

    def __init__(self, d: int, n_blocks: int, rot: str, seed: int, units: prm.Units | None = None, form: str = "next",
                 k: int = prm.K, path_blocks=None) -> None:
        super().__init__()
        assert rot in ROTS, f"rot {rot!r}: one of {ROTS}"
        assert rot == "plain" or units is not None, "the path rotation reads the unit rule"
        self.d, self.n_blocks, self.rot, self.seed = int(d), int(n_blocks), rot, int(seed)
        if path_blocks is not None:              # sorted, each once; the full and the empty set allowed (the laws' ends)
            assert rot == "path", "a path block set names where the path rotation goes: --ninth_head=path"
            path_blocks = tuple(sorted({int(b) for b in path_blocks}))
            assert all(1 <= b <= self.n_blocks for b in path_blocks), f"path blocks {path_blocks}: 1 .. {self.n_blocks}"
        self.path_blocks = path_blocks
        # skip_init: the layers' default init would draw from the run's generator; their weights come from birth() alone
        self.qkv = nn.ModuleList(nn.utils.skip_init(nn.Linear, self.d, 3 * HD, bias=False) for _ in range(self.n_blocks))
        self.o = nn.ModuleList(nn.utils.skip_init(nn.Linear, HD, self.d, bias=False) for _ in range(self.n_blocks))
        with torch.no_grad():
            for i in range(self.n_blocks):
                w, o = birth(self.d, self.seed, i + 1)
                self.qkv[i].weight.copy_(w)
                self.o[i].weight.copy_(o)
        self.path = Path16(units, form, k=k) if rot == "path" else None
        self.off = False
        self.capture = False                 # a read's capture of each block's rotated q9, k9 (room_read.look)
        self.caps = [None] * self.n_blocks

    @property
    def form(self):
        return None if self.path is None else self.path.form

    @property
    def units(self):
        return None if self.path is None else self.path.units

    def set(self, x: torch.Tensor) -> None:
        if self.path is not None:
            self.path.set(x)

    def clear(self) -> None:
        if self.path is not None:
            self.path.clear()

    def rotate(self, q: torch.Tensor, k: torch.Tensor, anchor: ar.AnchorStreams, i: int | None = None) -> tuple:
        """q, k (B, 1, n, 32) -> rotated: the path's phases on all sixteen pairs, or the block's anchored angles (plain; off; a block
        outside path_blocks). i: the block (from 0), needed when path_blocks names a set."""
        B, n = q.shape[0], q.shape[-2]
        path = self.path is not None and not self.off
        if path and self.path_blocks is not None:   # 0.1.95: the path on the named blocks only
            assert i is not None, "a path block set: the block's index decides its rotation"
            path = i + 1 in self.path_blocks
        if path:
            pq, pk = self.path.phi_q, self.path.phi_k
            assert pq is not None and tuple(pq.shape[:2]) == (B, n), \
                f"the path phases are {None if pq is None else tuple(pq.shape[:2])}, the forward is {(B, n)}: set them from the rows first"
            aq, ak = pq.to(q.device)[:, None], pk.to(k.device)[:, None]                              # (B, 1, n, 16)
        else:
            aq, ak = anchor.angles(B, n, 0, q.device)
        return (rope_half(q, aq.cos().to(q.dtype), aq.sin().to(q.dtype)),
                rope_half(k, ak.cos().to(k.dtype), ak.sin().to(k.dtype)))

    def head(self, i: int, x: torch.Tensor, anchor: ar.AnchorStreams) -> torch.Tensor:
        """Block i's ninth head on its attention input x (B, n, d) -> its output (B, n, d), to be added to the block's."""
        B, n, _ = x.shape
        q, k, v = self.qkv[i](x).view(B, n, 3, 1, HD).permute(2, 0, 3, 1, 4)                      # each (B, 1, n, 32)
        q, k = self.rotate(q, k, anchor, i)
        if self.capture:
            self.caps[i] = (q, k)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.o[i](y.transpose(1, 2).reshape(B, n, HD))


def patch(model: nn.Module, nh: NinthHeads) -> int:
    """Add the ninth head to every anchored block's attention output (the attention module's forward wrapped as an instance attribute:
    no submodule registered, the state dict unchanged) and, for the path form, set its phases from every embedded row. -> the blocks."""
    n = 0
    for b in model.blocks:
        a = b.attn
        if getattr(a, "_anchor", None) is None:
            continue
        assert a.qkv.weight.shape[1] == nh.d and a.qkv.weight.shape[0] == 3 * HD * a.h, (tuple(a.qkv.weight.shape), a.h, nh.d)
        inner, i = a.forward, n

        def forward(x, _inner=inner, _i=i, _a=a):
            return _inner(x) + nh.head(_i, x, _a._anchor)
        a.forward = forward
        n += 1
    assert n == nh.n_blocks, f"{n} anchored blocks, {nh.n_blocks} ninth heads"
    if nh.path is not None:
        emb = model.embed
        inner_f = emb.forward

        def eforward(idx, *args, **kwargs):
            nh.set(idx)
            return inner_f(idx, *args, **kwargs)
        emb.forward = eforward
    return n


def params(nh: NinthHeads) -> int:
    return sum(p.numel() for p in nh.parameters())


def state(nh: NinthHeads) -> dict:
    """The side file's tensors: every block's qkv and o weights, and the path's phases (the seeded draw, checked when read back)."""
    out = {k: v.detach().cpu().contiguous() for k, v in nh.state_dict().items()}
    if nh.path is not None:
        out["table"] = nh.path.table.detach().cpu().contiguous()
    return out


def load(nh: NinthHeads, sd: dict) -> None:
    """The side file into nh: the weights strict, the phases equal to the seeded draw."""
    sd = dict(sd)
    t = sd.pop("table", None)
    assert (t is None) == (nh.path is None), "the side file carries the path's phases exactly when the head carries the path"
    if t is not None:
        assert torch.equal(t.to(nh.path.table.device), nh.path.table), "the stored phases are not the seeded draw"
    nh.load_state_dict({k: v.to(nh.qkv[0].weight.device) for k, v in sd.items()}, strict=True)


def scale_field(nh: NinthHeads) -> dict:
    """The heads as scale.json's ninth_head field (rotary_scale.Core installs them from this and extra_head.safetensors)."""
    f = dict(rot=nh.rot, seed=nh.seed, blocks=nh.n_blocks, d=nh.d, hd=HD, birth_tag=BIRTH_TAG)
    if nh.path is not None:
        f.update(form=nh.form, k=nh.path.k, pairs=PAIRS, vocab=prm.VOCAB, seed_tag=seed_tag(nh.path.k), rule="entropy",
                 theta=nh.units.theta, floor=nh.units.floor, table=prm.TABLE_FILE, table_sha256=prm.TABLE_SHA256)
        if nh.path_blocks is not None:           # 0.1.95: the path's blocks (absent = every block, 0.1.94's field)
            f["path_blocks"] = list(nh.path_blocks)
    return f


def check_field(f: dict) -> None:
    """A stored ninth_head field is this module's head (the rotation, the birth and, for the path, the draw, the rule, the table and the
    path's blocks when named)."""
    assert f.get("rot") in ROTS and f.get("hd") == HD and f.get("birth_tag") == BIRTH_TAG and int(f.get("blocks", 0)) >= 1, f
    if f["rot"] == "path":
        assert f.get("form") in prm.FORMS and f.get("pairs") == PAIRS and f.get("vocab") == prm.VOCAB, f
        assert f.get("seed_tag") == seed_tag(int(f["k"])) and f.get("rule") == "entropy", f
        assert float(f.get("theta")) == prm.THETA and float(f.get("floor")) == prm.FLOOR and f.get("table_sha256") == prm.TABLE_SHA256, f
    if "path_blocks" in f:
        pb = f["path_blocks"]
        assert f["rot"] == "path" and isinstance(pb, list) and pb == sorted(set(pb)), f
        assert all(isinstance(b, int) and 1 <= b <= int(f["blocks"]) for b in pb), f


def digest(nh: NinthHeads) -> str:
    h = hashlib.sha256()
    for k, v in sorted(state(nh).items()):
        h.update(k.encode())
        h.update(v.float().numpy().tobytes())
    return h.hexdigest()


def record(nh: NinthHeads) -> dict:
    """The heads as a record field: scale_field's, the parameters and the trained state's digest."""
    return dict(scale_field(nh), params=params(nh), digest=digest(nh))


def scores(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """(B, 1, n, 32) q and k -> (B, n, n) the head's raw scores <q(t), k(u)> / sqrt(32) (the tests' and the reads' form)."""
    return (q[:, 0].float() @ k[:, 0].float().transpose(-1, -2)) / math.sqrt(HD)
