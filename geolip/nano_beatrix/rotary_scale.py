# -*- coding: utf-8 -*-
"""rotary_scale.py - THE SCALE TEST'S MEMBERS (0.1.71) on a grid core (bed_train --grid=ag): real arms and toy members trained alone on
the frozen core, then merged under the product gate, and the read that grades them.

THE MEMBERS. Member i owns address a_i (grid_rows.owned_order(): the first 48 addresses of a fixed permutation, in merge order; the other
16 are never owned) and answers a question on its record in its own code: f_i, a crc-seeded derangement of a-z ("SCALE|code|i"), byte by
byte (the END unchanged), at the question's seven answer positions (its SEP and six value bytes, teacher-forced on the plain row: the
member turns the core's prediction of each value byte into its code).
    real arms    the arm module of record (amoe-lora 0.2.12, RelayPatchwork) at LAWFUL 16 x 8: 16 slots of D 8 through an aleph read of
                 K 16 anchors (tau .1), squared ReLU, LayerNorm, hidden 256; the output weight AND bias born at zero (exact-inert), the
                 gate born at -3. The write Delta_i(h) = sigmoid(g_i) consume_i(m_hat_i(proj_i h)) (RelayPatchwork returns h + Delta).
    toy members  the night's module at hidden 256: LayerNorm, Linear, GELU, Linear (the output born zero): the members' replication arm.
Each member trains ALONE: its own mean cross-entropy on its own questions, its own pure Adam (betas .9 / .98, eps 1e-8, lr 1e-3, 50
warm-up steps, the cosine to a tenth; a member absent from a batch takes no step), the core frozen (its digest asserted at the end).
    LAST SITE  after the final block, before the final norm: all 48 real arms and 48 toy members in one paired pass a step on member-mode
               rows (18 owned compartments a row, a per-batch permutation of the 48 cycled over the rows, and 6 never-owned; the
               questions ask the owned records), so each member meets 3 of its questions a step; disjoint parameters and no position
               mixing after the site: each gradient is its solo gradient.
    MID SITE   after block 3 of 6: the first 16 owners as real arms on row-owned batches (each row an ordinary grid row whose one
               question, at a drawn slot, asks its member's record; only that member writes, only there; rows do not interact, so each
               gradient is its solo gradient although blocks 4-6 mix positions); blocks 4-6 recomputed in full.
The trunk runs under bf16 autocast while the members train (the core's training precision); the members and the readout run in fp32.
THE GATE (the S-line's): a request r = (r1, r2) at a question's answer positions (zero everywhere else: no request, every member abstains)
    k_l(a, b) = <L9[a], L9[b]> (nine orthonormal level anchors; a digit outside 0-7 reads 0),  u_i = U k_1(r1, i1) k_2(r2, i2),
    w_i = sinh(u_i) / (1 + sum_j (cosh(u_j) - 1))   (REST; FULL: sum_j cosh(u_j)),   h <- h + sum_i w_i Delta_i(h)
THE REQUEST SOURCES (read on the bare core): LAYOUT (the true address), INDEX (the two bytes the core writes at the asked key's index
entry, teacher-forced on the row), INDEXC (the true index with one pair of the row's questions trading addresses: must fail), HEADP (the
address head's per-digit scores s_l = sigmoid(z_l) in place of the kernels), HEADPQ999 (the scores through the absolute floor
f = max(0, s - tau) / (1 - tau), tau per level = the 99.9th percentile of the per-question largest wrong digit score on the calibration
rows), G2 (w_i = [r = a_i], the oracle), G0 (w_i = 1, must fail). On the anchored core the questions' query streams carry the written
index at the read (the training form carries the true address).
THE READ (fp32, TF32 off): each source x U 5 / 8 / 10 x m 16 / 32 / 48 under REST and FULL; merged / solo (the owner alone at its source's
solo weight tanh(U prod)) / unowned / bare; PROTECTED (merged >= solo - .05), QUIET (unowned >= bare - .02 and KL(bare || merged) <= .05
nats a position), EXACT (|merged - solo| <= .01 and KL <= .001); the leaked write over an owned write, the unowned margins, flips at
2 / 4 / 8x the leak, the owner's weight, the off-band weight, the request right rates, the owner-weight scan. Accuracies are per address,
then averaged over the addresses (the S-line's convention).
0.1.72, THE TWO-PASS READ (the find form; FIND_SOURCES): every source puts its OWN address in the questions' streams and in the gate. Pass 1
(every question at the null anchor) reads the index the core writes and the head at each index entry's SEP, on the row as composed and on
the free-running row (grid_rows.free_running: the core's own index bytes at their fixed point); pass 2 reads the answers with the source's
address in the streams. The sources: LAYOUT, INDEX, INDEXC as above; INDEXF (the index written on the free-running row); HEADPF (the head
at the question's SEP on the free-running row, INDEXF in the stream); HEADI / HEADIF (the head at the asked key's index entry on either row,
its .5 decode in the stream, its scores in the gate); the _Q999 forms through the floor calibrated in the same form. run_core(forms=...)
takes either read or both beside the same stacks; read0 reads a stored core with its stored stacks (no training).
0.1.80, THE LOOK (the readout tie): the request read off where one attention head looks. In pass 1 the designated head's own causal
attention (its block's rotated q and k from the forward, captured by pointer.HeadCapture: the forward unchanged) is summed over each
record's frame; the request is the address of the record holding more than half of that mass (at most one can; judged by its own mass,
never ranked: the house's .5 decode), both digits from that one record, else the null anchor (every member abstains). The sources: LOOK
(at the asked key's index entry's SEP, the row as composed), LOOKF (the same on the free-running row), LOOKQF (at the question's own SEP,
the free-running row); each exact in the gate. The head: the core's own (scale.json's ptr: the pointer term's block and head) or a given
(block, head); look_read reads them without members."""
from __future__ import annotations

import hashlib
import json
import math
import os
import queue
import threading
import time
import zlib

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from . import grid_rows as gr
    from . import anchored_rope as ar
    from . import pointer as pt
    from . import key_code as kcm
    from . import key_rotary as krm
    from . import __version__ as _PKG_VERSION
except ImportError:                                      # run as a file: the folder on the path
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import grid_rows as gr  # noqa: E402
    import anchored_rope as ar  # noqa: E402
    import pointer as pt  # noqa: E402
    import key_code as kcm  # noqa: E402
    import key_rotary as krm  # noqa: E402
    _PKG_VERSION = "file"
from geolip.alephllm.data.special_tokens import DOC  # noqa: E402

N_OWN = 48
M_SET = (16, 32, 48)
M_MID = 16
U_SET = (5.0, 8.0, 10.0)
Z_SET = ("REST", "FULL")
MID_SITE = 3                                             # the write enters blocks[3]: after block 3 of 6
SOURCES = ("LAYOUT", "INDEX", "INDEXC", "HEADP", "HEADPQ999")
ARM_SPEC = dict(n_slots=16, K=16, D=8, tau=0.1, hidden=256, gate_init=-3.0)
TOY_HID = 256
AMOE_PIN = "0.2.12"
NA = gr.VLEN + 1                                         # a question's answer positions: its SEP and six value bytes
FLIPS = (2, 4, 8)
QS = (0.9, 0.99, 0.999)
A_GRID = (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
BETAS, EPS, LR, WARM = (0.9, 0.98), 1e-8, 1e-3, 50
ROWS_LAST, ROWS_MID = 8, 8                               # a batch's rows (the bed's micro-batch)
OWN_PER_ROW, NEVER_PER_ROW = 18, 6
MEMBER_ROW = {"AG": (OWN_PER_ROW, NEVER_PER_ROW), "AG48": (36, 12)}   # 0.1.83: a last-site row's owned and never-owned compartments
COMPILE = True                                           # the stacked writes compiled (static shapes; the gate below)
COMPILE_MAX_SHAPES = 16                                  # the session's shapes: 2 trainers, the trace, the reads, the text bits
_COMPILED, _SHAPES = {}, set()


def crc(tag: str) -> int:
    return zlib.crc32(tag.encode())


def lr_at(step: int, steps: int, base: float = LR, warm: int = WARM) -> float:
    """The S-line's member schedule: linear warm-up, then the cosine to a tenth."""
    if step < warm:
        return base * (step + 1) / warm
    p = (step - warm) / max(1, steps - warm)
    return base * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * p)))


# ------------------------------------------------------------------ the members' tasks
def code_table(i: int) -> np.ndarray:
    """Member i's code: a crc-seeded derangement of a-z (no letter maps to itself); every other byte maps to itself. -> (256,) int64."""
    rng = np.random.default_rng(crc(f"SCALE|code|{i}"))
    while True:
        p = rng.permutation(26)
        if (p != np.arange(26)).all():
            break
    t = np.arange(256, dtype=np.int64)
    t[97:123] = 97 + p
    return t


def code_tables(n: int = N_OWN) -> torch.Tensor:
    return torch.from_numpy(np.stack([code_table(i) for i in range(n)]))


def owned_sets() -> tuple:
    own, never = gr.owned_order(N_OWN)
    return list(own), list(never)


# ------------------------------------------------------------------ the gate (the S-line's code path)
def simplex_lift(n_anchor: int, P: int) -> torch.Tensor:
    """(n_anchor, 2 P + 1) fp64: the night's lifted simplex code, orthonormal when n_anchor <= 2 P + 1."""
    N = 2 * P + 1
    c = torch.arange(n_anchor, dtype=torch.float64)[:, None]
    k = torch.arange(1, P + 1, dtype=torch.float64)[None, :]
    ang = 2 * math.pi * k * c / N
    v = torch.cat([ang.cos(), ang.sin()], 1) / math.sqrt(P)
    b = 1.0 / math.sqrt(N)
    return torch.cat([math.sqrt(1 - b * b) * v, torch.full((n_anchor, 1), b, dtype=torch.float64)], 1)


def level_code(dev) -> torch.Tensor:
    """The level anchors in fp32: rows 0-7 the digits, row 8 the null anchor (orthonormal; a request digit outside 0-7 reads row 8, whose
    kernel with every member's digit is 0)."""
    L9 = simplex_lift(gr.RADIX + 1, 4)
    assert float((L9 @ L9.T - torch.eye(gr.RADIX + 1, dtype=torch.float64)).abs().max()) < 1e-10
    return L9.float().to(dev)


def norm_w(u: torch.Tensor, Z: str) -> torch.Tensor:
    """w = sinh(u) / Z over the last axis: FULL sum cosh, REST 1 + sum (cosh - 1), MEAN the mean cosh."""
    s, ch = torch.sinh(u), torch.cosh(u)
    if Z == "FULL":
        d = ch.sum(-1, keepdim=True)
    elif Z == "REST":
        d = 1.0 + (ch - 1.0).sum(-1, keepdim=True)
    elif Z == "MEAN":
        d = ch.mean(-1, keepdim=True)
    else:
        raise ValueError(Z)
    return s / d


def exact_factors(req: torch.Tensor, mem: torch.Tensor, L9: torch.Tensor) -> list:
    """req (Q, L) request digits (outside 0-7 = no address), mem (m, L) the members' digits -> L kernels (Q, m) by the L9 dot products."""
    r = torch.where((req >= 0) & (req < gr.RADIX), req, torch.full_like(req, gr.RADIX))
    return [L9[r[:, l]] @ L9[mem[:, l]].T for l in range(req.shape[1])]


def floor_fac(s: torch.Tensor, tau) -> torch.Tensor:
    """The absolute floor: max(0, s - tau) / (1 - tau); tau 0 returns s; a score at or under tau gives exactly 0 (the member abstains).
    A tau of 1 (the calibration's largest wrong score saturated at 1) floors every score to 0: no member writes (the formula's 0 / 0)."""
    if float(tau) >= 1.0:
        return torch.zeros_like(s)
    return (s - tau).clamp(min=0.0) / (1.0 - tau)


def head_factors(s: torch.Tensor, mem: torch.Tensor, tau=None) -> list:
    """s (Q, L, R) the head's per-digit scores, mem (m, L) -> L factors (Q, m): each member's digit score, floored when tau (per level)."""
    out = []
    for l in range(s.shape[1]):
        f = s[:, l, mem[:, l]]
        out.append(f if tau is None else floor_fac(f, float(tau[l])))
    return out


def u_of(fac: list, U: float, m: int | None = None) -> torch.Tensor:
    u = U * (fac[0] if m is None else fac[0][:, :m])
    for g in fac[1:]:
        u = u * (g if m is None else g[:, :m])
    return u


def own_weight(Z: str, U: float, m: int) -> float:
    """The owner's merged weight on an exact gate (every other member at u = 0)."""
    s, c = math.sinh(U), math.cosh(U)
    if Z == "FULL":
        return s / (c + m - 1)
    if Z == "REST":
        return s / c
    return m * s / (c + m - 1)


# ------------------------------------------------------------------ the stacked members
def _arm_write(h, P, C, W1, B1, LW, LB, W2, B2, G, S: int, D: int, tau: float):
    """h (M, n, d) -> (M, n, d): each member's RelayPatchwork write on its own rows (amoe's AlephAddress.m_hat, stacked)."""
    M_, n, _ = h.shape
    x = torch.einsum("mnd,med->mne", h, P).reshape(M_, n, S, D)
    A = F.normalize(C, dim=-1)
    u = torch.einsum("mnsd,mkd->mnsk", F.normalize(x, dim=-1), A) / tau
    mx = u.abs().amax(-1, keepdim=True)
    ep, en = torch.exp(u - mx), torch.exp(-u - mx)
    f = torch.einsum("mnsk,mkd->mnsd", ep - en, A) / (ep + en).sum(-1, keepdim=True)
    z = torch.einsum("mne,mhe->mnh", f.reshape(M_, n, S * D), W1) + B1[:, None]
    z = F.relu(z) ** 2
    z = F.layer_norm(z, (z.shape[-1],), eps=1e-5) * LW[:, None] + LB[:, None]
    return torch.sigmoid(G)[:, None, None] * (torch.einsum("mnh,mdh->mnd", z, W2) + B2[:, None])


def _toy_write(h, LW, LB, W1, B1, W2, B2):
    """h (M, n, d) -> (M, n, d): the night's member (LayerNorm, Linear, GELU, Linear), stacked."""
    z = F.layer_norm(h, (h.shape[-1],), eps=1e-5) * LW[:, None] + LB[:, None]
    a = F.gelu(torch.einsum("mnd,mhd->mnh", z, W1) + B1[:, None])
    return torch.einsum("mnh,mdh->mnd", a, W2) + B2[:, None]


def _raise_dynamo_limits():
    import torch._dynamo.config as dc
    for names, floor in ((("recompile_limit", "cache_size_limit"), 64), (("accumulated_recompile_limit", "accumulated_cache_size_limit"), 4096)):
        for name in names:
            if hasattr(dc, name):
                setattr(dc, name, max(int(getattr(dc, name)), int(floor)))


def _run_write(fn, h, *params, **kw):
    """The write through torch.compile (static shapes) for the first COMPILE_MAX_SHAPES (function, shape) pairs; eager otherwise (exact)."""
    key = (fn.__name__, tuple(h.shape), str(h.device))
    if COMPILE and h.is_cuda and (key in _SHAPES or len(_SHAPES) < COMPILE_MAX_SHAPES):
        _SHAPES.add(key)
        if fn.__name__ not in _COMPILED:
            _raise_dynamo_limits()
            _COMPILED[fn.__name__] = torch.compile(fn, dynamic=False)
        return _COMPILED[fn.__name__](h, *params, **kw)
    return fn(h, *params, **kw)


class StackedArms(nn.Module):
    """M arm modules (amoe's RelayPatchwork at LAWFUL 16 x 8, exact-inert) with stacked parameters."""

    KEYS = ("proj.weight", "addr.codebook", "consume.0.weight", "consume.0.bias", "consume.2.weight", "consume.2.bias",
            "consume.3.weight", "consume.3.bias", "gate")

    def __init__(self, M: int, d: int, spec: dict = ARM_SPEC):
        super().__init__()
        self.M, self.d, self.spec = M, d, dict(spec)
        S, K, D, H = spec["n_slots"], spec["K"], spec["D"], spec["hidden"]
        self.proj = nn.Parameter(torch.zeros(M, S * D, d))
        self.codebook = nn.Parameter(torch.zeros(M, K, D))
        self.register_buffer("home", torch.zeros(M, K, D))
        self.w1 = nn.Parameter(torch.zeros(M, H, S * D))
        self.b1 = nn.Parameter(torch.zeros(M, H))
        self.lnw = nn.Parameter(torch.ones(M, H))
        self.lnb = nn.Parameter(torch.zeros(M, H))
        self.w2 = nn.Parameter(torch.zeros(M, d, H))
        self.b2 = nn.Parameter(torch.zeros(M, d))
        self.gate = nn.Parameter(torch.full((M,), float(spec["gate_init"])))

    @staticmethod
    def amoe_modules(M: int, d: int, tag: str) -> list:
        """M of amoe's RelayPatchwork at this spec, member j seeded crc(f"{tag}|{j}"), the output bias zeroed (the exact-inert form)."""
        import amoe
        from amoe.core.adapter import AdapterSpec, RelayPatchwork
        assert amoe.__version__ == AMOE_PIN, f"the arm module of record is amoe-lora {AMOE_PIN}, found {amoe.__version__}"
        s = ARM_SPEC
        out = []
        for j in range(M):
            torch.manual_seed(crc(f"{tag}|{j}"))
            a = RelayPatchwork(d, AdapterSpec(n_slots=s["n_slots"], K=s["K"], D=s["D"], tau=s["tau"], hidden=s["hidden"],
                                              gate_init=s["gate_init"], zero_init_head=True))
            with torch.no_grad():
                a.consume[-1].bias.zero_()
            out.append(a)
        return out

    @classmethod
    def build(cls, M: int, d: int, tag: str) -> "StackedArms":
        return cls(M, d).load_members([{k: v.detach().clone() for k, v in a.state_dict().items()} for a in cls.amoe_modules(M, d, tag)])

    @torch.no_grad()
    def load_members(self, sds: list) -> "StackedArms":
        """sds: RelayPatchwork state dicts in stack order."""
        for j, sd in enumerate(sds):
            self.proj[j].copy_(sd["proj.weight"])
            self.codebook[j].copy_(sd["addr.codebook"])
            self.home[j].copy_(sd["addr.home"])
            self.w1[j].copy_(sd["consume.0.weight"])
            self.b1[j].copy_(sd["consume.0.bias"])
            self.lnw[j].copy_(sd["consume.2.weight"])
            self.lnb[j].copy_(sd["consume.2.bias"])
            self.w2[j].copy_(sd["consume.3.weight"])
            self.b2[j].copy_(sd["consume.3.bias"])
            self.gate[j].copy_(sd["gate"].reshape(()))
        return self

    def member(self, j: int) -> dict:
        """Member j as a RelayPatchwork state dict (amoe's keys): it loads into RelayPatchwork(d, the LAWFUL spec) and attaches as any arm."""
        return {"proj.weight": self.proj[j], "addr.codebook": self.codebook[j], "addr.home": self.home[j],
                "consume.0.weight": self.w1[j], "consume.0.bias": self.b1[j], "consume.2.weight": self.lnw[j], "consume.2.bias": self.lnb[j],
                "consume.3.weight": self.w2[j], "consume.3.bias": self.b2[j], "gate": self.gate[j]}

    def params(self, idx=None) -> tuple:
        p = (self.proj, self.codebook, self.w1, self.b1, self.lnw, self.lnb, self.w2, self.b2, self.gate)
        return p if idx is None else tuple(t[idx] for t in p)

    def write(self, h: torch.Tensor, idx=None) -> torch.Tensor:
        """h (M', n, d), one slab per member (idx: which members, None = all M) -> the members' writes (M', n, d)."""
        s = self.spec
        return _run_write(_arm_write, h, *self.params(idx), S=s["n_slots"], D=s["D"], tau=float(s["tau"]))

    def write_all(self, h: torch.Tensor) -> torch.Tensor:
        """h (Q, d) -> (Q, M, d): every member on every position."""
        return self.write(h[None].expand(self.M, *h.shape).contiguous()).transpose(0, 1)


class StackedToys(nn.Module):
    """M of the night's members (capacity_bench.Member at d, hid) with stacked parameters; member j seeded crc(f"{tag}|{j}")."""

    def __init__(self, M: int, d: int, hid: int = TOY_HID):
        super().__init__()
        self.M, self.d, self.hid = M, d, hid
        self.lnw = nn.Parameter(torch.ones(M, d))
        self.lnb = nn.Parameter(torch.zeros(M, d))
        self.w1 = nn.Parameter(torch.zeros(M, hid, d))
        self.b1 = nn.Parameter(torch.zeros(M, hid))
        self.w2 = nn.Parameter(torch.zeros(M, d, hid))
        self.b2 = nn.Parameter(torch.zeros(M, d))

    @classmethod
    def build(cls, M: int, d: int, tag: str, hid: int = TOY_HID) -> "StackedToys":
        stk = cls(M, d, hid)
        with torch.no_grad():
            for j in range(M):
                torch.manual_seed(crc(f"{tag}|{j}"))
                ln, w1, w2 = nn.LayerNorm(d), nn.Linear(d, hid), nn.Linear(hid, d)
                nn.init.zeros_(w2.weight)
                nn.init.zeros_(w2.bias)
                stk.lnw[j].copy_(ln.weight)
                stk.lnb[j].copy_(ln.bias)
                stk.w1[j].copy_(w1.weight)
                stk.b1[j].copy_(w1.bias)
                stk.w2[j].copy_(w2.weight)
                stk.b2[j].copy_(w2.bias)
        return stk

    def params(self, idx=None) -> tuple:
        p = (self.lnw, self.lnb, self.w1, self.b1, self.w2, self.b2)
        return p if idx is None else tuple(t[idx] for t in p)

    def write(self, h: torch.Tensor, idx=None) -> torch.Tensor:
        return _run_write(_toy_write, h, *self.params(idx))

    def write_all(self, h: torch.Tensor) -> torch.Tensor:
        return self.write(h[None].expand(self.M, *h.shape).contiguous()).transpose(0, 1)


class StackedAdam:
    """torch.optim.Adam's update (pure Adam: no weight decay) per member along the first axis: a member absent from a step takes no
    step (its moments and its step count unchanged), exactly as its own optimizer would (the S-line's StackedAdam)."""

    def __init__(self, params, M: int, betas=BETAS, eps: float = EPS):
        self.params = list(params)
        self.b1, self.b2 = betas
        self.eps = eps
        self.m = [torch.zeros_like(p) for p in self.params]
        self.v = [torch.zeros_like(p) for p in self.params]
        self.t = torch.zeros(M, device=self.params[0].device, dtype=torch.float64)

    @torch.no_grad()
    def step(self, present: torch.Tensor, lr: float) -> None:
        self.t += present.double()
        bc1 = (1 - self.b1 ** self.t.clamp(min=1)).float()
        bc2s = ((1 - self.b2 ** self.t.clamp(min=1)) ** 0.5).float()
        for p, m, v in zip(self.params, self.m, self.v):
            g = p.grad
            if g is None:
                continue
            shp = (-1,) + (1,) * (p.dim() - 1)
            pm = present.view(shp)
            m.copy_(torch.where(pm, m.lerp(g, 1 - self.b1), m))
            v.copy_(torch.where(pm, v * self.b2 + (1 - self.b2) * g * g, v))
            denom = v.sqrt() / bc2s.view(shp) + self.eps
            p.sub_(torch.where(pm, (lr / bc1.view(shp)) * m / denom, torch.zeros_like(p)))

    def state(self) -> dict:
        return {"m": [t.detach().cpu() for t in self.m], "v": [t.detach().cpu() for t in self.v], "t": self.t.detach().cpu()}

    @torch.no_grad()
    def load(self, st: dict) -> None:
        for a, b in zip(self.m, st["m"]):
            a.copy_(b)
        for a, b in zip(self.v, st["v"]):
            a.copy_(b)
        self.t.copy_(st["t"])


# ------------------------------------------------------------------ the core
def _digest(model: nn.Module, extra: torch.Tensor | None = None, key_code: torch.Tensor | None = None,
            key_rot: torch.Tensor | None = None) -> str:
    """The core's state as one hash; extra (0.1.76: the value-path tables), key_code (0.1.81: the key code's tables) and key_rot (0.1.82:
    the key rotation's phases) join it when the core carries them (a core without any hashes as before)."""
    h = hashlib.sha256()
    for k, v in sorted(model.state_dict().items()):
        h.update(k.encode())
        h.update(v.detach().cpu().contiguous().numpy().tobytes())
    if extra is not None:
        h.update(b"value_addr")
        h.update(extra.detach().cpu().contiguous().numpy().tobytes())
    if key_code is not None:
        h.update(b"key_code")
        h.update(key_code.detach().cpu().contiguous().numpy().tobytes())
    if key_rot is not None:
        h.update(b"key_rot")
        h.update(key_rot.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


class Core:
    """A trained grid core from its weights folder (config.json, model.safetensors, scale.json, addr_head.safetensors), frozen: the
    anchored rotation patched in when scale.json names it, the value-path tables (0.1.76) and the key code (0.1.81) in the embedding and
    the key rotation (0.1.82) in the told block when the folder carries their files, the address head beside it. look (0.1.80): the head the look sources read -
    "ptr" = the block and head the core's pointer term trained (scale.json's ptr; none on a core without the term), a (block, head) pair
    with the block numbered from 1 as the reads print it (the control on a core without the term), or None (no look sources)."""

    def __init__(self, folder: str, dev, look="ptr") -> None:
        from safetensors.torch import load_file
        from geolip.alephllm.model.alephlm import AlephLM
        from geolip.alephllm.presets import AlephLMConfig
        self.folder, self.dev = folder, torch.device(dev)
        self.tag = os.path.basename(os.path.normpath(folder))
        c = json.load(open(os.path.join(folder, "config.json"), encoding="utf-8"))
        self.cfg = AlephLMConfig(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in c.items()})
        self.model = AlephLM(self.cfg).to(self.dev)
        self.model.load_state_dict({k: v.to(self.dev) for k, v in load_file(os.path.join(folder, "model.safetensors")).items()}, strict=True)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        m = self.model
        assert m.fusion is None and not len(getattr(m, "lookup_arms", {})) and m.last_fix is None and m.head.mode == "", \
            "the members' sites are written for the byte trunk without fusion, lookup arms, the last fix or a dial head"
        self.n_blocks = len(m.blocks)
        self.scale = json.load(open(os.path.join(folder, "scale.json"), encoding="utf-8"))
        self.abytes = int(self.scale.get("abytes", 2))            # 0.1.74: the address form the core was trained on (its rows are built so)
        self.layout = self.scale.get("grid_layout", "AG")         # 0.1.83: the grid layout and the keys' form it was trained on (the same)
        self.key_form = self.scale.get("grid_keys", "std")
        assert self.layout in gr.LAYOUTS and self.key_form in gr.KEY_FORMS, (self.layout, self.key_form)
        self.code = self.scale.get("grid_code", gr.CODE)          # 0.1.91: the address code it was trained on (its rows, streams and head)
        self.radix, self.levels = gr.radix_levels(self.code)
        self.anchor = None
        if isinstance(self.scale.get("rotary"), dict):            # 0.1.73: at the split the core was trained with (ALHENA's 2 or ALHENA-W's 4)
            assert ar.code_of(self.scale["rotary"]) == self.code, "scale.json's rotation and grid code disagree"
            self.anchor = ar.AnchorStreams(ar.content_of(self.scale["rotary"]), self.code)
            ar.patch(self.model, self.anchor)
        self.value_addr = None
        vp = os.path.join(folder, "value_addr.safetensors")
        assert os.path.exists(vp) == ("value_addr" in self.scale), "scale.json names the value-path tables exactly when their file is there"
        if os.path.exists(vp):                                    # 0.1.76: the value-path tables on the same streams, in every embedding
            assert self.anchor is not None, "the value-path tables read the anchored rotation's streams"
            self.value_addr = ar.ValueAddress(self.cfg.d_model, self.anchor).to(self.dev)
            self.value_addr.table.data.copy_(load_file(vp)["table"].to(self.dev))
            self.value_addr.table.requires_grad_(False)
            ar.patch_value(self.model, self.value_addr)
        self.key_code = None
        kp = os.path.join(folder, "key_code.safetensors")
        assert os.path.exists(kp) == ("key_code" in self.scale), "scale.json names the key code's tables exactly when their file is there"
        if os.path.exists(kp):                                    # 0.1.81: the key code at the end-marker, after the value-path tables
            kf = self.scale["key_code"]
            assert kf.get("k") == kcm.K and kf.get("vocab") == kcm.VOCAB and kf.get("d") == self.cfg.d_model, kf
            self.key_code = kcm.KeyCode(self.cfg.d_model, kf["form"]).to(self.dev)
            t = load_file(kp)["table"]
            assert tuple(t.shape) == tuple(self.key_code.table.shape), (tuple(t.shape), tuple(self.key_code.table.shape))
            with torch.no_grad():
                self.key_code.table.copy_(t.to(self.dev))         # the stored rows (the fixed form's: its seeded draw, as trained)
            self.key_code.table.requires_grad_(False)
            kcm.patch_key(self.model, self.key_code)
        self.key_rot = None
        rp = os.path.join(folder, "key_rot.safetensors")
        assert os.path.exists(rp) == ("key_rot" in self.scale), "scale.json names the key rotation's phases exactly when their file is there"
        if os.path.exists(rp):                                    # 0.1.82: the key in the told head's rotation, before the look's capture
            kf = self.scale["key_rot"]
            assert kf.get("pairs") in krm.PAIR_FORMS and kf.get("k") == gr.KEY_MAX[self.key_form] and kf.get("vocab") == kcm.VOCAB, kf
            assert kf.get("seed_tag", krm.SEED_TAG) == krm.seed_tag(int(kf["k"]), int(kf["pairs"])), kf   # 0.1.83 / 0.1.87: the draw
            assert kf.get("norm", 0) in krm.NORMS, kf                                       # 0.1.88: 2 = the records' coded SEPs only
            assert self.anchor is not None, "the key rotation turns pairs of the anchored rotation"
            self.key_rot = krm.KeyRotary(int(kf["block"]), int(kf["head"]), k=int(kf["k"]), pairs=int(kf["pairs"]),   # 0.1.87: the
                                         norm=int(kf.get("norm", 0)), rho=float(kf.get("rho", krm.RHO)),           # binding's forms
                                         code=self.code).to(self.dev)                                               # 0.1.91: its code
            t = load_file(rp)["table"]
            assert tuple(t.shape) == tuple(self.key_rot.table.shape), (tuple(t.shape), tuple(self.key_rot.table.shape))
            with torch.no_grad():
                self.key_rot.table.copy_(t.to(self.dev))          # the stored phases (the seeded draw, as trained)
            krm.patch(self.model, self.key_rot)
        self.head = None
        hp = os.path.join(folder, "addr_head.safetensors")
        if os.path.exists(hp):
            self.head = nn.Linear(self.cfg.d_model, self.levels * self.radix).to(self.dev)
            self.head.load_state_dict({k: v.to(self.dev) for k, v in load_file(hp).items()})
            self.head.eval()
            for p in self.head.parameters():
                p.requires_grad_(False)
        self.digest = self.state_digest()
        self.d = self.cfg.d_model
        self.head_at = self.scale.get("head_at", "question")      # 0.1.72: where the head was trained (the stored cores: the questions)
        self.free = {}                                             # 0.1.72: the free-running rows read so far (free_rows)
        self.look, self.look_cap = None, None                     # 0.1.80: the look's head and its capture (after the anchored patch)
        if look == "ptr":
            p = self.scale.get("ptr")
            look = (int(p["block"]), int(p["head"])) if isinstance(p, dict) else None
        if look is not None:
            b, hh = int(look[0]), int(look[1])
            assert 1 <= b <= self.n_blocks, f"the look's block {b}: 1 .. {self.n_blocks} (numbered from 1)"
            attn = m.blocks[b - 1].attn
            assert hasattr(attn, "_rotate") and 0 <= hh < attn.h, f"block {b} head {hh} is not a softmax block's head"
            self.look = (b, hh)
            self.look_cap = pt.HeadCapture(attn, stop=False, head=hh)   # keeps the forward's own rotated q and k; the output unchanged
            self.look_cap.on = False                                       # on in pass 1 only (batch_views)

    def state_digest(self) -> str:
        """The frozen core's state (the model's state dict, and the value-path tables and the key code's tables when it carries them) as
        one hash."""
        return _digest(self.model, None if self.value_addr is None else self.value_addr.table,
                       None if self.key_code is None else self.key_code.table,
                       None if self.key_rot is None else self.key_rot.table)

    def set_streams(self, metas: list, request=None) -> None:
        """The anchored core's streams for these rows: request None = the true address at the questions, "null" = the null anchor, else a
        per-row list of per-question digit pairs. A house-rotary core reads no streams."""
        if self.anchor is None:
            return
        dk, dq = [], []
        for j, m in enumerate(metas):
            req = None if request is None else ([(self.radix,) * self.levels] * len(m["questions"]) if request == "null" else request[j])
            k_, q_ = gr.streams(m, gr.CTX, req)
            dk.append(k_)
            dq.append(q_)
        self.anchor.set(torch.from_numpy(np.stack(dk).astype(np.int64)).to(self.dev),
                        torch.from_numpy(np.stack(dq).astype(np.int64)).to(self.dev))

    def clear(self) -> None:
        if self.anchor is not None:
            self.anchor.clear()

    def trunk(self, x: torch.Tensor, lo: int = 0, hi: int | None = None, h: torch.Tensor | None = None) -> torch.Tensor:
        """The blocks [lo, hi) of the library's byte trunk (_trunk without lookup arms or fusion); h None = the embedding of x."""
        m = self.model
        if h is None:
            assert lo == 0
            h = m.embed(x)                                # 0.1.82: the patched embedding sets the key rotation's phases from x
        elif self.key_rot is not None:
            self.key_rot.set(x)                           # 0.1.82: a forward from a given state reads the phases of the same rows
        cls = m._byte_cls_table[x] if getattr(m, "byte_table", False) else None
        for b in m.blocks[lo:hi]:
            h = b(h, byte_cls=cls)
        return h

    def readout(self, h: torch.Tensor) -> torch.Tensor:
        """The craft's own readout: the final norm and the head (AlephLM.forward's tail)."""
        return self.model.head(self.model.nf(h))

    def head_scores(self, h_sep: torch.Tensor) -> torch.Tensor:
        """The address head on the final norm's state at the questions' SEPs (0.1.72: or at the index entries' SEPs) -> (Q, 2, 8) sigmoid
        scores (fp32)."""
        z = self.head(self.model.nf(h_sep).float())
        return torch.sigmoid(z.float().view(-1, self.levels, self.radix))


# ------------------------------------------------------------------ the rows
def load_packed(bed_dir: str) -> np.ndarray:
    """The bed's training stream: the corpus with DOC after every document (bed_train's packing)."""
    corpus = np.fromfile(os.path.join(bed_dir, "bed_corpus.bin"), dtype=np.uint8)
    offs = np.load(os.path.join(bed_dir, "bed_corpus_offsets.npy"))
    return np.insert(corpus, offs[1:], np.uint8(DOC))


def load_heldout(path: str, ctx: int = gr.CTX) -> list:
    """bed_train's held-out documents: every document of at least ctx + 64 bytes, in file order."""
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            b = json.loads(line)["text"].encode("utf-8")
            if len(b) >= ctx + 64:
                out.append(b)
    return out


def gauge_rows(seed: int, name: str, n: int, docs: list, abytes: int = 2, hard: bool = False, layout: str = "AG",
               key_form: str = "std", code: str = gr.CODE) -> list:
    """bed_train's gauge rows (ag_set): n grid rows from the generator [seed, crc("SCALE|<name>")], the filler the documents `docs` in
    turn behind a DOC byte. The members' read uses the cores' own read rows ("read", the held-out documents) and calibration rows
    ("calib", the held-out documents reversed). abytes (0.1.74): the core's address form. hard (0.1.75): key-hard rows (grid_rows.hard_keys
    drawn from the generator before each row, as bed_train's ag_rows; "hard" with the held-out documents = the trainer's hard-row read).
    layout and key_form (0.1.83): the core's grid layout and keys' form (scale.json; AG and std = 0.1.82's rows). code (0.1.91): the core's
    address code (scale.json's grid_code; 8x2 = the rows above)."""
    rng = np.random.default_rng([seed, gr.crc(f"SCALE|{name}")])
    out = []
    for j in range(n):
        keys = gr.hard_keys(rng, layout, key_form) if hard else None
        out.append(gr.compose_ag(rng, bytes([DOC]) + docs[j % len(docs)], gr.CTX, abytes=abytes, keys=keys, layout=layout, key_form=key_form,
                                 code=code))
    return out


def member_rows(rng, packed: np.ndarray, owned: list, never: list, n_rows: int = ROWS_LAST, abytes: int = 2, layout: str = "AG",
                key_form: str = "std") -> list:
    """A last-site batch: each row 18 owned compartments (a per-batch permutation of the owned cycled over the rows) and 6 never-owned,
    its questions asking the owned records in a drawn order. 0.1.83: at the layout's counts (MEMBER_ROW: AG48 36 owned + 12 never-owned)
    and its questions' room. 0.1.84: the asked are the row's first `ask` owned IN THE CYCLED ORDER (then drawn into the questions' order),
    so every member is asked equally often in a batch, as train_last's member mode asserts (AG48: 8 rows x 24 asked = 4 a member; 0.1.83
    asked a drawn 24 of the 36 and broke the equality); AG asks all 18 and its rows are 0.1.82's byte for byte. -> [(row, meta)]."""
    n_own, n_nev = MEMBER_ROW[layout]
    ask = gr.lay(layout)["ask"]
    perm = rng.permutation(len(owned))
    seq = np.concatenate([perm] * (n_rows * n_own // len(owned) + 2))
    out = []
    for r in range(n_rows):
        own = [owned[int(k)] for k in seq[r * n_own:(r + 1) * n_own]]
        nev = [int(a) for a in rng.choice(never, size=n_nev, replace=False)]
        addrs = sorted(own + nev)
        asked = [addrs.index(a) for a in own][:ask]                  # 0.1.84: the first `ask` in the cycled order (AG: all 18)
        asked = [asked[int(i)] for i in rng.permutation(len(asked))]
        p = int(rng.integers(0, len(packed) - gr.CTX - 2))
        out.append(gr.compose_ag(rng, packed[p:p + gr.CTX + 1].tobytes(), gr.CTX, addrs=addrs, asked=asked, abytes=abytes, layout=layout,
                                 key_form=key_form))
    return out


def owned_row(rng, src: bytes, addr: int, abytes: int = 2, layout: str = "AG", key_form: str = "std") -> tuple:
    """A mid-site row: an ordinary grid row that shows `addr`'s compartment, its 24 questions in a drawn order (one asks addr's record).
    0.1.83: at the layout (AG48: 47 other compartments; the questions ask addr's record and 23 of the others, in a drawn order); AG =
    0.1.82's row. -> (row, meta, the question index asking addr's record)."""
    L = gr.lay(layout)
    others = rng.choice([a for a in range(gr.N_ADDR) if a != addr], size=L["comps"] - 1, replace=False)
    addrs = [addr] + [int(a) for a in others]
    asked = None
    if L["ask"] < L["comps"]:                            # the member's record among the asked (AG asks every record: None, 0.1.82's draw)
        mine = sorted(addrs).index(addr)
        rest = rng.choice([i for i in range(L["comps"]) if i != mine], size=L["ask"] - 1, replace=False)
        asked = [int(i) for i in rng.permutation([mine] + [int(i) for i in rest])]
    row, meta = gr.compose_ag(rng, src, gr.CTX, addrs=addrs, asked=asked, abytes=abytes, layout=layout, key_form=key_form)
    qi = next(i for i, q in enumerate(meta["questions"]) if meta["recs"][q["rec"]]["addr"] == addr)
    return row, meta, qi


def to_x(rows: list, dev) -> torch.Tensor:
    return torch.from_numpy(np.stack([np.frombuffer(r, dtype=np.uint8) for r, _ in rows]).astype(np.int64)).to(dev)


def question_table(metas: list, owned: list) -> dict:
    """Per question of the rows (row order, then question order): its row, its answer positions (Q, 7), its address, its owner's index
    (-1 = never owned), its true digits, its index entry's address positions (Q, the row's address form: 2 digit bytes or 1 byte)."""
    rank = {a: i for i, a in enumerate(owned)}
    row, pos, addr, own, ipos = [], [], [], [], []
    for j, m in enumerate(metas):
        for q, ps, e in zip(m["questions"], gr.answer_positions(m), gr.index_positions(m)):
            a = m["recs"][q["rec"]]["addr"]
            row.append(j)
            pos.append(ps)
            addr.append(a)
            own.append(rank.get(a, -1))
            ipos.append(e)
    addr = np.array(addr)
    code = gr.code_of(metas[0]) if metas else gr.CODE              # 0.1.91: the rows' address code (one per read)
    assert all(gr.code_of(m) == code for m in metas), "one address code a read"
    dig = (np.stack([addr // gr.RADIX, addr % gr.RADIX], 1) if code == gr.CODE
           else np.array([gr.digits(int(a), code) for a in addr], dtype=np.int64).reshape(len(addr), gr.radix_levels(code)[1]))
    return dict(row=np.array(row), pos=np.array(pos), addr=addr, own=np.array(own), dig=dig, ipos=np.array(ipos))


# ------------------------------------------------------------------ the trainers
def _ckpt_save(path: str, step: int, stacks: dict, opts: dict, extra: dict) -> None:
    tmp = path + ".tmp"
    torch.save({"step": step, "stacks": {k: {n: t.detach().cpu() for n, t in s.state_dict().items()} for k, s in stacks.items()},
                "opts": {k: o.state() for k, o in opts.items()}, "extra": extra}, tmp)
    os.replace(tmp, path)


def _ckpt_load(path: str, stacks: dict, opts: dict) -> tuple:
    st = torch.load(path, map_location="cpu", weights_only=False)
    for k, s in stacks.items():
        s.load_state_dict(st["stacks"][k])
    for k, o in opts.items():
        o.load(st["opts"][k])
    return st["step"], st["extra"]


class _Prefetch:
    """The next step's rows compose in one worker thread while this step runs (the batches are identical either way: each step seeds
    its own generator)."""

    def __init__(self, fn, start: int, steps: int) -> None:
        self.fn, self.steps = fn, steps
        self.pool = __import__("concurrent.futures").futures.ThreadPoolExecutor(max_workers=1)
        self.nxt = self.pool.submit(fn, start) if start < steps else None

    def get(self, step: int):
        out = self.nxt.result()
        self.nxt = self.pool.submit(self.fn, step + 1) if step + 1 < self.steps else None
        return out


def train_last(core: Core, stacks: dict, steps: int, seed: int, packed: np.ndarray, owned: list, never: list, log=print,
               ckpt: str | None = None, ckpt_every: int = 2000, trace_every: int = 2000) -> dict:
    """The last-site paired trainer: every stack in `stacks` (name -> StackedArms / StackedToys over the same N_OWN owners) trains on the
    same member-mode rows, each member alone (its own mean CE on its own answer positions, its own Adam). -> the training record."""
    assert getattr(core, "code", gr.CODE) == gr.CODE, "the members' rows and gates are written for the 8x2 address code"
    dev = core.dev
    codes = code_tables(N_OWN).to(dev)
    opts = {k: StackedAdam(list(s.parameters()), s.M) for k, s in stacks.items()}
    q_step = ROWS_LAST * min(MEMBER_ROW[core.layout][0], gr.lay(core.layout)["ask"]) // N_OWN     # 0.1.83: the layout's asked owned
    start, rec = 0, dict(site="last", steps=steps, rows_per_step=ROWS_LAST, questions_per_member_step=q_step,
                         curve={k: [] for k in stacks}, trace=[], ms_per_step=None, absent_member_steps=0)
    if ckpt and os.path.exists(ckpt):
        start, rec = _ckpt_load(ckpt, stacks, opts)
        log(f"[members last] resumed at step {start} from {os.path.basename(ckpt)}")
    salt = gr.crc("SCALE|memb|last")

    def make(step):
        rows = member_rows(np.random.default_rng([seed, salt, step]), packed, owned, never, abytes=core.abytes, layout=core.layout, key_form=core.key_form)
        return rows, question_table([m for _, m in rows], owned)
    pf = _Prefetch(make, start, steps)
    present = torch.ones(N_OWN, dtype=torch.bool, device=dev)
    t0, tl = time.time(), time.time()
    for step in range(start, steps):
        rows, qt = pf.get(step)
        x = to_x(rows, dev)
        lr = lr_at(step, steps)
        with torch.no_grad():
            core.set_streams([m for _, m in rows])                     # the training form: the true address at the questions
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                h = core.trunk(x[:, :-1])
            core.clear()
            own = qt["own"]
            sel = np.nonzero(own >= 0)[0]
            order = sel[np.argsort(own[sel], kind="stable")]
            per = len(order) // N_OWN
            assert per * N_OWN == len(order) and (np.bincount(own[order], minlength=N_OWN) == per).all(), "member mode: equal questions a member"
            r_ = torch.from_numpy(np.repeat(qt["row"][order], NA)).to(dev)
            p_ = torch.from_numpy(qt["pos"][order].reshape(-1)).to(dev)
            hg = h[r_, p_].float().view(N_OWN, per * NA, core.d)
            y = x[:, 1:][r_, p_].view(N_OWN, per * NA)
            tgt = torch.gather(codes, 1, y)                                # each member's code of the true bytes (the END unchanged)
        loss_total = 0.0
        for k, s in stacks.items():
            w = s.write(hg)
            lg = core.readout(hg + w)
            ce = F.cross_entropy(lg.reshape(-1, lg.shape[-1]).float(), tgt.reshape(-1), reduction="none").view(N_OWN, -1).mean(1)
            for p in s.parameters():
                p.grad = None
            ce.sum().backward()
            opts[k].step(present, lr)
            if (step + 1) % 100 == 0 or step == start:
                rec["curve"][k].append([step + 1, float(ce.detach().mean())])
            loss_total += float(ce.detach().mean())
        if step == start + 199:
            rec["ms_per_step"] = (time.time() - t0) * 1000 / 200
            log(f"[members last] {rec['ms_per_step']:.1f} ms a step over the first 200 steps -> the {steps:,} steps about "
                f"{rec['ms_per_step'] * (steps - start) / 60000:.1f} min")
        if (step + 1) % 500 == 0 or step + 1 == steps:
            el = time.time() - t0
            log(f"[members last] {step + 1:,}/{steps:,}: mean CE " + " / ".join(f"{k} {rec['curve'][k][-1][1]:.4f}" for k in stacks)
                + f"; {el / 60:.1f} min spent, ~{el * (steps - step - 1) / max(1, step + 1 - start) / 60:.1f} min left")
        if trace_every and ((step + 1) % trace_every == 0 or step + 1 == steps):
            rec["trace"].append(dict(step=step + 1, **{k: _solo_batch(core, s, owned, never, packed, seed, codes) for k, s in stacks.items()}))
            log(f"[members last] trace at {step + 1:,}: solo on one batch " + " / ".join(f"{k} {rec['trace'][-1][k]:.4f}" for k in stacks))
        if ckpt and ((step + 1) % ckpt_every == 0) and step + 1 < steps:
            _ckpt_save(ckpt, step + 1, stacks, opts, rec)
    rec["wall_s"] = time.time() - t0
    assert core.state_digest() == core.digest, "the core moved while the members trained"
    return rec


@torch.no_grad()
def _solo_batch(core: Core, stk, owned: list, never: list, packed: np.ndarray, seed: int, codes: torch.Tensor) -> float:
    """The trace's solo read: one member-mode batch from its own generator, each owner alone at weight 1 (whole coded answers)."""
    dev = core.dev
    rows = member_rows(np.random.default_rng([seed, gr.crc("SCALE|memb|trace")]), packed, owned, never, abytes=core.abytes, layout=core.layout, key_form=core.key_form)
    qt = question_table([m for _, m in rows], owned)
    x = to_x(rows, dev)
    core.set_streams([m for _, m in rows])
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
        h = core.trunk(x[:, :-1])
    core.clear()
    sel = np.nonzero(qt["own"] >= 0)[0]
    r_ = torch.from_numpy(np.repeat(qt["row"][sel], NA)).to(dev)
    p_ = torch.from_numpy(qt["pos"][sel].reshape(-1)).to(dev)
    o = torch.from_numpy(qt["own"][sel]).to(dev)
    hq = h[r_, p_].float().view(len(sel), NA, core.d)
    y = x[:, 1:][r_, p_].view(len(sel), NA)
    tgt = codes[o[:, None], y]
    w = stk.write(hq, o)
    ok = (core.readout(hq + w).argmax(-1) == tgt).all(-1)
    return float(ok.double().mean())


def train_mid(core: Core, arms: StackedArms, steps: int, seed: int, packed: np.ndarray, owned: list, log=print,
              ckpt: str | None = None, ckpt_every: int = 4000) -> dict:
    """The mid-site row-owned trainer: `arms` over the first M_MID owners after block MID_SITE; each batch of ROWS_MID rows holds one row a
    member (half the members a step, every member once in each pair of steps), only that member writes, at its question's answer
    positions; blocks MID_SITE.. recomputed in full; each member's own mean CE at those positions, its own Adam. steps = batch steps."""
    assert getattr(core, "code", gr.CODE) == gr.CODE, "the members' rows and gates are written for the 8x2 address code"
    dev = core.dev
    M = arms.M
    assert M == 2 * ROWS_MID, (M, ROWS_MID)
    codes = code_tables(N_OWN)[:M].to(dev)
    opt = StackedAdam(list(arms.parameters()), M)
    start, rec = 0, dict(site="mid", block=MID_SITE, batch_steps=steps, own_steps_per_member=steps // 2, rows_per_step=ROWS_MID,
                         curve=[], ms_per_step=None)
    if ckpt and os.path.exists(ckpt):
        start, rec = _ckpt_load(ckpt, {"arms": arms}, {"arms": opt})
        log(f"[members mid] resumed at step {start} from {os.path.basename(ckpt)}")
    salt, osalt = gr.crc("SCALE|memb|mid"), gr.crc("SCALE|memb|mid|order")

    def make(step):
        order = np.random.default_rng([seed, osalt, step // 2]).permutation(M)
        mem = order[:ROWS_MID] if step % 2 == 0 else order[ROWS_MID:]
        rng = np.random.default_rng([seed, salt, step])
        rows, qis = [], []
        for j in mem:
            p = int(rng.integers(0, len(packed) - gr.CTX - 2))
            row, meta, qi = owned_row(rng, packed[p:p + gr.CTX + 1].tobytes(), owned[int(j)], abytes=core.abytes, layout=core.layout, key_form=core.key_form)
            rows.append((row, meta))
            qis.append(qi)
        return rows, np.array(mem), qis
    pf = _Prefetch(make, start, steps)
    t0 = time.time()
    for step in range(start, steps):
        rows, mem, qis = pf.get(step)
        x = to_x(rows, dev)
        metas = [m for _, m in rows]
        lr = lr_at(step, steps)
        pos = torch.tensor([gr.answer_positions(m)[qi] for m, qi in zip(metas, qis)], device=dev)        # (B, 7)
        rr = torch.arange(len(rows), device=dev)[:, None].expand_as(pos)
        midx = torch.from_numpy(mem).to(dev)
        core.set_streams(metas)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            h3 = core.trunk(x[:, :-1], hi=MID_SITE)
        hq = h3[rr, pos].float()
        w = arms.write(hq, midx)                                           # (B, 7, d): each row's own member
        h3w = h3.float().index_put((rr, pos), hq + w)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            h6 = core.trunk(x[:, :-1], lo=MID_SITE, h=h3w)
        core.clear()
        lg = core.readout(h6[rr, pos].float())
        y = x[:, 1:][rr, pos]
        tgt = codes[midx[:, None], y]
        ce = F.cross_entropy(lg.reshape(-1, lg.shape[-1]).float(), tgt.reshape(-1), reduction="none").view(len(rows), -1).mean(1)
        for p in arms.parameters():
            p.grad = None
        ce.sum().backward()
        present = torch.zeros(M, dtype=torch.bool, device=dev)
        present[midx] = True
        opt.step(present, lr)
        if (step + 1) % 100 == 0 or step == start:
            rec["curve"].append([step + 1, float(ce.detach().mean())])
        if step == start + 199:
            rec["ms_per_step"] = (time.time() - t0) * 1000 / 200
            log(f"[members mid] {rec['ms_per_step']:.1f} ms a step over the first 200 steps -> the {steps:,} batch steps about "
                f"{rec['ms_per_step'] * (steps - start) / 60000:.1f} min")
        if (step + 1) % 1000 == 0 or step + 1 == steps:
            el = time.time() - t0
            log(f"[members mid] {step + 1:,}/{steps:,}: mean CE {rec['curve'][-1][1]:.4f}; {el / 60:.1f} min spent, "
                f"~{el * (steps - step - 1) / max(1, step + 1 - start) / 60:.1f} min left")
        if ckpt and ((step + 1) % ckpt_every == 0) and step + 1 < steps:
            _ckpt_save(ckpt, step + 1, {"arms": arms}, {"arms": opt}, rec)
    rec["wall_s"] = time.time() - t0
    assert core.state_digest() == core.digest, "the core moved while the members trained"
    return rec


# ------------------------------------------------------------------ the read
def _swap_requests(dig: np.ndarray, rows: np.ndarray, rng) -> np.ndarray:
    """INDEXC: the true digits with one pair of each row's questions trading addresses."""
    out = dig.copy()
    for j in np.unique(rows):
        qi = np.nonzero(rows == j)[0]
        a, b = rng.choice(qi, size=2, replace=False)
        out[[a, b]] = out[[b, a]]
    return out


@torch.no_grad()
def batch_context(core: Core, rows: list, owned: list, swap_rng, site: int | None = None) -> dict:
    """One read batch on the bare core (fp32): the written index, the states at the answer positions (the final ones, and after block
    `site` when given), the bare logits there, the head's scores, every request source's digits."""
    dev = core.dev
    metas = [m for _, m in rows]
    qt = question_table(metas, owned)
    x = to_x(rows, dev)
    xin = x[:, :-1]
    ip = torch.from_numpy(qt["ipos"]).to(dev)
    rq = torch.from_numpy(qt["row"]).to(dev)
    core.set_streams(metas, "null" if core.anchor is not None else None)  # the index precedes the questions: its writing reads no request
    h = core.trunk(xin)
    pred = torch.stack([core.readout(h[rq, ip[:, i]]).argmax(-1) for i in range(ip.shape[1])], 1)  # the entry's address bytes (its form)
    written = gr.decode_written(pred.cpu().numpy(), ip.shape[1])
    if core.anchor is not None:                                            # the questions carry the written index
        req = []
        for j in range(len(rows)):
            req.append([tuple(int(d) for d in written[k]) for k in np.nonzero(qt["row"] == j)[0]])
        core.set_streams(metas, req)
        h = core.trunk(xin) if site is None else None
    h_site = None
    if site is not None:
        h_site = core.trunk(xin, hi=site)
        h = core.trunk(xin, lo=site, h=h_site)
    core.clear()
    r_ = torch.from_numpy(np.repeat(qt["row"], NA)).to(dev)
    p_ = torch.from_numpy(qt["pos"].reshape(-1)).to(dev)
    Q = len(qt["row"])
    out = dict(qt=qt, x=x, metas=metas, written=written, rq=rq, r_=r_, p_=p_,
               hq=h[r_, p_].view(Q, NA, core.d), raw=x[:, 1:][r_, p_].view(Q, NA),
               index_right=float((pred == x[:, 1:][rq[:, None], ip]).all(-1).double().mean()),
               swapped=_swap_requests(qt["dig"], qt["row"], swap_rng))
    if site is not None:
        out["h_site"], out["h_site_q"] = h_site, h_site[r_, p_].view(Q, NA, core.d)
    out["bare_logits"] = core.readout(out["hq"])
    if core.head is not None:
        out["s"] = core.head_scores(out["hq"][:, 0])                       # the head at each question's SEP
    return out


@torch.no_grad()
def calibrate(core: Core, rows: list, owned: list, mb: int = ROWS_LAST) -> dict:
    """The floor's tau per level: the quantiles of the per-question largest wrong digit score on the calibration rows (never the read's),
    read as the read reads (fp32; the anchored core's questions at the written index)."""
    if core.head is None:
        return {}
    lw = []
    for i in range(0, len(rows), mb):
        ctx = batch_context(core, rows[i:i + mb], owned, np.random.default_rng(0))
        dig = torch.from_numpy(ctx["qt"]["dig"]).to(core.dev)
        lw.append(ctx["s"].scatter(-1, dig[..., None], -1.0).amax(-1).cpu())
    lw = torch.cat(lw).double()
    tau = {str(q): [float(torch.quantile(lw[:, l], q)) for l in range(gr.LEVELS)] for q in QS}
    return dict(rows=len(rows), questions=int(lw.shape[0]), tau=tau,
                largest_wrong_at_1=[float((lw[:, l] >= 1.0).double().mean()) for l in range(gr.LEVELS)],
                tau_999_saturated=any(t >= 1.0 for t in tau[str(0.999)]))


def _cells(core: Core):
    out = []
    srcs = [s for s in SOURCES if core.head is not None or not s.startswith("HEAD")]
    for src in srcs:
        for U in U_SET:
            for Z in Z_SET:
                for m in M_SET:
                    out.append((f"{src}_U{int(U)}_{Z}_m{m}", src, U, Z, m))
    for m in M_SET:
        out.append((f"G2_m{m}", "G2", 0.0, "-", m))
        out.append((f"G0_m{m}", "G0", 0.0, "-", m))
    return out, srcs


def _factors(ctx: dict, src: str, mem: torch.Tensor, L9: torch.Tensor, tau999) -> list:
    dev = mem.device
    qt = ctx["qt"]
    if src == "LAYOUT":
        return exact_factors(torch.from_numpy(qt["dig"]).to(dev), mem, L9)
    if src == "INDEX":
        return exact_factors(torch.from_numpy(ctx["written"]).to(dev), mem, L9)
    if src == "INDEXC":
        return exact_factors(torch.from_numpy(ctx["swapped"]).to(dev), mem, L9)
    if src == "HEADP":
        return head_factors(ctx["s"], mem)
    if src == "HEADPQ999":
        return head_factors(ctx["s"], mem, tau999)
    raise ValueError(src)


def _gate(kind: str, U: float, Z: str, m: int, fac: dict, qt: dict, owned_t: torch.Tensor, dev) -> torch.Tensor:
    if kind == "G0":
        return torch.ones(len(qt["row"]), m, device=dev)
    if kind == "G2":
        return (torch.from_numpy(qt["addr"]).to(dev)[:, None] == owned_t[None, :m]).float()
    return norm_w(u_of(fac[kind], U, m), Z)


def _margin(logits: torch.Tensor, lab: torch.Tensor) -> torch.Tensor:
    top = logits.gather(-1, lab[..., None])[..., 0]
    return top - logits.scatter(-1, lab[..., None], float("-inf")).amax(-1)


class _Acc:
    """Per-address sums (64 addresses) and scalar sums."""

    def __init__(self, dev) -> None:
        self.dev, self.v, self.s = dev, {}, {}

    def add(self, key: str, addr: torch.Tensor, val: torch.Tensor) -> None:
        if key not in self.v:
            self.v[key] = torch.zeros(gr.N_ADDR, dtype=torch.float64, device=self.dev)
        self.v[key] += torch.bincount(addr, weights=val.double(), minlength=gr.N_ADDR)

    def sc(self, key: str, val: float) -> None:
        self.s[key] = self.s.get(key, 0.0) + float(val)

    def mx(self, key: str, val: float, fn=max) -> None:
        self.s[key] = fn(self.s[key], float(val)) if key in self.s else float(val)


def _mean_over(acc: _Acc, key: str, cnt: torch.Tensor, addrs: list) -> float:
    a = torch.tensor(addrs, device=acc.dev)
    return float((acc.v[key][a] / cnt[a].clamp(min=1)).mean()) if key in acc.v else 0.0


@torch.no_grad()
def read_last(core: Core, stacks: dict, rows: list, cal: dict, owned: list, never: list, seed: int, mb: int = ROWS_LAST, log=print) -> dict:
    """The last-site read on the read rows: every cell of every stack, the solo reads, the owner-weight scan, the request sources'
    diagnostics. stacks: name -> a stack over the N_OWN owners. -> the record."""
    assert getattr(core, "code", gr.CODE) == gr.CODE, "the members' rows and gates are written for the 8x2 address code"
    dev = core.dev
    L9 = level_code(dev)
    owned_t = torch.tensor(owned, device=dev)
    mem = torch.stack([owned_t // gr.RADIX, owned_t % gr.RADIX], 1)
    codes = code_tables(N_OWN).to(dev)
    tau999 = (cal.get("tau") or {}).get(str(0.999))
    cells, srcs = _cells(core)
    swap_rng = np.random.default_rng([seed, gr.crc("SCALE|swap")])
    cnt = torch.zeros(gr.N_ADDR, dtype=torch.float64, device=dev)
    A = {k: _Acc(dev) for k in stacks}
    B = _Acc(dev)
    diag = dict(index_right=0.0, batches=0, questions=0, req_right={s: 0 for s in srcs}, owner_u={s: 0.0 for s in srcs},
                offband_u_max={s: 0.0 for s in srcs}, floor_owner_zero=0, floor_wrong_pass=0, n_owned_q=0)
    t0 = time.time()
    for bi in range(0, len(rows), mb):
        ctx = batch_context(core, rows[bi:bi + mb], owned, swap_rng)
        qt = ctx["qt"]
        Q = len(qt["row"])
        addr = torch.from_numpy(qt["addr"]).to(dev)
        o = torch.from_numpy(qt["own"]).to(dev)
        nv = o < 0
        qo = (o >= 0).nonzero().flatten()
        cnt += torch.bincount(addr, minlength=gr.N_ADDR).double()
        hq, raw, lb = ctx["hq"], ctx["raw"], ctx["bare_logits"]
        lpb = F.log_softmax(lb, -1)
        pb = lb.argmax(-1)
        B.add("bare_raw", addr, (pb == raw).all(-1))
        mb_ = _margin(lb[nv], raw[nv])
        B.sc("bare_margin_sum", mb_.sum())
        B.sc("bare_n_pos", mb_.numel())
        B.mx("bare_margin_min", mb_.min(), min)
        fac = {s: _factors(ctx, s, mem, L9, tau999) for s in srcs}
        dig = torch.from_numpy(qt["dig"]).to(dev)
        diag["index_right"] += ctx["index_right"]
        diag["batches"] += 1
        diag["questions"] += Q
        diag["n_owned_q"] += int(qo.numel())
        for s in srcs:
            if s in ("LAYOUT", "INDEX", "INDEXC"):
                r = {"LAYOUT": qt["dig"], "INDEX": ctx["written"], "INDEXC": ctx["swapped"]}[s]
                diag["req_right"][s] += int((r == qt["dig"]).all(-1).sum())
            else:
                diag["req_right"][s] += int((ctx["s"].argmax(-1) == dig).all(-1).sum())
            pr = fac[s][0] * fac[s][1]
            diag["owner_u"][s] += float(pr[qo, o[qo]].sum())
            off = pr.clone()
            off[qo, o[qo]] = float("-inf")
            diag["offband_u_max"][s] += float(off.amax(-1).clamp(min=0).sum())
            if s == "HEADPQ999":
                diag["floor_owner_zero"] += int((pr[qo, o[qo]] == 0).sum())
                diag["floor_wrong_pass"] += int((off > 0).any(-1).sum())
        for name, stk in stacks.items():
            acc = A[name]
            W = stk.write_all(hq.reshape(-1, core.d)).view(Q, NA, stk.M, core.d)
            Wo = W[qo, :, o[qo]]                                            # (Qo, 7, d) each owner's own write
            tgt = codes[o[qo][:, None], raw[qo]]
            # solo: the owner alone at weight 1 ("ONE") and at each source's solo weight tanh(U prod)
            hit = (core.readout(hq[qo] + Wo).argmax(-1) == tgt).all(-1)
            acc.add("solo_ONE", addr[qo], hit)
            for s in srcs:
                pr = (fac[s][0] * fac[s][1])[qo, o[qo]]
                for U in U_SET:
                    ws = torch.tanh(U * pr)
                    acc.add(f"solo_{s}_U{int(U)}", addr[qo], (core.readout(hq[qo] + ws[:, None, None] * Wo).argmax(-1) == tgt).all(-1))
            for a in A_GRID:
                acc.add(f"scan_{a}", addr[qo], (core.readout(hq[qo] + a * Wo).argmax(-1) == tgt).all(-1))
            sw = Wo.norm(dim=-1)
            for key, kind, U, Z, m in cells:
                w = _gate(kind, U, Z, m, fac, qt, owned_t, dev)
                dh = torch.einsum("qm,qpmd->qpd", w, W[:, :, :m])
                la = core.readout(hq + dh)
                pa = la.argmax(-1)
                inm = (o >= 0) & (o < m)
                qi = inm.nonzero().flatten()
                acc.add(f"{key}|hit_own", addr[qi], (pa[qi] == codes[o[qi][:, None], raw[qi]]).all(-1))
                acc.add(f"{key}|hit_raw", addr[nv], (pa[nv] == raw[nv]).all(-1))
                du, lau = dh[nv], la[nv]
                acc.sc(f"{key}|kl", (lpb[nv].exp() * (lpb[nv] - F.log_softmax(lau, -1))).sum())
                acc.sc(f"{key}|leak", du.norm(dim=-1).sum())
                mu = _margin(lau, raw[nv])
                acc.sc(f"{key}|msum", mu.sum())
                acc.mx(f"{key}|mmin", mu.min(), min)
                for sc in FLIPS:
                    ps = core.readout(hq[nv] + sc * du).argmax(-1)
                    acc.sc(f"{key}|flip{sc}", (ps != raw[nv]).sum())
                    acc.sc(f"{key}|flipb{sc}", (ps != pb[nv]).sum())
                sel = inm[qo]
                acc.sc(f"{key}|own_write", dh[qi].norm(dim=-1).sum())
                acc.sc(f"{key}|solo_write", sw[sel].sum())
                acc.sc(f"{key}|n_own_pos", int(qi.numel()) * NA)
                acc.sc(f"{key}|ownw", w[qi, o[qi]].sum())
                acc.sc(f"{key}|n_own", qi.numel())
                offm = torch.ones_like(w, dtype=torch.bool)
                offm[qi, o[qi]] = False
                acc.sc(f"{key}|off", w.abs()[offm].sum())
                acc.sc(f"{key}|n_q", Q)
        if ((bi // mb) + 1) % 8 == 0 or bi + mb >= len(rows):
            el = time.time() - t0
            done = min(len(rows), bi + mb)
            log(f"[read last] {done}/{len(rows)} rows: {el:.0f}s spent, ~{el * (len(rows) - done) / done:.0f}s left")
    nu = max(1.0, B.s["bare_n_pos"])
    res = dict(rows=len(rows), questions=diag["questions"], questions_per_address_min=int(cnt.min()),
               index_right=diag["index_right"] / max(1, diag["batches"]),
               request_right={s: diag["req_right"][s] / max(1, diag["questions"]) for s in srcs},
               owner_u_over_U={s: diag["owner_u"][s] / max(1, diag["n_owned_q"]) for s in srcs},
               largest_offband_u_over_U_mean={s: diag["offband_u_max"][s] / max(1, diag["questions"]) for s in srcs},
               floor=dict(owners_floored_to_0=diag["floor_owner_zero"] / max(1, diag["n_owned_q"]),
                          questions_passing_a_wrong_member=diag["floor_wrong_pass"] / max(1, diag["questions"])),
               calibration=cal, bare=dict(unowned=_mean_over(B, "bare_raw", cnt, never), owned=_mean_over(B, "bare_raw", cnt, owned),
                                          owned_per_owner=[float(B.v["bare_raw"][a] / max(1.0, float(cnt[a]))) for a in owned],
                                          margin_unowned_mean=B.s["bare_margin_sum"] / nu, margin_unowned_min=B.s["bare_margin_min"]),
               stacks={}, wall_s=time.time() - t0)
    for name, acc in A.items():
        r = dict(solo={}, a_scan={}, cells={})
        solo_keys = ["ONE"] + [f"{s}_U{int(U)}" for s in srcs for U in U_SET]
        for k in solo_keys:
            r["solo"][k] = {str(m): _mean_over(acc, f"solo_{k}", cnt, owned[:m]) for m in M_SET}
        r["solo_per_owner"] = [float(acc.v["solo_ONE"][a] / max(1.0, float(cnt[a]))) for a in owned]
        for a in A_GRID:
            r["a_scan"][repr(a)] = {str(m): _mean_over(acc, f"scan_{a}", cnt, owned[:m]) for m in M_SET}
        for key, kind, U, Z, m in cells:
            sk = "ONE" if kind in ("G0", "G2") else f"{kind}_U{int(U)}"
            s = acc.s
            n_own_pos = max(1.0, s[f"{key}|n_own_pos"])
            sw_ = s[f"{key}|solo_write"] / n_own_pos
            c = dict(m=m, solo_key=sk, solo_mean=r["solo"][sk][str(m)], merged_mean=_mean_over(acc, f"{key}|hit_own", cnt, owned[:m]),
                     unowned_raw_merged=_mean_over(acc, f"{key}|hit_raw", cnt, never), unowned_raw_bare=res["bare"]["unowned"],
                     kl_unowned=s[f"{key}|kl"] / nu, leak_write_unowned=s[f"{key}|leak"] / nu, solo_write_owned=sw_,
                     leak_over_own_write=(s[f"{key}|leak"] / nu) / max(1e-12, sw_),
                     margin_unowned_mean=s[f"{key}|msum"] / nu, margin_unowned_min=s[f"{key}|mmin"],
                     flip_rate_unowned={str(sc): s[f"{key}|flip{sc}"] / nu for sc in FLIPS},
                     flip_vs_bare_unowned={str(sc): s[f"{key}|flipb{sc}"] / nu for sc in FLIPS},
                     owned_write_over_solo_write=s[f"{key}|own_write"] / max(1e-12, s[f"{key}|solo_write"]),
                     own_gate_weight_mean=s[f"{key}|ownw"] / max(1.0, s[f"{key}|n_own"]),
                     offband_gate_mass_per_question=s[f"{key}|off"] / max(1.0, s[f"{key}|n_q"]))
            c["PROTECTED"] = c["merged_mean"] >= c["solo_mean"] - 0.05
            c["EXACT"] = abs(c["merged_mean"] - c["solo_mean"]) <= 0.01 and c["kl_unowned"] <= 0.001
            c["QUIET"] = c["unowned_raw_merged"] >= c["unowned_raw_bare"] - 0.02 and c["kl_unowned"] <= 0.05
            r["cells"][key] = c
        res["stacks"][name] = r
    res["nan_fields"] = _nan_census(res)
    return res


def _nan_census(rec, path: str = "") -> list:
    """Every non-finite number in a record (the read collects, never raises mid-run): the paths, for the log and the grading."""
    out = []
    if isinstance(rec, dict):
        for k, v in rec.items():
            out += _nan_census(v, f"{path}/{k}")
    elif isinstance(rec, (list, tuple)):
        for i, v in enumerate(rec):
            out += _nan_census(v, f"{path}[{i}]")
    elif isinstance(rec, float) and not math.isfinite(rec):
        out.append(path)
    return out


@torch.no_grad()
def read_mid(core: Core, arms: StackedArms, rows: list, cal: dict, owned: list, never: list, seed: int, mb: int = ROWS_MID,
             log=print, quick: bool = False) -> dict:
    """The mid-site read (m = M_MID) on the read rows: every source x U x Z cell and G2 / G0 (the merged writes after block MID_SITE at
    every question's answer positions, blocks MID_SITE.. recomputed), and the solo reads with each owned question's owner ALONE in its
    row (one forward a question, chunked). quick: U 5 only (the CPU shape check). -> the record."""
    assert getattr(core, "code", gr.CODE) == gr.CODE, "the members' rows and gates are written for the 8x2 address code"
    dev = core.dev
    L9 = level_code(dev)
    m = arms.M
    owned_m = owned[:m]
    owned_t = torch.tensor(owned, device=dev)
    mem = torch.stack([owned_t[:m] // gr.RADIX, owned_t[:m] % gr.RADIX], 1)
    codes = code_tables(m).to(dev)
    tau999 = (cal.get("tau") or {}).get(str(0.999))
    srcs = [s for s in SOURCES if core.head is not None or not s.startswith("HEAD")]
    us = U_SET[:1] if quick else U_SET
    cells = [(f"{s}_U{int(U)}_{Z}_m{m}", s, U, Z) for s in srcs for U in us for Z in Z_SET]
    cells += [(f"G2_m{m}", "G2", 0.0, "-"), (f"G0_m{m}", "G0", 0.0, "-")]
    solo_keys = ["ONE"] + [f"{s}_U{int(U)}" for s in srcs for U in us]
    swap_rng = np.random.default_rng([seed, gr.crc("SCALE|swap")])
    cnt = torch.zeros(gr.N_ADDR, dtype=torch.float64, device=dev)
    acc, B = _Acc(dev), _Acc(dev)
    t0 = time.time()
    for bi in range(0, len(rows), mb):
        ctx = batch_context(core, rows[bi:bi + mb], owned, swap_rng, site=MID_SITE)
        qt = ctx["qt"]
        Q = len(qt["row"])
        addr = torch.from_numpy(qt["addr"]).to(dev)
        o_all = torch.from_numpy(qt["own"]).to(dev)
        o = torch.where(o_all < m, o_all, torch.full_like(o_all, -2))       # -2 = owned beyond the mid set (neither merged nor unowned)
        nv = o_all < 0
        qo = (o >= 0).nonzero().flatten()
        cnt += torch.bincount(addr, minlength=gr.N_ADDR).double()
        raw, lb = ctx["raw"], ctx["bare_logits"]
        lpb = F.log_softmax(lb, -1)
        pb = lb.argmax(-1)
        B.add("bare_raw", addr, (pb == raw).all(-1))
        B.sc("n_pos", int(nv.sum()) * NA)
        fac = {s: _factors(ctx, s, mem, L9, tau999) for s in srcs}
        h3q = ctx["h_site_q"]
        W = arms.write_all(h3q.reshape(-1, core.d)).view(Q, NA, m, core.d)
        streams_req = None
        if core.anchor is not None:
            streams_req = [[tuple(int(d) for d in ctx["written"][k]) for k in np.nonzero(qt["row"] == j)[0]] for j in range(len(ctx["metas"]))]

        def suffix(h3w):
            core.set_streams(ctx["metas"], streams_req)
            h6 = core.trunk(ctx["x"][:, :-1], lo=MID_SITE, h=h3w)
            core.clear()
            return core.readout(h6[ctx["r_"], ctx["p_"]].view(Q, NA, core.d))
        for key, kind, U, Z in cells:
            w = _gate(kind, U, Z, m, fac, qt, owned_t, dev)
            dh = torch.einsum("qm,qpmd->qpd", w, W)
            h3w = ctx["h_site"].index_put((ctx["r_"], ctx["p_"]), (h3q + dh).reshape(-1, core.d))
            la = suffix(h3w)
            pa = la.argmax(-1)
            acc.add(f"{key}|hit_own", addr[qo], (pa[qo] == codes[o[qo][:, None], raw[qo]]).all(-1))
            acc.add(f"{key}|hit_raw", addr[nv], (pa[nv] == raw[nv]).all(-1))
            acc.sc(f"{key}|kl", (lpb[nv].exp() * (lpb[nv] - F.log_softmax(la[nv], -1))).sum())
            acc.sc(f"{key}|leak", dh[nv].norm(dim=-1).sum())
            acc.sc(f"{key}|ownw", w[qo, o[qo]].sum())
            acc.sc(f"{key}|n_own", qo.numel())
        # the solo reads: each owned question's owner alone in its row, at each solo weight
        prs = {s: (fac[s][0] * fac[s][1])[qo, o[qo]] for s in srcs}
        for sk in solo_keys:
            if sk == "ONE":
                ws = torch.ones(qo.numel(), device=dev)
            else:
                s, U = sk.rsplit("_U", 1)
                ws = torch.tanh(float(U) * prs[s])
            hit = torch.zeros(qo.numel(), dtype=torch.bool, device=dev)
            live = (ws != 0).nonzero().flatten()
            for c0 in range(0, live.numel(), mb):
                part = live[c0:c0 + mb]
                qq = qo[part]
                rws = torch.from_numpy(qt["row"]).to(dev)[qq]
                hs = ctx["h_site"][rws].clone()
                pos = torch.from_numpy(qt["pos"]).to(dev)[qq]                  # (b, 7)
                rr = torch.arange(len(qq), device=dev)[:, None].expand_as(pos)
                hs[rr, pos] = h3q[qq] + ws[part][:, None, None] * W[qq, :, o[qq]]
                metas = [ctx["metas"][int(j)] for j in rws]
                req = None if streams_req is None else [streams_req[int(j)] for j in rws]
                core.set_streams(metas, req)
                h6 = core.trunk(ctx["x"][rws, :-1], lo=MID_SITE, h=hs)
                core.clear()
                la = core.readout(h6[rr, pos])
                hit[part] = (la.argmax(-1) == codes[o[qq][:, None], raw[qq]]).all(-1)
            acc.add(f"solo_{sk}", addr[qo], hit)
        if ((bi // mb) + 1) % 8 == 0 or bi + mb >= len(rows):
            el = time.time() - t0
            done = min(len(rows), bi + mb)
            log(f"[read mid] {done}/{len(rows)} rows: {el:.0f}s spent, ~{el * (len(rows) - done) / done:.0f}s left")
    nu = max(1.0, B.s["n_pos"])
    bare_unowned = _mean_over(B, "bare_raw", cnt, never)
    res = dict(rows=len(rows), m=m, block=MID_SITE, bare=dict(unowned=bare_unowned, owned=_mean_over(B, "bare_raw", cnt, owned_m)),
               solo={k: _mean_over(acc, f"solo_{k}", cnt, owned_m) for k in solo_keys},
               solo_per_owner=[float(acc.v["solo_ONE"][a] / max(1.0, float(cnt[a]))) for a in owned_m], cells={}, wall_s=None)
    for key, kind, U, Z in cells:
        sk = "ONE" if kind in ("G0", "G2") else f"{kind}_U{int(U)}"
        s = acc.s
        c = dict(solo_key=sk, solo_mean=res["solo"][sk], merged_mean=_mean_over(acc, f"{key}|hit_own", cnt, owned_m),
                 unowned_raw_merged=_mean_over(acc, f"{key}|hit_raw", cnt, never), unowned_raw_bare=bare_unowned,
                 kl_unowned=s[f"{key}|kl"] / nu, leak_write_unowned=s[f"{key}|leak"] / nu,
                 own_gate_weight_mean=s[f"{key}|ownw"] / max(1.0, s[f"{key}|n_own"]))
        c["PROTECTED"] = c["merged_mean"] >= c["solo_mean"] - 0.05
        c["EXACT"] = abs(c["merged_mean"] - c["solo_mean"]) <= 0.01 and c["kl_unowned"] <= 0.001
        c["QUIET"] = c["unowned_raw_merged"] >= c["unowned_raw_bare"] - 0.02 and c["kl_unowned"] <= 0.05
        res["cells"][key] = c
    res["wall_s"] = time.time() - t0
    res["nan_fields"] = _nan_census(res)
    return res


# ------------------------------------------------------------------ 0.1.72: the two-pass read (the find form)
# Every source of this form puts ITS OWN address in the questions' query streams and in the gate (the form above puts the written index in
# every source's streams). A source reads one of two rows - "T" the row as composed (teacher-forced), "F" the free-running row
# (grid_rows.free_running: the core's own index bytes at their fixed point) - with one stream at its questions:
#   true      the true address                         written   the index the core writes on the T row (its argmax there)
#   swapped   the true address, one pair of each row's questions trading       indexf    the index the core writes on the F row
#   dec_i / dec_if   the .5 decode (grid_rows.decode_scores) of the head's scores at the asked key's index entry on the T / F row, read
#                    in PASS 1 (every question at the null anchor; the index precedes the questions, so nothing there reads a request)
# and one gate: exact (the kernels of the stream's address), qhead (the head's per-digit scores at the question's SEP in the source's
# own view), ihead (the head's scores at the asked key's index entry, pass 1), the _q999 forms through the absolute floor with tau
# calibrated in the same form. G2 and G0 read the T row with the true address. A house-rotary core reads no streams: its sources differ
# only by row and gate.
# 0.1.80, the look's streams (look_of, PASS 1 with the core's look head): look_i / look_if the record holding more than half of the head's
# attention at the asked key's index entry's SEP on the T / F row, look_qf the same at the question's own SEP on the F row; its address,
# both digits from that one record, else the null anchor.
FIND_SOURCES = {
    "LAYOUT": ("T", "true", "exact"), "INDEX": ("T", "written", "exact"), "INDEXC": ("T", "swapped", "exact"),
    "INDEXF": ("F", "indexf", "exact"), "HEADPF": ("F", "indexf", "qhead"), "HEADPFQ999": ("F", "indexf", "qhead_q999"),
    "HEADI": ("T", "dec_i", "ihead"), "HEADIF": ("F", "dec_if", "ihead"), "HEADIFQ999": ("F", "dec_if", "ihead_q999"),
    "LOOK": ("T", "look_i", "exact"), "LOOKF": ("F", "look_if", "exact"), "LOOKQF": ("F", "look_qf", "exact"),
}
SRC_FIND = ("LAYOUT", "INDEX", "INDEXC", "HEADPF", "INDEXF", "HEADI", "HEADIF", "HEADIFQ999")   # a core with the head at the index entries
SRC_QHEAD = ("LAYOUT", "INDEX", "INDEXC", "INDEXF", "HEADPF", "HEADPFQ999", "HEADI", "HEADIF")  # a head at the questions only
SRC_NOHEAD = ("LAYOUT", "INDEX", "INDEXC", "INDEXF")
SRC_FIND_MID = ("LAYOUT", "INDEXC", "INDEXF", "HEADIF", "HEADIFQ999")
SRC_QHEAD_MID = ("LAYOUT", "INDEXC", "INDEXF", "HEADPF")
SRC_NOHEAD_MID = ("LAYOUT", "INDEXC", "INDEXF")
SRC_LOOK = ("LOOK", "LOOKF", "LOOKQF")                   # 0.1.80: on a core with a look head, after the sets above
SRC_LOOK_MID = ("LOOKF",)
LOOK_STREAMS = {"look_i": ("T", "index"), "look_if": ("F", "index"), "look_qf": ("F", "question")}   # stream -> (row, query position)
LOOK_THR = 0.5                                           # more than half of the head's mass: at most one record can hold it
FIND_SOURCES_BY_STREAM = {v[1]: s for s, v in FIND_SOURCES.items() if v[1] in LOOK_STREAMS}   # look_i -> LOOK, ...


def find_sources(core: Core, site: str = "last") -> tuple:
    """The two-pass form's sources for this core: the head at the index entries (SRC_FIND), at the questions only (SRC_QHEAD), none;
    0.1.80: and the look's (SRC_LOOK; the mid site SRC_LOOK_MID) on a core with a look head."""
    if core.head is None:
        base = SRC_NOHEAD if site == "last" else SRC_NOHEAD_MID
    elif core.head_at == "both":
        base = SRC_FIND if site == "last" else SRC_FIND_MID
    else:
        base = SRC_QHEAD if site == "last" else SRC_QHEAD_MID
    if getattr(core, "look", None) is not None:
        base = base + (SRC_LOOK if site == "last" else SRC_LOOK_MID)
    return base


def look_name(core_tag: str, look) -> str:
    """0.1.80: the look read's record name - the core's tag and the head it reads (block numbered from 1)."""
    return f"{core_tag}_look_b{int(look[0])}h{int(look[1])}"


@torch.no_grad()
def look_of(cap, metas: list, rows_idx, pos, asked, thr: float = LOOK_THR) -> dict:
    """0.1.80, THE LOOK: at input position pos[j] of batch row rows_idx[j], the captured head's causal softmax (its block's own rotated q
    and k from the latest forward; the eval kernel's scale 1 / sqrt(head dim)) summed over each record's frame [lo, frame_end) of that
    row. The decode: the record holding more than `thr` of the mass (thr >= .5: at most one), judged by its own mass and never ranked
    against the others; its address's digits, else the null anchor at both levels. Beside it (a read, never a request) the largest-mass
    record. asked[j]: the record the question asks (its index in the row's recs).
    -> dict(dig (Q, 2) int64, rec (Q,) the decoded record or -1, mass (Q,) its mass (0 when none), top (Q,) the largest-mass record,
    top_mass, second (Q,) the second frame's mass, true_mass (Q,) the asked record's mass), numpy."""
    assert thr >= 0.5, "below one half two records could both pass: the decode would have to rank them"
    q, k, h = cap.q, cap.k, cap.head
    assert q is not None and k is not None, "the look reads the latest forward's capture (pass 1 with the capture on)"
    rows_idx, pos, asked = (np.asarray(x, dtype=np.int64) for x in (rows_idx, pos, asked))
    Q = len(rows_idx)
    hd = q.shape[-1]
    code = gr.code_of(metas[0]) if metas else gr.CODE              # 0.1.91: the rows' address code (its levels; its radix the null anchor)
    rx, nl = gr.radix_levels(code)
    o = dict(dig=np.full((Q, nl), rx, dtype=np.int64), rec=np.full(Q, -1, dtype=np.int64), mass=np.zeros(Q),
             top=np.zeros(Q, dtype=np.int64), top_mass=np.zeros(Q), second=np.zeros(Q), true_mass=np.zeros(Q))
    for r in np.unique(rows_idx):
        js = np.nonzero(rows_idx == r)[0]
        recs = metas[int(r)]["recs"]
        p_ = torch.as_tensor(pos[js], device=q.device)
        assert int(p_.max()) < k.shape[2], "a query position past the forward's length"
        s = (q[int(r), h, p_].float() @ k[int(r), h].float().transpose(0, 1)) / math.sqrt(hd)        # (c, n)
        t = torch.arange(s.shape[-1], device=q.device)
        a = torch.softmax(s.masked_fill(t[None, :] > p_[:, None], float("-inf")), -1).double()      # the causal row
        cs = F.pad(a.cumsum(-1), (1, 0))                                                            # cs[:, j] = mass on [0, j)
        lo = torch.tensor([x["lo"] for x in recs], device=q.device)
        hi = torch.tensor([x["frame_end"] for x in recs], device=q.device)
        fm = (cs[:, hi] - cs[:, lo]).cpu().numpy()                                                  # (c, records) frame masses
        ix = np.arange(len(js))
        above = fm > thr                                                       # each record against one fixed number (grid_rows.decode_scores)
        one = above.sum(-1) == 1
        rec = np.where(one, above.argmax(-1), -1)                              # the one record above it, else none
        o["rec"][js] = rec
        o["mass"][js] = np.where(one, fm[ix, np.maximum(rec, 0)], 0.0)
        o["true_mass"][js] = fm[ix, asked[js]]
        order = np.argsort(-fm, axis=-1, kind="stable")                        # the largest-mass form: a read beside, never the request
        o["top"][js], o["top_mass"][js], o["second"][js] = order[:, 0], fm[ix, order[:, 0]], fm[ix, order[:, 1]]
        for jj, j in enumerate(js):
            if rec[jj] >= 0:
                o["dig"][j] = gr.digits(int(recs[int(rec[jj])]["addr"]), code)
    return o


@torch.no_grad()
def free_rows(core: Core, rows: list) -> tuple:
    """The rows at their free-running fixed point (grid_rows.free_running), written as the read reads (fp32; every question at the null
    anchor), cached on the core by the row's bytes. -> (rows [(row, meta)], passes, converged), passes and converged per row."""
    todo = [i for i, (r, _) in enumerate(rows) if r not in core.free]
    if todo:
        def greedy(now):
            x = to_x(now, core.dev)
            core.set_streams([m for _, m in now], "null")
            h = core.trunk(x[:, :-1])
            core.clear()
            pred = x[:, 1:].clone()                                        # the row's own bytes elsewhere (never read)
            for j, (_, m) in enumerate(now):
                ip = torch.tensor([p for pair in gr.index_positions(m) for p in pair], device=core.dev)
                pred[j, ip] = core.readout(h[j, ip]).argmax(-1)
            return pred.cpu().numpy()
        fr, k, conv = gr.free_running([rows[i] for i in todo], greedy)
        for i, (r2, _), cv in zip(todo, fr, conv):
            core.free[rows[i][0]] = (r2, k, cv)
    got = [core.free[r] for r, _ in rows]
    return [(g[0], m) for g, (_, m) in zip(got, rows)], [g[1] for g in got], [g[2] for g in got]


def _req_lists(qt: dict, n_rows: int, dig: np.ndarray) -> list:
    """Per row, its questions' digit pairs in question order (a stream request) from per-question digits (Q, 2) in the table's order."""
    return [[tuple(int(d) for d in dig[k]) for k in np.nonzero(qt["row"] == j)[0]] for j in range(n_rows)]


@torch.no_grad()
def batch_views(core: Core, rows: list, owned: list, swap_rng, srcs, site: int | None = None, gates: bool = True, looks=()) -> dict:
    """One read batch in the two-pass form (fp32). Pass 1 on each row the sources read (every question at the null anchor): the index
    the core writes, and the head's scores at each index entry's SEP. Then one forward per (row, stream) the sources need, the source's
    own address in its questions' streams: the states at the answer positions (and after block `site`), the bare logits there, the head
    at the questions' SEPs. gates: the T row with the true address too (G2, G0, the solo and scan reads). looks (0.1.80): look streams to
    read in pass 1 beside the sources' own (no view of their own). -> the context."""
    dev = core.dev
    metas = [m for _, m in rows]
    qt = question_table(metas, owned)
    Q = len(qt["row"])
    rq = torch.from_numpy(qt["row"]).to(dev)
    ip = torch.from_numpy(qt["ipos"]).to(dev)
    r_ = torch.from_numpy(np.repeat(qt["row"], NA)).to(dev)
    p_ = torch.from_numpy(qt["pos"].reshape(-1)).to(dev)
    xs = {"T": to_x(rows, dev)}
    out = dict(qt=qt, metas=metas, rq=rq, r_=r_, p_=p_, xs=xs, swapped=_swap_requests(qt["dig"], qt["row"], swap_rng), p1={}, stream={})
    if any(FIND_SOURCES[s][0] == "F" for s in srcs):
        fr, passes, conv = free_rows(core, rows)
        xs["F"] = to_x(fr, dev)
        out.update(free=fr, passes=passes, converged=conv)
    need = sorted(({FIND_SOURCES[s][1] for s in srcs} | set(looks)) & set(LOOK_STREAMS)) if core.look is not None else []   # 0.1.80
    asked = np.array([q["rec"] for m in metas for q in m["questions"]], dtype=np.int64)                         # the asked records
    out["look"] = {}
    for rk, x in xs.items():                                               # PASS 1: every question at the null anchor
        core.set_streams(metas, "null")
        mine = [sn for sn in need if LOOK_STREAMS[sn][0] == rk]
        if mine:
            core.look_cap.on = True                                        # the look head's q and k of this forward (nothing else)
        h = core.trunk(x[:, :-1])
        core.clear()
        if mine:
            core.look_cap.on = False
            for sn in mine:                                                # at the asked key's index entry's SEP, or the question's own
                at = qt["ipos"][:, 0] if LOOK_STREAMS[sn][1] == "index" else qt["pos"][:, 0]
                out["look"][sn] = look_of(core.look_cap, metas, qt["row"], at, asked)
            core.look_cap.clear()
        pred = torch.stack([core.readout(h[rq, ip[:, i]]).argmax(-1) for i in range(ip.shape[1])], 1)  # the entry's address bytes
        out["p1"][rk] = dict(written=gr.decode_written(pred.cpu().numpy(), ip.shape[1], getattr(core, "code", gr.CODE)),
                             s_e=core.head_scores(h[rq, ip[:, 0]]) if core.head is not None else None)  # the head at the SEP
    st = out["stream"]
    st["true"], st["written"], st["swapped"] = qt["dig"], out["p1"]["T"]["written"], out["swapped"]
    if "F" in xs:
        st["indexf"] = np.array([d for r, m in out["free"] for d in gr.row_request(r, m)], dtype=np.int64).reshape(
            -1, gr.radix_levels(getattr(core, "code", gr.CODE))[1])
    if core.head is not None:
        for rk in xs:
            st["dec_i" if rk == "T" else "dec_if"] = np.array(gr.decode_scores(out["p1"][rk]["s_e"].cpu().numpy()), dtype=np.int64)
    for sn, lk in out["look"].items():
        st[sn] = lk["dig"]
    out["asked"] = asked
    anchored = core.anchor is not None
    view_of = {s: (FIND_SOURCES[s][0], FIND_SOURCES[s][1] if anchored else "-") for s in srcs}
    if gates:
        view_of["G"] = ("T", "true" if anchored else "-")
    views = {}
    for vk in sorted(set(view_of.values())):
        rk, sn = vk
        x = xs[rk]
        req = None if sn in ("true", "-") else _req_lists(qt, len(metas), st[sn])
        core.set_streams(metas, req)
        hs = None
        if site is None:
            h = core.trunk(x[:, :-1])
        else:
            hs = core.trunk(x[:, :-1], hi=site)
            h = core.trunk(x[:, :-1], lo=site, h=hs)
        core.clear()
        hq = h[r_, p_].view(Q, NA, core.d)
        v = dict(hq=hq, lb=core.readout(hq), raw=x[:, 1:][r_, p_].view(Q, NA), req=req, x=x)
        if core.head is not None:
            v["s_q"] = core.head_scores(hq[:, 0])
        if site is not None:
            v["h_site"], v["h_site_q"] = hs, hs[r_, p_].view(Q, NA, core.d)
        views[vk] = v
    out["views"], out["view_of"] = views, view_of
    return out


def _view_tag(vk: tuple) -> str:
    return f"{vk[0]}:{vk[1]}"


def _factors_find(ctx: dict, src: str, mem: torch.Tensor, L9: torch.Tensor, taus: dict) -> list:
    rk, sn, g = FIND_SOURCES[src]
    if g == "exact":
        return exact_factors(torch.from_numpy(ctx["stream"][sn]).to(mem.device), mem, L9)
    s = ctx["views"][ctx["view_of"][src]]["s_q"] if g.startswith("qhead") else ctx["p1"][rk]["s_e"]
    return head_factors(s, mem, taus.get(src) if g.endswith("q999") else None)


def _scores_of(ctx: dict, src: str) -> torch.Tensor:
    """A head source's per-digit scores (Q, 2, 8): at the question's SEP in its own view, or at the asked key's index entry (pass 1)."""
    rk, _, g = FIND_SOURCES[src]
    return ctx["views"][ctx["view_of"][src]]["s_q"] if g.startswith("qhead") else ctx["p1"][rk]["s_e"]


@torch.no_grad()
def calibrate_find(core: Core, rows: list, owned: list, srcs, mb: int = ROWS_LAST) -> dict:
    """The floors' tau per level in the two-pass form, for each floored source of `srcs`: the quantiles of the per-question largest wrong
    digit score of the source's own scores (HEADPFQ999: the head at the question's SEP on the free-running row with the index written
    there in the stream; HEADIFQ999: the head at the asked key's index entry, pass 1 on the free-running row) on the calibration rows
    (never the read's). -> {source: {tau, saturation}}."""
    fams = [s for s in srcs if FIND_SOURCES[s][2].endswith("q999")]
    if core.head is None or not fams:
        return {}
    lw = {s: [] for s in fams}
    for i in range(0, len(rows), mb):
        ctx = batch_views(core, rows[i:i + mb], owned, np.random.default_rng(0), fams, gates=False)
        dig = torch.from_numpy(ctx["qt"]["dig"]).to(core.dev)
        for s in fams:
            lw[s].append(_scores_of(ctx, s).scatter(-1, dig[..., None], -1.0).amax(-1).cpu())
    out = {}
    for s in fams:
        t = torch.cat(lw[s]).double()
        tau = {str(q): [float(torch.quantile(t[:, l], q)) for l in range(gr.LEVELS)] for q in QS}
        out[s] = dict(rows=len(rows), questions=int(t.shape[0]), tau=tau,
                      largest_wrong_at_1=[float((t[:, l] >= 1.0).double().mean()) for l in range(gr.LEVELS)],
                      tau_999_saturated=any(x >= 1.0 for x in tau[str(0.999)]))
    return out


def _find_cells(srcs, m_set=M_SET) -> list:
    out = [(f"{s}_U{int(U)}_{Z}_m{m}", s, U, Z, m) for s in srcs for U in U_SET for Z in Z_SET for m in m_set]
    for m in m_set:
        out.append((f"G2_m{m}", "G2", 0.0, "-", m))
        out.append((f"G0_m{m}", "G0", 0.0, "-", m))
    return out


def _find_diag(ctx: dict, srcs, fac: dict, diag: dict, qo, o) -> None:
    """The request sources' diagnostics on one batch: each source's address right (its stream's digits; a question head: its argmax, and
    its agreement with the index written on the free-running row), the index heads' argmax right, the owner's u and the largest off-band
    u, the floors' owners at 0 and wrong members passing; the free-running row's index right and its fixed-point check."""
    qt = ctx["qt"]
    dig = qt["dig"]
    diag["questions"] += len(qt["row"])
    diag["n_owned_q"] += int(qo.numel())
    diag["batches"] += 1
    diag["written_right"] += int((ctx["stream"]["written"] == dig).all(-1).sum())
    if "indexf" in ctx["stream"]:
        diag["indexf_right"] += int((ctx["stream"]["indexf"] == dig).all(-1).sum())
        diag["fixed_point_agree"] += int((ctx["p1"]["F"]["written"] == ctx["stream"]["indexf"]).all(-1).sum())
        diag["passes"] += list(ctx["passes"])
        diag["converged"] += [bool(c) for c in ctx["converged"]]
    for s in srcs:
        _, sn, g = FIND_SOURCES[s]
        d = diag["src"].setdefault(s, dict(right=0, argmax_right=0, agree_indexf=0, owner_u=0.0, offband_u_max=0.0, floor_owner_zero=0,
                                           floor_wrong_pass=0, truth=0.0, wrong_max=0.0))
        if g != "exact":                                                   # a head's scores: the truth's mean, the largest wrong one's
            sc = _scores_of(ctx, s)
            dt = torch.from_numpy(dig).to(sc.device)
            d["truth"] += float(sc.gather(-1, dt[..., None])[..., 0].mean(-1).sum())
            d["wrong_max"] += float(sc.scatter(-1, dt[..., None], -1.0).amax(-1).amax(-1).sum())
        if g.startswith("qhead"):
            am = _scores_of(ctx, s).argmax(-1).cpu().numpy()
            d["right"] += int((am == dig).all(-1).sum())
            d["agree_indexf"] += int((am == ctx["stream"]["indexf"]).all(-1).sum()) if "indexf" in ctx["stream"] else 0
        else:
            d["right"] += int((ctx["stream"][sn] == dig).all(-1).sum())
        if g.startswith("ihead"):
            d["argmax_right"] += int((_scores_of(ctx, s).argmax(-1).cpu().numpy() == dig).all(-1).sum())
        pr = fac[s][0] * fac[s][1]
        d["owner_u"] += float(pr[qo, o[qo]].sum())
        off = pr.clone()
        off[qo, o[qo]] = float("-inf")
        d["offband_u_max"] += float(off.amax(-1).clamp(min=0).sum())
        if g.endswith("q999"):
            d["floor_owner_zero"] += int((pr[qo, o[qo]] == 0).sum())
            d["floor_wrong_pass"] += int((off > 0).any(-1).sum())
        if sn in LOOK_STREAMS:                                             # 0.1.80: the look's own counts
            _look_add(d.setdefault("look", _look_new()), ctx, sn)


LOOK_COUNTS = ("questions", "no_request", "wrong_record", "right", "decoded_mass", "true_mass", "top_right", "split", "duplicate",
               "wrong_same_ending")


def _look_new() -> dict:
    return {k: 0.0 for k in LOOK_COUNTS}


def _look_add(acc: dict, ctx: dict, sn: str) -> None:
    """0.1.80: one batch's look counts for stream sn: no request (no record above one half), a request naming a wrong record, right; the
    decoded frame's mass (summed over the decoded), the asked record's mass; the largest-mass form right; the split looks (the second
    frame holding at least half the first's); the duplicate looks (a record decoded for two questions of one row); the wrong requests
    whose record's key shares the asked key's last two characters (the key's two-digit ending)."""
    lk, asked, metas, row = ctx["look"][sn], ctx["asked"], ctx["metas"], ctx["qt"]["row"]
    rec = lk["rec"]
    wrong = (rec >= 0) & (rec != asked)
    acc["questions"] += len(rec)
    acc["no_request"] += int((rec < 0).sum())
    acc["wrong_record"] += int(wrong.sum())
    acc["right"] += int((rec == asked).sum())
    acc["decoded_mass"] += float(lk["mass"].sum())
    acc["true_mass"] += float(lk["true_mass"].sum())
    acc["top_right"] += int((lk["top"] == asked).sum())
    acc["split"] += int((lk["second"] >= 0.5 * lk["top_mass"]).sum())
    for j in np.unique(row):
        rj = rec[(row == j) & (rec >= 0)]
        _, cnt = np.unique(rj, return_counts=True)
        acc["duplicate"] += int(cnt[cnt > 1].sum())
    for j in np.nonzero(wrong)[0]:
        rs_ = metas[int(row[j])]["recs"]
        acc["wrong_same_ending"] += int(rs_[int(rec[j])]["key"][-2:] == rs_[int(asked[j])]["key"][-2:])


def _look_out(acc: dict) -> dict:
    n = max(1.0, acc["questions"])
    dec = max(1.0, acc["questions"] - acc["no_request"])
    return dict(questions=int(acc["questions"]), right=acc["right"] / n, no_request=acc["no_request"] / n, wrong_record=acc["wrong_record"] / n,
                decoded_mass_mean=acc["decoded_mass"] / dec, true_mass_mean=acc["true_mass"] / n, largest_mass_form_right=acc["top_right"] / n,
                split=acc["split"] / n, duplicate=acc["duplicate"] / n,
                wrong_same_ending=acc["wrong_same_ending"] / max(1.0, acc["wrong_record"]), wrong_records=int(acc["wrong_record"]))


def _find_diag_out(diag: dict, srcs) -> dict:
    nq, no = max(1, diag["questions"]), max(1, diag["n_owned_q"])
    out = dict(written_index_right=diag["written_right"] / nq, questions=diag["questions"])
    if diag["passes"]:
        out.update(indexf_right=diag["indexf_right"] / nq, fixed_point_agree=diag["fixed_point_agree"] / nq,
                   passes=dict(mean=float(np.mean(diag["passes"])), max=int(max(diag["passes"]))),
                   converged=float(np.mean(diag["converged"])))
    out["sources"] = {}
    for s in srcs:
        d, g = diag["src"][s], FIND_SOURCES[s][2]
        r = dict(view=_view_tag((FIND_SOURCES[s][0], FIND_SOURCES[s][1])), request_right=d["right"] / nq, owner_u_over_U=d["owner_u"] / no,
                 largest_offband_u_over_U_mean=d["offband_u_max"] / nq)
        if g.startswith("qhead") and diag["passes"]:
            r["argmax_agrees_with_indexf"] = d["agree_indexf"] / nq
        if g.startswith("ihead"):
            r["argmax_right"] = d["argmax_right"] / nq
        if g != "exact":
            r["mean_truth_score"], r["mean_largest_wrong_score"] = d["truth"] / nq, d["wrong_max"] / nq
        if g.endswith("q999"):
            r["floor"] = dict(owners_floored_to_0=d["floor_owner_zero"] / no, questions_passing_a_wrong_member=d["floor_wrong_pass"] / nq)
        if "look" in d:
            r["look"] = _look_out(d["look"])
        out["sources"][s] = r
    return out


def _new_find_diag() -> dict:
    return dict(questions=0, n_owned_q=0, batches=0, written_right=0, indexf_right=0, fixed_point_agree=0, passes=[], converged=[], src={})


@torch.no_grad()
def read_last_find(core: Core, stacks: dict, rows: list, cal: dict, owned: list, never: list, seed: int, srcs=None, mb: int = ROWS_LAST,
                   log=print) -> dict:
    """The last-site read in the two-pass form: every source x U x Z x m and G2 / G0 for every stack, each source in its own view (its
    own address in the streams and the gate; the bare core and the solo reads in that same view), the owner-weight scan on the T row with
    the true address, the sources' diagnostics. cal: calibrate_find's record. -> the record (read_last's fields, plus each cell's view)."""
    assert getattr(core, "code", gr.CODE) == gr.CODE, "the members' rows and gates are written for the 8x2 address code"
    dev = core.dev
    srcs = list(srcs or find_sources(core, "last"))
    L9 = level_code(dev)
    owned_t = torch.tensor(owned, device=dev)
    mem = torch.stack([owned_t // gr.RADIX, owned_t % gr.RADIX], 1)
    codes = code_tables(N_OWN).to(dev)
    taus = {s: ((cal.get(s) or {}).get("tau") or {}).get(str(0.999)) for s in srcs if FIND_SOURCES[s][2].endswith("q999")}
    cells = _find_cells(srcs)
    swap_rng = np.random.default_rng([seed, gr.crc("SCALE|swap")])
    cnt = torch.zeros(gr.N_ADDR, dtype=torch.float64, device=dev)
    A = {k: _Acc(dev) for k in stacks}
    B = _Acc(dev)
    diag = _new_find_diag()
    tags = {}
    t0 = time.time()
    for bi in range(0, len(rows), mb):
        ctx = batch_views(core, rows[bi:bi + mb], owned, swap_rng, srcs)
        qt, views, vof = ctx["qt"], ctx["views"], ctx["view_of"]
        Q = len(qt["row"])
        addr = torch.from_numpy(qt["addr"]).to(dev)
        o = torch.from_numpy(qt["own"]).to(dev)
        nv = o < 0
        qo = (o >= 0).nonzero().flatten()
        cnt += torch.bincount(addr, minlength=gr.N_ADDR).double()
        B.sc("bare_n_pos", int(nv.sum()) * NA)
        for vk, v in views.items():                                        # the bare core in every view
            tg = tags.setdefault(vk, _view_tag(vk))
            v["lpb"], v["pb"] = F.log_softmax(v["lb"], -1), v["lb"].argmax(-1)
            B.add(f"bare_raw|{tg}", addr, (v["pb"] == v["raw"]).all(-1))
            mb_ = _margin(v["lb"][nv], v["raw"][nv])
            B.sc(f"bare_margin_sum|{tg}", mb_.sum())
            B.mx(f"bare_margin_min|{tg}", mb_.min(), min)
        fac = {s: _factors_find(ctx, s, mem, L9, taus) for s in srcs}
        _find_diag(ctx, srcs, fac, diag, qo, o)
        vG = views[vof["G"]]
        for name, stk in stacks.items():
            acc = A[name]
            Wv = {vk: stk.write_all(v["hq"].reshape(-1, core.d)).view(Q, NA, stk.M, core.d) for vk, v in views.items()}
            sw = {vk: W[qo, :, o[qo]].norm(dim=-1) for vk, W in Wv.items()}
            tgt = codes[o[qo][:, None], vG["raw"][qo]]
            WoG = Wv[vof["G"]][qo, :, o[qo]]
            acc.add("solo_ONE", addr[qo], (core.readout(vG["hq"][qo] + WoG).argmax(-1) == tgt).all(-1))
            for s in srcs:                                                 # the solo reads, each in its source's own view
                v, Wo = views[vof[s]], Wv[vof[s]][qo, :, o[qo]]
                pr = (fac[s][0] * fac[s][1])[qo, o[qo]]
                for U in U_SET:
                    ws = torch.tanh(U * pr)
                    acc.add(f"solo_{s}_U{int(U)}", addr[qo], (core.readout(v["hq"][qo] + ws[:, None, None] * Wo).argmax(-1) == tgt).all(-1))
            for a in A_GRID:
                acc.add(f"scan_{a}", addr[qo], (core.readout(vG["hq"][qo] + a * WoG).argmax(-1) == tgt).all(-1))
            for key, kind, U, Z, m in cells:
                vk = vof["G"] if kind in ("G0", "G2") else vof[kind]
                v, W = views[vk], Wv[vk]
                w = _gate(kind, U, Z, m, fac, qt, owned_t, dev)
                dh = torch.einsum("qm,qpmd->qpd", w, W[:, :, :m])
                la = core.readout(v["hq"] + dh)
                pa = la.argmax(-1)
                inm = (o >= 0) & (o < m)
                qi = inm.nonzero().flatten()
                acc.add(f"{key}|hit_own", addr[qi], (pa[qi] == codes[o[qi][:, None], v["raw"][qi]]).all(-1))
                acc.add(f"{key}|hit_raw", addr[nv], (pa[nv] == v["raw"][nv]).all(-1))
                du, lau = dh[nv], la[nv]
                acc.sc(f"{key}|kl", (v["lpb"][nv].exp() * (v["lpb"][nv] - F.log_softmax(lau, -1))).sum())
                acc.sc(f"{key}|leak", du.norm(dim=-1).sum())
                mu = _margin(lau, v["raw"][nv])
                acc.sc(f"{key}|msum", mu.sum())
                acc.mx(f"{key}|mmin", mu.min(), min)
                for sc in FLIPS:
                    ps = core.readout(v["hq"][nv] + sc * du).argmax(-1)
                    acc.sc(f"{key}|flip{sc}", (ps != v["raw"][nv]).sum())
                    acc.sc(f"{key}|flipb{sc}", (ps != v["pb"][nv]).sum())
                sel = inm[qo]
                acc.sc(f"{key}|own_write", dh[qi].norm(dim=-1).sum())
                acc.sc(f"{key}|solo_write", sw[vk][sel].sum())
                acc.sc(f"{key}|n_own_pos", int(qi.numel()) * NA)
                acc.sc(f"{key}|ownw", w[qi, o[qi]].sum())
                acc.sc(f"{key}|n_own", qi.numel())
                offm = torch.ones_like(w, dtype=torch.bool)
                offm[qi, o[qi]] = False
                acc.sc(f"{key}|off", w.abs()[offm].sum())
                acc.sc(f"{key}|n_q", Q)
        if ((bi // mb) + 1) % 8 == 0 or bi + mb >= len(rows):
            el = time.time() - t0
            done = min(len(rows), bi + mb)
            log(f"[read last, two-pass] {done}/{len(rows)} rows: {el:.0f}s spent, ~{el * (len(rows) - done) / done:.0f}s left")
    nu = max(1.0, B.s["bare_n_pos"])
    bare = {tg: dict(unowned=_mean_over(B, f"bare_raw|{tg}", cnt, never), owned=_mean_over(B, f"bare_raw|{tg}", cnt, owned),
                     margin_unowned_mean=B.s[f"bare_margin_sum|{tg}"] / nu, margin_unowned_min=B.s[f"bare_margin_min|{tg}"])
            for tg in tags.values()}
    tg_G = _view_tag(("T", "true" if core.anchor is not None else "-"))
    res = dict(form="two-pass", sources=srcs, rows=len(rows), questions=diag["questions"], questions_per_address_min=int(cnt.min()),
               diag=_find_diag_out(diag, srcs), calibration=cal,
               bare=dict(bare[tg_G], owned_per_owner=[float(B.v[f"bare_raw|{tg_G}"][a] / max(1.0, float(cnt[a]))) for a in owned]),
               bare_by_view=bare, stacks={}, wall_s=time.time() - t0)
    for name, acc in A.items():
        r = dict(solo={}, a_scan={}, cells={})
        for k in ["ONE"] + [f"{s}_U{int(U)}" for s in srcs for U in U_SET]:
            r["solo"][k] = {str(m): _mean_over(acc, f"solo_{k}", cnt, owned[:m]) for m in M_SET}
        r["solo_per_owner"] = [float(acc.v["solo_ONE"][a] / max(1.0, float(cnt[a]))) for a in owned]
        for a in A_GRID:
            r["a_scan"][repr(a)] = {str(m): _mean_over(acc, f"scan_{a}", cnt, owned[:m]) for m in M_SET}
        for key, kind, U, Z, m in cells:
            sk = "ONE" if kind in ("G0", "G2") else f"{kind}_U{int(U)}"
            tg = tg_G if kind in ("G0", "G2") else _view_tag((FIND_SOURCES[kind][0], FIND_SOURCES[kind][1] if core.anchor is not None else "-"))
            s = acc.s
            n_own_pos = max(1.0, s[f"{key}|n_own_pos"])
            sw_ = s[f"{key}|solo_write"] / n_own_pos
            c = dict(m=m, view=tg, solo_key=sk, solo_mean=r["solo"][sk][str(m)], merged_mean=_mean_over(acc, f"{key}|hit_own", cnt, owned[:m]),
                     unowned_raw_merged=_mean_over(acc, f"{key}|hit_raw", cnt, never), unowned_raw_bare=bare[tg]["unowned"],
                     kl_unowned=s[f"{key}|kl"] / nu, leak_write_unowned=s[f"{key}|leak"] / nu, solo_write_owned=sw_,
                     leak_over_own_write=(s[f"{key}|leak"] / nu) / max(1e-12, sw_),
                     margin_unowned_mean=s[f"{key}|msum"] / nu, margin_unowned_min=s[f"{key}|mmin"],
                     flip_rate_unowned={str(sc): s[f"{key}|flip{sc}"] / nu for sc in FLIPS},
                     flip_vs_bare_unowned={str(sc): s[f"{key}|flipb{sc}"] / nu for sc in FLIPS},
                     owned_write_over_solo_write=s[f"{key}|own_write"] / max(1e-12, s[f"{key}|solo_write"]),
                     own_gate_weight_mean=s[f"{key}|ownw"] / max(1.0, s[f"{key}|n_own"]),
                     offband_gate_mass_per_question=s[f"{key}|off"] / max(1.0, s[f"{key}|n_q"]))
            c["PROTECTED"] = c["merged_mean"] >= c["solo_mean"] - 0.05
            c["EXACT"] = abs(c["merged_mean"] - c["solo_mean"]) <= 0.01 and c["kl_unowned"] <= 0.001
            c["QUIET"] = c["unowned_raw_merged"] >= c["unowned_raw_bare"] - 0.02 and c["kl_unowned"] <= 0.05
            r["cells"][key] = c
        res["stacks"][name] = r
    res["nan_fields"] = _nan_census(res)
    return res


@torch.no_grad()
def read_mid_find(core: Core, arms: StackedArms, rows: list, cal: dict, owned: list, never: list, seed: int, srcs=None,
                  mb: int = ROWS_MID, log=print, quick: bool = False) -> dict:
    """The mid-site read (m = M_MID) in the two-pass form: every source x U x Z cell and G2 / G0, each source in its own view (the merged
    writes after block MID_SITE at every question's answer positions, blocks MID_SITE.. recomputed with the view's streams), and the
    solo reads with each owned question's owner ALONE in its row in the same view. quick: U 5 only. -> the record."""
    assert getattr(core, "code", gr.CODE) == gr.CODE, "the members' rows and gates are written for the 8x2 address code"
    dev = core.dev
    srcs = list(srcs or find_sources(core, "mid"))
    L9 = level_code(dev)
    m = arms.M
    owned_m = owned[:m]
    owned_t = torch.tensor(owned, device=dev)
    mem = torch.stack([owned_t[:m] // gr.RADIX, owned_t[:m] % gr.RADIX], 1)
    codes = code_tables(m).to(dev)
    taus = {s: ((cal.get(s) or {}).get("tau") or {}).get(str(0.999)) for s in srcs if FIND_SOURCES[s][2].endswith("q999")}
    us = U_SET[:1] if quick else U_SET
    cells = [(f"{s}_U{int(U)}_{Z}_m{m}", s, U, Z) for s in srcs for U in us for Z in Z_SET]
    cells += [(f"G2_m{m}", "G2", 0.0, "-"), (f"G0_m{m}", "G0", 0.0, "-")]
    solo_keys = ["ONE"] + [f"{s}_U{int(U)}" for s in srcs for U in us]
    swap_rng = np.random.default_rng([seed, gr.crc("SCALE|swap")])
    cnt = torch.zeros(gr.N_ADDR, dtype=torch.float64, device=dev)
    acc, B = _Acc(dev), _Acc(dev)
    diag = _new_find_diag()
    tags = {}
    t0 = time.time()
    for bi in range(0, len(rows), mb):
        ctx = batch_views(core, rows[bi:bi + mb], owned, swap_rng, srcs, site=MID_SITE)
        qt, views, vof = ctx["qt"], ctx["views"], ctx["view_of"]
        Q = len(qt["row"])
        addr = torch.from_numpy(qt["addr"]).to(dev)
        o_all = torch.from_numpy(qt["own"]).to(dev)
        o = torch.where(o_all < m, o_all, torch.full_like(o_all, -2))       # -2 = owned beyond the mid set (neither merged nor unowned)
        nv = o_all < 0
        qo = (o >= 0).nonzero().flatten()
        cnt += torch.bincount(addr, minlength=gr.N_ADDR).double()
        B.sc("n_pos", int(nv.sum()) * NA)
        for vk, v in views.items():
            tg = tags.setdefault(vk, _view_tag(vk))
            v["lpb"], v["pb"] = F.log_softmax(v["lb"], -1), v["lb"].argmax(-1)
            B.add(f"bare_raw|{tg}", addr, (v["pb"] == v["raw"]).all(-1))
        fac = {s: _factors_find(ctx, s, mem, L9, taus) for s in srcs}
        _find_diag(ctx, srcs, fac, diag, qo, o)
        Wv = {vk: arms.write_all(v["h_site_q"].reshape(-1, core.d)).view(Q, NA, m, core.d) for vk, v in views.items()}

        def suffix(v, h3w):
            core.set_streams(ctx["metas"], v["req"])
            h6 = core.trunk(v["x"][:, :-1], lo=MID_SITE, h=h3w)
            core.clear()
            return core.readout(h6[ctx["r_"], ctx["p_"]].view(Q, NA, core.d))
        for key, kind, U, Z in cells:
            vk = vof["G"] if kind in ("G0", "G2") else vof[kind]
            v, W = views[vk], Wv[vk]
            w = _gate(kind, U, Z, m, fac, qt, owned_t, dev)
            dh = torch.einsum("qm,qpmd->qpd", w, W)
            h3w = v["h_site"].index_put((ctx["r_"], ctx["p_"]), (v["h_site_q"] + dh).reshape(-1, core.d))
            la = suffix(v, h3w)
            pa = la.argmax(-1)
            acc.add(f"{key}|hit_own", addr[qo], (pa[qo] == codes[o[qo][:, None], v["raw"][qo]]).all(-1))
            acc.add(f"{key}|hit_raw", addr[nv], (pa[nv] == v["raw"][nv]).all(-1))
            acc.sc(f"{key}|kl", (v["lpb"][nv].exp() * (v["lpb"][nv] - F.log_softmax(la[nv], -1))).sum())
            acc.sc(f"{key}|leak", dh[nv].norm(dim=-1).sum())
            acc.sc(f"{key}|ownw", w[qo, o[qo]].sum())
            acc.sc(f"{key}|n_own", qo.numel())
        prs = {s: (fac[s][0] * fac[s][1])[qo, o[qo]] for s in srcs}
        for sk in solo_keys:                                               # the solo reads: the owner alone in its row, in the view
            if sk == "ONE":
                vk, ws = vof["G"], torch.ones(qo.numel(), device=dev)
            else:
                s, U = sk.rsplit("_U", 1)
                vk, ws = vof[s], torch.tanh(float(U) * prs[s])
            v, W = views[vk], Wv[vk]
            hit = torch.zeros(qo.numel(), dtype=torch.bool, device=dev)
            live = (ws != 0).nonzero().flatten()
            for c0 in range(0, live.numel(), mb):
                part = live[c0:c0 + mb]
                qq = qo[part]
                rws = torch.from_numpy(qt["row"]).to(dev)[qq]
                hs = v["h_site"][rws].clone()
                pos = torch.from_numpy(qt["pos"]).to(dev)[qq]
                rr = torch.arange(len(qq), device=dev)[:, None].expand_as(pos)
                hs[rr, pos] = v["h_site_q"][qq] + ws[part][:, None, None] * W[qq, :, o[qq]]
                core.set_streams([ctx["metas"][int(j)] for j in rws], None if v["req"] is None else [v["req"][int(j)] for j in rws])
                h6 = core.trunk(v["x"][rws, :-1], lo=MID_SITE, h=hs)
                core.clear()
                hit[part] = (core.readout(h6[rr, pos]).argmax(-1) == codes[o[qq][:, None], v["raw"][qq]]).all(-1)
            acc.add(f"solo_{sk}", addr[qo], hit)
        if ((bi // mb) + 1) % 8 == 0 or bi + mb >= len(rows):
            el = time.time() - t0
            done = min(len(rows), bi + mb)
            log(f"[read mid, two-pass] {done}/{len(rows)} rows: {el:.0f}s spent, ~{el * (len(rows) - done) / done:.0f}s left")
    nu = max(1.0, B.s["n_pos"])
    bare = {tg: dict(unowned=_mean_over(B, f"bare_raw|{tg}", cnt, never), owned=_mean_over(B, f"bare_raw|{tg}", cnt, owned_m))
            for tg in tags.values()}
    tg_G = _view_tag(("T", "true" if core.anchor is not None else "-"))
    res = dict(form="two-pass", sources=srcs, rows=len(rows), m=m, block=MID_SITE, diag=_find_diag_out(diag, srcs), bare=bare[tg_G],
               bare_by_view=bare, solo={k: _mean_over(acc, f"solo_{k}", cnt, owned_m) for k in solo_keys},
               solo_per_owner=[float(acc.v["solo_ONE"][a] / max(1.0, float(cnt[a]))) for a in owned_m], cells={}, wall_s=None)
    for key, kind, U, Z in cells:
        sk = "ONE" if kind in ("G0", "G2") else f"{kind}_U{int(U)}"
        tg = tg_G if kind in ("G0", "G2") else _view_tag((FIND_SOURCES[kind][0], FIND_SOURCES[kind][1] if core.anchor is not None else "-"))
        s = acc.s
        c = dict(view=tg, solo_key=sk, solo_mean=res["solo"][sk], merged_mean=_mean_over(acc, f"{key}|hit_own", cnt, owned_m),
                 unowned_raw_merged=_mean_over(acc, f"{key}|hit_raw", cnt, never), unowned_raw_bare=bare[tg]["unowned"],
                 kl_unowned=s[f"{key}|kl"] / nu, leak_write_unowned=s[f"{key}|leak"] / nu,
                 own_gate_weight_mean=s[f"{key}|ownw"] / max(1.0, s[f"{key}|n_own"]))
        c["PROTECTED"] = c["merged_mean"] >= c["solo_mean"] - 0.05
        c["EXACT"] = abs(c["merged_mean"] - c["solo_mean"]) <= 0.01 and c["kl_unowned"] <= 0.001
        c["QUIET"] = c["unowned_raw_merged"] >= c["unowned_raw_bare"] - 0.02 and c["kl_unowned"] <= 0.05
        res["cells"][key] = c
    res["wall_s"] = time.time() - t0
    res["nan_fields"] = _nan_census(res)
    return res


@torch.no_grad()
def text_bits(core: Core, stacks: dict, docs: list, n_docs: int = 128, mb: int = 8) -> dict:
    """The held-out text bits a byte (the bed's eval rows: DOC + the first ctx bytes of the first n_docs held-out documents), fp32: the bare
    core; the gated merge (these rows carry no addressed question, so the gate is zero at every position and no write enters: the
    difference is 0 by construction, computed); G0 with every member of each stack writing at every byte at the last site (a control
    that can fail)."""
    dev = core.dev
    tot = {"bare": 0.0, "merged": 0.0}
    tot.update({f"G0_{k}": 0.0 for k in stacks})
    n = 0
    for i in range(0, min(n_docs, len(docs)), mb):
        rows = [bytes([DOC]) + d[:gr.CTX] for d in docs[i:i + mb]]
        x = torch.from_numpy(np.stack([np.frombuffer(r, dtype=np.uint8) for r in rows]).astype(np.int64)).to(dev)
        core.clear()
        h = core.trunk(x[:, :-1])
        y = x[:, 1:].reshape(-1)
        lb = core.readout(h)
        nll = F.cross_entropy(lb.reshape(-1, lb.shape[-1]), y, reduction="sum")
        tot["bare"] += float(nll)
        tot["merged"] += float(F.cross_entropy(core.readout(h + torch.zeros_like(h)).reshape(-1, lb.shape[-1]), y, reduction="sum"))
        for k, stk in stacks.items():
            hf = h.reshape(-1, core.d)
            dh = torch.zeros_like(hf)
            for c0 in range(0, hf.shape[0], 2048):
                dh[c0:c0 + 2048] = stk.write_all(hf[c0:c0 + 2048]).sum(1)
            tot[f"G0_{k}"] += float(F.cross_entropy(core.readout((hf + dh).view_as(h)).reshape(-1, lb.shape[-1]), y, reduction="sum"))
        n += y.numel()
    out = {k: v / n / math.log(2) for k, v in tot.items()}
    out["merged_minus_bare"] = out["merged"] - out["bare"]
    out.update({f"G0_{k}_minus_bare": out[f"G0_{k}"] - out["bare"] for k in stacks})
    out["docs"] = min(n_docs, len(docs))
    return out


# ------------------------------------------------------------------ the card's checks: compiled = eager at step 1, the price lines
def compile_check(core: Core, packed: np.ndarray, owned: list, never: list, seed: int = 1) -> dict:
    """Law D at step 1 on the card: the compiled stacked writes against eager on one member-mode batch (the last site's grouped shape)
    and on the mid site's shape (8 rows x 7 positions, one member a row): the writes and every gradient, as the largest difference over
    the largest value. The members are moved off their born zero first so every gradient is live."""
    global COMPILE
    dev = core.dev
    rows = member_rows(np.random.default_rng([seed, crc("SCALE|compile")]), packed, owned, never, abytes=core.abytes, layout=core.layout, key_form=core.key_form)
    qt = question_table([m for _, m in rows], owned)
    x = to_x(rows, dev)
    with torch.no_grad():
        core.set_streams([m for _, m in rows])
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            h = core.trunk(x[:, :-1])
        core.clear()
        sel = np.nonzero(qt["own"] >= 0)[0]
        order = sel[np.argsort(qt["own"][sel], kind="stable")]
        per = len(order) // N_OWN
        r_ = torch.from_numpy(np.repeat(qt["row"][order], NA)).to(dev)
        p_ = torch.from_numpy(qt["pos"][order].reshape(-1)).to(dev)
        hg = h[r_, p_].float().view(N_OWN, per * NA, core.d)
        tgt = torch.gather(code_tables(N_OWN).to(dev), 1, x[:, 1:][r_, p_].view(N_OWN, per * NA))
    out = {}
    g = torch.Generator(device="cpu").manual_seed(7)
    cases = (("arms_last", StackedArms.build(N_OWN, core.d, "SCALE|compile|arm"), None, hg, tgt),
             ("toys_last", StackedToys.build(N_OWN, core.d, "SCALE|compile|toy"), None, hg, tgt),
             ("arms_mid", StackedArms.build(M_MID, core.d, "SCALE|compile|mid"), torch.arange(ROWS_MID, device=dev) * 2, hg[:ROWS_MID, :NA],
              tgt[:ROWS_MID, :NA]))
    for name, stk, idx, hh, tt in cases:
        stk = stk.to(dev)
        with torch.no_grad():
            for p in stk.parameters():
                p.add_(0.02 * torch.randn(p.shape, generator=g).to(dev))
        res = {}
        for mode in (False, True):
            COMPILE = mode
            for p in stk.parameters():
                p.grad = None
            w = stk.write(hh, idx)
            lg = core.readout(hh + w)
            F.cross_entropy(lg.reshape(-1, lg.shape[-1]).float(), tt.reshape(-1)).backward()
            res[mode] = (w.detach().clone(), [p.grad.detach().clone() for p in stk.parameters()])
        COMPILE = True
        rel = lambda a, b: float((a - b).abs().max() / b.abs().max().clamp(min=1e-30))      # noqa: E731
        out[name] = dict(write=rel(res[True][0], res[False][0]), grad=max(rel(a, b) for a, b in zip(res[True][1], res[False][1])))
    return out


def price(core: Core, packed: np.ndarray, owned: list, never: list, seed: int = 1, steps: int = 200, log=print) -> dict:
    """The members' price lines on this card: `steps` steps of each trainer on throwaway stacks (the first steps carry the compile)."""
    a, t = StackedArms.build(N_OWN, core.d, "SCALE|price|arm").to(core.dev), StackedToys.build(N_OWN, core.d, "SCALE|price|toy").to(core.dev)
    t0 = time.time()
    train_last(core, {"arms": a, "toys": t}, steps, seed, packed, owned, never, log=log, trace_every=0)
    last_ms = (time.time() - t0) * 1000 / steps
    m = StackedArms.build(M_MID, core.d, "SCALE|price|mid").to(core.dev)
    t0 = time.time()
    train_mid(core, m, steps, seed, packed, owned, log=log)
    mid_ms = (time.time() - t0) * 1000 / steps
    return dict(steps=steps, last_ms_per_step=last_ms, mid_ms_per_step=mid_ms)


# ------------------------------------------------------------------ the session: one core, both sites, shipped as it lands
class Shipper(threading.Thread):
    """Uploads folders to a Hub repo in the background with retries (the done mark precedes the ship; the card never waits)."""

    def __init__(self, repo: str, log=print) -> None:
        super().__init__(daemon=True)
        self.repo, self.log, self.q = repo, log, queue.Queue()
        self.start()

    def submit(self, folder: str, path_in_repo: str) -> None:
        self.q.put((folder, path_in_repo))

    def run(self) -> None:
        from huggingface_hub import HfApi
        api = HfApi(token=os.environ.get("HF_TOKEN"))
        while True:
            folder, dst = self.q.get()
            try:
                for k in range(6):
                    try:
                        api.upload_folder(folder_path=folder, path_in_repo=dst, repo_id=self.repo,
                                          commit_message=f"scale members: {os.path.basename(folder)}")
                        self.log(f"[ship] {dst} shipped")
                        break
                    except Exception as e:  # noqa: BLE001
                        self.log(f"[ship] {dst} attempt {k + 1} failed: {type(e).__name__}; retrying in {30 * 2 ** k} s")
                        time.sleep(30 * 2 ** k)
            finally:
                self.q.task_done()

    def drain(self, timeout: float = 1800.0) -> bool:
        t0 = time.time()
        while self.q.unfinished_tasks and time.time() - t0 < timeout:
            time.sleep(1.0)
        return not self.q.unfinished_tasks


def _save_stack(path: str, stk: nn.Module, meta: dict) -> None:
    from safetensors.torch import save_file
    save_file({k: v.detach().cpu().contiguous() for k, v in stk.state_dict().items()}, path,
              metadata={"meta": json.dumps(meta)})


def _rm(path: str) -> None:
    if os.path.exists(path):
        os.remove(path)


def run_core(core_folder: str, out_dir: str, seed: int, bed_dir: str, heldout_path: str, last_steps: int = 16000,
             mid_steps: int = 32000, read_batches: int = 64, calib_batches: int = 32, dev: str = "cuda", log=print,
             shipper: Shipper | None = None, hub_dir: str = "bed/scale_members", quick: bool = False, forms=("scale",),
             look="ptr") -> dict:
    """One grid core's members, resume-first: the last site (48 real arms + 48 toy members, `last_steps`), its read and the text bits;
    the mid site (16 real arms, `mid_steps` batch steps), its read. Writes <out_dir>/<core tag>_members/ (the stacks as safetensors, the
    records as JSON, a done mark) and ships it when a shipper is given. quick: the CPU dry run (the mid read at U 5, two text documents).
    forms (0.1.72): the reads to take - "scale" (the form above: last.json, mid.json) and / or "find" (the two-pass form: last_find.json,
    mid_find.json, beside the same stacks); the members train once whatever the forms. look (0.1.80): Core's look head (the find form's
    look sources on a core with one; None = 0.1.79's sources)."""
    assert forms and set(forms) <= {"scale", "find"}, forms
    core = Core(core_folder, dev, look=look)
    name = f"{core.tag}_members_l{last_steps}_m{mid_steps}"
    od = os.path.join(out_dir, name)
    os.makedirs(od, exist_ok=True)
    done = os.path.join(od, "done.json")
    if os.path.exists(done):
        log(f"[members] {name}: done before")
        return json.load(open(done, encoding="utf-8"))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    owned, never = owned_sets()
    packed = load_packed(bed_dir)
    held = load_heldout(heldout_path)
    read_rows = gauge_rows(seed, "read", read_batches * ROWS_LAST, held, abytes=core.abytes, layout=core.layout, key_form=core.key_form)
    cal_rows = gauge_rows(seed, "calib", calib_batches * ROWS_LAST, held[::-1], abytes=core.abytes, layout=core.layout, key_form=core.key_form)
    t0 = time.time()
    meta = dict(core=core.tag, seed=seed, owned=owned, never=never, arm_spec=ARM_SPEC, amoe=AMOE_PIN, toy_hidden=TOY_HID,
                codes="SCALE|code|<i>: a crc-seeded derangement of a-z", betas=list(BETAS), eps=EPS, lr=LR, warm=WARM)
    if core.look is not None:
        meta["look"] = dict(block=core.look[0], head=core.look[1], thr=LOOK_THR)                 # 0.1.80: the look sources' head
    log(f"[members] {name}: core {core.tag} ({'anchored' if core.anchor is not None else 'house'} rotation, head "
        f"{'off' if core.head is None else 'at the questions and the index entries' if core.head_at == 'both' else 'at the questions'}); "
        f"last site {last_steps:,} steps, mid site {mid_steps:,} batch steps; read {len(read_rows)} rows, calibration {len(cal_rows)} rows; "
        f"read forms {list(forms)}")
    paths_last = {"scale": os.path.join(od, "last.json"), "find": os.path.join(od, "last_find.json")}
    f_arms, f_toys = (os.path.join(od, f"{k}_last_m{N_OWN}_st{last_steps}.safetensors") for k in ("arms", "toys"))
    if any(not os.path.exists(paths_last[f]) for f in forms):
        if os.path.exists(f_arms) and os.path.exists(f_toys) and not os.path.exists(os.path.join(od, "last_ckpt.pt")):
            arms, toys = _load_stack(StackedArms(N_OWN, core.d), f_arms, core.dev), _load_stack(StackedToys(N_OWN, core.d), f_toys, core.dev)
            tr = None                                                      # trained in an earlier session: its record holds the training
            log(f"[members] the last site's stacks loaded from {os.path.basename(f_arms)}, {os.path.basename(f_toys)}")
        else:
            arms = StackedArms.build(N_OWN, core.d, f"SCALE|arm|s{seed}").to(core.dev)
            toys = StackedToys.build(N_OWN, core.d, f"SCALE|toy|s{seed}").to(core.dev)
            tr = train_last(core, {"arms": arms, "toys": toys}, last_steps, seed, packed, owned, never, log=log,
                            ckpt=os.path.join(od, "last_ckpt.pt"))
            _save_stack(f_arms, arms, dict(meta, site="last", steps=last_steps))
            _save_stack(f_toys, toys, dict(meta, site="last", steps=last_steps))
        tb = None
        if "scale" in forms and not os.path.exists(paths_last["scale"]):
            cal = calibrate(core, cal_rows, owned)
            log(f"[members] calibration: tau 0.999 per level {cal.get('tau', {}).get('0.999')}")
            rd = read_last(core, {"arms": arms, "toys": toys}, read_rows, cal, owned, never, seed, log=log)
            tb = text_bits(core, {"arms": arms, "toys": toys}, held, n_docs=2 if quick else 128)
            json.dump(dict(meta, train=tr, read=rd, text_bits=tb), open(paths_last["scale"], "w", encoding="utf-8"), indent=1)
            _rm(os.path.join(od, "last_ckpt.pt"))                          # the training is on record: the checkpoint goes
            _log_last(rd, tb, log)
            tr = None                                                      # one record carries the training
        if "find" in forms and not os.path.exists(paths_last["find"]):
            srcs = find_sources(core, "last")
            cal = calibrate_find(core, cal_rows, owned, srcs)
            log(f"[members] the two-pass calibration: tau 0.999 per level " + str({s: c["tau"]["0.999"] for s, c in cal.items()}))
            rd = read_last_find(core, {"arms": arms, "toys": toys}, read_rows, cal, owned, never, seed, srcs=srcs, log=log)
            if tb is None:
                tb = (json.load(open(paths_last["scale"], encoding="utf-8"))["text_bits"] if os.path.exists(paths_last["scale"]) else
                      text_bits(core, {"arms": arms, "toys": toys}, held, n_docs=2 if quick else 128))
            json.dump(dict(meta, train=tr, read=rd, text_bits=tb), open(paths_last["find"], "w", encoding="utf-8"), indent=1)
            _rm(os.path.join(od, "last_ckpt.pt"))
            _log_last_find(rd, tb, log)
        del arms, toys
    paths_mid = {"scale": os.path.join(od, "mid.json"), "find": os.path.join(od, "mid_find.json")}
    f_mid = os.path.join(od, f"arms_mid_b{MID_SITE}_m{M_MID}_st{mid_steps}.safetensors")
    if any(not os.path.exists(paths_mid[f]) for f in forms):
        if os.path.exists(f_mid) and not os.path.exists(os.path.join(od, "mid_ckpt.pt")):
            arms, tr = _load_stack(StackedArms(M_MID, core.d), f_mid, core.dev), None
            log(f"[members] the mid site's stack loaded from {os.path.basename(f_mid)}")
        else:
            arms = StackedArms.build(M_MID, core.d, f"SCALE|midarm|s{seed}").to(core.dev)
            tr = train_mid(core, arms, mid_steps, seed, packed, owned, log=log, ckpt=os.path.join(od, "mid_ckpt.pt"))
            _save_stack(f_mid, arms, dict(meta, site=f"after block {MID_SITE}", batch_steps=mid_steps))
        if "scale" in forms and not os.path.exists(paths_mid["scale"]):
            last = json.load(open(paths_last["scale"], encoding="utf-8"))
            rd = read_mid(core, arms, read_rows, last["read"]["calibration"], owned, never, seed, log=log, quick=quick)
            json.dump(dict(meta, train=tr, read=rd), open(paths_mid["scale"], "w", encoding="utf-8"), indent=1)
            _rm(os.path.join(od, "mid_ckpt.pt"))
            _log_mid(rd, log)
            tr = None
        if "find" in forms and not os.path.exists(paths_mid["find"]):
            last = json.load(open(paths_last["find"], encoding="utf-8"))
            rd = read_mid_find(core, arms, read_rows, last["read"]["calibration"], owned, never, seed, log=log, quick=quick)
            json.dump(dict(meta, train=tr, read=rd), open(paths_mid["find"], "w", encoding="utf-8"), indent=1)
            _rm(os.path.join(od, "mid_ckpt.pt"))
            _log_mid(rd, log)
        del arms
    out = dict(name=name, core=core.tag, wall_s=time.time() - t0)
    if forms != ("scale",):
        out["forms"] = list(forms)
    json.dump(out, open(done, "w", encoding="utf-8"), indent=1)
    if shipper is not None:
        shipper.submit(od, f"{hub_dir}/{name}")
    return out


def _load_stack(stk: nn.Module, path: str, dev) -> nn.Module:
    """A stack saved by _save_stack, loaded into a fresh stack of the same shape."""
    from safetensors.torch import load_file
    stk.load_state_dict(load_file(path))
    return stk.to(dev)


# ------------------------------------------------------------------ 0.1.72: READ 0 - stored cores and their stored stacks, no training
def fetch_stored(repo: str, core_tag: str, members_name: str, local_dir: str) -> tuple:
    """A stored grid core (bed/weights/<core_tag>/) and its members' folder (bed/scale_members/<members_name>/) from the Hub repo, into
    local_dir (tokenless for a public repo). -> (core folder, members folder)."""
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id=repo, local_dir=local_dir,
                      allow_patterns=[f"bed/weights/{core_tag}/*", f"bed/scale_members/{members_name}/*"])
    cf, mf = os.path.join(local_dir, "bed", "weights", core_tag), os.path.join(local_dir, "bed", "scale_members", members_name)
    for f in ("config.json", "model.safetensors", "scale.json"):
        assert os.path.exists(os.path.join(cf, f)), f"the stored core lacks {f}: {cf}"
    return cf, mf


def _continuity(stored: dict, new: dict) -> dict:
    """The stored read's registered cells (REST, G2, G0) against the same cells read again: the marks equal, the largest differences."""
    keys = ("merged_mean", "solo_mean", "unowned_raw_merged", "kl_unowned", "leak_over_own_write")
    n, same, worst = 0, 0, {k: 0.0 for k in keys}
    for name, r in stored["stacks"].items():
        for key, c in r["cells"].items():
            if "_FULL_" in key or name not in new["stacks"] or key not in new["stacks"][name]["cells"]:
                continue
            d = new["stacks"][name]["cells"][key]
            n += 1
            same += all(c[mk] == d[mk] for mk in ("PROTECTED", "QUIET", "EXACT"))
            for k in keys:
                worst[k] = max(worst[k], abs(float(c[k]) - float(d[k])))
    return dict(cells=n, marks_equal=same, largest_difference=worst)


def read0(core_folder: str, members_folder: str, out_path: str, seed: int, heldout_path: str, last_steps: int = 16000,
          read_batches: int = 64, calib_batches: int = 32, dev: str = "cuda", log=print, forms=("scale", "find"), look="ptr") -> dict:
    """READ 0: a stored grid core and its stored last-site stacks (48 real arms, 48 toy members), read with no training. "scale": the
    scale test's form again (its stored cells re-read: the continuity check against the members folder's last.json); "find": the
    two-pass form (the free-running index, the head's sources; 0.1.80: the look's on a core with a look head, and the continuity check
    against the folder's last_find.json). One record at out_path, resume-first (an existing record is returned)."""
    if os.path.exists(out_path):
        log(f"[read 0] {os.path.basename(out_path)}: done before")
        return json.load(open(out_path, encoding="utf-8"))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    t0 = time.time()
    core = Core(core_folder, dev, look=look)
    owned, never = owned_sets()
    held = load_heldout(heldout_path)
    read_rows = gauge_rows(seed, "read", read_batches * ROWS_LAST, held, abytes=core.abytes, layout=core.layout, key_form=core.key_form)
    cal_rows = gauge_rows(seed, "calib", calib_batches * ROWS_LAST, held[::-1], abytes=core.abytes, layout=core.layout, key_form=core.key_form)
    stacks = {"arms": _load_stack(StackedArms(N_OWN, core.d), os.path.join(members_folder, f"arms_last_m{N_OWN}_st{last_steps}.safetensors"), core.dev),
              "toys": _load_stack(StackedToys(N_OWN, core.d), os.path.join(members_folder, f"toys_last_m{N_OWN}_st{last_steps}.safetensors"), core.dev)}
    rec = dict(core=core.tag, members=os.path.basename(os.path.normpath(members_folder)), seed=seed, head_at=core.head_at,
               rotation="anchored" if core.anchor is not None else "house", head=core.head is not None, rows=len(read_rows),
               calib_rows=len(cal_rows), forms=list(forms))
    if core.look is not None:
        rec["look"] = dict(block=core.look[0], head=core.look[1], thr=LOOK_THR)                  # 0.1.80
    log(f"[read 0] {core.tag}: {rec['rotation']} rotation, head {'on' if rec['head'] else 'off'}; {len(read_rows)} read rows, "
        f"{len(cal_rows)} calibration rows; forms {list(forms)}"
        + (f"; the look at block {core.look[0]} head {core.look[1]}" if core.look is not None else ""))
    if "scale" in forms:
        cal = calibrate(core, cal_rows, owned)
        rd = read_last(core, stacks, read_rows, cal, owned, never, seed, log=log)
        rec["scale"] = rd
        sp = os.path.join(members_folder, "last.json")
        if os.path.exists(sp):
            rec["continuity"] = _continuity(json.load(open(sp, encoding="utf-8"))["read"], rd)
            log(f"[read 0] the scale form against the stored read: {rec['continuity']}")
        _log_last(rd, {}, log)
    if "find" in forms:
        srcs = find_sources(core, "last")
        calf = calibrate_find(core, cal_rows, owned, srcs)
        rdf = read_last_find(core, stacks, read_rows, calf, owned, never, seed, srcs=srcs, log=log)
        rec["find"] = rdf
        fp = os.path.join(members_folder, "last_find.json")
        if os.path.exists(fp):                                              # 0.1.80: the stored two-pass read's cells re-read
            rec["continuity_find"] = _continuity(json.load(open(fp, encoding="utf-8"))["read"], rdf)
            log(f"[read 0] the find form against the stored read: {rec['continuity_find']}")
        _log_last_find(rdf, None, log)
    rec["wall_s"] = time.time() - t0
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    json.dump(rec, open(out_path, "w", encoding="utf-8"), indent=1)
    log(f"[read 0] wrote {out_path} ({rec['wall_s'] / 60:.1f} min)")
    return rec


LOOK_VIEWS = ("LAYOUT", "INDEXF", "LOOKF", "LOOKQF")      # 0.1.80: the look read's views (the answers through each source's address)
LOOK_REQS = ("INDEX", "INDEXF", "LOOK", "LOOKF", "LOOKQF")  # its requests (pass 1 alone for INDEX and LOOK)


@torch.no_grad()
def look_read(core_folder: str, out_path: str, seed: int, heldout_path: str, look="ptr", read_batches: int = 64, hard_batches: int = 16,
              dev: str = "cuda", log=print) -> dict:
    """0.1.80, THE LOOK READ (no members): the core's read rows (the members' own: gauge_rows "read" on the held-out documents) and its
    key-hard rows (gauge_rows "hard", hard=True), every question at the null anchor in pass 1: per request source its address right
    (the asked record's), the share with no request and the share naming a wrong record; the answers whole through the views of
    LAYOUT, INDEXF, LOOKF and LOOKQF (the bare core with that source's address in the questions' streams); each look stream's counts
    (_look_out); the free-running fixed point. look: Core's look head ("ptr" or a (block, head) pair). One record at out_path,
    resume-first (an existing record is returned)."""
    if os.path.exists(out_path):
        log(f"[look] {os.path.basename(out_path)}: done before")
        return json.load(open(out_path, encoding="utf-8"))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    t0 = time.time()
    core = Core(core_folder, dev, look=look)
    assert core.look is not None, "the look read needs a look head: the core's pointer term (scale.json's ptr) or a given (block, head)"
    assert core.anchor is not None, "the look sources put their address in the anchored core's streams"
    owned, _ = owned_sets()
    held = load_heldout(heldout_path)
    sets = {"read": gauge_rows(seed, "read", read_batches * ROWS_LAST, held, abytes=core.abytes, layout=core.layout, key_form=core.key_form,
                               code=core.code)}
    if hard_batches:
        sets["hard"] = gauge_rows(seed, "hard", hard_batches * ROWS_LAST, held, abytes=core.abytes, hard=True, layout=core.layout,
                                  key_form=core.key_form, code=core.code)
    rec = dict(core=core.tag, seed=seed, version=_PKG_VERSION, look=dict(block=core.look[0], head=core.look[1], thr=LOOK_THR,
               from_ptr=look == "ptr"), sets={k: len(v) for k, v in sets.items()}, views=list(LOOK_VIEWS), requests=list(LOOK_REQS))
    n_all = sum(len(v) for v in sets.values())
    done_rows = 0
    log(f"[look] {core.tag}: block {core.look[0]} head {core.look[1]} ({'its pointer term' if look == 'ptr' else 'given'}); "
        + ", ".join(f"{k} {len(v)} rows" for k, v in sets.items()))
    for name, rows in sets.items():
        req = {s: dict(right=0, null=0) for s in LOOK_REQS}
        ans = {s: 0 for s in LOOK_VIEWS}
        lk = {sn: _look_new() for sn in LOOK_STREAMS}
        nq, passes, conv = 0, [], []
        swap_rng = np.random.default_rng([seed, gr.crc("SCALE|swap")])
        for bi in range(0, len(rows), ROWS_LAST):
            ctx = batch_views(core, rows[bi:bi + ROWS_LAST], owned, swap_rng, LOOK_VIEWS, gates=False, looks=("look_i",))
            dig, st = ctx["qt"]["dig"], ctx["stream"]
            nq += len(dig)
            for s in LOOK_REQS:
                sd = st[FIND_SOURCES[s][1]]
                req[s]["right"] += int((sd == dig).all(-1).sum())
                req[s]["null"] += int((sd == core.radix).any(-1).sum())
            for s in LOOK_VIEWS:
                v = ctx["views"][ctx["view_of"][s]]
                ans[s] += int((v["lb"].argmax(-1) == v["raw"]).all(-1).sum())
            for sn in LOOK_STREAMS:
                _look_add(lk[sn], ctx, sn)
            passes += list(ctx["passes"])
            conv += [bool(c) for c in ctx["converged"]]
            done_rows += len(ctx["metas"])
            if (bi // ROWS_LAST + 1) % 16 == 0 or bi + ROWS_LAST >= len(rows):
                el = time.time() - t0
                log(f"[look] {name}: {min(len(rows), bi + ROWS_LAST)}/{len(rows)} rows ({done_rows}/{n_all} in all): {el:.0f}s spent, "
                    f"~{el * (n_all - done_rows) / max(1, done_rows):.0f}s left")
        n = max(1, nq)
        rec[name] = dict(rows=len(rows), questions=nq,
                         request={s: dict(right=r["right"] / n, no_request=r["null"] / n, wrong=1.0 - (r["right"] + r["null"]) / n)
                                  for s, r in req.items()},
                         answers={s: a / n for s, a in ans.items()},
                         look={FIND_SOURCES_BY_STREAM[sn]: _look_out(a) for sn, a in lk.items()},
                         free_running=dict(passes_mean=float(np.mean(passes)), passes_max=int(max(passes)), converged=float(np.mean(conv))))
        r = rec[name]
        log(f"[look] {name} ({nq} questions): requests right " + " ".join(f"{s} {x['right']:.4f}" for s, x in r["request"].items())
            + "; no request " + " ".join(f"{s} {r['look'][s]['no_request']:.4f}" for s in SRC_LOOK)
            + "; a wrong record " + " ".join(f"{s} {r['look'][s]['wrong_record']:.4f}" for s in SRC_LOOK)
            + "; answers " + " ".join(f"{s} {x:.4f}" for s, x in r["answers"].items()))
    rec["wall_s"] = time.time() - t0
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    json.dump(rec, open(out_path, "w", encoding="utf-8"), indent=1)
    log(f"[look] wrote {out_path} ({rec['wall_s'] / 60:.1f} min)")
    return rec



def _log_last(rd: dict, tb: dict, log=print) -> None:
    log(f"[read last] index right {rd['index_right']:.4f}; requests right " + " ".join(f"{k} {v:.4f}" for k, v in rd["request_right"].items())
        + f"; bare owned {rd['bare']['owned']:.4f} unowned {rd['bare']['unowned']:.4f}")
    cal = rd.get("calibration") or {}
    if cal.get("tau_999_saturated"):
        log(f"[read last] FLAG the floor's tau saturated at 1 (the largest wrong digit score at 1 on {cal.get('largest_wrong_at_1')} of the "
            f"calibration questions a level): HEADPQ999 silences every member, so its cells' marks are vacuous")
    if rd.get("nan_fields"):
        log(f"[read last] FLAG {len(rd['nan_fields'])} non-finite fields, first {rd['nan_fields'][:3]}")
    for name, r in rd["stacks"].items():
        so = r["solo_per_owner"]
        log(f"[read last] {name}: solo ONE mean {np.mean(so):.4f} (10th percentile {np.quantile(so, 0.1):.4f}, min {min(so):.4f})")
        for key in ("LAYOUT_U5_REST_m48", "INDEX_U5_REST_m48", "HEADP_U5_REST_m48", "HEADPQ999_U5_REST_m48", "INDEXC_U5_REST_m48",
                    "G2_m48", "G0_m48"):
            c = r["cells"].get(key)
            if c:
                log(f"[read last] {name} {key}: solo {c['solo_mean']:.4f} merged {c['merged_mean']:.4f} unowned {c['unowned_raw_merged']:.4f}"
                    f" (bare {c['unowned_raw_bare']:.4f}) KL {c['kl_unowned']:.2e} leak/own {c['leak_over_own_write']:.2e} "
                    f"P/Q/E {c['PROTECTED']}/{c['QUIET']}/{c['EXACT']}")
    log("[text bits] " + " ".join(f"{k} {v:.4f}" for k, v in tb.items() if isinstance(v, float)))


def _log_last_find(rd: dict, tb: dict | None, log=print) -> None:
    """The two-pass read's lines: the free-running index, each source's address right, the floors, the headline cells."""
    d = rd["diag"]
    log(f"[read last, two-pass] the index written: teacher-forced row {d['written_index_right']:.4f}"
        + (f", free-running row {d['indexf_right']:.4f} (passes mean {d['passes']['mean']:.1f} max {d['passes']['max']}, converged "
           f"{d['converged']:.4f}, fixed point {d['fixed_point_agree']:.4f})" if "indexf_right" in d else "")
        + "; addresses right " + " ".join(f"{s} {r['request_right']:.4f}" for s, r in d["sources"].items())
        + "".join(f"; {s} agrees with the free-running index {r['argmax_agrees_with_indexf']:.4f}" for s, r in d["sources"].items()
                  if "argmax_agrees_with_indexf" in r))
    for s, c in (rd.get("calibration") or {}).items():
        if c.get("tau_999_saturated"):
            log(f"[read last, two-pass] FLAG {s}'s tau saturated at 1: its cells' marks are vacuous")
    if rd.get("nan_fields"):
        log(f"[read last, two-pass] FLAG {len(rd['nan_fields'])} non-finite fields, first {rd['nan_fields'][:3]}")
    for name, r in rd["stacks"].items():
        so = r["solo_per_owner"]
        log(f"[read last, two-pass] {name}: solo ONE mean {np.mean(so):.4f} (10th percentile {np.quantile(so, 0.1):.4f}, min {min(so):.4f})")
        for s in rd["sources"] + ["G2", "G0"]:
            key = f"{s}_U5_REST_m48" if s not in ("G2", "G0") else f"{s}_m48"
            c = r["cells"].get(key)
            if c:
                log(f"[read last, two-pass] {name} {key} ({c['view']}): solo {c['solo_mean']:.4f} merged {c['merged_mean']:.4f} unowned "
                    f"{c['unowned_raw_merged']:.4f} (bare {c['unowned_raw_bare']:.4f}) KL {c['kl_unowned']:.2e} leak/own "
                    f"{c['leak_over_own_write']:.2e} P/Q/E {c['PROTECTED']}/{c['QUIET']}/{c['EXACT']}")
    if tb:
        log("[text bits] " + " ".join(f"{k} {v:.4f}" for k, v in tb.items() if isinstance(v, float)))


def _log_mid(rd: dict, log=print) -> None:
    so = rd["solo_per_owner"]
    log(f"[read mid] solo ONE mean {np.mean(so):.4f} (min {min(so):.4f}); bare owned {rd['bare']['owned']:.4f}")
    for key, c in rd["cells"].items():
        if "_U5_REST" in key or key.startswith("G"):
            log(f"[read mid] {key}: solo {c['solo_mean']:.4f} merged {c['merged_mean']:.4f} unowned {c['unowned_raw_merged']:.4f} "
                f"KL {c['kl_unowned']:.2e} P/Q/E {c['PROTECTED']}/{c['QUIET']}/{c['EXACT']}")
