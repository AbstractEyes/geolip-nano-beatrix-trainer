# -*- coding: utf-8 -*-
"""capacity.py - THE BINDING'S CAPACITY READ (0.1.85). The key rotation (key_rotary) binds the whole key into the told head's eight key
pairs: at an index entry's coded SEP m the head scores a coded SEP n as

    s(m, n) = (1 / sqrt(hd)) sum_{j<8} Re[ q_j(m) conj(k_j(n)) e^{i (phi_j(m) - phi_j(n))} ]

and a 4,096-byte row holds at most ~92 competing coded SEPs. This read measures the binding past the row, with no training. The stored
core reads R rows at its own layout and keys' form (the look read's row maker, half of them with the key-hard keys); the trunk runs through
the told block only (the later blocks never touch the told head's q and k) and the look capture keeps the head's rotated q and k at every
coded SEP. Each index entry's true score (toward its asked record's SEP) is then set against

    REAL      pools: the coded SEPs (records' and index entries') of the OTHER rows whose keys differ from the entry's, N drawn without
              replacement (DRAWS draws an entry);
    EMULATED  pools: a pooled SEP's key content unrotated by its own phases and turned by the phases of a fresh key (to 10^6
              competitors; the same bank for every entry, never an entry's own key);

and the leak - one minus the true record's mass under the head's softmax over the true SEP and the N competitors, key pairs only -

    leak_N = W_N / (1 + W_N),   W_N = sum_n e^{s(m, n) - s(m, t)}

is recorded as its mean, median, 99th percentile and max over the entries, with the look error (leak >= 1/2) and the argmax error (a
competitor at or above the true score), beside the moment of random phases at the core's own margin

    E[leak_N] ~ N I0(s)^8 / (e^{8 s} + N I0(s)^8),   s = mean over entries of (true - mean of the row's other records) / 8.

The moment is an annealed average: W_N is dominated by rare near-aligned competitors, so at N well below their inverse probability the
typical leak (the median) sits far under it. EMULATED keys: the core's own form when its key space holds the bank ten times over (the long
keys: a cue word + six digits, 16,000,000), otherwise the form's shape with the word drawn as random lowercase letters of a cue word's
length (the standard keys: a cue word + three digits, 16,000 in all); every byte at every distance has its fixed phase, so the shape keeps
the SEP coded and the phase differences of distinct keys uniform, and the content is always a real record's. CPU or card; deterministic
for a given core, held-out text, R and seed."""
from __future__ import annotations

import json
import math
import os
import time

import numpy as np
import torch

from geolip.alephllm.data.special_tokens import CUE, SEP
from . import __version__ as _PKG_VERSION
from . import bed_frames as bf
from . import grid_rows as gr
from . import key_rotary as krm
from . import rotary_scale as rs

SEED_TAG = "CAPACITY|v1"
NS = (100, 1000, 10000, 100000, 1000000)                 # the competitor counts read (EMULATED: all; REAL: those the pool holds)
DRAWS = 8                                                # REAL draws an entry at each N
J = krm.J                                                # the key pairs (8)
KD = list(krm.KEY_DIMS)                                  # their dims in the head (rotate-half: pair j = dims j and j + PAIRS)
N_DIGITS = {"std": 3, "long": 6}
N_FORM = {"std": len(bf.WORDS) * 1000, "long": len(bf.WORDS) * 1000000}
LETTERS = np.frombuffer(b"abcdefghijklmnopqrstuvwxyz", dtype=np.uint8)


def turn(v: torch.Tensor, ang: torch.Tensor) -> torch.Tensor:
    """The key dims (..., 2 J) - first the J pairs' first components, then their second (the head's rotate-half order) - turned by the
    angles (..., J): (x1, x2) -> (x1 cos - x2 sin, x2 cos + x1 sin), the rotation rope_half applies."""
    c, s = torch.cos(ang), torch.sin(ang)
    x1, x2 = v[..., :J], v[..., J:]
    return torch.cat([x1 * c - x2 * s, x2 * c + x1 * s], dim=-1)


def key_phases(kr: krm.KeyRotary, keys: list, chunk: int = 20000) -> torch.Tensor:
    """The phases (len(keys), J) a key's SEP takes in a row (CUE key SEP), through the rotation's own phases(): every key at the reach
    (its SEP coded), the same float values wherever the key stands."""
    width = kr.k + 2
    out = []
    for i in range(0, len(keys), chunk):
        part = keys[i:i + chunk]
        a = np.full((len(part), width), 32, dtype=np.int64)          # left-padded with spaces: the SEP is the last column of every row
        for r, key in enumerate(part):
            assert 1 <= len(key) <= kr.k, (key, kr.k)
            b = bytes([CUE]) + key + bytes([SEP])
            a[r, width - len(b):] = np.frombuffer(b, dtype=np.uint8)
        phi, on = kr.phases(torch.from_numpy(a))
        assert bool(on[:, -1].all()), "a key's SEP is coded at the reach"
        out.append(phi[:, -1].float())
    return torch.cat(out)


def bank_keys(rng: np.random.Generator, key_form: str, n: int, exclude: set) -> list:
    """n distinct keys for the EMULATED bank, none in `exclude` (the read's own keys): the form itself when its space holds ten banks
    (the long keys), else the form's shape - a word of random lowercase letters of a cue word's length + the form's digits."""
    assert key_form in N_DIGITS, key_form
    out, seen = [], set(exclude)
    if N_FORM[key_form] >= 10 * n:
        while len(out) < n:
            for i in rng.choice(N_FORM[key_form], size=n - len(out) + 1024, replace=False):
                k = bf.key_long(int(i)) if key_form == "long" else bf.key_name(int(i))
                if k not in seen:
                    seen.add(k)
                    out.append(k)
                    if len(out) == n:
                        break
        return out
    lens = np.array([len(w) for w in bf.WORDS])
    nd = N_DIGITS[key_form]
    while len(out) < n:
        m = n - len(out) + 1024
        wl = lens[rng.integers(len(lens), size=m)]
        letters = LETTERS[rng.integers(26, size=(m, int(lens.max())))]
        nums = rng.integers(10 ** nd, size=m)
        for i in range(m):
            k = letters[i, :wl[i]].tobytes() + f"{int(nums[i]):0{nd}d}".encode()
            if k not in seen:
                seen.add(k)
                out.append(k)
                if len(out) == n:
                    break
    return out


def moment(s: float, n: int) -> float:
    """The moment of random phases: N I0(s)^J / (e^{J s} + N I0(s)^J), in log space."""
    a = math.log(n) + J * math.log(float(np.i0(s))) - J * s
    return 1.0 / (1.0 + math.exp(-a)) if a > -700 else 0.0


def read_rows(core: rs.Core, held: list, R: int, seed: int = 1) -> list:
    """R rows at the core's layout, keys' form and address form through the look read's row maker (the held-out documents as filler):
    the first half standard ('CAPACITY|v1'), the second key-hard ('CAPACITY|v1|hard'), each from the generator [seed, crc32('SCALE|<name>')]."""
    a = rs.gauge_rows(seed, SEED_TAG, R // 2, held, abytes=core.abytes, layout=core.layout, key_form=core.key_form)
    b = rs.gauge_rows(seed, SEED_TAG + "|hard", R - R // 2, held, abytes=core.abytes, hard=True, layout=core.layout,
                      key_form=core.key_form)
    return a + b


@torch.no_grad()
def capture(core: rs.Core, rows: list, batch: int = 4, log=print) -> dict:
    """The told head's rotated q and k (key dims) at the rows' coded SEPs, through the trunk to the told block: per index entry its q, its
    asked record's k, the row's other records' k (the in-row baseline); per pooled SEP (every record's and entry's) its k, its phases, its
    row and its key."""
    kr = core.key_rot
    assert kr is not None and core.look == (kr.block, kr.head), "the read needs the key rotation and its told head as the look"
    h, kd = kr.head, torch.tensor(KD)
    qs, ts, inrow, q_row, q_key = [], [], [], [], []
    pk, pphi, prow, pkey = [], [], [], []
    t0 = time.time()
    for i in range(0, len(rows), batch):
        part = rows[i:i + batch]
        metas = [mt for _, mt in part]
        x = rs.to_x(part, core.dev)[:, :-1]
        core.set_streams(metas)
        core.look_cap.on = True
        core.trunk(x, 0, kr.block)
        q = core.look_cap.q[:, h][..., kd].float().cpu()                 # (B, n, 2 J) rotated
        k = core.look_cap.k[:, h][..., kd].float().cpu()
        phi, on = kr.phi.float().cpu(), kr.on.cpu()
        core.look_cap.on = False
        core.look_cap.clear()
        core.clear()
        for b, mt in enumerate(metas):
            r = i + b
            rec_sep = [rc["v0"] - 1 for rc in mt["recs"]]
            ent_sep = [e["a_pos"] - 1 for e in mt["index"]]
            row = x[b]
            assert all(int(row[p]) == SEP and bool(on[b, p]) for p in rec_sep + ent_sep), "every record's and entry's SEP is coded"
            keys = [rc["key"].encode() for rc in mt["recs"]]
            for e, p in zip(mt["index"], ent_sep):
                o = e["rec"]
                qs.append(q[b, p])
                ts.append(k[b, rec_sep[o]])
                inrow.append(k[b, [rec_sep[j] for j in range(len(rec_sep)) if j != o]])
                q_row.append(r)
                q_key.append(keys[o])
            for p, key in [(rec_sep[j], keys[j]) for j in range(len(rec_sep))] + [(p, keys[e["rec"]]) for e, p in zip(mt["index"], ent_sep)]:
                pk.append(k[b, p])
                pphi.append(phi[b, p])
                prow.append(r)
                pkey.append(key)
        if log and (i // batch) % 8 == 0:
            done = min(i + batch, len(rows))
            el = time.time() - t0
            log(f"[capacity] {core.tag[:40]}: {done}/{len(rows)} rows read, {el:.0f} s, ~{el / done * (len(rows) - done):.0f} s left")
    return dict(q=torch.stack(qs), t=torch.stack(ts), inrow=torch.stack(inrow), q_row=torch.tensor(q_row), q_key=q_key,
                pool_k=torch.stack(pk), pool_phi=torch.stack(pphi), pool_row=torch.tensor(prow), pool_key=pkey)


def _stats(W: np.ndarray, maxd: np.ndarray) -> dict:
    """The leak statistics over the entries from W_N (Q,) and the largest competitor score minus the true (Q,)."""
    leak = W / (1.0 + W)
    return dict(mean=float(leak.mean()), median=float(np.median(leak)), p99=float(np.quantile(leak, 0.99)), max=float(leak.max()),
                look_error=float((W >= 1.0).mean()), argmax_error=float((maxd >= 0.0).mean()), entries=int(len(W)))


@torch.no_grad()
def emulated(q: torch.Tensor, st: torch.Tensor, bank: torch.Tensor, ns=NS, chunk: int = 32) -> tuple:
    """W_N and the largest (s_n - s_t) at every N for each entry, against the bank's first N vectors (the same bank for every entry)."""
    sc = 1.0 / math.sqrt(2 * krm.ar.PAIRS)
    bounds = [0] + list(ns)
    W = np.zeros((len(q), len(ns)))
    M = np.full((len(q), len(ns)), -np.inf)
    bt = bank.T.contiguous()
    for i in range(0, len(q), chunk):
        d = (q[i:i + chunk] @ bt) * sc - st[i:i + chunk, None]          # (C, bank)
        acc_w = torch.zeros(len(d), dtype=torch.float64)
        acc_m = torch.full((len(d),), -float("inf"))
        for g in range(len(ns)):
            seg = d[:, bounds[g]:bounds[g + 1]]
            acc_w += torch.exp(seg).sum(1, dtype=torch.float64)
            acc_m = torch.maximum(acc_m, seg.max(1).values)
            W[i:i + chunk, g] = acc_w.numpy()
            M[i:i + chunk, g] = acc_m.numpy()
    return W, M


@torch.no_grad()
def real(cap: dict, st: torch.Tensor, ns, draws: int = DRAWS, seed: int = 1, chunk: int = 64) -> tuple:
    """W_N and the largest (s_n - s_t) at every N each pool can hold, for each entry and draw: N of the entry's allowed pool (the other
    rows' coded SEPs whose keys differ from the entry's) without replacement. -> (W (Q, draws, len(ns)), M (same), the smallest pool)."""
    sc = 1.0 / math.sqrt(2 * krm.ar.PAIRS)
    g = torch.Generator().manual_seed(int(gr.crc(f"{SEED_TAG}|real|{seed}")))
    keys = {k: i for i, k in enumerate(sorted(set(cap["pool_key"]) | set(cap["q_key"])))}
    pkey = torch.tensor([keys[k] for k in cap["pool_key"]])
    qkey = torch.tensor([keys[k] for k in cap["q_key"]])
    Q, P = len(cap["q"]), len(cap["pool_k"])
    W = np.zeros((Q, draws, len(ns)))
    M = np.full((Q, draws, len(ns)), -np.inf)
    smallest = P
    pt = cap["pool_k"].T.contiguous()
    for i in range(0, Q, chunk):
        sl = slice(i, min(i + chunk, Q))
        d = (cap["q"][sl] @ pt) * sc - st[sl, None]                      # (C, P)
        ok = (cap["pool_row"][None, :] != cap["q_row"][sl, None]) & (pkey[None, :] != qkey[sl, None])
        smallest = min(smallest, int(ok.sum(1).min()))
        w = torch.where(ok, torch.exp(d.double()), torch.zeros((), dtype=torch.float64))
        dm = torch.where(ok, d, torch.full_like(d, -float("inf")))
        for r in range(draws):
            u = torch.rand(d.shape, generator=g)
            u = torch.where(ok, u, torch.full_like(u, 2.0))                # the disallowed sort last
            order = torch.argsort(u, dim=1)
            cw = torch.cumsum(torch.gather(w, 1, order), dim=1)
            cm = torch.cummax(torch.gather(dm, 1, order), dim=1).values
            for j, n in enumerate(ns):
                if n <= cw.shape[1]:
                    W[sl, r, j] = cw[:, n - 1].numpy()
                    M[sl, r, j] = cm[:, n - 1].numpy()
    return W, M, smallest


def record_name(tag: str, R: int, seed: int) -> str:
    return f"{tag}_capacity_r{R}_s{seed}.json"


def read_core(folder: str, held: list, R: int = 192, seed: int = 1, ns=NS, draws: int = DRAWS, bank_n: int | None = None,
              batch: int = 4, dev: str = "cpu", log=print) -> dict:
    """The capacity read of one stored core (its weights folder): the record above."""
    t0 = time.time()
    torch.backends.cuda.matmul.allow_tf32 = False
    core = rs.Core(folder, dev)
    kr = core.key_rot
    rows = read_rows(core, held, R, seed)
    cap = capture(core, rows, batch=batch, log=log)
    t_cap = time.time() - t0
    sc = 1.0 / math.sqrt(2 * krm.ar.PAIRS)
    st = (cap["q"] * cap["t"]).sum(-1) * sc                                          # the true scores (Q,)
    so = (cap["q"][:, None, :] * cap["inrow"]).sum(-1) * sc                           # the row's other records (Q, comps - 1)
    margin = st - so.mean(1)
    s_core = float(margin.mean()) / J
    qa = torch.sqrt(cap["q"][:, :J] ** 2 + cap["q"][:, J:] ** 2)                      # the true pairs' magnitudes |q_j| |k_j| / sqrt(hd)
    ta = torch.sqrt(cap["t"][:, :J] ** 2 + cap["t"][:, J:] ** 2)
    pair_mag = (qa * ta * sc).mean(0)
    w_in = torch.exp((so - st[:, None]).double()).sum(1).numpy()
    # EMULATED: the pooled contents unrotated by their own phases, turned by fresh keys' phases
    nb = int(bank_n or max(ns))
    rng = np.random.default_rng(gr.crc(f"{SEED_TAG}|bank|{seed}|{core.key_form}"))
    k0 = turn(cap["pool_k"], -cap["pool_phi"])
    bkeys = bank_keys(rng, core.key_form, nb, set(cap["q_key"]) | set(cap["pool_key"]))
    bphi = key_phases(kr, bkeys)
    content = torch.from_numpy(rng.integers(len(k0), size=nb))
    bank = turn(k0[content], bphi)
    t_bank = time.time() - t0
    W_e, M_e = emulated(cap["q"], st, bank, [n for n in ns if n <= nb])
    W_r, M_r, smallest = real(cap, st, [n for n in ns if n <= len(cap["pool_k"])], draws=draws, seed=seed)
    out_e, out_r = {}, {}
    for j, n in enumerate([n for n in ns if n <= nb]):
        out_e[str(n)] = dict(_stats(W_e[:, j], M_e[:, j]), moment=moment(s_core, n))
    for j, n in enumerate([n for n in ns if n <= len(cap["pool_k"])]):
        if n <= smallest:
            out_r[str(n)] = _stats(W_r[:, :, j].reshape(-1), M_r[:, :, j].reshape(-1))
    pct = lambda v, p: float(np.quantile(v.numpy() if torch.is_tensor(v) else v, p))   # noqa: E731
    return dict(core=core.tag, version=_PKG_VERSION, seed=seed, rows=R, hard_rows=R - R // 2, layout=core.layout, key_form=core.key_form,
                reach=kr.k, told=dict(block=kr.block, head=kr.head), entries=int(len(st)), pool=int(len(cap["pool_k"])),
                pool_smallest_allowed=int(smallest), bank=nb, draws=draws,
                bank_keys="form" if N_FORM[core.key_form] >= 10 * nb else "shape",
                s_core=s_core, margin=dict(mean=float(margin.mean()), p01=pct(margin, .01), p10=pct(margin, .1), p50=pct(margin, .5)),
                true_score=dict(mean=float(st.mean()), p01=pct(st, .01), p10=pct(st, .1), p50=pct(st, .5)),
                pair_magnitude=[float(v) for v in pair_mag], inrow=_stats(w_in, (so - st[:, None]).max(1).values.numpy()),
                emulated=out_e, real=out_r, wall=dict(capture=round(t_cap, 1), bank=round(t_bank - t_cap, 1),
                                                      total=round(time.time() - t0, 1)))


def run(folder: str, held: list, out_dir: str, R: int = 192, seed: int = 1, log=print, **kw) -> dict:
    """Resume-first: the record at out_dir/<core>_capacity_r<R>_s<seed>.json is returned when it exists, else read and written."""
    tag = os.path.basename(os.path.normpath(folder))
    path = os.path.join(out_dir, record_name(tag, R, seed))
    if os.path.exists(path):
        if log:
            log(f"[capacity] {tag[:40]}: done before")
        return json.load(open(path, encoding="utf-8"))
    rec = read_core(folder, held, R=R, seed=seed, log=log, **kw)
    os.makedirs(out_dir, exist_ok=True)
    tmp = path + ".tmp"
    json.dump(rec, open(tmp, "w", encoding="utf-8"), indent=1)
    os.replace(tmp, path)
    return rec
