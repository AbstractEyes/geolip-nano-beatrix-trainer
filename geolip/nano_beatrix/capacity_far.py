# -*- coding: utf-8 -*-
"""capacity_far.py - THE FAR READ (0.1.89): the tail read (capacity.tail_core) carried past a million competing keys. tail_core holds the
whole EMULATED bank and a (32, N) block of scores in memory, so it stops near 10^6 on a small runtime; the far read computes the same
numbers streamed over the bank, on a grid of N to 10^8 (FAR_NS):

    THE BANK    capacity.bank_of's draw exactly - the same generator and the same draws in the same order. The keys of the shape form (a
                word of random lowercase letters of a cue word's length and the form's digits) are drawn in numpy rounds and kept by
                first occurrence, as bank_keys' loop keeps them (shape_bank: key for key); their phases are the gathers key_phases sums
                at the key's end-marker, summed in the same order (shape_phases: bit for bit); the pooled contents are drawn after them.
                The keys and phases depend on the read's rows (their keys are excluded), the seed, the key form and the rotation's
                table only, so the cores of one layout, form and rotation share them (far_bank's cache).
    THE SCORES  per block of the bank, for every index entry m: the base and M conditions' scores against the true record, their
                largest kept and e^{s(m, n) - s(m, t)} summed in float64 into each grid segment, and the base's argmax over the whole
                bank; W_N and the largest at N are the segments' running sums and maxima - capacity.emulated's numbers, up to the
                float64 sum's order and the float32 product's blocking.
    E           from the base: E sets every true score to the core's median true score med, so
                    W_N^E(m) = e^{s(m, t) - med} W_N(m),    max_N^E(m) = max_N(m) + (s(m, t) - med)
                (tail_core reads it as a third pass; the two agree to the float32 rounding of the score's difference).

The record (<core>_capfar_r<R>_s<seed>.json) carries tail_core's fields at every N of the grid, and per condition the onset (the smallest
N with a look error) and the half point (the N where the look error reaches one half, log-linear between the grid points around it;
None past the grid, with the error at the largest N), the host (CPUs, threads, the block) and the walls. compare_tail sets a far record
against a stored tail record of the same bank (the replication gate). Progress lines carry the job size, the time spent and the time
left. CPU; deterministic for a given core, held-out text, R, seed and grid."""
from __future__ import annotations

import hashlib
import json
import math
import os
import time

import numpy as np
import torch

from . import __version__ as _PKG_VERSION
from . import bed_frames as bf
from . import capacity as cp
from . import grid_rows as gr
from . import key_code as kcm
from . import key_rotary as krm
from . import rotary_scale as rs

FAR_NS = (1000, 4096, 10 ** 4, 10 ** 5, 10 ** 6, 2 * 10 ** 6, 5 * 10 ** 6, 10 ** 7, 2 * 10 ** 7, 5 * 10 ** 7, 10 ** 8)
BLOCK = 1 << 16                                          # the bank's block: (entries, BLOCK) float32 scores at a time
CODE_BASE = 37                                           # a key's code: its bytes in base 37 (a..z 1..26, 0..9 27..36; no byte 0)
MAX_CODE_LEN = 12                                        # 37^12 < 2^64: a key of at most 12 bytes codes to one uint64
_VAL = np.zeros(256, dtype=np.uint64)
_VAL[np.frombuffer(b"abcdefghijklmnopqrstuvwxyz", dtype=np.uint8)] = np.arange(1, 27, dtype=np.uint64)
_VAL[np.frombuffer(b"0123456789", dtype=np.uint8)] = np.arange(27, 37, dtype=np.uint64)


class ShapeKeys:
    """Keys of the shape form as arrays: letters (n, L) uint8 (the word's letters first, the rest unused), wl (n,) the word's length,
    nums (n,) the digits' value, nd the digits. key(i) is the key's bytes."""

    def __init__(self, letters: np.ndarray, wl: np.ndarray, nums: np.ndarray, nd: int) -> None:
        assert letters.ndim == 2 and len(letters) == len(wl) == len(nums), (letters.shape, wl.shape, nums.shape)
        self.letters, self.wl, self.nums, self.nd = letters, wl, nums, int(nd)

    def __len__(self) -> int:
        return len(self.wl)

    def key(self, i: int) -> bytes:
        return self.letters[i, :int(self.wl[i])].tobytes() + f"{int(self.nums[i]):0{self.nd}d}".encode()

    def lengths(self) -> np.ndarray:
        return self.wl.astype(np.int64) + self.nd


def bytes_code(key: bytes) -> int | None:
    """A key's code (its bytes in base 37), or None when a byte lies outside a..z / 0..9 or it is longer than MAX_CODE_LEN: no key of the
    shape form equals it."""
    if not 1 <= len(key) <= MAX_CODE_LEN:
        return None
    v = 0
    for b in key:
        d = int(_VAL[b])
        if d == 0:
            return None
        v = v * CODE_BASE + d
    return v


def shape_codes(letters: np.ndarray, wl: np.ndarray, nums: np.ndarray, nd: int) -> np.ndarray:
    """Each shape key's code (uint64): the word's letters, then the digits (the most significant first), in base 37 - bytes_code of the
    key's bytes; distinct keys, distinct codes."""
    assert letters.shape[1] + nd <= MAX_CODE_LEN, (letters.shape, nd)
    base = np.uint64(CODE_BASE)
    code = np.zeros(len(wl), dtype=np.uint64)
    for c in range(letters.shape[1]):
        code = np.where(wl > c, code * base + _VAL[letters[:, c]], code)
    for i in range(nd):
        dig = (nums // (10 ** (nd - 1 - i))) % 10
        code = code * base + (np.uint64(27) + dig.astype(np.uint64))
    return code


def shape_bank(rng: np.random.Generator, key_form: str, n: int, exclude: set) -> ShapeKeys:
    """capacity.bank_keys' shape path in numpy: n distinct keys, none in exclude, drawn from rng exactly as the loop draws them (a round of
    m = n - kept + 1024: the words' lengths, the letters (m, the longest word), the digits) and kept in the loop's order - each round's
    first occurrences of keys neither kept before nor excluded, in draw order, up to n."""
    assert key_form in cp.N_DIGITS, key_form
    assert cp.N_FORM[key_form] < 10 * n, "a key space ten banks deep draws the form itself: capacity.bank_keys"
    lens = np.array([len(w) for w in bf.WORDS])
    nd = cp.N_DIGITS[key_form]
    L = int(lens.max())
    seen = np.unique(np.array([c for c in (bytes_code(k) for k in exclude) if c is not None], dtype=np.uint64))
    parts, have = [], 0
    while have < n:
        m = n - have + 1024
        wl = lens[rng.integers(len(lens), size=m)]
        letters = cp.LETTERS[rng.integers(26, size=(m, L))]
        nums = rng.integers(10 ** nd, size=m)
        codes = shape_codes(letters, wl, nums, nd)
        u, first = np.unique(codes, return_index=True)
        take = np.sort(first[~np.isin(u, seen, assume_unique=True)])[:n - have]
        parts.append((letters[take], wl[take].astype(np.uint8), nums[take]))
        seen = np.union1d(seen, codes[take])
        have += len(take)
    return ShapeKeys(*(np.concatenate([p[i] for p in parts]) for i in range(3)), nd)


@torch.no_grad()
def shape_phases(kr: krm.KeyRotary, keys: ShapeKeys, chunk: int = 1 << 20) -> torch.Tensor:
    """capacity.key_phases for shape keys, bit for bit: the phases (n, pairs) at each key's end-marker - column k the byte k + 1 before
    the SEP inside the key (the digits from the last, then the word's letters from its last), the null row past it - the gathers summed in
    key_phases' order, then mod 2 pi."""
    K = kr.k
    ln = keys.lengths()
    assert len(keys) and 1 <= int(ln.min()) and int(ln.max()) <= K, "every key's SEP coded at the reach"
    t = torch.cat([kr.table, kr.table.new_zeros(kr.pairs, K, 1)], dim=-1)       # (pairs, K, VOCAB + 1), as KeyRotary.phases builds it
    out = torch.empty(len(keys), kr.pairs)
    nd = keys.nd
    for a in range(0, len(keys), chunk):
        b = min(a + chunk, len(keys))
        wl = keys.wl[a:b].astype(np.int64)
        let = keys.letters[a:b]
        off = np.full((b - a, K), kcm.VOCAB, dtype=np.int64)
        for k in range(min(nd, K)):                                                # the digits, the last first
            off[:, k] = ord("0") + (keys.nums[a:b] // (10 ** k)) % 10
        rows = np.arange(b - a)
        for k in range(nd, K):                                                     # the word's letters, its last first
            j = wl - 1 - (k - nd)
            inw = j >= 0
            off[inw, k] = let[rows[inw], j[inw]]
        off_t = torch.from_numpy(off)
        phi = None
        for k in range(K):
            p = t[:, k][:, off_t[:, k]]                                            # (pairs, b - a)
            phi = p if phi is None else phi + p
        out[a:b] = torch.remainder(phi, krm.TWO_PI).T
    return out


def bank_fingerprint(core: rs.Core, cap: dict, seed: int, nb: int) -> str:
    """The bank's identity: the key form, the seed, the bank's size, the pool's size, the rotation's reach, pairs and table, and the read's
    keys (the excluded set) - two cores of one fingerprint draw one bank of keys, phases and pooled indices."""
    kr = core.key_rot
    h = hashlib.sha256()
    for part in (core.key_form, seed, nb, len(cap["pool_k"]), kr.k, kr.pairs):
        h.update(f"{part}|".encode())
    h.update(kr.table.detach().cpu().contiguous().numpy().tobytes())
    for k in sorted(set(cap["q_key"]) | set(cap["pool_key"])):
        h.update(k + b"|")
    return h.hexdigest()


@torch.no_grad()
def far_bank(core: rs.Core, cap: dict, seed: int, nb: int, cache: dict | None = None) -> tuple:
    """capacity.bank_of's draw without the vectors: (k0 (pool, 2 J), bphi (nb, J), content (nb,), cached) - the bank's vector n is
    turn(k0[content[n]], bphi[n]). The shape path vectorized (shape_bank, shape_phases), the form path capacity's own. cache (a dict):
    the phases and the pooled indices kept under the bank's fingerprint."""
    kr = core.key_rot
    k0 = cp.turn(cap["pool_k"], -cap["pool_phi"])
    fp = bank_fingerprint(core, cap, seed, nb)
    if cache is not None and fp in cache:
        return (k0, *cache[fp], True)
    rng = np.random.default_rng(gr.crc(f"{cp.SEED_TAG}|bank|{seed}|{core.key_form}"))
    exclude = set(cap["q_key"]) | set(cap["pool_key"])
    if cp.N_FORM[core.key_form] >= 10 * nb:
        bphi = cp.key_phases(kr, cp.bank_keys(rng, core.key_form, nb, exclude))
    else:
        bphi = shape_phases(kr, shape_bank(rng, core.key_form, nb, exclude))
    content = torch.from_numpy(rng.integers(len(k0), size=nb))
    if cache is not None:
        cache[fp] = (bphi, content)
    return k0, bphi, content, False


@torch.no_grad()
def far_scores(q: torch.Tensor, t: torch.Tensor, st: torch.Tensor, k0: torch.Tensor, bphi: torch.Tensor, content: torch.Tensor, ns,
               block: int = BLOCK, log=None, tag: str = "") -> dict:
    """W_N and the largest (s_n - s_t) at every N of ns for the base and M conditions, streamed over the bank in blocks; E from the base;
    the base's argmax over the whole bank and its value. -> {"base": (W, M), "M": (W, M), "E": (W, M), "arg": (Q,), "best": (Q,)}."""
    sc = 1.0 / math.sqrt(2 * krm.ar.PAIRS)
    ns = [int(n) for n in ns]
    assert ns == sorted(ns) and len(set(ns)) == len(ns) and ns[0] >= 1, ns
    nb = ns[-1]
    assert len(bphi) >= nb and len(content) >= nb, (len(bphi), len(content), nb)
    ta = cp.pair_norms(t)
    qm = q * torch.cat([ta, ta], dim=-1)
    bounds = [0] + ns
    Q, G = len(q), len(ns)
    seg = {c: torch.zeros(Q, G, dtype=torch.float64) for c in ("base", "M")}
    top = {c: torch.full((Q, G), -float("inf")) for c in ("base", "M")}
    best = torch.full((Q,), -float("inf"))
    arg = torch.zeros(Q, dtype=torch.long)
    t0, last = time.time(), 0.0
    for a in range(0, nb, block):
        b = min(a + block, nb)
        bank = cp.turn(k0[content[a:b]], bphi[a:b])                               # the bank's vectors a .. b - 1, as bank_of turns them
        for cond, qq, kk in (("base", q, bank), ("M", qm, cp.unit_pairs(bank))):
            d = (qq @ kk.T) * sc - st[:, None]                                    # (Q, b - a)
            spans = [(g, max(bounds[g], a) - a, min(bounds[g + 1], b) - a) for g in range(G)]
            spans = [(g, lo, hi) for g, lo, hi in spans if lo < hi]
            if cond == "base":
                v, i = d.max(1)
                up = v > best
                best = torch.where(up, v, best)
                arg = torch.where(up, i + a, arg)
            for g, lo, hi in spans:
                mx = v if (cond == "base" and lo == 0 and hi == b - a) else d[:, lo:hi].max(1).values
                top[cond][:, g] = torch.maximum(top[cond][:, g], mx)
            d.exp_()                                                              # the scores' exponentials in place (float32, as emulated)
            for g, lo, hi in spans:
                seg[cond][:, g] += d[:, lo:hi].sum(1, dtype=torch.float64)
        el = time.time() - t0
        if log and (el - last >= 60 or b == nb):
            last = el
            log(f"[capacity far] {tag[:40]}: {b:,}/{nb:,} competitors, {el:.0f} s, ~{el / b * (nb - b):.0f} s left")
    out = {c: (torch.cumsum(seg[c], dim=1).numpy(), torch.cummax(top[c], dim=1).values.numpy()) for c in ("base", "M")}
    med = float(np.median(st.numpy()))
    shift = st.double() - med
    out["E"] = (out["base"][0] * torch.exp(shift).numpy()[:, None], out["base"][1] + (st - med).numpy()[:, None])
    out["arg"], out["best"] = arg, best
    return out


def onset_half(ns, looks) -> dict:
    """The smallest N with a look error above 0 (None: none on the grid) and the N where it reaches one half, log-linear between the grid
    points around it (None: not on the grid; the error at the largest N beside it)."""
    ns, looks = [int(n) for n in ns], [float(e) for e in looks]
    onset = next((n for n, e in zip(ns, looks) if e > 0), None)
    half = None
    for j, (n, e) in enumerate(zip(ns, looks)):
        if e >= 0.5:
            if j == 0:
                half = float(n)
            else:
                n0, e0 = ns[j - 1], looks[j - 1]
                f = (0.5 - e0) / (e - e0)
                half = float(10 ** (math.log10(n0) + f * (math.log10(n) - math.log10(n0))))
            break
    return dict(onset=onset, half=half, look_at_max=looks[-1])


@torch.no_grad()
def far_core(folder: str, held: list, R: int = 192, seed: int = 1, ns=FAR_NS, block: int = BLOCK, batch: int = 4, dev: str = "cpu",
             cache: dict | None = None, log=print) -> dict:
    """THE FAR READ of one stored core: tail_core's rows, bank and record at every N of ns, streamed (the module's head)."""
    t0 = time.time()
    torch.backends.cuda.matmul.allow_tf32 = False
    ns = tuple(int(n) for n in ns)
    core = rs.Core(folder, dev)
    kr = core.key_rot
    rows = cp.read_rows(core, held, R, seed)
    cap = cp.capture(core, rows, batch=batch, log=log)
    t_cap = time.time() - t0
    sc = 1.0 / math.sqrt(2 * krm.ar.PAIRS)
    st = (cap["q"] * cap["t"]).sum(-1) * sc
    so = (cap["q"][:, None, :] * cap["inrow"]).sum(-1) * sc
    margin = st - so.mean(1)
    s_core = float(margin.mean()) / kr.pairs
    is_rec, own, row_recs, off = [], [], [], 0                    # the pool in capture's order: a row's records' SEPs, then its entries'
    for _, mt in rows:
        nr, ne = len(mt["recs"]), len(mt["index"])
        is_rec += [True] * nr + [False] * ne
        own += [off + e["rec"] for e in mt["index"]]
        row_recs += [nr] * ne
        off += nr + ne
    is_rec, own = torch.tensor(is_rec), torch.tensor(own)
    assert off == len(cap["pool_k"]) and len(own) == len(st) and torch.equal(cap["pool_k"][own], cap["t"]), "the pool's layout"
    nb = int(max(ns))
    k0, bphi, content, cached = far_bank(core, cap, seed, nb, cache)
    t_bank = time.time() - t0
    if log:
        log(f"[capacity far] {core.tag[:40]}: the bank of {nb:,} ({'cached' if cached else 'drawn'}) in {t_bank - t_cap:.0f} s")
    sco = far_scores(cap["q"], cap["t"], st, k0, bphi, content, ns, block=block, log=log, tag=core.tag)
    t_sco = time.time() - t0
    rank_m, rank_t = cp._pct_rank(margin.numpy()), cp._pct_rank(st.numpy())
    out, far = {}, {}
    for name in ("base", "M", "E"):
        W, M = sco[name]
        out[name] = {}
        for j, n in enumerate(ns):
            look = W[:, j] >= 1.0
            out[name][str(n)] = dict(cp._stats(W[:, j], M[:, j]), moment=cp.moment(s_core, n, kr.pairs),
                                     look_by_margin_pct=cp._by_band(rank_m, look), look_by_true_pct=cp._by_band(rank_t, look))
        far[name] = onset_half(ns, [out[name][str(n)]["look_error"] for n in ns])
    ta = cp.pair_norms(cap["t"])
    pl = cp.pair_norms(cap["pool_k"])
    ratio = pl / ta.median(0).values
    qs = lambda v, p: [float(x) for x in torch.quantile(v, p, dim=0)]   # noqa: E731
    pool_ratio = {g: dict(n=int(sel.sum()), p50=qs(ratio[sel], .5), p99=qs(ratio[sel], .99), max=[float(x) for x in ratio[sel].max(0).values])
                  for g, sel in (("all", torch.ones(len(pl), dtype=torch.bool)), ("records", is_rec), ("entries", ~is_rec))}
    P, n_rec, rr = len(pl), int(is_rec.sum()), np.array(row_recs, dtype=np.float64)
    W_b = sco["base"][0][:, -1]
    idx = np.nonzero(W_b >= 1.0)[0]
    kinds, shares, lmean, lmax = dict.fromkeys(cp.SOURCES, 0), [], [], []
    for m in idx.tolist():
        s = int(content[int(sco["arg"][m])])
        same_row = int(cap["pool_row"][s]) == int(cap["q_row"][m])
        kind = (cp.SOURCES[0] if s == int(own[m]) else cp.SOURCES[1] if bool(is_rec[s]) and same_row else
                cp.SOURCES[2] if bool(is_rec[s]) else cp.SOURCES[3])
        kinds[kind] += 1
        shares.append(float(np.exp(np.float64(sco["best"][m].item())) / W_b[m]))
        r = pl[s] / ta[m]
        lmean.append(float(r.mean()))
        lmax.append(float(r.max()))
    med = lambda v: float(np.median(v)) if len(v) else None   # noqa: E731
    top = dict(n=int(len(idx)), source=kinds,
               bank_share={cp.SOURCES[0]: 1.0 / P, cp.SOURCES[1]: float(np.mean(rr - 1)) / P, cp.SOURCES[2]: float(np.mean(n_rec - rr)) / P,
                           cp.SOURCES[3]: (P - n_rec) / P},
               share_of_W_median=med(shares), length_ratio_mean_median=med(lmean), length_ratio_max_median=med(lmax))
    return dict(core=core.tag, version=_PKG_VERSION, seed=seed, rows=R, hard_rows=R - R // 2, layout=core.layout, key_form=core.key_form,
                reach=kr.k, told=cp._told(kr), entries=int(len(st)), pool=P, bank=nb,
                bank_keys="form" if cp.N_FORM[core.key_form] >= 10 * nb else "shape", ns=list(ns), s_core=s_core,
                true_score_median=float(np.median(st.numpy())), base=out["base"], M=out["M"], E=out["E"], far=far,
                pool_length_ratio=pool_ratio, true_pair_length=dict(p01=qs(ta, .01), p50=qs(ta, .5), p99=qs(ta, .99)), top_competitor=top,
                host=dict(cpus=os.cpu_count(), threads=torch.get_num_threads(), block=block, bank_cached=cached),
                wall=dict(capture=round(t_cap, 1), bank=round(t_bank - t_cap, 1), scores=round(t_sco - t_bank, 1),
                          total=round(time.time() - t0, 1)))


def far_name(tag: str, R: int, seed: int) -> str:
    return f"{tag}_capfar_r{R}_s{seed}.json"


def run_far(folder: str, held: list, out_dir: str, R: int = 192, seed: int = 1, log=print, **kw) -> dict:
    """Resume-first: the record at out_dir/<core>_capfar_r<R>_s<seed>.json is returned when it exists, else read and written."""
    tag = os.path.basename(os.path.normpath(folder))
    path = os.path.join(out_dir, far_name(tag, R, seed))
    if os.path.exists(path):
        if log:
            log(f"[capacity far] {tag[:40]}: done before")
        return json.load(open(path, encoding="utf-8"))
    rec = far_core(folder, held, R=R, seed=seed, log=log, **kw)
    os.makedirs(out_dir, exist_ok=True)
    tmp = path + ".tmp"
    json.dump(rec, open(tmp, "w", encoding="utf-8"), indent=1)
    os.replace(tmp, path)
    return rec


def _close(a: float, b: float, rtol: float) -> bool:
    return a == b or abs(a - b) <= rtol * max(abs(a), abs(b))


def compare_tail(far: dict, tail: dict, rtol: float = 1e-5, rtol_e: float = 1e-4) -> list:
    """A far record against a tail record of the same rows and bank (the replication gate): every N of the tail's grid - the base and M
    statistics within rtol relative, E's within rtol_e, the look and argmax errors, the entries and the band counts exact, the moment to
    1e-9; the margin, the true scores' median, the pooled and true pair lengths within rtol; the strongest competitors' count and sources
    exact, their medians within rtol. -> the mismatches (empty: the gate passes)."""
    bad = []
    for f in ("core", "rows", "layout", "key_form", "reach", "entries", "pool", "bank", "bank_keys"):
        if far.get(f) != tail.get(f):
            bad.append((f, far.get(f), tail.get(f)))
    for f in ("s_core", "true_score_median"):
        if not _close(far[f], tail[f], rtol):
            bad.append((f, far[f], tail[f]))
    for name in ("base", "M", "E"):
        tol = rtol_e if name == "E" else rtol
        for n, tv in tail[name].items():
            fv = far[name].get(n)
            if fv is None:
                bad.append((name, n, "missing"))
                continue
            for f in ("mean", "median", "p99", "max"):
                if not _close(fv[f], tv[f], tol):
                    bad.append((name, n, f, fv[f], tv[f]))
            for f in ("look_error", "argmax_error", "entries", "look_by_margin_pct", "look_by_true_pct"):
                if fv[f] != tv[f]:
                    bad.append((name, n, f, fv[f], tv[f]))
            if not _close(fv["moment"], tv["moment"], 1e-9):
                bad.append((name, n, "moment", fv["moment"], tv["moment"]))
    for g, tv in tail["pool_length_ratio"].items():
        fv = far["pool_length_ratio"][g]
        if fv["n"] != tv["n"] or not all(_close(x, y, rtol) for k in ("p50", "p99", "max") for x, y in zip(fv[k], tv[k])):
            bad.append(("pool_length_ratio", g))
    if not all(_close(x, y, rtol) for k in ("p01", "p50", "p99") for x, y in zip(far["true_pair_length"][k], tail["true_pair_length"][k])):
        bad.append(("true_pair_length",))
    ft, tt = far["top_competitor"], tail["top_competitor"]
    if ft["n"] != tt["n"] or ft["source"] != tt["source"]:
        bad.append(("top_competitor", ft["n"], ft["source"], tt["n"], tt["source"]))
    for f in ("share_of_W_median", "length_ratio_mean_median", "length_ratio_max_median"):
        a, b = ft[f], tt[f]
        if (a is None) != (b is None) or (a is not None and not _close(a, b, rtol)):
            bad.append(("top_competitor", f, a, b))
    return bad
