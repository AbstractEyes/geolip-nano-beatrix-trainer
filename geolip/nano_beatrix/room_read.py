# -*- coding: utf-8 -*-
"""room_read.py - 0.1.93, THE ROOM READ (no training): what a head's rotation pairs carry in a stored core, and where its attention goes at
the positions whose unit path occurred earlier in the row.

    THE ROOM. For a head h and a set J of its pairs (pair j = dims j and j + 16), in every softmax block after the block's own rotation (the
      anchored rotation and the key and path rotations' wrappers), q_h and k_h are zeroed on the dims {j, j + 16 : j in J}, so the removed
      pairs add nothing to the head's scores
          s_h(t, u) = sum_{j not in J} <q_hj(t), k_hj(u)> / sqrt(32);
      the OUTPUT removal zeroes head h's values (v_h = 0: the head writes nothing to its block). A removal's cost is the held-out bits with
      it minus the core's own on the same rows, overall, at the repeated paths and elsewhere, and by window of the row.
    THE LOOK. A_bh(t, u) = softmax over u <= t of s_bh(t, u), formed in fp32 from the block's own rotated q and k. At a position t whose
      path occurred earlier in the row, the continuation set C(t) = {u <= t : u >= 1, path(u - 1) = path(t)} holds the bytes after the
      earlier copies of t's path (it is non-empty exactly where path_rotary.repeat_mask is True), and
          m_bh = mean over the repeated t of sum_{u in C(t)} A_bh(t, u),   beside the uniform share  mean |C(t)| / (t + 1),
      split by the path's length (1, 2-3, 4 or more bytes), with the share of repeated positions whose largest weight lands in C(t).

The rows are the text read's (rotary_scale.text_read: DOC + the first ctx bytes of the held-out documents, the craft's forward under bf16
autocast on the card). The pair groups come from the core's own split (anchored_rope.frequencies: the relative band, the content pairs,
the anchor levels). read() writes one record a core, resume-first. Nothing here touches training: the wrappers are instance attributes
installed on a reading core, pass-through when no removal is set."""
from __future__ import annotations

import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

try:
    from . import grid_rows as gr
    from . import path_rotary as prm
    from . import rotary_scale as rs
    from . import __version__ as _PKG_VERSION
except ImportError:                                      # run as a file: the folder on the path
    import grid_rows as gr  # noqa: E402
    import path_rotary as prm  # noqa: E402
    import rotary_scale as rs  # noqa: E402
    _PKG_VERSION = "file"

LENGTHS = (("1", 1, 1), ("2-3", 2, 3), ("4+", 4, 1 << 30))   # the path-length classes of the look
HALF = 16                                                   # pair j = dims j and j + HALF in a head of 32


def groups(core) -> dict:
    """The core's pair groups from its own split: rel (the relative band), content, anchors (every anchor level), scores (all sixteen)."""
    assert core.anchor is not None, "the room read is written for an anchored core"
    kind = core.anchor.kind.tolist()
    g = dict(rel=tuple(j for j, k in enumerate(kind) if k == 0), content=tuple(j for j, k in enumerate(kind) if k == 1),
             anchors=tuple(j for j, k in enumerate(kind) if k >= 2), scores=tuple(range(len(kind))))
    assert len(g["scores"]) == HALF and len(g["anchors"]) == 8 and g["rel"] and g["content"], g
    return g


def plan(core, kind: str, path_head: int = 7) -> list:
    """The removals of the registered read (history/plans/2026-10-27_pillar_room_read_plan.md section 1), as (name, head, pairs, out):
    kind "plain" - every head's anchor pairs, the path head's relative band, content pairs, scores and output; kind "path" - the path
    head's anchor pairs and output; kind "look" (0.1.94) - none (the model's own bits and the look alone)."""
    g = groups(core)
    if kind == "look":
        return []
    H = core.model.blocks[0].attn.h
    if kind == "plain":
        out = [(f"h{h}_anchors", h, g["anchors"], False) for h in range(H)]
        out += [(f"h{path_head}_{n}", path_head, g[n], False) for n in ("rel", "content", "scores")]
        return out + [(f"h{path_head}_out", path_head, (), True)]
    assert kind == "path", kind
    return [(f"h{path_head}_anchors", path_head, g["anchors"], False), (f"h{path_head}_out", path_head, (), True)]


class Removal:
    """The read's wrappers on every softmax block of a model (installed after the core's own, so they act on what the kernel reads):
    set(head, pairs, out) removes the named pairs of one head from its scores (q and k zeroed on the pairs' dims after the block's
    rotation) and, with out, its output (its values zeroed), in every block; clear() makes the wrappers pass the tensors through untouched."""

    def __init__(self, model: torch.nn.Module) -> None:
        self.head, self.pairs, self.out = None, (), False
        self.n = 0
        for b in model.blocks:
            a = b.attn
            if not (hasattr(a, "_rotate") and hasattr(a, "_split") and hasattr(a, "h")):
                continue
            inner_r, inner_s = a._rotate, a._split

            def _rotate(q, k, offset: int = 0, _inner=inner_r):
                qo, ko = _inner(q, k, offset)
                if self.head is None or not self.pairs:
                    return qo, ko
                m = self._mask(qo)
                return qo * m, ko * m

            def _split(x, n, _inner=inner_s):
                q, k, v = _inner(x, n)
                if self.head is None or not self.out:
                    return q, k, v
                mv = torch.ones(1, v.shape[1], 1, 1, dtype=v.dtype, device=v.device)
                mv[:, self.head] = 0
                return q, k, v * mv
            a._rotate, a._split = _rotate, _split
            self.n += 1
        assert self.n, "no softmax block to read"

    def _mask(self, t: torch.Tensor) -> torch.Tensor:
        H, hd = t.shape[1], t.shape[-1]
        assert hd == 2 * HALF and 0 <= self.head < H, (hd, self.head, H)
        m = torch.ones(1, H, 1, hd, dtype=t.dtype, device=t.device)
        for j in self.pairs:
            m[0, self.head, 0, j] = 0
            m[0, self.head, 0, j + HALF] = 0
        return m

    def set(self, head: int, pairs=(), out: bool = False) -> None:
        assert all(0 <= int(j) < HALF for j in pairs) and (pairs or out), (pairs, out)
        self.head, self.pairs, self.out = int(head), tuple(int(j) for j in pairs), bool(out)

    def clear(self) -> None:
        self.head, self.pairs, self.out = None, (), False


def _rows(docs: list, dev) -> torch.Tensor:
    rows = [bytes([rs.DOC]) + d[:gr.CTX] for d in docs]
    return torch.from_numpy(np.stack([np.frombuffer(r, dtype=np.uint8) for r in rows]).astype(np.int64)).to(dev)


@torch.no_grad()
def text_bits(core, docs: list, units, mb: int = 16) -> dict:
    """The held-out bits a byte as text_read forms them (the same rows, batches and autocast), with the repeat split and the windows."""
    from geolip.alephllm.eval import position_bits as pb
    tot, n, rep, pos = 0.0, 0, prm.RepeatSplit(), None
    for i in range(0, len(docs), mb):
        x = _rows(docs[i:i + mb], core.dev)
        core.clear()
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=core.dev.type == "cuda"):
            lg = core.model(x[:, :-1])[0].float()
        nll = F.cross_entropy(lg.reshape(-1, lg.shape[-1]), x[:, 1:].reshape(-1), reduction="none").view(x.shape[0], -1)
        pos = pb.merge(pos, pb.bits_by_position(nll))
        nl = nll.cpu().numpy()
        tot += float(nl.sum())
        n += nl.size
        rep.add(nl, prm.repeat_mask(units, x[:, :-1]))
    return dict(bits=tot / n / math.log(2.0), by_repeat=rep.out(), by_position=pb.finish(pos))


def path_ids(units, x: torch.Tensor) -> np.ndarray:
    """x (B, n) bytes -> (B, n) int64: each position's unit path (the bytes from its unit's start through it, whole, as repeat_mask keys
    it) as an id; equal paths in a row share an id."""
    d = units.depth(x).cpu().numpy()
    xb = x.detach().cpu().numpy().astype(np.uint8)
    out = np.zeros(d.shape, dtype=np.int64)
    for b in range(xb.shape[0]):
        row, ids, db = xb[b].tobytes(), {}, d[b]
        for t in range(xb.shape[1]):
            out[b, t] = ids.setdefault(row[t - int(db[t]) + 1:t + 1], len(ids))
    return out


def continuation(ids: torch.Tensor) -> torch.Tensor:
    """ids (n,) -> (n, n) bool: M[t, u] = u <= t and u >= 1 and id(u - 1) = id(t) - the bytes after the earlier copies of t's path."""
    n = ids.shape[0]
    prev = torch.full_like(ids, -1)
    prev[1:] = ids[:-1]
    return (prev[None, :] == ids[:, None]) & torch.ones(n, n, dtype=torch.bool, device=ids.device).tril()


def attention(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """q, k (H, n, hd) -> (H, n, n) fp32: the causal softmax attention the kernel forms (scale 1 / sqrt(hd))."""
    q, k = q.float(), k.float()
    s = q @ k.transpose(-1, -2) / math.sqrt(q.shape[-1])
    n = s.shape[-1]
    s.masked_fill_(~torch.ones(n, n, dtype=torch.bool, device=s.device).tril(), float("-inf"))
    return s.softmax(-1)


@torch.no_grad()
def look(core, docs: list, units, rows: int = 16, mb: int = 4) -> dict:
    """THE LOOK on the first `rows` documents: for every block and head, m (the mean mass on C(t) at the repeated positions), the uniform
    share, the lift, the share of repeated positions whose largest weight lands in C(t), and m by path length. 0.1.94: a core with the
    ninth head (ninth_head.py) reads it too, as each block's last head (index 8)."""
    attns = [b.attn for b in core.model.blocks if hasattr(b.attn, "_rotate")]
    caps, saved = [None] * len(attns), []
    for i, a in enumerate(attns):
        inner = a._rotate
        saved.append(inner)

        def _rotate(q, k, offset: int = 0, _inner=inner, _i=i):
            qo, ko = _inner(q, k, offset)
            caps[_i] = (qo, ko)
            return qo, ko
        a._rotate = _rotate
    nin = getattr(core, "ninth", None)                     # 0.1.94: the ninth head's own rotated q and k, read as head 8
    if nin is not None:
        nin.capture = True
    H = attns[0].h + (0 if nin is None else 1)
    acc = np.zeros((len(attns), H, 2))                     # [block, head]: sum of m, sum of the argmax-in-C indicator
    acc_len = np.zeros((len(attns), H, len(LENGTHS)))
    n_len = np.zeros(len(LENGTHS))
    base_sum, n_rep, n_pos = 0.0, 0, 0
    try:
        for i in range(0, rows, mb):
            x = _rows(docs[i:min(i + mb, rows)], core.dev)
            xin = x[:, :-1]
            core.clear()
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=core.dev.type == "cuda"):
                core.model(xin)
            ids = torch.from_numpy(path_ids(units, xin)).to(core.dev)
            dep = units.depth(xin)
            n = xin.shape[1]
            tpos = torch.arange(n, device=core.dev)
            for r in range(xin.shape[0]):
                M = continuation(ids[r])
                rep = M.any(-1)
                R = int(rep.sum())
                n_pos += n
                if R == 0:
                    continue
                cnt = M.sum(-1).float()
                base_sum += float((cnt / (tpos + 1).float())[rep].sum())
                n_rep += R
                lens = dep[r][rep]
                lsel = [(lens >= lo) & (lens <= hi) for _, lo, hi in LENGTHS]
                n_len += np.array([int(s.sum()) for s in lsel])
                Mr = M[rep]                                                   # (R, n)
                for bi, (qo, ko) in enumerate(caps):
                    if nin is not None:
                        qo, ko = torch.cat([qo, nin.caps[bi][0].to(qo.dtype)], dim=1), torch.cat([ko, nin.caps[bi][1].to(ko.dtype)], dim=1)
                    A = attention(qo[r], ko[r])                               # (H, n, n)
                    Ar = A[:, rep]                                            # (H, R, n)
                    mass = (Ar * Mr[None]).sum(-1)                            # (H, R)
                    amax = Ar.argmax(-1)                                      # (H, R)
                    hit = Mr.gather(1, amax.T).T                              # (H, R): the largest weight in C(t)
                    acc[bi, :, 0] += mass.sum(-1).double().cpu().numpy()
                    acc[bi, :, 1] += hit.sum(-1).double().cpu().numpy()
                    for li, s in enumerate(lsel):
                        acc_len[bi, :, li] += mass[:, s].sum(-1).double().cpu().numpy()
                    del A, Ar, mass
    finally:
        for a, inner in zip(attns, saved):
            a._rotate = inner
        if nin is not None:
            nin.capture, nin.caps = False, [None] * nin.n_blocks
    base = base_sum / max(n_rep, 1)
    blocks = []
    for bi in range(len(attns)):
        heads = []
        for h in range(H):
            m = acc[bi, h, 0] / max(n_rep, 1)
            heads.append(dict(m=m, lift=m / base if base > 0 else None, argmax_share=acc[bi, h, 1] / max(n_rep, 1),
                              by_length={nm: (acc_len[bi, h, li] / n_len[li] if n_len[li] else None) for li, (nm, _, _) in enumerate(LENGTHS)}))
        blocks.append(heads)
    return dict(rows=rows, n_repeat=n_rep, repeat_share=n_rep / max(n_pos, 1), uniform=base,
                n_by_length={nm: int(n_len[li]) for li, (nm, _, _) in enumerate(LENGTHS)}, blocks=blocks)


def read(core_folder: str, out_path: str, heldout_path: str, kind: str, look_rows: int = 16, n_docs: int = 128, mb: int = 16,
         dev: str = "cuda", log=print) -> dict:
    """THE ROOM READ of one stored core: its own held-out bits (the text read's), each removal of plan(kind) and its cost, and the look
    (look_rows 0 = none). One record at out_path, resume-first (an existing record is returned)."""
    if os.path.exists(out_path):
        log(f"[room] {os.path.basename(out_path)}: done before")
        return json.load(open(out_path, encoding="utf-8"))
    t0 = time.time()
    core = rs.Core(core_folder, dev, look=None)
    pr = core.path_rot
    units = pr.units if pr is not None else prm.Units().to(core.dev)
    docs = rs.load_heldout(heldout_path)[:n_docs]
    assert len(docs) == n_docs, f"the held-out file holds {len(docs)} documents of at least {gr.CTX + 64} bytes, the read asks {n_docs}"
    rm = Removal(core.model)
    base = text_bits(core, docs, units, mb)
    rec = dict(core=core.tag, version=_PKG_VERSION, kind=kind, docs=n_docs, ctx=gr.CTX, groups=groups(core),
               rule=dict(rule="entropy", theta=units.theta, floor=units.floor, table_sha256=prm.TABLE_SHA256),
               path=(None if pr is None else dict(form=pr.form, head=pr.head, k=pr.k)), base=base, removed={})
    nin = getattr(core, "ninth", None)
    if nin is not None:                              # 0.1.94: the ninth heads (the look reads each as its block's head 8)
        rec["ninth"] = dict(rot=nin.rot, form=nin.form)
    for name, h, pairs, out in plan(core, kind):
        rm.set(h, pairs, out)
        try:
            r = text_bits(core, docs, units, mb)
        finally:
            rm.clear()
        b, rb = r["by_repeat"], base["by_repeat"]
        r.update(head=h, pairs=list(pairs), out=out, cost=r["bits"] - base["bits"], cost_repeat=b["repeat"] - rb["repeat"],
                 cost_other=b["other"] - rb["other"])
        rec["removed"][name] = r
        log(f"[room] {core.tag[-40:]} {name}: {r['bits']:.4f} ({r['cost']:+.4f}; at the repeated paths {r['cost_repeat']:+.4f}, elsewhere "
            f"{r['cost_other']:+.4f})")
    if look_rows:
        rec["look"] = look(core, docs, units, rows=look_rows)
    rec["wall_s"] = time.time() - t0
    log(f"[room] {core.tag}: base {base['bits']:.4f}, {len(rec['removed'])} removals" + (f", the look on {look_rows} rows" if look_rows else "")
        + f" ({rec['wall_s']:.0f} s)")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    json.dump(rec, open(out_path, "w", encoding="utf-8"), indent=1)
    return json.load(open(out_path, encoding="utf-8"))                   # the record as stored (a resumed read returns the same)


def room_name(core_tag: str) -> str:
    return f"{core_tag}_room"
