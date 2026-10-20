# -*- coding: utf-8 -*-
"""thought_frames.py - THE ART THOUGHT DATASET rendered into the frame grammar, and its loader (0.1.62; the plan's section 2 and 7).

A thought row (the dataset's JSON Lines: the hidden spec, the thoughts, the wordings, the questions) becomes one document in the caption register:

  DOC MODE c SEP t SEP                       the head (the caption documents' head with 't' for thought in place of 'q')
  THINK<letter> [THINK a ...]... END         THE CHAIN FRAME: one sized, sliced thinking frame whose size letter is the depth (simple c/d =
                                             128/256, medium e = 512, deep f/g = 1,024/2,048); every thought a CHILD thinking frame (a run of
                                             32-byte slices) whose first record is its kind, CUE step SEP see|name|place|..., followed by the
                                             thought's text across the run's slices; the children are siblings under the one parent, no gaps
  DATA<letter> [DATA a ...]... END           THE SPEC FRAME: the records as the caption pack's symbolic trie (one child run a system, the
                                             records packed into 29-byte slices, each relation its own child), the parent the smallest size
                                             that holds them
  CUE say SEP caption \\n <text> \\n           THE WORDINGS, each introduced by CUE say SEP kind (caption, prompt, tags, critique)
  CUE key SEP value END                      THE QUESTION (one of the row's, drawn), as the caption documents ask

in four LAYOUTS (the plan's section 7): A thought-first (the chain, the spec, the wordings, the question), B caption-first (the wordings,
the spec, the question; no chain), C frameless (the wordings and the question only: the body must think unaided), D question-first (the
question's key, then the chain, the spec, the wordings, and the question answered). One layout is SEALED (held out of training: D by
default) for the travel read. A window (thought_window) packs documents back to back to the context as the caption windows do. The loader
(ThoughtPack / load) reads the JSON Lines shards of the Hub repo's layout (data/<depth>/<domain>/*.jsonl for train, held/*.jsonl for the
reads; a train row with sealed values or a sealed layout is dropped). The framed diet's 'thought' kind (framed_diet.FORMS['p50t']) draws
these windows beside the carry, frames, caption and note rows."""
from __future__ import annotations

import glob
import json
import os
import zlib

import numpy as np

from . import bed_frames as bf
from . import frames_data as FD
from geolip.alephllm.data.special_tokens import CUE, DATA, DOC, END, SEP, THINK

VERSION = "0.1.65"
# 0.1.65: the dataset's own tools (the vocabulary and its seals, the row validator, the sampler, the generation prompt, the seed contract)
# live with the dataset (its repo's tools folder), not in the model's trainer; the trainer keeps what training needs: this renderer, the
# window, the loader, and the two schema facts below (a spec's records, the slices a chain takes), which the dataset's validator shares.
CHAIN_MAX = {"simple": 256, "medium": 512, "deep": 2048}       # the depth's largest thinking frame: the thoughts fit it


def records(spec: dict) -> list:
    """A spec's records as (key, value) pairs: "system attribute" -> value for the systems, and for each relation three records
    "rel<i> a", "rel<i> rel", "rel<i> b"."""
    out = []
    for system, attrs in spec.items():
        if system == "relation":
            for i, r in enumerate(attrs or [], 1):
                out += [(f"rel{i} a", str(r.get("a", ""))), (f"rel{i} rel", str(r.get("rel", ""))), (f"rel{i} b", str(r.get("b", "")))]
        elif isinstance(attrs, dict):
            for a, v in attrs.items():
                out.append((f"{system} {a}", str(v)))
    return out


def chain_slices(thoughts: list) -> int:
    """The 32-byte slices a row's thoughts take in the chain frame: each thought a run of ceil((7 + len(kind) + len(text)) / 29) slices
    (its kind record CUE step SEP kind, the newline after it, then the text)."""
    return sum(max(1, -(-(7 + len(str(t.get("step", ""))) + len(str(t.get("text", "")).strip().encode("utf-8", "replace"))) // 29))
               for t in thoughts)
PACK_ENV = "NANO_THOUGHT_PACK"
DATASET_REPO = "AbstractPhil/thought-symbolic-image-captions"    # 0.1.67: the live dataset (public, Apache-2.0); ensure() fetches its shards
HELD_SERIES_FRACTION = 0.1          # 0.1.67: when the dataset marks no row held, the loader holds one series in ten out of training (the reads)
HEAD = bytes([DOC]) + bf.tag_bytes("caption", "mode") + bytes([SEP]) + b"t" + bytes([SEP])
SLICE, ROOM = 32, 29                                   # the sliced frame: 32-byte slices, 29 bytes of body each
DEPTH_SIZES = {"simple": (128, 256), "medium": (512,), "deep": (1024, 2048)}      # the chain frame's sizes by depth: its letter
PARENT_SIZES = (64, 128, 256, 512, 1024, 2048)
MAX_SYSTEM_SLICES = 16                                 # a system's child run holds up to sixteen slices of records (the rest dropped, counted)
WORDING_ORDER = ("caption", "prompt", "tags", "critique")
LAYOUTS = ("A", "B", "C", "D")
THOUGHT = dict(layouts=(("A", 0.5), ("B", 0.2), ("C", 0.3)), sealed="D", ask="any")
SPECIALS = frozenset({DOC, DATA, END, THINK, CUE, SEP, 247, 245, 254, 253, 251})


class _NoGaps:
    """The builders' random source with every gap drawn as zero, so the children follow each other and the thoughts' text stays in order."""

    def __init__(self, rng):
        self.rng = rng

    def integers(self, lo, hi=None, **kw):
        return lo if hi is not None else 0

    def __getattr__(self, k):
        return getattr(self.rng, k)


def _clean(text: str) -> bytes:
    b = bytearray(text.encode("utf-8", "replace"))
    for i, c in enumerate(b):
        if c in SPECIALS:
            b[i] = 10
    return bytes(b)


def _slices_of(recs: list, cap: int) -> tuple:
    """The records packed into 29-byte slices (a record never split), at most cap slices. -> (slices, dropped)."""
    slices, cur, used = [], [], 0
    for k, v in recs:
        n = len(bf.record_bytes(k, v))
        assert n <= ROOM, (k, v)
        if used + n > ROOM:
            slices.append(cur)
            cur, used = [], 0
        cur.append((k, v))
        used += n
    if cur:
        slices.append(cur)
    kept = slices[:cap]
    return kept, sum(len(s) for s in slices[cap:])


def spec_frame(rng, recs: list, max_slices: int = MAX_SYSTEM_SLICES) -> tuple:
    """THE SPEC FRAME: the records (bytes pairs) as the symbolic trie: a child run a system (a relation's three records their own system
    'relN'), in the order the systems first appear, nested in the smallest parent that holds them; one system alone gives a flat sliced
    frame. -> (frame bytes, the records written, the records dropped)."""
    groups, order = {}, []
    for k, v in recs:
        s = k.split(b" ")[0]
        if s not in groups:
            groups[s] = []
            order.append(s)
        groups[s].append((k, v))
    state = FD._state()
    fill = bf.Filler(b" " * 8192, 0)
    if len(order) <= 1:
        slices, dropped = _slices_of(recs, max_slices)
        kept = [kv for s in slices for kv in s]
        # 0.1.68: the frame's size is the next GENERIC size at or above the slices (three slices -> 128, not 96: a sized open tag names
        # its size in one letter, and 96 has none; the spare slices carry filler). Multi-system rows are unchanged.
        size = next(S for S in (SLICE,) + PARENT_SIZES if S >= SLICE * max(1, len(slices)))
        fb = bf.build_frame(_NoGaps(rng), bf.spec("data", size, records=len(kept), keys=[k for k, _ in kept], values=[v for _, v in kept]), 0,
                            fill, "caption", state)
        return fb, kept, dropped
    kids, kept, dropped = [], [], 0
    for s in order:
        slices, d = _slices_of(groups[s], max_slices)
        dropped += d
        rs = [kv for sl in slices for kv in sl]
        kids.append(bf.spec("data", 32 * len(slices), records=len(rs), keys=[k for k, _ in rs], values=[v for _, v in rs]))
        kept += rs
    need = sum(c["size"] // SLICE for c in kids)
    size = next((S for S in PARENT_SIZES if S // SLICE >= need), None)
    assert size, f"the spec needs {need} slices, more than the largest frame holds"
    fb = bf.build_frame(_NoGaps(rng), bf.spec("data", size, records=0, children=kids), 0, fill, "caption", state)
    return fb, kept, dropped


def chain_frame(rng, thoughts: list, depth: str) -> tuple:
    """THE CHAIN FRAME: the thoughts as child thinking frames under one parent sized by the depth. Each child's slices: its kind as the
    first record (CUE step SEP kind), then its text across the run; a text longer than the depth's largest frame affords is cut (counted).
    -> (frame bytes, meta: size, letter, slices per thought, bytes cut)."""
    sizes = DEPTH_SIZES[depth]
    recs = [(b"step", str(t.get("step", "see")).encode("ascii", "replace")) for t in thoughts]
    texts = [_clean(str(t.get("text", "")).strip()) for t in thoughts]        # a thought's outer spaces carry nothing (the padding is spaces)
    rec_n = [len(bf.record_bytes(k, v)) + 1 for k, v in recs]                 # the record and the newline that ends its value
    need = [max(1, -(-(r + len(x)) // ROOM)) for r, x in zip(rec_n, texts)]
    assert sum(need) == chain_slices(thoughts), "the slice count and the renderer's differ"
    cap = sizes[-1] // SLICE
    cut = 0
    if sum(need) > cap:                                             # shrink the longest thoughts until the chain fits the largest frame
        while sum(need) > cap:
            j = max(range(len(need)), key=lambda i: need[i])
            need[j] -= 1
        for j, (r, x) in enumerate(zip(rec_n, texts)):
            room = need[j] * ROOM - r
            if len(x) > room:
                cut += len(x) - room
                texts[j] = x[:room]
    size = next(S for S in sizes if S // SLICE >= sum(need))
    kids = [bf.spec("think", SLICE * n, records=1, keys=[k], values=[v]) for n, (k, v) in zip(need, recs)]
    src = b"".join(b"\n" + x + b" " * (n * ROOM - r - len(x)) for n, r, x in zip(need, rec_n, texts)) + b" " * size
    state = FD._state()
    fb = bf.build_frame(_NoGaps(rng), bf.spec("think", size, records=0, children=kids), 0, bf.Filler(src, 0), "caption", state)
    assert len(fb) == size + 3, (len(fb), size)
    return fb, dict(size=size, letter=bf.SIZE_LETTER[size].decode(), slices=need, cut=cut, thoughts=len(thoughts))


def wordings_bytes(wordings: dict) -> bytes:
    out = bytearray()
    for kind in WORDING_ORDER:
        text = str((wordings or {}).get(kind) or "").strip()
        if text:
            out += bytes([CUE]) + b"say" + bytes([SEP]) + kind.encode() + b"\n" + _clean(text) + b"\n"
    return bytes(out)


def question_bytes(q: dict, answered: bool = True) -> bytes:
    k, v = str(q.get("key", "")).encode("ascii", "replace"), str(q.get("value", "")).encode("ascii", "replace")
    return bytes([CUE]) + k + bytes([SEP]) + (v + bytes([END]) if answered else b"")


def render(rng, row: dict, layout: str = "A", ask="any") -> tuple:
    """One row as one document in a layout. ask: 'any' draws one of the row's questions, an int picks one. -> (bytes, meta)."""
    assert layout in LAYOUTS, layout
    recs = [(k.encode("ascii", "replace"), v.encode("ascii", "replace")) for k, v in records(row["spec"])]
    qs = row.get("questions") or []
    q = qs[int(rng.integers(len(qs))) if ask == "any" else int(ask)] if qs else None
    meta = dict(id=row.get("id"), depth=row.get("depth"), layout=layout, asked=(q["key"], q["value"], q.get("needs")) if q else None, records=len(recs))
    out = bytearray(HEAD)
    chain = spec = None
    if layout in ("A", "D"):
        chain, cm = chain_frame(rng, row.get("thoughts") or [], row["depth"])
        meta["chain"] = cm
    if layout in ("A", "B", "D"):
        spec, kept, dropped = spec_frame(rng, recs)
        meta.update(written=len(kept), dropped=dropped)
    if layout == "A":
        out += chain + b"\n" + spec + b"\n" + wordings_bytes(row.get("wordings"))
    elif layout == "B":
        out += wordings_bytes(row.get("wordings")) + spec + b"\n"
    elif layout == "C":
        out += wordings_bytes(row.get("wordings"))
    else:
        out += (question_bytes(q, answered=False) + b"\n" if q else b"") + chain + b"\n" + spec + b"\n" + wordings_bytes(row.get("wordings"))
    if q:
        out += question_bytes(q)
    return bytes(out), meta


def _pick(rng, table) -> str:
    u, acc = rng.random(), 0.0
    for k, w in table:
        acc += w
        if u < acc:
            return k
    return table[-1][0]


def thought_window(rng, pack, form: dict = THOUGHT, ctx: int = FD.ROW_CTX) -> tuple:
    """Thought documents back to back until the window is full, the last one cut, each in a layout drawn from the form's shares (the
    sealed layout never drawn here). -> (ctx + 1 bytes, the documents' metas)."""
    layouts = tuple((k, w) for k, w in form["layouts"] if k != form.get("sealed"))
    out, metas = bytearray(), []
    while len(out) < ctx + 1:
        i = pack.draw(rng)
        d, m = render(rng, pack.row(i), _pick(rng, layouts), form.get("ask", "any"))
        m["start"], m["doc"] = len(out), i
        out += d
        metas.append(m)
    return bytes(out[:ctx + 1]), metas


# ------------------------------------------------------------------ the loader
class ThoughtPack:
    """The rows of a split in memory: row(i), draw(rng) uniform, n."""

    def __init__(self, rows: list, split: str = "train"):
        self.rows, self.split = list(rows), split
        self.n = len(self.rows)

    @classmethod
    def from_folder(cls, folder: str, split: str = "train") -> "ThoughtPack":
        """0.1.67: the train split = the data shards' rows that is_held() leaves in; the held split = the rows it holds out plus any
        held/*.jsonl shard (the dataset's own held files, when it ships them). The two splits are disjoint and together are every row."""
        rows = []
        for p in sorted(glob.glob(os.path.join(folder, "data", "**", "*.jsonl"), recursive=True)):
            rows += _read_jsonl(p)
        if split == "held":
            held = [r for r in rows if is_held(r)]
            for p in sorted(glob.glob(os.path.join(folder, "held", "*.jsonl"))):
                held += _read_jsonl(p)
            seen, uniq = set(), []                     # a row shipped both marked in data/ and in held/ counts once
            for r in held:
                k = r.get("id") or id(r)
                if k not in seen:
                    seen.add(k)
                    uniq.append(r)
            return cls(uniq, split)
        return cls([r for r in rows if not is_held(r)], split)

    def row(self, i: int) -> dict:
        return self.rows[i]

    def draw(self, rng) -> int:
        assert self.n, "an empty thought pack"
        return int(rng.integers(self.n))

    def depths(self) -> dict:
        """The rows by depth (the dataset's validator, in its own repo, is what checks them)."""
        out = {}
        for r in self.rows:
            out[r.get("depth")] = out.get(r.get("depth"), 0) + 1
        return out


_PACKS: dict = {}


def _read_jsonl(path: str) -> list:
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def is_held(row: dict) -> bool:
    """0.1.67: a row the loader keeps out of training: the dataset's own marks (held values or a held layout), else one series in ten by a
    stable hash of the series id (a series never splits across train and held; the pilot marks no row, so the reads need this split; the
    stream children and the reads apply the same rule, so what the body trains on and what it is read on never overlap)."""
    h = row.get("held") or {}
    if h.get("values") or h.get("layout"):
        return True
    sid = str(row.get("series") or row.get("id") or "")
    return zlib.crc32(sid.encode("utf-8")) % 1000 < int(round(HELD_SERIES_FRACTION * 1000))


def load(folder: str | None = None, split: str = "train") -> ThoughtPack:
    """The pack of a split, cached per folder (the stream children read the folder from NANO_THOUGHT_PACK)."""
    folder = folder or os.environ.get(PACK_ENV, "")
    assert folder and os.path.isdir(folder), f"the thought pack is missing: set {PACK_ENV} to a folder holding data/**/*.jsonl"
    key = (os.path.abspath(folder), split)
    if key not in _PACKS:
        _PACKS[key] = ThoughtPack.from_folder(folder, split)
    return _PACKS[key]


def ensure(folder: str, token=None, log=print) -> dict:
    """0.1.67: the thought pack at `folder`, fetched from the Hub when its shards are absent (the dataset's data/ shards, vocab/ and card;
    tokenless when no token: a public repo). -> the counts: rows, train, held, by depth (both splits)."""
    if not glob.glob(os.path.join(folder, "data", "**", "*.jsonl"), recursive=True):
        from huggingface_hub import snapshot_download
        os.makedirs(folder, exist_ok=True)
        snapshot_download(DATASET_REPO, repo_type="dataset", token=token if token else False, local_dir=folder,
                          allow_patterns=["data/**", "vocab/**", "README.md"])
    _PACKS.pop((os.path.abspath(folder), "train"), None)
    _PACKS.pop((os.path.abspath(folder), "held"), None)
    tr, hd = load(folder, "train"), load(folder, "held")
    st = dict(folder=folder, repo=DATASET_REPO, rows=tr.n + hd.n, train=tr.n, held=hd.n, depths=tr.depths(), held_depths=hd.depths())
    log(f"[thought] the pack at {folder}: {st['rows']:,} rows, {st['train']:,} train and {st['held']:,} held by series; train by depth {st['depths']}, held {st['held_depths']}")
    return st


def prompt_of(doc: bytes, key: str) -> bytes | None:
    """0.1.67: a rendered document cut right after the LAST 'CUE key SEP' (the question asked at its end), so the bytes that follow are
    the value the body has to produce; None when the key is not asked in it."""
    mark = bytes([CUE]) + key.encode("ascii", "replace") + bytes([SEP])
    i = doc.rfind(mark)
    return None if i < 0 else doc[:i + len(mark)]


# ------------------------------------------------------------------ reading a rendered document back (the tests, the reads)
def records_in(doc: bytes) -> list:
    """Every CUE key SEP value record in a document, the value ending at the next special byte or newline, trailing spaces stripped (the
    frames' filler), in order. The 'step' and 'say' records come back too (their values end at the newline before their text)."""
    out, i, n = [], 0, len(doc)
    while i < n:
        if doc[i] != CUE:
            i += 1
            continue
        j = i + 1
        while j < n and doc[j] != SEP and doc[j] not in SPECIALS:
            j += 1
        if j >= n or doc[j] != SEP:
            i = j
            continue
        k = doc[i + 1:j]
        m = j + 1
        while m < n and doc[m] not in SPECIALS and doc[m] != 10:
            m += 1
        out.append((k, doc[j + 1:m].rstrip(b" ")))
        i = m
    return out


def chain_text(doc: bytes) -> list:
    """The thoughts of a document's chain frame: for each child run, (kind, text) with the text joined across the run's slices and the
    padding stripped (the parent's own filler slices skipped)."""
    i = doc.find(bytes([THINK]))
    if i < 0:
        return []
    size = {v: k for k, v in bf.SIZE_LETTER.items()}[doc[i + 1:i + 2]]
    body = doc[i + 2:i + 2 + size]
    kinds, texts, cur = [], [], None
    for s in range(0, size, SLICE):
        sl = body[s:s + SLICE]
        assert sl[0] == THINK and sl[-1] == END, "not a thinking slice"
        b = sl[2:-1]
        if b[:1] == bytes([CUE]):
            j = b.find(bytes([SEP]))
            m = b.find(b"\n", j + 1)
            m = len(b) if m < 0 else m
            if cur is not None:
                texts.append(cur)
            kinds.append(b[j + 1:m].decode())
            cur = bytearray(b[m + 1:])
        elif cur is not None:
            cur += b
    if cur is not None:
        texts.append(cur)
    return [(k, bytes(t).rstrip(b" ").lstrip(b" ")) for k, t in zip(kinds, texts)]
