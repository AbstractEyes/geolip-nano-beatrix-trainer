# -*- coding: utf-8 -*-
"""caption_pack.py - THE CAPTION PACK as the nano bed's caption documents (0.1.55; plans/2026-10-19_caption_finetune_plan.md).

The pack is the dataset the Beatrix caption arm trained on (plans/2026-10-05_caption_combo_pack_arm.md): on the Hub at
AbstractPhil/alephllm-mini-beatrix-training under mini-beatrix-3/caption_pack/pack_v2/<source>/<source>-NNNNN.jsonl, one JSON row per
image with a `split` (train / held), the age protocol applied at its build. Photograph rows (dff, scf) carry `attributes` (hair, eye,
expression, makeup, jewelry, gender, age_band) in real wordings, `tags` (the booru register) and four captions (short, long, fused,
intent); illustration rows (danbooru) carry `tags` and three captions (short, medium, long); web-image rows (cc12m) three captions
(caption, short, long); COCO rows a caption and its structure (subjects with attributes).

A row becomes a DOCUMENT: its RECORDS (key, value) in real words, the frame's content (the attribute word as the key; the wording
without its noun and its scaffolding as the value: "dark brown hair" -> hair: dark brown; "with a thoughtful expression" -> expression:
thoughtful; from the tags where a photograph's attributes lack one; for an illustration the first hair colour, eye colour and
expression tag; for COCO the subjects' first attributes), and its WORDINGS (the captions, one drawn per document: the same image said
several ways). fetch() brings the shards here, build() writes the compact cache (a byte blob and an index per split), load()
memory-maps it; the stream children find the folder through the environment variable NANO_CAPTION_PACK (as the frames corpus through
NANO_FRAMES_CORPUS)."""
from __future__ import annotations

import json
import os
import re
import time

import numpy as np

REPO = "AbstractPhil/alephllm-mini-beatrix-training"
PREFIX = "mini-beatrix-3/caption_pack/pack_v2"
PACK_ENV = "NANO_CAPTION_PACK"
# the shards read per source (the first ones; a source's rows are in its own id order): about 1.6 GB, enough documents for the phase
SOURCES = {"dff": dict(kind="photo", files=2), "scf": dict(kind="photo", files=1), "danbooru": dict(kind="danbooru", files=2),
           "cc12m": dict(kind="cc12m", files=1), "coco": dict(kind="coco", files=1)}
SRC_ID = {"dff": 0, "scf": 1, "danbooru": 2, "cc12m": 3, "coco": 4}
KIND_OF = {0: "photo", 1: "photo", 2: "danbooru", 3: "cc12m", 4: "coco"}
# a document's kind is drawn at the Beatrix caption arm's pack shares (plans/2026-10-05_caption_combo_pack_arm.md THE STREAM:
# photographs 45%, CC12M 27.5%, danbooru 25%, COCO 2.5%), here over documents
KIND_WEIGHTS = (("photo", 0.45), ("cc12m", 0.275), ("danbooru", 0.25), ("coco", 0.025))
CAPS = {"photo": ("short", "long", "fused", "intent"), "danbooru": ("short", "medium", "long"), "cc12m": ("caption", "short", "long"),
        "coco": ("caption",)}
CAP_SLOTS = 4                                   # wordings kept per document
CAP_BYTES = 1200                                # a caption is cut at a word boundary within this many bytes
KEY_MAX, VAL_MAX, REC_MAX = 12, 20, 27          # the note form's record: key + value within 27 bytes (one 32-byte slice holds it)
IDX_COLS = 3 + 2 * CAP_SLOTS                    # source, records offset, records length, caption offsets, caption lengths
COLOURS = frozenset(("aqua", "black", "blonde", "blue", "brown", "green", "grey", "gray", "orange", "pink", "purple", "red", "silver",
                     "white", "yellow", "violet", "amber", "golden", "teal", "lavender", "platinum blonde", "light brown", "dark brown",
                     "light blue", "dark blue", "dark", "light", "ginger", "auburn", "strawberry blonde", "multicolored", "two-tone",
                     "gradient", "streaked", "colored", "heterochromia", "hazel", "crimson", "scarlet", "turquoise", "emerald", "magenta"))
EXPRESSIONS = frozenset(("smile", "frown", "grin", "open mouth", "closed mouth", "blush", "tears", "crying", "angry", "sad", "happy",
                         "surprised", "laughing", "smirk", "pout", "scowl", "expressionless", "embarrassed", "wink", "closed eyes",
                         "half-closed eyes", "sleepy", "nervous", "serious", "shy", "smug", "annoyed", "disgust", "scared", "worried",
                         "sweatdrop", "light smile", "evil smile", "grimace", "parted lips", "clenched teeth", "tongue out", "wide-eyed",
                         "crazy smile", "nervous smile", "seductive smile", "forced smile", "false smile", "one eye closed"))
_PREFIX = re.compile(r"^(?:with an? |with |wearing an? |wearing |has an? |has |having an? |having |in an? |in |an? |the )+")
_NOUN = {"hair": (" hair",), "eyes": (" eyes", " eye"), "expression": (" expression", " look", " face"), "makeup": (" makeup", " make-up"),
         "jewelry": (" jewelry", " jewellery")}
# THE SYMBOLIC TRIE (Phil 2026-10-19 02:0x: "symbolic associative trie is likely the best course of action for utilizing systemic
# association frames, system of systems sort of behavior"): a record's key is a PATH of symbols from a closed vocabulary, the system
# first, the attribute second ("look hair", "scene mood", "wear clothing"); records of one system share their prefix and, in the frame,
# their child frame (frames_data.caption_frame nests one child frame per system inside the document's frame). The systems: person
# (gender, age), look (hair, eyes, expression, makeup, jewelry), wear (clothing, accessory), body (skin, pose, form), scene (setting,
# mood, layout), shot (light, angle, framing), thing (a COCO subject by name).
_ATTR_KEYS = (("hair", "look hair"), ("eye", "look eyes"), ("expression", "look expression"), ("makeup", "look makeup"),
              ("jewelry", "look jewelry"), ("gender", "person gender"), ("age_band", "person age"))
_TAG_KEYS = {"hair": "look hair", "eyes": "look eyes", "expression": "look expression"}
_STRATA_KEYS = (("clothing", "wear clothing"), ("accessory", "wear accessory"), ("skin", "body skin"), ("pose", "body pose"), ("body", "body form"))
_SCENE_KEYS = (("setting", "scene setting"), ("mood", "scene mood"), ("layout", "scene layout"))
_SHOT = ((re.compile(r"\b(light|lit|lighting|sunlight|backlit|spotlight)\b"), "shot light"), (re.compile(r"\b(angle|view|looking)\b"), "shot angle"),
         (re.compile(r"\b(shot|framed|framing|length|portrait|close-up)\b"), "shot framing"))
SYSTEMS = ("person", "look", "wear", "body", "scene", "shot", "thing")


def norm_value(key: str, phrase) -> bytes | None:
    """The record's value: the wording in lower case without its scaffolding ("with a", "wearing") and without the key's own noun
    ("... hair", "... eyes", "... expression"), ASCII, cut at a word boundary to the record's room (key + value within 27 bytes, the
    value within 20). None when nothing usable remains."""
    leaf = key.split(" ")[-1]
    s = re.sub(r"\s+", " ", str(phrase or "")).strip().lower().strip(" .,;:")
    s = _PREFIX.sub("", s)
    for suf in _NOUN.get(leaf, ()):
        if s.endswith(suf) and len(s) - len(suf) >= 2:
            s = s[:-len(suf)].strip()
            break
    s = s.strip(" .,;:")
    if s == "no":                                   # "wearing no jewelry" -> jewelry: none
        s = "none"
    try:
        b = s.encode("ascii")
    except UnicodeEncodeError:
        return None
    room = min(VAL_MAX, REC_MAX - len(key))
    if len(b) > room:
        cut = b[:room].rsplit(b" ", 1)[0].rstrip(b" ,;:.")
        if len(cut) < 3:
            return None
        b = cut
    if not (1 <= len(b) <= room) or any(c < 32 or c > 126 for c in b):
        return None
    return b


def tag_records(tags) -> dict:
    """The first hair colour, eye colour and expression among booru tags ("blonde hair, blue eyes, smile, ..."): key -> wording."""
    out = {}
    for t in (x.strip().lower() for x in str(tags or "").split(",")):
        if not t:
            continue
        if "hair" not in out and t.endswith(" hair") and t[:-5] in COLOURS:
            out["hair"] = t[:-5]
        elif "eyes" not in out and t.endswith(" eyes") and t[:-5] in COLOURS:
            out["eyes"] = t[:-5]
        elif "expression" not in out and t in EXPRESSIONS:
            out["expression"] = t
    return out


def source_of(row: dict) -> str:
    return str(row.get("id", "")).split(":")[0]


def records_of(row: dict) -> list:
    """A row's records [(path key bytes, value bytes)], keys unique, in the row's own order (the frame shuffles them and groups them by
    system). Photographs: the attributes (person, look), the tags where an attribute is missing, the strata (wear, body), the scene, one
    photography clause (shot); illustrations: the look from the tags; COCO: the subjects (thing <name>); web images: none."""
    src, out = source_of(row), {}

    def put(key, phrase):
        if key not in out:
            v = norm_value(key, phrase)
            if v:
                out[key] = v

    if src in ("dff", "scf"):
        a = row.get("attributes") or {}
        for col, key in _ATTR_KEYS:
            if a.get(col):
                put(key, a[col])
        for leaf, w in tag_records(row.get("tags")).items():
            put(_TAG_KEYS[leaf], w)
        strata = row.get("strata") or {}
        for st, key in _STRATA_KEYS:
            vals = strata.get(st) or []
            if vals:
                put(key, vals[0])
        sc = row.get("scene") or {}
        for f, key in _SCENE_KEYS:
            if sc.get(f) and sc.get(f) != "unknown":
                put(key, sc[f])
        for clause in [c.strip() for c in str(row.get("photo") or "").split(";") if c.strip()][:4]:
            for rx, key in _SHOT:
                if key not in out and rx.search(clause.lower()):
                    put(key, clause)
                    break
    elif src == "danbooru":
        for leaf, w in tag_records(row.get("tags")).items():
            put(_TAG_KEYS[leaf], w)
    elif src == "coco":
        try:
            st = json.loads(row.get("structure") or "{}")
        except ValueError:
            st = {}
        for subj in (st.get("subjects") or [])[:3]:
            name = re.sub(r"\s+", " ", str(subj.get("name") or "")).strip().lower()
            attrs = [str(x) for x in (subj.get("attributes") or []) if x]
            if name and attrs and 1 <= len(name) <= 8 and name.isascii() and " " not in name:
                put(f"thing {name}", attrs[0])
    return [(k.encode("ascii"), v) for k, v in out.items()]


def system_of(key: bytes) -> bytes:
    """The first symbol of a path key: the system a record belongs to."""
    return key.split(b" ")[0]


def clean_caption(s) -> bytes | None:
    """A caption's bytes: whitespace collapsed, cut at a word boundary within CAP_BYTES; None under 8 bytes."""
    s = re.sub(r"\s+", " ", str(s or "")).strip()
    b = s.encode("utf-8")
    if len(b) > CAP_BYTES:
        cut = b[:CAP_BYTES].rsplit(b" ", 1)[0]
        b = cut if len(cut) >= 8 else b[:CAP_BYTES]
    return b if len(b) >= 8 else None


def captions_of(row: dict) -> list:
    """A row's wordings (bytes), in the pack's field order for its kind, the empty ones left out."""
    kind = KIND_OF[SRC_ID[source_of(row)]]
    caps = [clean_caption(row.get(f)) for f in CAPS[kind]]
    return [c for c in caps if c][:CAP_SLOTS]


def shard_paths(src: str) -> list:
    return [f"{PREFIX}/{src}/{src}-{i:05d}.jsonl" for i in range(SOURCES[src]["files"])]


def fetch(folder: str, token=None, max_rows: int | None = None, log=print) -> dict:
    """The shards into <folder>/raw: whole (hf_hub_download) or, with max_rows, the first max_rows lines of each source's first shard
    streamed (a smoke's pack). -> source -> local paths."""
    raw = os.path.join(folder, "raw")
    os.makedirs(raw, exist_ok=True)
    got = {}
    for src in SOURCES:
        if max_rows:
            p = os.path.join(raw, f"{src}-head{max_rows}.jsonl")
            if not os.path.exists(p):
                from huggingface_hub import HfFileSystem
                fs = HfFileSystem(token=token or False)
                tmp = p + ".part"
                with fs.open(f"{REPO}/{shard_paths(src)[0]}", "rb") as fh, open(tmp, "wb") as out:
                    n = 0
                    for line in fh:
                        out.write(line)
                        n += 1
                        if n >= max_rows:
                            break
                os.replace(tmp, p)
                log(f"[pack] {src}: {n} rows streamed")
            got[src] = [p]
        else:
            from huggingface_hub import hf_hub_download
            ps = []
            for rel in shard_paths(src):
                ps.append(hf_hub_download(REPO, rel, token=token or False, local_dir=raw))
            log(f"[pack] {src}: {len(ps)} shards")
            got[src] = ps
    return got


def _files(folder: str, split: str) -> tuple:
    suf = "" if split == "train" else "_held"
    return os.path.join(folder, f"caption_pack{suf}.bin"), os.path.join(folder, f"caption_pack{suf}_index.npy")


def build(folder: str, token=None, max_rows: int | None = None, log=print, files: dict | None = None) -> dict:
    """The cache from the shards: per split a byte blob (each document's records as key NUL value NUL ..., then its captions) and an
    int64 index [n, IDX_COLS]; caption_pack.json with the counts. files: source -> local jsonl paths already here (a test's pack);
    else fetch(). -> the stats."""
    t0 = time.time()
    files = files or fetch(folder, token, max_rows, log)
    blobs = {"train": bytearray(), "held": bytearray()}
    idx = {"train": [], "held": []}
    counts = {}
    for src, paths in files.items():
        for p in paths:
            with open(p, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    split = "held" if row.get("split") == "held" else "train"
                    caps = captions_of(row)
                    if not caps:
                        counts[f"{src}_no_caption"] = counts.get(f"{src}_no_caption", 0) + 1
                        continue
                    recs = records_of(row)
                    if not recs:
                        counts[f"{src}_no_records"] = counts.get(f"{src}_no_records", 0) + 1
                    blob = blobs[split]
                    rec_bytes = b"".join(k + b"\x00" + v + b"\x00" for k, v in recs)
                    r = [SRC_ID[src], len(blob), len(rec_bytes)]
                    blob += rec_bytes
                    offs, lens = [], []
                    for c in caps:
                        offs.append(len(blob))
                        lens.append(len(c))
                        blob += c
                    while len(offs) < CAP_SLOTS:
                        offs.append(0)
                        lens.append(0)
                    idx[split].append(r + offs + lens)
                    counts[f"{src}_{split}"] = counts.get(f"{src}_{split}", 0) + 1
    for split in ("train", "held"):
        bp, ip = _files(folder, split)
        arr = np.asarray(idx[split], dtype=np.int64).reshape(-1, IDX_COLS)
        np.frombuffer(bytes(blobs[split]), dtype=np.uint8).tofile(bp)
        np.save(ip, arr)
    stats = dict(utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), repo=REPO, prefix=PREFIX, max_rows=max_rows,
                 sources={s: [os.path.relpath(p, folder) for p in ps] for s, ps in files.items()}, counts=counts,
                 train_docs=len(idx["train"]), held_docs=len(idx["held"]), train_bytes=len(blobs["train"]), held_bytes=len(blobs["held"]),
                 cap_bytes=CAP_BYTES, cap_slots=CAP_SLOTS, kind_weights=KIND_WEIGHTS, seconds=round(time.time() - t0, 1))
    json.dump(stats, open(os.path.join(folder, "caption_pack.json"), "w", encoding="utf-8"), indent=1)
    log(f"[pack] built: {stats['train_docs']:,} training documents ({stats['train_bytes'] / 1e6:.0f} MB), {stats['held_docs']:,} held, {stats['seconds']} s")
    return stats


class Pack:
    """The cache of one split: documents by index; records(i), captions(i), kind(i), draw(rng) by the kind weights."""

    def __init__(self, folder: str, split: str = "train"):
        bp, ip = _files(folder, split)
        self.folder, self.split = folder, split
        self.blob = np.memmap(bp, dtype=np.uint8, mode="r") if os.path.getsize(bp) else np.zeros(0, dtype=np.uint8)
        self.index = np.load(ip)
        self.n = int(len(self.index))
        kinds = [k for k, _ in KIND_WEIGHTS]
        self.by_kind = {k: np.flatnonzero(np.isin(self.index[:, 0], [s for s, kk in KIND_OF.items() if kk == k])) for k in kinds}
        self.kinds = [k for k, w in KIND_WEIGHTS if len(self.by_kind[k])]
        w = np.array([w for k, w in KIND_WEIGHTS if len(self.by_kind[k])], dtype=float)
        self.weights = w / w.sum() if len(w) else w

    def records(self, i: int) -> list:
        r = self.index[i]
        raw = bytes(self.blob[int(r[1]):int(r[1] + r[2])])
        parts = raw.split(b"\x00")[:-1] if raw else []
        return list(zip(parts[0::2], parts[1::2]))

    def captions(self, i: int) -> list:
        r = self.index[i]
        return [bytes(self.blob[int(r[3 + j]):int(r[3 + j] + r[3 + CAP_SLOTS + j])]) for j in range(CAP_SLOTS) if r[3 + CAP_SLOTS + j] > 0]

    def kind(self, i: int) -> str:
        return KIND_OF[int(self.index[i, 0])]

    def draw(self, rng) -> int:
        """A document index: its kind by the pack's shares, then uniform within the kind."""
        assert self.n, "an empty pack"
        k = self.kinds[int(rng.choice(len(self.kinds), p=self.weights))]
        ids = self.by_kind[k]
        return int(ids[int(rng.integers(len(ids)))])


_PACKS: dict = {}


def load(folder: str | None = None, split: str = "train") -> Pack:
    """The pack of a split, cached per folder (the stream children read the folder from NANO_CAPTION_PACK)."""
    folder = folder or os.environ.get(PACK_ENV, "")
    assert folder and os.path.exists(_files(folder, split)[1]), \
        f"the caption pack is missing: set {PACK_ENV} to a folder holding caption_pack_index.npy (caption_pack.build)"
    key = (os.path.abspath(folder), split)
    if key not in _PACKS:
        _PACKS[key] = Pack(folder, split)
    return _PACKS[key]


def ensure(folder: str, token=None, max_rows: int | None = None, log=print) -> dict:
    """Build the cache when absent; -> caption_pack.json's stats."""
    sp = os.path.join(folder, "caption_pack.json")
    if not (os.path.exists(sp) and os.path.exists(_files(folder, "train")[1]) and os.path.exists(_files(folder, "held")[1])):
        os.makedirs(folder, exist_ok=True)
        return build(folder, token, max_rows, log)
    return json.load(open(sp, encoding="utf-8"))
