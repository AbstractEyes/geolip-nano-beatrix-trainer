# -*- coding: utf-8 -*-
"""thought_pack.py - THE ART THOUGHT DATASET's tooling (0.1.59; plans/2026-10-19_art_thought_dataset_plan.md; the dataset
AbstractPhil/thought-symbolic-image-captions). Text only: a row is one act of thinking about an image nobody sees. This module holds
what the generating side needs before a row is written and what the checking side needs after: the systems trie and its closed
vocabulary (the pack's seven systems and five added: form, relation, mood, intent, story), the SEALED held-out values (a tenth of every
closed list, never given to the generating model, kept for the reads), the ROW VALIDATOR (the plan's eight rules and the depth counts),
the SERIES check (a simple row's facts are a subset of the medium's, the medium's of the deep's), the PROMPT the generating model writes
rows against (the schema, the rules, the lists it may draw from), and the pilot CHECKER (a JSON Lines file -> the problems by rule).
The renderer into the frame grammar (depth = the THINK frame's size letter, the kind = the frame's first record, parallel thoughts =
sibling frames, the facts = DATA frames by system) and the loader for the framed diet follow in the next version.

The depths (the plan's section 3): simple (an observation or a naming; 1-4 records, 1-2 thoughts, 1-2 questions; a 128-256-byte thinking
frame), medium (subjects, relations, form, a definition, the behavior shown; 6-14 records, 3-5 thoughts, 3-5 questions; 512), deep (the
whole composition; 14-30 records, 6-10 thoughts, 5-8 questions incl. one unknown and one needing the chain; 1,024-2,048).

Command line: --vocab=<folder> writes vocab/trie.json and vocab/held.json; --check=<rows.jsonl> [--held=vocab/held.json] prints the
problems by rule and the depth counts; --prompt=1 --depth=medium --domain=photograph [--series=1 --split=held] prints the prompt for
one call; --sample=N [--depth= --series=1 --seed=] prints drawn specs (the facts; the model adds mood, intent, story) as JSON lines;
--doc=<path> writes the prompt document (docs/thought_generation_prompt.md in this repo)."""
from __future__ import annotations

import json
import os
import re
import sys
from collections import Counter

import numpy as np

from .caption_pack import COLOURS, EXPRESSIONS

VERSION = "0.1.59"
REPO = "AbstractPhil/thought-symbolic-image-captions"
DEPTHS = ("simple", "medium", "deep")
DOMAINS = ("photograph", "painting", "illustration", "sculpture", "design")
STEPS = ("see", "name", "place", "count", "relate", "form", "light", "colour", "mood", "mean", "intend", "story", "compare", "change",
         "check", "say")
NEEDS = ("record", "chain", "unknown")
VAL_MAX, KEY_MAX = 20, 24                       # a record's value and key, in bytes (the 32-byte slice holds CUE key SEP value in 29)
# the systems trie: system -> its attributes (the key is "system attribute"); relation is a list of (a, rel, b)
SYSTEMS = {"person": ("count", "gender", "age", "role"),
           "look": ("hair", "eyes", "expression", "makeup", "jewelry", "skin"),
           "wear": ("clothing", "accessory", "colour"),
           "body": ("pose", "form", "gesture"),
           "scene": ("setting", "time", "weather", "layout", "mood"),
           "shot": ("light", "angle", "framing", "lens", "focus"),
           "thing": ("name", "count", "colour", "state", "position"),
           "form": ("composition", "balance", "line", "shape", "space", "rhythm", "contrast", "palette", "texture", "medium", "style", "period"),
           "mood": ("feeling", "tone", "energy", "temperature", "evidence"),
           "intent": ("purpose", "audience", "message", "genre"),
           "story": ("moment", "before", "after", "cause", "teller")}
RELATIONS = ("left of", "right of", "above", "below", "behind", "in front of", "beside", "holding", "facing", "looking at",
             "larger than", "smaller than", "same colour as", "touching", "inside", "under", "leaning on", "reflected in")
# THE DEPTH COUNTS (the plan's section 3): records, thoughts, questions a row (inclusive ranges)
COUNTS = {"simple": dict(records=(1, 4), thoughts=(1, 2), questions=(1, 2), wordings=1),
          "medium": dict(records=(6, 14), thoughts=(3, 5), questions=(3, 5), wordings=1),
          "deep": dict(records=(14, 30), thoughts=(6, 10), questions=(5, 8), wordings=3)}
FRAME_SIZES = {"simple": (128, 256), "medium": (512,), "deep": (1024, 2048)}
RULES = {1: "every claim is in the spec or is marked as an inference with its evidence; nothing refers to pixels or to seeing the file",
         2: "record values are plain lower-case words of at most 20 bytes, keys from the trie; longer language lives in the thoughts and the wordings",
         3: "the depth sets the counts of records, thoughts and questions",
         4: "every question's value is a record's value, 'unknown' (a key the spec does not hold), or (needs 'chain') a value the thoughts state",
         5: "at least one wording a row, three for deep (caption, prompt, tags); the tag wording uses the booru register where it exists",
         6: "a mood names its evidence: the key of the record that shows it (look expression, shot light, rel1 rel)",
         7: "a series shares its spec by subset: simple within medium within deep",
         8: "no real person's name, no copyrighted image described from memory: the spec is invented or comes from a licensed description"}
BANNED = ("pixel", "the image file", "i can see", "as seen in the image", "in this photo i", "looking at the image", "the jpeg", "the png")

# ------------------------------------------------------------------ the closed vocabulary (a tenth of every list is sealed as held-out)
def _union(base, more) -> list:
    """The pack's list plus additions, each value once, the pack's order sorted then the additions in their order."""
    out = sorted(base)
    out += [m for m in more if m not in base]
    return out


VOCAB = {
    "colour": _union(COLOURS - {"gray"}, ["cream", "beige", "tan", "gold", "navy", "olive", "maroon", "coral", "lilac", "rust", "ivory", "charcoal", "indigo", "peach", "mint", "mustard", "plum", "sand", "slate", "wine"]),
    "expression": _union(EXPRESSIONS, ["wry", "serene", "stern", "pensive", "amused", "weary", "startled", "proud", "tender", "bored", "defiant", "dreamy", "wince", "gaze", "calm", "fierce", "mournful", "curious"]),
    "hair": ["short", "long", "curly", "straight", "wavy", "braided", "bob", "bun", "ponytail", "buzzed", "shaved", "messy", "slicked", "afro", "locs", "bangs", "pigtails", "undercut", "wet", "windblown"],
    "gender": ["woman", "man", "girl", "boy", "person", "nonbinary person"],
    "age": ["child", "teen", "20s", "30s", "40s", "50s", "60s", "elderly", "baby", "young adult"],
    "role": ["worker", "dancer", "musician", "soldier", "farmer", "nurse", "student", "chef", "rider", "swimmer", "painter", "monk", "vendor", "guard", "traveller", "reader", "mother", "father", "bride", "athlete"],
    "clothing": ["raincoat", "suit", "dress", "kimono", "hoodie", "uniform", "overalls", "gown", "jacket", "sweater", "coat", "robe", "armour", "apron", "cloak", "tunic", "jeans", "sari", "parka", "vest"],
    "accessory": ["hat", "glasses", "scarf", "umbrella", "watch", "bag", "cane", "gloves", "mask", "crown", "headphones", "belt", "cap", "veil", "necklace", "earrings", "ring", "bracelet", "pin", "none"],
    "pose": ["standing", "sitting", "walking", "running", "lying", "kneeling", "leaning", "crouching", "jumping", "dancing", "turning", "reaching", "resting", "climbing", "floating", "bowing", "stretching", "crossed arms", "hands in pockets", "arms raised"],
    "gesture": ["waving", "pointing", "holding", "covering face", "hands clasped", "thumbs up", "shrug", "beckoning", "saluting", "clapping"],
    "setting": ["market street", "forest", "beach", "kitchen", "studio", "rooftop", "subway", "cathedral", "desert", "harbour", "cafe", "library", "garden", "mountain pass", "alley", "ballroom", "farm", "classroom", "river bank", "snowfield", "greenhouse", "stadium", "workshop", "bridge", "cliff", "meadow", "laboratory", "temple", "bus stop", "hospital ward", "lighthouse", "vineyard", "attic", "ruins", "pier", "swamp", "courtyard", "cave", "skate park", "night market"],
    "time": ["dawn", "morning", "noon", "afternoon", "golden hour", "dusk", "blue hour", "night", "midnight", "late evening"],
    "weather": ["rain", "snow", "fog", "clear", "overcast", "storm", "haze", "wind", "drizzle", "sleet", "heat shimmer", "frost"],
    "layout": ["single subject", "pair", "crowd", "empty", "cluttered", "sparse", "layered depth", "flat", "panoramic", "tight"],
    "light": ["soft window", "hard sun", "backlit", "rim light", "neon", "candle", "overcast", "spotlight", "golden", "moonlight", "fluorescent", "firelight", "studio flash", "dappled", "silhouette", "lantern", "screen glow", "street lamp", "low key", "high key", "from the left", "from the right", "from above", "from below"],
    "angle": ["eye level", "low angle", "high angle", "overhead", "worm's eye", "dutch tilt", "over the shoulder", "profile", "three quarter", "frontal"],
    "framing": ["close-up", "extreme close-up", "portrait", "waist up", "full body", "wide", "extreme wide", "macro", "detail", "medium shot", "two shot", "establishing"],
    "lens": ["24mm", "35mm", "50mm", "85mm", "135mm", "telephoto", "fisheye", "tilt-shift", "pinhole", "anamorphic"],
    "focus": ["on the eyes", "deep", "shallow", "soft", "selective", "on the hands", "on the background", "motion blur", "tack sharp", "bokeh"],
    "composition": ["rule of thirds", "centred", "symmetry", "golden spiral", "leading lines", "frame within frame", "diagonal", "triangular", "layered", "negative space", "subject right third", "subject left third", "horizon high", "horizon low", "radial", "grid", "s-curve", "fill the frame", "off balance", "mirrored"],
    "balance": ["balanced", "left heavy", "right heavy", "top heavy", "bottom heavy", "tense", "settled"],
    "line": ["vertical", "horizontal", "diagonal", "curved", "jagged", "converging", "broken", "flowing"],
    "shape": ["round", "angular", "organic", "geometric", "blocky", "spindly", "massive", "delicate"],
    "space": ["deep", "flat", "crowded", "open", "enclosed", "vast", "intimate", "airless"],
    "rhythm": ["repeating", "irregular", "alternating", "accelerating", "still", "staccato", "flowing"],
    "contrast": ["high", "low", "medium", "tonal", "colour", "harsh", "gentle"],
    "texture": ["smooth", "rough", "grainy", "glossy", "matte", "cracked", "velvet", "wet", "dusty", "woven", "brushy", "polished"],
    "medium": ["photograph", "oil painting", "watercolour", "ink", "charcoal", "pencil", "gouache", "acrylic", "fresco", "woodcut", "etching", "digital painting", "3d render", "pixel art", "collage", "bronze", "marble", "clay", "pastel", "tempera", "linocut", "gelatin silver print", "cyanotype", "polaroid", "film still", "mosaic", "tapestry", "stained glass", "vector art", "lithograph"],
    "style": ["realism", "impressionism", "expressionism", "cubism", "surrealism", "baroque", "renaissance", "ukiyo-e", "art nouveau", "art deco", "minimalism", "pop art", "bauhaus", "anime", "comic", "noir", "romanticism", "abstract", "fauvism", "pointillism", "brutalism", "folk art", "photorealism", "gothic", "rococo", "futurism", "dada", "street art", "naive", "symbolism"],
    "period": ["ancient", "medieval", "renaissance", "baroque", "19th century", "early 20th", "mid century", "1980s", "1990s", "contemporary", "futuristic", "timeless"],
    "feeling": ["calm", "tense", "amused", "wistful", "joyful", "lonely", "proud", "anxious", "serene", "playful", "grim", "hopeful", "weary", "defiant", "tender", "eerie", "triumphant", "bored", "curious", "mournful", "giddy", "stern", "cozy", "restless", "awed", "bitter", "gentle", "fierce", "shy", "nostalgic"],
    "tone": ["warm", "cold", "dry", "lyrical", "ironic", "solemn", "brash", "quiet", "sweet", "harsh"],
    "energy": ["still", "slow", "steady", "brisk", "frantic", "explosive", "drifting", "pulsing"],
    "temperature": ["warm", "cool", "neutral", "mixed"],
    "purpose": ["editorial", "advertising", "personal record", "archival", "fine art", "book illustration", "concept art", "album cover", "poster", "scientific record", "fashion", "product", "documentary", "devotional", "propaganda", "children's book", "game art", "portrait commission", "study", "protest"],
    "audience": ["a travel feature", "collectors", "children", "a newspaper", "a gallery", "the family", "a fashion house", "a game studio", "a church", "the public square", "students", "a museum", "a magazine cover", "a film crew", "a scientific journal"],
    "genre": ["portrait", "landscape", "still life", "street", "documentary", "fashion", "product", "architecture", "wildlife", "abstract", "sport", "food", "interior", "aerial", "underwater", "macro", "war", "history painting", "genre scene", "self portrait", "cityscape", "seascape", "nude study", "botanical", "scientific"],
    "moment": ["the instant before", "the instant after", "mid action", "a pause", "a departure", "an arrival", "a discovery", "a farewell", "a wait", "a celebration", "a loss", "a plain afternoon"],
}
VOCAB["eyes"] = VOCAB["colour"]
VOCAB["wear colour"] = VOCAB["colour"]
VOCAB["thing colour"] = VOCAB["colour"]
VOCAB["relation"] = list(RELATIONS)
# the closed lists by key; an attribute without a list is OPEN (any plain words of at most VAL_MAX bytes)
KEY_LIST = {"look hair": "hair", "look eyes": "eyes", "look expression": "expression", "person gender": "gender", "person age": "age",
            "person role": "role", "wear clothing": "clothing", "wear accessory": "accessory", "wear colour": "colour", "body pose": "pose",
            "body gesture": "gesture", "scene setting": "setting", "scene time": "time", "scene weather": "weather", "scene layout": "layout",
            "scene mood": "feeling", "shot light": "light", "shot angle": "angle", "shot framing": "framing", "shot lens": "lens",
            "shot focus": "focus", "thing colour": "colour", "form composition": "composition", "form balance": "balance", "form line": "line",
            "form shape": "shape", "form space": "space", "form rhythm": "rhythm", "form contrast": "contrast", "form texture": "texture",
            "form medium": "medium", "form style": "style", "form period": "period", "mood feeling": "feeling", "mood tone": "tone",
            "mood energy": "energy", "mood temperature": "temperature", "intent purpose": "purpose", "intent audience": "audience",
            "intent genre": "genre", "story moment": "moment"}


def keys() -> list:
    """Every key of the trie ("system attribute"), relation as its own key."""
    return [f"{s} {a}" for s, attrs in SYSTEMS.items() for a in attrs] + ["relation"]


def _base(name: str) -> str:
    return "colour" if name in ("eyes", "wear colour", "thing colour") else name


def list_of(key: str) -> str | None:
    """The sealed list a record key's value is checked against: its closed list, 'relation' for a rel<i> rel record, 'thing' for a
    thing's name or a relation's ends; None for an open key (its values are never sealed)."""
    if key in KEY_LIST:
        return _base(KEY_LIST[key])
    if key.startswith("rel") and key.endswith(" rel"):
        return "relation"
    if key == "thing name" or (key.startswith("rel") and key[-2:] in (" a", " b")):
        return "thing"
    return None


def sealed_used(recs: list, held: dict) -> list:
    """The (key, value) records whose value is sealed in the value's own list."""
    return [(k, v) for k, v in recs if list_of(k) in held and v in held[list_of(k)]]


def seal(frac: float = 0.1, seed: int = 2026) -> dict:
    """THE SEALED HELD-OUT VALUES: a tenth of every closed list (at least one value a list), drawn once from this seed, never given to
    the generating model's training rows, kept for the reads. -> {list name: [values]}."""
    rng = np.random.default_rng(seed)
    protected = {v for _, v in records(EXAMPLE["spec"])}    # the schema's example is a training row: its values stay open
    out = {}
    for name, vals in VOCAB.items():
        if name in ("eyes", "wear colour", "thing colour"):
            continue                                         # the colour list is sealed once, under "colour"
        pool = [v for v in vals if v not in protected]
        k = max(1, int(round(len(vals) * frac)))
        pick = sorted(pool[i] for i in rng.choice(len(pool), size=k, replace=False))
        out[name] = pick
    return out


def lists_for(held: dict, split: str = "train") -> dict:
    """The lists a generating model may draw from: the training lists without the sealed values (split 'train'), or the sealed values
    alone (split 'held', for the reads' rows)."""
    out = {}
    for name, vals in VOCAB.items():
        sealed = set(held.get(_base(name), ()))
        out[name] = [v for v in vals if (v in sealed) == (split == "held")]
    return out


def write_vocab(folder: str, frac: float = 0.1, seed: int = 2026) -> dict:
    """vocab/trie.json (the systems, the keys, the lists, the counts, the rules) and vocab/held.json (the seals) under `folder`."""
    os.makedirs(os.path.join(folder, "vocab"), exist_ok=True)
    held = seal(frac, seed)
    trie = dict(version=VERSION, repo=REPO, depths=DEPTHS, domains=DOMAINS, steps=STEPS, needs=NEEDS, systems=SYSTEMS, relations=RELATIONS,
                keys=keys(), key_list=KEY_LIST, lists=VOCAB, counts=COUNTS, frame_sizes=FRAME_SIZES, rules=RULES, val_max=VAL_MAX,
                key_max=KEY_MAX, held_frac=frac, held_seed=seed)
    with open(os.path.join(folder, "vocab", "trie.json"), "w", encoding="utf-8") as fh:
        json.dump(trie, fh, indent=1, ensure_ascii=False)
    with open(os.path.join(folder, "vocab", "held.json"), "w", encoding="utf-8") as fh:
        json.dump(dict(version=VERSION, frac=frac, seed=seed, held=held), fh, indent=1, ensure_ascii=False)
    return held


# ------------------------------------------------------------------ the records of a row
def records(spec: dict) -> list:
    """A spec's records as (key, value) pairs: "system attribute" -> value for the systems, and for each relation three records
    "rel<i> a", "rel<i> rel", "rel<i> b" (the renderer's form; a relation counts as one record in the depth counts)."""
    out = []
    for system, attrs in spec.items():
        if system == "relation":
            for i, r in enumerate(attrs or [], 1):
                out += [(f"rel{i} a", str(r.get("a", ""))), (f"rel{i} rel", str(r.get("rel", ""))), (f"rel{i} b", str(r.get("b", "")))]
        elif isinstance(attrs, dict):
            for a, v in attrs.items():
                out.append((f"{system} {a}", str(v)))
    return out


def n_records(spec: dict) -> int:
    return sum(1 for s, attrs in spec.items() if s != "relation" and isinstance(attrs, dict) for _ in attrs) + len(spec.get("relation") or [])


# ------------------------------------------------------------------ the spec sampler (the facts; the model adds mood, intent, story)
THINGS = ["umbrella", "bicycle", "lantern", "fish stall", "window", "mirror", "dog", "cat", "horse", "boat", "car", "book", "cup", "chair",
          "table", "tree", "lamp", "door", "bench", "sign", "flag", "bottle", "flower", "clock", "bird", "ladder", "rope", "basket",
          "fence", "wall", "statue", "fountain", "train", "kite", "drum", "violin", "sword", "candle", "suitcase", "camera"]
VOCAB["thing"] = THINGS
# the order the sampler fills a spec in: the first attributes of the first systems come first, so a simple spec is a person seen
FILL = [("person", "gender"), ("look", "hair"), ("scene", "setting"), ("wear", "clothing"), ("look", "expression"), ("shot", "framing"),
        ("scene", "time"), ("wear", "colour"), ("shot", "light"), ("person", "age"), ("body", "pose"), ("form", "composition"),
        ("thing", "name"), ("scene", "weather"), ("shot", "angle"), ("form", "palette"), ("thing", "colour"), ("form", "contrast"),
        ("look", "eyes"), ("shot", "lens"), ("form", "medium"), ("shot", "focus"), ("form", "style"), ("person", "role"),
        ("wear", "accessory"), ("thing", "position"), ("form", "texture"), ("body", "gesture"), ("scene", "layout"), ("form", "line")]
SAMPLE_RECORDS = {"simple": (1, 4), "medium": (6, 10), "deep": (12, 22)}      # the sampler's share; the model's mood/intent/story fill the rest
SAMPLE_RELATIONS = {"simple": 0, "medium": 1, "deep": 2}


def sample_spec(rng, depth: str, lists: dict | None = None) -> dict:
    """A hidden image's FACTS drawn from the lists (consistent by construction: the colour of the clothing is one colour, the thing
    has a name before a colour, a relation joins the person and the thing). The interpretive systems (mood with its evidence, intent,
    story) are the generating model's to add. lists: the training lists (lists_for(held, 'train')) or the held lists."""
    lists = lists or lists_for(seal(), "train")
    lo, hi = SAMPLE_RECORDS[depth]
    n = int(rng.integers(lo, hi + 1))                        # the records in all: attributes + the count + the relations
    k_rel = SAMPLE_RELATIONS[depth]
    with_count = depth != "simple"
    n_attrs = n - k_rel - (1 if with_count else 0)
    things = lists["thing"]
    spec = {}
    for system, attr in FILL[:n_attrs]:
        key = f"{system} {attr}"
        if key == "thing name":
            val = things[int(rng.integers(len(things)))]
        elif key == "form palette":
            c = lists["colour"]
            for _ in range(20):
                i, j = rng.choice(len(c), size=2, replace=False)
                val = f"{c[i]} and {c[j]}"
                if len(val) <= VAL_MAX:
                    break
        elif key == "thing position":
            val = ("left", "right", "centre", "foreground", "background")[int(rng.integers(5))]
        else:
            lst = lists[KEY_LIST[key]] if key in KEY_LIST else lists["colour"]
            val = lst[int(rng.integers(len(lst)))]
        spec.setdefault(system, {})[attr] = val
    if with_count:
        spec.setdefault("person", {})["count"] = "1"
    rels = []
    a = spec.get("person", {}).get("gender", "figure")
    b = spec.get("thing", {}).get("name") or things[int(rng.integers(len(things)))]
    for _ in range(k_rel):
        rel = lists["relation"][int(rng.integers(len(lists["relation"])))]
        if not rels:
            rels.append({"a": a, "rel": rel, "b": b})
        else:
            c = things[int(rng.integers(len(things)))]
            rels.append({"a": b, "rel": rel, "b": c})
    if rels:
        spec["relation"] = rels
    return spec


def sample_series(rng, lists: dict | None = None) -> dict:
    """Three nested specs on one hidden image: deep drawn, medium a subset of it, simple a subset of the medium (rule 7 by
    construction). -> {"simple": spec, "medium": spec, "deep": spec}."""
    deep = sample_spec(rng, "deep", lists)
    def cut(spec, depth):
        lo, hi = SAMPLE_RECORDS[depth]
        n = int(rng.integers(lo, hi + 1))
        k = SAMPLE_RELATIONS[depth] if spec.get("relation") else 0
        with_count = depth != "simple" and "count" in spec.get("person", {})
        n_attrs = n - k - (1 if with_count else 0)
        out, kept = {}, 0
        for system, attr in FILL:
            if kept >= n_attrs:
                break
            if system in spec and attr in spec[system]:
                out.setdefault(system, {})[attr] = spec[system][attr]
                kept += 1
        if with_count:
            out.setdefault("person", {})["count"] = spec["person"]["count"]
        if k:
            out["relation"] = spec["relation"][:k]
        return out
    medium = cut(deep, "medium")
    return {"simple": cut(medium, "simple"), "medium": medium, "deep": deep}


def _ascii(s: str) -> bool:
    try:
        b = s.encode("ascii")
    except UnicodeEncodeError:
        return False
    return all(32 <= c < 127 for c in b)


def validate(row: dict, held: dict | None = None) -> list:
    """The plan's rules on one row. -> [(rule, message)], empty when the row passes. held: the seals; a training row (held.values empty)
    must not use a sealed value, a held row must use at least one."""
    p = []
    depth = row.get("depth")
    if depth not in DEPTHS:
        return [(3, f"depth {depth!r} is not one of {DEPTHS}")]
    for f in ("id", "series", "rank", "domain", "spec", "thoughts", "wordings", "questions", "held", "source"):
        if f not in row:
            p.append((0, f"missing field {f!r}"))
    if p:
        return p
    if row["domain"] not in DOMAINS:
        p.append((0, f"domain {row['domain']!r} is not one of {DOMAINS}"))
    spec = row["spec"]
    # rule 2: keys and values
    recs = records(spec)
    known = set(keys())
    for system, attrs in spec.items():
        if system not in SYSTEMS and system != "relation":
            p.append((2, f"unknown system {system!r}"))
        elif system == "relation":
            for r in attrs or []:
                if not isinstance(r, dict) or not all(k in r for k in ("a", "rel", "b")):
                    p.append((2, f"a relation needs a, rel, b: {r!r}"))
                elif r["rel"] not in RELATIONS:
                    p.append((2, f"relation {r['rel']!r} is not in the list"))
        else:
            for a in attrs:
                if f"{system} {a}" not in known:
                    p.append((2, f"unknown key {system + ' ' + a!r}"))
    for k, v in recs:
        if not isinstance(v, str) or not v or v != v.strip() or v != v.lower() or not _ascii(v):
            p.append((2, f"value of {k!r} is not plain lower-case ASCII: {v!r}"))
        elif len(v.encode("ascii")) > VAL_MAX:
            p.append((2, f"value of {k!r} is {len(v)} bytes, over {VAL_MAX}: {v!r}"))
        if len(k.encode("ascii", "replace")) > KEY_MAX:
            p.append((2, f"key {k!r} is over {KEY_MAX} bytes"))
        lst = KEY_LIST.get(k)
        if lst and v not in VOCAB[lst] and v != "none":
            p.append((2, f"value of {k!r} is not in the closed list {lst!r}: {v!r}"))
    # rule 3: the counts
    c = COUNTS[depth]
    nr, nt, nq = n_records(spec), len(row["thoughts"]), len(row["questions"])
    for name, n, (lo, hi) in (("records", nr, c["records"]), ("thoughts", nt, c["thoughts"]), ("questions", nq, c["questions"])):
        if not lo <= n <= hi:
            p.append((3, f"{depth}: {n} {name}, the depth wants {lo}-{hi}"))
    # the thoughts: a step from the list, a text
    text_all = " ".join(str(t.get("text", "")) for t in row["thoughts"]).lower()
    for t in row["thoughts"]:
        if t.get("step") not in STEPS:
            p.append((1, f"thought step {t.get('step')!r} is not one of {STEPS}"))
        if not str(t.get("text", "")).strip():
            p.append((1, "a thought without text"))
    # rule 1: no pixels, no seeing the file
    words = " ".join([text_all] + [str(v) for v in row["wordings"].values()]).lower()
    for b in BANNED:
        if b in words:
            p.append((1, f"the text speaks of the image as seen: {b!r}"))
    # rule 4: the questions
    recmap = {}
    for k, v in recs:
        recmap.setdefault(k, set()).add(v)
    for q in row["questions"]:
        k, v, need = q.get("key"), q.get("value"), q.get("needs")
        if need not in NEEDS:
            p.append((4, f"question needs {need!r} is not one of {NEEDS}"))
            continue
        if need == "record":
            if k not in recmap or v not in recmap[k]:
                p.append((4, f"record question {k!r} = {v!r} is not a record of the spec"))
        elif need == "unknown":
            if v != "unknown":
                p.append((4, f"an unknown question's value must be 'unknown', not {v!r}"))
            if k in recmap:
                p.append((4, f"unknown question {k!r} is a key the spec holds"))
        else:
            if not v or str(v).lower() not in text_all:
                p.append((4, f"chain question {k!r} = {v!r}: the thoughts never state the value"))
            elif k in recmap and v in recmap[k]:
                p.append((4, f"chain question {k!r} = {v!r} is a record of the spec (needs 'record')"))
    if depth == "deep" and not any(q.get("needs") == "unknown" for q in row["questions"]):
        p.append((4, "a deep row asks one unknown question"))
    if depth == "deep" and not any(q.get("needs") == "chain" for q in row["questions"]):
        p.append((4, "a deep row asks one question that needs the chain"))
    # rule 5: the wordings
    w = {k: v for k, v in row["wordings"].items() if str(v or "").strip()}
    if len(w) < c["wordings"]:
        p.append((5, f"{depth}: {len(w)} wordings, the depth wants {c['wordings']}"))
    if depth == "deep" and not all(k in w for k in ("caption", "prompt", "tags")):
        p.append((5, "a deep row carries caption, prompt and tags"))
    # rule 6: a mood names its evidence
    mood = spec.get("mood") or {}
    if mood:
        ev = str(mood.get("evidence", "")).strip()
        if not ev:
            p.append((6, "the mood has no evidence"))
        elif ev not in recmap or ev.startswith("mood "):
            p.append((6, f"the mood's evidence {ev!r} is not a record of the spec"))
    # the held field and the seals
    h = row["held"]
    if not isinstance(h, dict) or not all(k in h for k in ("layout", "wordings", "values")):
        p.append((0, "held needs layout, wordings, values"))
    elif held:
        used = sorted({v for _, v in sealed_used(recs, held)})
        if h["values"] and not used:
            p.append((7, "a held row uses no sealed value"))
        if not h["values"] and used:
            p.append((7, f"a training row uses sealed values: {used}"))
    src = row["source"]
    if not isinstance(src, dict) or not src.get("kind") or not src.get("license"):
        p.append((8, "source needs kind and license"))
    return p


def validate_series(rows: list) -> list:
    """Rule 7 across a series: the rows sorted by rank; each spec's records within the next's. -> [(7, message)]."""
    p = []
    rows = sorted(rows, key=lambda r: int(r.get("rank", 0)))
    for a, b in zip(rows, rows[1:]):
        ra, rb = set(records(a["spec"])), set(records(b["spec"]))
        if not ra <= rb:
            p.append((7, f"{a['id']} ({a['depth']}) holds records {sorted(ra - rb)[:4]} that {b['id']} ({b['depth']}) lacks"))
        if a["depth"] == b["depth"]:
            p.append((7, f"{a['id']} and {b['id']} share the depth {a['depth']}"))
    return p


def check_file(path: str, held: dict | None = None) -> dict:
    """The pilot checker: every row of a JSON Lines file validated, the series checked. -> dict(rows, ok, unparsed, by_rule, by_depth,
    problems); rows = the parsed rows, ok = those with no problem, unparsed = the lines that are not JSON."""
    rows, problems, by_rule, by_depth, bad, unparsed = [], [], Counter(), Counter(), 0, 0
    with open(path, encoding="utf-8") as fh:
        for n, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                problems.append((n, 0, f"not JSON: {e}"))
                by_rule[0] += 1
                unparsed += 1
                continue
            rows.append(row)
            by_depth[row.get("depth")] += 1
            ps = validate(row, held)
            if ps:
                bad += 1
            for rule, msg in ps:
                problems.append((n, rule, msg))
                by_rule[rule] += 1
    series = {}
    for r in rows:
        series.setdefault(r.get("series"), []).append(r)
    for sid, members in series.items():
        if len(members) > 1:
            for rule, msg in validate_series(members):
                problems.append((sid, rule, msg))
                by_rule[rule] += 1
    return dict(rows=len(rows), ok=len(rows) - bad, unparsed=unparsed, by_rule=dict(sorted(by_rule.items())), by_depth=dict(by_depth),
                problems=problems)


# ------------------------------------------------------------------ the prompt the generating model writes rows against
EXAMPLE = {
    "id": "art-000123", "series": "s-00041", "depth": "medium", "rank": 2, "domain": "photograph", "genre": "street portrait",
    "spec": {"person": {"gender": "woman"}, "look": {"hair": "short", "expression": "wry"},
             "wear": {"clothing": "raincoat", "colour": "yellow"},
             "scene": {"setting": "market street", "time": "dusk", "weather": "rain"},
             "shot": {"light": "neon", "framing": "waist up"},
             "form": {"composition": "subject right third"},
             "relation": [{"a": "woman", "rel": "in front of", "b": "fish stall"}],
             "mood": {"feeling": "amused", "evidence": "look expression"}},
    "thoughts": [{"step": "see", "text": "A woman in a yellow raincoat, short hair, a wry half smile."},
                 {"step": "place", "text": "She stands on a market street at dusk, in the rain, in front of a fish stall."},
                 {"step": "form", "text": "Neon lights her face; she sits in the right third of the frame, seen from the waist up."},
                 {"step": "mean", "text": "The wry expression reads as amused rather than weary. With dusk and neon the frame is dark around a single subject: an inference from the time and the light, not a record."},
                 {"step": "say", "text": "The composed wording follows."}],
    "wordings": {"caption": "A woman with short hair and a wry smile, in a yellow raincoat, on a rainy market street at dusk, lit by neon, in front of a fish stall.",
                 "prompt": "street portrait, woman, short hair, wry smile, yellow raincoat, rainy market street at dusk, neon light, fish stall behind, waist up, subject right third",
                 "tags": "1girl, short hair, smile, raincoat, rain, dusk, neon lights, market, portrait, upper body"},
    "questions": [{"key": "look hair", "value": "short", "needs": "record"},
                  {"key": "rel1 b", "value": "fish stall", "needs": "record"},
                  {"key": "scene layout", "value": "single subject", "needs": "chain"},
                  {"key": "wear accessory", "value": "unknown", "needs": "unknown"}],
    "held": {"layout": None, "wordings": False, "values": []},
    "source": {"kind": "synthetic", "model": "sonnet", "seed_dataset": None, "license": "CC BY 4.0"}}


def prompt(depth: str, domain: str, held: dict | None = None, split: str = "train", series: bool = False, n: int = 1, seed_spec: dict | None = None) -> str:
    """The instruction for one generation call: the task, the schema with the example, the rules, the depth's counts, the lists the
    model may draw from (the training lists without the sealed values, or the sealed values for a held row), the series form."""
    held = held or seal()
    lists = lists_for(held, split)
    c = COUNTS[depth]
    want = (f"{n} series of three rows (ranks 1, 2, 3 = simple, medium, deep) on ONE hidden image each, the simple row's facts a subset "
            f"of the medium's, the medium's of the deep's; the wordings grow with the depth" if series else
            f"{n} row(s) of depth {depth}: {c['records'][0]}-{c['records'][1]} records, {c['thoughts'][0]}-{c['thoughts'][1]} thoughts, "
            f"{c['questions'][0]}-{c['questions'][1]} questions, at least {c['wordings']} wording(s)"
            + (" including caption, prompt and tags, one unknown question and one that needs the chain" if depth == "deep" else ""))
    spec_line = ("Start from THIS hidden spec: keep every record as given; add mood (a feeling with its evidence), intent and story as the "
                 "depth asks; add no other facts:\n" + json.dumps(seed_spec, ensure_ascii=False) if seed_spec else
                 "Invent the hidden spec first (consistent: counts match the subjects, every colour is bound to a part, the light has a direction, "
                 "a palette names two colours), then write the thoughts, then the wordings, then the questions.")
    rules = "\n".join(f"  {k}. {v}" for k, v in RULES.items())
    lists_txt = "\n".join(f"  {k}: {', '.join(v)}" for k, v in lists.items() if v)
    return f"""You write rows of THE ART THOUGHT DATASET ({REPO}): text only, one act of thinking about an image that nobody sees.
Domain: {domain}. Write {want}. Output JSON Lines, one JSON object a row, nothing else.

{spec_line}

THE SCHEMA (an example row of depth medium):
{json.dumps(EXAMPLE, ensure_ascii=False)}

The fields: id (art-NNNNNN), series (s-NNNNN), depth (simple | medium | deep), rank (1 | 2 | 3 within a series), domain, genre, spec
(the hidden image as records: system -> attribute -> value, plus relation as a list of {{a, rel, b}}), thoughts (a list of steps, each
a step kind from {', '.join(STEPS)} and a text of one to four sentences), wordings (caption, prompt, tags, critique: what an artist
would say; a prompt is the comma-separated phrase list that would recreate the picture; tags use the booru register where it exists),
questions (key, value, needs: record = the value is a record of the spec; chain = the value is stated by the thoughts and is not a
record of the spec, under a trie key; unknown = a key the spec does not hold, the value 'unknown'), held (layout null, wordings false,
values {'[the sealed values used]' if split == 'held' else '[]'}),
source (kind synthetic, model, seed_dataset null or its name, license).

THE RULES (a row that breaks one is rejected by the validator):
{rules}
The systems and their attributes: {json.dumps(SYSTEMS)}
The relations: {', '.join(RELATIONS)}
Record values: plain lower-case words, at most {VAL_MAX} bytes, from these lists where a list exists (an attribute without a list is open):
{lists_txt}
A question's key is a trie key ("look hair") or a relation record ("rel1 a", "rel1 rel", "rel1 b").
Depth sets the frame the thought will live in: simple 128-256 bytes, medium 512, deep 1,024-2,048; write each thought's text to fit:
simple thoughts one or two sentences, deep thoughts up to four.
Before you answer, re-read every row against the rules and fix it."""


PROMPT_DOC = f"""# Writing rows of the art thought dataset

The dataset: {REPO}. The plan: history/plans/2026-10-19_art_thought_dataset_plan.md in the research record. The tooling:
`geolip.nano_beatrix.thought_pack` (this document is written by it; the tests keep them equal).

A row is one act of THINKING ABOUT AN IMAGE that nobody sees: a hidden image specification in the systems trie, a chain of thoughts
at a stated depth, the wordings an artist would use, and questions whose answers sit in the records. Text only; nothing refers to
pixels or to seeing a file.

## The depths
| depth | the thought holds | records | thoughts | questions | wordings | frame |
|---|---|---|---|---|---|---|
| simple | one observation or one naming | 1-4 | 1-2 | 1-2 | 1 | 128-256 bytes |
| medium | subjects with attributes, their relations, the form, a definition, the behavior shown | 6-14 | 3-5 | 3-5 | 1 | 512 bytes |
| deep | the whole composition, mood and intent with evidence, a comparison or a change, the story, a critique | 14-30 | 6-10 | 5-8 (one unknown, one needing the chain) | 3 (caption, prompt, tags) | 1,024-2,048 bytes |

Rows come in SERIES of three on one hidden image: ranks 1, 2, 3 = simple, medium, deep, each spec a subset of the next.

## The rules
{chr(10).join(f'{k}. {v}' for k, v in RULES.items())}

## The systems trie
{chr(10).join(f'- {s}: {", ".join(a)}' for s, a in SYSTEMS.items())}
- relation: a list of {{a, rel, b}} with rel from: {', '.join(RELATIONS)}

## The schema (an example row of depth medium)
```json
{json.dumps(EXAMPLE, ensure_ascii=False, indent=1)}
```

## The protocol
1. The vocabulary file and the sealed held-out values first (`python -m geolip.nano_beatrix.thought_pack --vocab=<folder>`): a tenth of
   every closed list is sealed; training rows never use a sealed value, the held rows for the reads use them.
2. The spec before the thought: draw or receive the hidden spec, then the thoughts, then the wordings, then the questions.
3. A series is written together so the subsets hold.
4. The diversity knobs are set per batch and logged: the depth mix, the domain, the genres, the register, the share of unknown
   questions, the share of negative facts ("accessory: none"), relations in every medium and deep row.
5. Every row passes the validator (`--check=<rows.jsonl>`); the batch log keeps the problems by rule.
6. A pilot of 2,000 rows comes before volume.

## The prompt
`python -m geolip.nano_beatrix.thought_pack --prompt=1 --depth=medium --domain=photograph` prints the instruction for one call (the
schema, the rules, the lists the model may draw from); `--series=1` asks for series of three; `--split=held` writes rows on the sealed
values for the reads. `--sample=N --series=1 --seed=S` draws N nested spec triples (the facts) for the model to think from, so the lists
are covered evenly; the model adds mood (with its evidence), intent and story.
"""


def write_doc(path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(PROMPT_DOC)


def main(argv=None) -> int:
    args = {a.split("=")[0].lstrip("-"): a.split("=", 1)[1] if "=" in a else "1" for a in (argv or sys.argv[1:])}
    if "vocab" in args:
        held = write_vocab(args["vocab"])
        print(f"vocab/trie.json and vocab/held.json written under {args['vocab']}: {len(keys())} keys, {sum(len(v) for v in held.values())} sealed values")
    if "check" in args:
        held = None
        if args.get("held"):
            held = json.load(open(args["held"], encoding="utf-8"))["held"]
        res = check_file(args["check"], held)
        print(f"{res['rows']} rows, {res['ok']} pass, {res['unparsed']} unparsed; by depth {res['by_depth']}; problems by rule {res['by_rule']}")
        for n, rule, msg in res["problems"][:int(args.get("show", "40"))]:
            print(f"  row {n} rule {rule}: {msg}")
    if "prompt" in args:
        held = json.load(open(args["held"], encoding="utf-8"))["held"] if args.get("held") else None
        print(prompt(args.get("depth", "medium"), args.get("domain", "photograph"), held, args.get("split", "train"), args.get("series") == "1",
                     int(args.get("n", "1"))))
    if "sample" in args:
        held = json.load(open(args["held"], encoding="utf-8"))["held"] if args.get("held") else seal()
        lists = lists_for(held, args.get("split", "train"))
        rng = np.random.default_rng(int(args.get("seed", "0")))
        for _ in range(int(args["sample"])):
            print(json.dumps(sample_series(rng, lists) if args.get("series") == "1" else sample_spec(rng, args.get("depth", "medium"), lists),
                             ensure_ascii=False))
    if "doc" in args:
        write_doc(args["doc"])
        print("written", args["doc"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
