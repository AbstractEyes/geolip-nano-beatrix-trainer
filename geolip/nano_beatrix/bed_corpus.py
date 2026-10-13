# -*- coding: utf-8 -*-
"""bed_corpus.py - the small bed's training text (2026-10-12): fineweb-edu sample-10BT (public, tokenless), the reserved head of
2,048 rows SKIPPED (that head is the held-out file heldout_fineweb_edu_head2048.jsonl), documents of 400..24,000 UTF-8 bytes taken
in stream order until TARGET bytes. Writes OUT/bed_corpus.bin (uint8, the documents back to back) and OUT/bed_corpus_offsets.npy
(int64, len = docs + 1). Progress lines carry the job size, the time spent and the time left.
Usage: bed_corpus.py OUT [TARGET_BYTES]"""
import os
import sys
import time

import numpy as np

os.environ["HF_TOKEN"] = ""
os.environ.pop("HF_HUB_OFFLINE", None)
OUT = sys.argv[1]
TARGET = int(float(sys.argv[2])) if len(sys.argv) > 2 else 160_000_000
SKIP = 2048
os.makedirs(OUT, exist_ok=True)
T0 = time.time()


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')} +{time.time() - T0:6.0f}s] {msg}", flush=True)


from datasets import load_dataset  # noqa: E402

log(f"opening fineweb-edu sample-10BT (streaming), skipping the reserved head of {SKIP:,} rows; target {TARGET / 1e6:.0f} MB")
ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
ds = ds.select_columns(["text"]).skip(SKIP)
chunks, offs, total, n_docs, n_seen = [], [0], 0, 0, 0
next_mark = 10_000_000
for row in ds:
    n_seen += 1
    b = str(row["text"]).encode("utf-8", errors="replace")
    if len(b) < 400 or len(b) > 24_000:
        continue
    chunks.append(b)
    total += len(b)
    offs.append(total)
    n_docs += 1
    if total >= next_mark:
        el = time.time() - T0
        rate = total / max(el, 1e-6)
        log(f"{total / 1e6:6.1f} / {TARGET / 1e6:.0f} MB, {n_docs:,} docs ({n_seen:,} rows seen); {rate / 1e6:.1f} MB/s; left ~{(TARGET - total) / rate:.0f}s")
        next_mark += 10_000_000
    if total >= TARGET:
        break
buf = np.frombuffer(b"".join(chunks), dtype=np.uint8)
np.save(os.path.join(OUT, "bed_corpus_offsets.npy"), np.asarray(offs, dtype=np.int64))
buf.tofile(os.path.join(OUT, "bed_corpus.bin"))
log(f"done: {total / 1e6:.1f} MB, {n_docs:,} documents, mean {total / max(n_docs, 1):.0f} bytes; rows seen {n_seen:,}")
