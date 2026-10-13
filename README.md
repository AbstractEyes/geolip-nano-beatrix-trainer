# geolip-nano-beatrix-trainer

The nano Beatrix trainer: small byte-level aleph crafts (under ten million parameters) trained from birth to measure what a craft
that small can hold and how fast it trains. The crafts, the trainer core and the model code live in the library
[AbstractEyes/alephllm](https://github.com/AbstractEyes/alephllm) (package `geolip.alephllm`); this repository holds the bed that
drives them, as `geolip.nano_beatrix`. What the bed produces (weights, run records, reports) lives on the Hub at
[AbstractPhil/alephllm-nano-beatrix-training](https://huggingface.co/AbstractPhil/alephllm-nano-beatrix-training).

## What is here

| module | what it does |
|---|---|
| `geolip/nano_beatrix/bed_train.py` | trains one arm of the bed from birth (or continues a saved craft) on fineweb-edu text with planted records or framed rows; reads the held-out bits a byte by byte class, by position in the window and by frame, the planted fetch at several distances and by frame, the expert dispatch census, the time cost; writes a JSON record and the weights |
| `geolip/nano_beatrix/bed_frames.py` | the framed row composer: frames of generic sizes opened and closed by the byte registry's tags, nested and overlapping frames, records to look up from the window or to recall from the weights, the frame masks and bits by frame |
| `geolip/nano_beatrix/bed_table.py` | the results index: one table per context from a folder of run records, the seed means, the bits by position |
| `geolip/nano_beatrix/bed_timing.py` | the back-to-back timing of the arms on one card (train step and inference row) |
| `geolip/nano_beatrix/bed_corpus.py` | builds the bed's corpus and held-out file from fineweb-edu |
| `geolip/nano_beatrix/bed_runner.py` | runs a grid of arms in sequence on one card (the launch point for a rented card): the grid expands from comma lists, finished runs are remembered so a restart resumes, a stop file ends the grid between runs, each finished run can ship its record and weights to the Hub repo. From Python, `run(spec, out, push, parallel=N)` streams every run with a grid bar and a step bar per running child fed by the trainer's own progress lines; `parallel` shares the card between runs (a 9M craft takes ~5 GB a run) |

The arms: an attention form (softmax with a learned position table, softmax with rotary, the splat hub with its aleph address, the
hub with rotary on its codes, the dual hub with two masses, hub and softmax blocks alternating) times a bank form (the aleph address
alone, an installed byte-class expert table, or both).

The nano twin pair has names: **Castor** is the rotary softmax body (`--arm=castor-<bank>`, the same craft as `smr`), the mortal
twin whose memory grows with the window; **Pollux** is the all-rotary alternating body (`--arm=pollux-<bank>`, the same as `altn`),
the immortal twin whose hub half keeps a constant decode memory. The twins' bank form of record is the byte-class table beside
the aleph address (`--arm=castor-both`, `--arm=pollux-both`): on two seeds at 4,096 it read the best text on both bodies and the
steadiest far copy on Pollux, at +9% (Pollux) and +23% (Castor) of a step over the table alone. The two full-size teacher forms the timing tool measures beside
them are **Virgil** (the splat hub body, 237M) and **Matelda** (the softmax control, 257M), the pair trained in the mini-beatrix repo.

## Install

```bash
pip install -r requirements.txt
pip install -e .
```

The library is pinned to a commit of its `feat/night-rigs` branch (the rotary flags, the byte-class table, the dual hub, the bits-by-position
gauge). The pin moves to a tagged release when that branch merges.

## Run

The bed reads its corpus (`<out>/bed/bed_corpus.bin` + offsets) and a held-out file (`heldout_fineweb_edu_head2048.jsonl` in `<out>` or
`<out>/bed`) from an output folder; `python -m geolip.nano_beatrix.bed_corpus <out>/bed` builds all three from fineweb-edu (needs the
`corpus` extra: `pip install "geolip-nano-beatrix-trainer[corpus]"`). On a rented card, `colab/nano_frame_grid.ipynb` does the whole
sequence in the house format: C1 pins and the decision block, C2 the preflight (the data, the plan, a short run for the price line, the
push probe), C3 the session with its bars, C4 the results table (`bed_table.render`). Open it in Colab straight from GitHub:
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AbstractEyes/geolip-nano-beatrix-trainer/blob/main/colab/nano_frame_grid.ipynb)
The second pass, `colab/nano_frame_pass2.ipynb`, runs the in-window capacity axis (4 / 8 / 16 records a lookup frame, frames to
1,024 bytes) and the scrutiny set (nesting 0.6, conflicting definitions 0.3, the question on a nested frame or a conflicted key) at
4,096 on both twins, two seeds, one row in four a plain document: two grids in one decision block, run in sequence by C3:
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AbstractEyes/geolip-nano-beatrix-trainer/blob/main/colab/nano_frame_pass2.ipynb)

Then:

```bash
python -m geolip.nano_beatrix.bed_train --out=<out> --arm=smr-fixed --seed=1 --ctx=4096 --mb=8 --steps=12000 --plant=1.0 --records=8 --dists=64,512,1024,2048,3072
```

Framed rows: add `--frames=dyn` (dynamic sectors of generic sizes; `--sizes=32,128,256 --frame_records=4 --nest=0.3 --overlap=0.0
--ask=any|nested|overlap|weights --facts=M --register=caption|math|code --tags=mode|esc`) or `--frames=A` (the fixed reference
layout; `--n_lookup --n_bulk --ask=lookup|bulk|weights`). Without `--frames` the rows are the planted rows, bit for bit.

Every run prints its job size, the time spent and the time left, and ends with the gauges. The record lands at `<out>/bed/bed_<tag>.json`
and the weights at `<out>/bed/<tag>/`. A continued run (`--init=<a previous tag>`) carries its starting craft in its own tag
(`..._from_<init>`), so two continuations of one recipe from different crafts never share a name. `python -m geolip.nano_beatrix.bed_table <out>/bed --pairs` builds the results index.

## Tests

```bash
pytest -q
```

The frame composer's tests run on the CPU without a model.

## Status

0.1.16 (2026-10-13): three gauges for the nano train's curriculum, pure functions in bed_frames and keys in every record: the canary
rows (a bare document; the register tag and a question cue before it; one flat 32-byte frame before it; bits a byte whole and by
window), the formation verdict (the near band's online value bits at the end against the register's uniform rate, log2 of its
alphabet, with the whole value in every lookup group; the first step under the rate) and the plain-collapse flag (any window past
the head above 2.5 bits a byte). Flags are log lines and record fields, never a halt.

0.1.15 (2026-10-13): the runner's bar labels carry the grid axes that vary (records a frame, a strict ask, nesting and conflicts, facts);
its child processes run with TensorFlow's logging off (the noise seen on a Colab image); both notebooks default to ONE run at a time,
because four at once only divided the RTX PRO 6000 (4 x ~214k tok/s = one run's 863k): parallel runs pay only when a single run leaves
the card idle, which the step bars' summed tok/s shows. Both notebooks pin v0.1.15.

0.1.14 (2026-10-13): the composer works one step ahead of the card (`--prefetch=1` by default, `--prefetch=0` the serial loop): the
batches are identical by construction (every step seeds its own rng) and were proven bit-for-bit on the CPU; on the 4090 at 1,024 x 32
under the mixed diet the framed step went from 369k to 399k tok/s (+8%). Both notebooks pin v0.1.14.

0.1.11 (2026-10-13): the bed as it ran the first night, the frame composer and its wiring, the runner with its Python entry point and bars,
the corpus and table tools callable, the Colab notebook in the house format; the twins named and their bank of record set. 92 to 100 tests on the CPU.

## Licence

MIT (LICENSE). The weights on the Hub carry their own licence (CC BY-NC-SA 4.0, stated there).
