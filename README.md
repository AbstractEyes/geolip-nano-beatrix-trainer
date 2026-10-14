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
| `geolip/nano_beatrix/bed_curriculum.py` | a curriculum train as chains of continuation blocks: each chain (an arm and a seed) goes through the stages in order, a stage runs in blocks, and each block's record decides by the stage's rule whether the stage takes another block or the chain moves on (the copy formed; the far reads still improving); the chains advance in rounds, one block at a time on the card; a restart replays the decisions from the records on disk or on the Hub; `run(STAGES, BASE, arms, seeds, out, push)` and `report(...)` (the boundary table) |
| `geolip/nano_beatrix/bed_pair.py` | how alike two crafts are on the same held-out rows: the KL between their next-byte distributions each way, how often their greedy next bytes agree, the debiased linear CKA of their residual streams block by block; the helpers that read a stored pair the same way (`load_craft`, `heldout_rows`, `solo_twins`) and the pair notebook's table (`report`) |

The arms: an attention form (softmax with a learned position table, softmax with rotary, the splat hub with its aleph address, the
hub with rotary on its codes, the dual hub with two masses, hub and softmax blocks alternating) times a bank form (the aleph address
alone, an installed byte-class expert table, or both). The dual-hub bodies: `hubd` (six dual hubs: each hub keeps a rotary address
mass, sharp on near records, and a plain one, distance-free, that reaches far records), `hubdb` (the same on the bigger address, two
books of 64 in 32), `altd` (Pollux's alternating layout with dual hubs in place of its rotary-only hubs), `d5s1` (five dual hubs and
one rotary softmax block, block 3) and `d4s2` (four dual hubs and two rotary softmax blocks, 1 and 3: `altd` with its last softmax block
a dual hub). With Canon layers (a short causal convolution with a residual on each block's mixer input and bank input, born zero;
Allen-Zhu, arXiv 2512.17351): `altdc` (`altd` with them), `hubdc` (`hubd` with them) and `polluxc` (Pollux with them).

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

The library is pinned to a commit of its `feat/canon-layers` branch (the `feat/night-rigs` branch's rotary flags, byte-class table, dual
hub and bits-by-position gauge, plus the Canon layers). The pin moves to a tagged release when those branches merge.

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
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AbstractEyes/geolip-nano-beatrix-trainer/blob/main/colab/nano_pass3.ipynb) (the third pass: the scrutiny reads by continuation, the sixteen-record second seeds, Pollux's hub address axis)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AbstractEyes/geolip-nano-beatrix-trainer/blob/main/colab/nano_pass4.ipynb) (the fourth pass: the plain, sized and sliced frames side by side, stage A at 1,024 then the 4,096 continuation, both twins, seeds 3 and 4)

The v4 train, `colab/nano_v4_train.ipynb`, takes both twins on two seeds through the curriculum block by block (`bed_curriculum`):
stage A, the copy, at 1,024 x 32 (12,000-step blocks, at most two); stage B, the long window and the count, at 4,096 x 8 (4,000-step
blocks while the reads of records more than 512 bytes back still improve, at most five); the anneal on the plain diet. Its frame form is
one setting in C1 (plain, sized or sliced), and C4 renders the boundary table:
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AbstractEyes/geolip-nano-beatrix-trainer/blob/main/colab/nano_v4_train.ipynb)

The twins trained together, `colab/nano_pair.ipynb`: Castor and Pollux in one process on the very same rows, with a cross term between
their next-byte predictions, in three forms (Pollux learns from Castor at the asked values' bytes; Pollux learns from Castor at every
byte; each learns from the other at every byte), seeds 1 and 2, the v4 train's stage A and one long-window block; C4 reads every pair
against the v4 train's solo blocks of the same seeds and reads how alike the twins end, paired and alone:
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AbstractEyes/geolip-nano-beatrix-trainer/blob/main/colab/nano_pair.ipynb)

Then:

```bash
python -m geolip.nano_beatrix.bed_train --out=<out> --arm=smr-fixed --seed=1 --ctx=4096 --mb=8 --steps=12000 --plant=1.0 --records=8 --dists=64,512,1024,2048,3072
```

Framed rows: add `--frames=dyn` (dynamic sectors of generic sizes; `--sizes=32,128,256 --frame_records=4 --nest=0.3 --overlap=0.0
--ask=any|nested|overlap|weights --facts=M --register=caption|math|code --tags=mode|esc`) or `--frames=A` (the fixed reference
layout; `--n_lookup --n_bulk --ask=lookup|bulk|weights`). Without `--frames` the rows are the planted rows, bit for bit.
With dynamic frames, `--sized=1` gives every open tag one byte naming its frame's generic size (a = 32, b = 64, c = 128, d = 256,
e = 512, f = 1,024, g = 2,048), so the close is countable; `--sliced=32` (with `--sized=1`) builds every frame larger than 32 bytes as
closed 32-byte slices back to back between its own open tag and END. Both are off by default, and the rows are then those of 0.1.20.
`--gauge_asks=nested,overlap` asks the nested and the redefined-key questions strictly on their own held-out rows after the main gauge,
whatever the training rows ask (`fetch_by_frame_asks` in the record).

A pair run: `--pair=pollux-both [--xc=one|mutual --xc_pos=value|all --xc_w=1 --xc_from=0 --pair_init=<the partner's starting craft>]`
trains a second craft beside `--arm` on the very same rows, each with its own Adam and the same schedule, plus a cross term: at the
selected bytes (the asked values' bytes, or every byte) the KL from one craft's next-byte distribution, held fixed, to the other's,
weighted like the byte's own target; `one`: the partner learns from the arm (the arm trains exactly as alone), `mutual`: each from the
other. Both crafts start from the run's seed, so with `--xc_w=0` each trains exactly as it would alone (a CPU smoke: the two records and
weights byte for byte those of the two solo runs). The run writes two records (the partner's first), each with a `pair` field: the
roles, the cross term's mean KL a kept byte, and the pair's reads on 32 held-out documents (`bed_pair.twin_reads`). Names: a pair
craft's tag carries `_pair-<partner>-<role><where>` (role t teaches, s learns, m both ways; where v the asked values, a every byte;
then `w<weight>` and `f<first step>` when not 1 and 0); a learning rate or warmup other than the defaults adds `_lr<rate>` / `_wu<steps>`.

Every run prints its job size, the time spent and the time left, and ends with the gauges. The record lands at `<out>/bed/bed_<tag>.json`
and the weights at `<out>/bed/<tag>/`. A continued run (`--init=<a previous tag>`) carries its starting craft in its own tag
(`..._from_<init>`), so two continuations of one recipe from different crafts never share a name. `python -m geolip.nano_beatrix.bed_table <out>/bed --pairs` builds the results index.

## Tests

```bash
pytest -q
```

The frame composer's tests run on the CPU without a model.

## Status

0.1.24 (2026-10-13): the pair run. `bed_train --pair=<arm>` trains two crafts on the very same rows with a cross term between their
next-byte predictions (one way or both ways, at the asked values' bytes or every byte; the module docstring and the run section above);
both start from the run's seed, so a zero-weight pair reproduces the two solo runs byte for byte (a CPU smoke: weights and every record
field but the clock); in the one-way forms the teaching craft trains exactly as alone (the same smoke: its weights byte for byte). The
trainer's single craft became a `Craft` and its gauges a per-craft `finish()`; the defaults are bit-identical to 0.1.23 (a CPU smoke of
each twin: the weights byte for byte, the record but the clock). `bed_pair` reads how alike two crafts are (the KL each way, the greedy
agreement, the debiased CKA by block) for a pair run and, from stored weights on the same rows, for two solo runs. The runner ships every
record a run writes (a pair: two) and fetches both starting crafts of a continuation pair (`init`, `pair_init`); the names carry a
learning rate and a warmup that are not the defaults. colab/nano_pair.ipynb runs the three forms against the v4 train's solo blocks.

0.1.23 (2026-10-13): the curriculum train. `bed_curriculum` runs chains of continuation blocks: a stage runs in blocks, each block a
continuation of the chain's previous block named by its stage (`_stA1`, `_stB3`), and each block's record decides the next by the
stage's rule: "formed" (the copy formed by the whole-value rule: move on; not formed: another block up to the stage's most, then move on
marked), "far_fall" (another block while the far band's online value bits fell by more than a threshold over the block's last window
of steps against the window before, exact from the curve's counts: `bed_frames.band_fall`), or one block. The decisions are pure
functions of the records, so a restart replays them and runs only the missing blocks; a block's record missing on disk is fetched from
the Hub when it is there; a record continued from another craft than the chain's previous block (a rule changed since it ran) stops the
walk with a message. The chains advance in rounds, one block at a time on the card, through one background shipper (`bed_runner.run`
takes a caller's shipper and then never waits on the Hub). The decisions and the boundary table ship at the end under
`bed/curriculum/`. `--gauge_asks=nested,overlap` reads the strict questions by the gauge alone (a craft trained on a strict ask learns a
stricter grammar, so the curriculum's rows ask any); a gauge row draws the asked feature at least as often as the scrutiny settings of
record (nesting 0.6, a key defined twice in 0.3 of the rows), so a craft trained on that ask with those settings is gauged on the very
same rows (a CPU smoke: identical groups). The defaults are bit-identical to 0.1.22 (a CPU smoke's weights byte for byte and its record
but the clock). colab/nano_v4_train.ipynb runs the train (both twins, two seeds; the frame form one setting).

0.1.22 (2026-10-13): the online value gauge reads windows, and a curriculum block names its stage. Every record's training curve
(`value_bits`, the value bytes' bits by distance band at each progress line) has been a mean since the run's first step: the reset meant
to make it a 400-step window sat inside the progress-line branch on a step it can never reach. `value_bits` keeps that meaning (every
stored record carries it); each curve line now also carries the counts (`value_n`), so any window between lines is exact, and the
window since the previous line (`value_bits_window`, which the progress line now prints). The formation verdict reports the near band
over the last 400 steps (`near_band_basis` names it); records before 0.1.22 carry the run-long mean there, and their "first step under
the rate" lags the real crossing. The verdict itself (the whole value) and every fetch are unchanged. `--stage=<name>` puts the stage name
in a continuation's tag in place of its starting craft's (a chain of blocks would otherwise nest every earlier tag into the next: the
second block's ran to 259 characters); the record keeps the starting craft. The defaults are bit-identical to 0.1.21 in the weights and
in every field 0.1.21 wrote. colab/nano_pass4.ipynb pins v0.1.22.

0.1.21 (2026-10-13): the sized and the sliced frames. The open tag of a frame costs 6.4-6.7 bits at every size, but the close tag
costs 0.04 bits at 32 bytes (the records fill the frame) and 2.3-3.8 bits at 128-1,024 bytes, where the size is unannounced. `--sized=1`
puts the frame's generic size in its open tag (the frame tag and one letter, as the register tag is MODE and one letter: no new special
byte); `--sliced=32` builds every larger frame as closed 32-byte slices between its own open tag and END (size + 3 bytes): a lookup
frame's records fill its first slices, a nested lookup frame becomes a run of lookup slices, and the fetch groups a record by the frame
it belongs to. The defaults are bit-identical to 0.1.20 (the composer's rows and metadata, and a CPU smoke's weights and record); the
composer costs the same in all three forms. The results table names each run's form and whether it continued. colab/nano_pass4.ipynb
runs the three forms side by side. pyproject's version follows the package's again (it had stayed at 0.1.13).

0.1.20 (2026-10-13): continuation stages as grids. `init` is a grid key and an init written with {arm} {seed} {ask} {hub_K} {hub_D}
is filled from each run's own values; a starting craft missing from the output folder is fetched from the Hub's bed/weights/ before
the run (tokenless; the push repo or --init_repo); the bar marks a continuation. colab/nano_pass3.ipynb runs the third pass: the
scrutiny reads by continuation, sixteen records on the second seeds, Pollux's hub address axis at 1,024 continued at 4,096. All
notebooks pin v0.1.20.

0.1.19 (2026-10-13): the formation verdict reads the whole value (a failure when every lookup group of eight trials or more writes the
value whole at or under 0.05; the near band's bits are reported, not required: a copy can form misaligned), and every framed record
carries the fetch with the bank off (`fetch_by_frame_bank_off`): whether the address or attention carries the lookup. Both notebooks pin v0.1.19.

0.1.18 (2026-10-13): only runs finished under a push are ever pending. The 0.1.17 resume shipped a notebook's 200-step price-line run
(run without a push) to the Hub as an unshipped record from before. Now a run finished under a push is listed in runner_pushlist.txt
before its ship; pending = listed and not shipped; `ship_pending(REPO, OUT, all_records=True)` is the by-hand salvage of a whole folder.
Both notebooks pin v0.1.18.

0.1.17 (2026-10-13): a Hub fault never stops a grid. A 504 from the Hub's commit API during a ship killed a twelve-run grid on a rented
card and lost three finished runs' ships (the ship ran in the main thread with no retry, and the done mark came after it). Now the done
mark is written the moment a run finishes; ships run in a background thread and retry a Hub fault with a doubling wait (15 s to 10 min,
eight attempts); shipped tags are remembered in runner_shipped.txt and anything left unshipped is re-shipped at the next start and at
the end (the result carries `unshipped`; `bed_runner.ship_pending(REPO, OUT)` ships by hand); the run's tag is one function
(names.run_tag) for the trainer and the runner, so a resume treats a run whose record exists on disk as done. Both notebooks pin v0.1.17.

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
