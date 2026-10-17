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
| `geolip/nano_beatrix/bodies.py` | the bed's bodies as one table: an arm name to the library config of its craft at a given context (`craft_cfg(arm, ctx)`); `bed_train` and `nano_v1` both build from it |
| `geolip/nano_beatrix/nano_v1.py` | the nano v1 crafts: four bodies, each from the birth of its strongest bed run, trained through Beatrix 2s's diet (16.1B bytes, thirteen phases) by the library's own trainer, round-robin by phase, with the bed's gauges (`bed_train --gauge_only`) read into the same TensorBoard every 4,000 steps and at every phase boundary; the training streams composed in child processes, a few batches ahead, with the library stream's exact batches and resume positions; `run(out, hf_token, ...)`, `bench(...)`, `table(...)`. The v1.1 crafts (`CRAFTS_V11`, the same names with `.1`): the same bodies, births and diet with the output layer's two readout matrices on AdamW (decay 0.1) instead of Muon |
| `geolip/nano_beatrix/lastfix_screen.py` | the last block's write under a bound, screened on the v1 bodies before their train: each arm (a body, a start, one of the library's `bank_last_fix` forms or none) trains through the start of the 2s diet with the 2s recipe; every 500 steps the full eval and the big-direction reads (the share of each block's output on the last block's mean direction, the last bank write's share and size, every block's rank, the final norm's gain and bias against the direction, the output layer's size and loudness) into the run's TensorBoard; at the end the drift memory on random bytes; the second stage's arms put the output layer's readout on AdamW (`TrainConfig.decay_params`); `run(out, hf_token, names)`, `table(records)`; 0.1.34: the head's dial arms (the library's `head_mode`: the aleph read as the output layer's confidence dial, a bounded per-position scale on the readout's logits born at 1), each at the v1.1 recipe against the readout-on-AdamW run of its body and start, with the head's toggle in bits, the dial's spread, saturation and correlation with the prediction's entropy, and the head address's health in the reads |
| `geolip/nano_beatrix/bed_optim.py` | the bed's optimizer: pure Adam by default (every run before 0.1.31), or AdamW with decoupled decay on a named set of weights (`mat`: every weight that maps the stream; `head`: the output layer's two readouts), every other parameter at decay 0 (Adam's update exactly); `bed_train --wd=0.1 [--wd_set=mat]`; the run's name and record carry it |
| `geolip/nano_beatrix/bed_rerun.py` | the bed's deciding tests again with AdamW in place of the pure Adam they ran on: every test's settings verbatim from its notebook, in parts that share no run (one part a card); `plan`, `price`, `run`, and `read` (each run beside the pure-Adam run of the same settings) |
| `geolip/nano_beatrix/automodel.py` | the finished crafts as transformers AutoModel packages in their repo: the library's model files vendored with the wrapper (`hf_template/modeling_nano_beatrix.py`) at the repo's top and in each craft's folder, each craft's config and final weights (bf16, and the fp32 variant), and the gates every package passes before anything is pushed (exact parity with the library; a transformers load in a process where the library cannot be imported); `release(out, repo, crafts, token)` |

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

The Canon bodies beside the twins, `colab/nano_canon_train.ipynb`: Pollux with Canon layers (`polluxc`) and Pollux's layout with dual
hubs and Canon layers (`altdc`) go through the v4 train's curriculum on the seeds still open, beside Castor and Pollux on the same
seeds, in phases run in order (the twins on seed 4; both Canon bodies on seed 3, continued from their stage-A blocks on the Hub; the
twins on seed 3). C2 prices every arm at each stage's rows on the card in hand and prints each chain's next block and the time it has
left; C4 renders each phase's boundary table:
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AbstractEyes/geolip-nano-beatrix-trainer/blob/main/colab/nano_canon_train.ipynb)

The nano v1 crafts, `colab/nano_v1_2s.ipynb`: four bodies (Pollux's layout with dual hubs and Canon layers, Pollux with Canon layers,
Pollux's layout with dual hubs, and Castor as the baseline), each started from exactly the weights its strongest bed run started from,
trained through Beatrix 2s's diet with the library's trainer and its curves (the health census, the full evals with the bits by
position band) plus the bed's frame, scaling and window gauges at intervals. C2 benches every craft's step on the card in hand, picks
the micro-batch shape and prints the length; C3 trains round-robin by phase and resumes from the repo; C4 shows the curves; C5, after
the diet, packages the crafts for transformers AutoModel in the same repo. The crafts' records go to
[AbstractPhil/alephllm-nano-castor-pollux](https://huggingface.co/AbstractPhil/alephllm-nano-castor-pollux):
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AbstractEyes/geolip-nano-beatrix-trainer/blob/main/colab/nano_v1_2s.ipynb)

Before that train, `colab/nano_v1_lastfix_screen.ipynb`: under the 2s recipe the v1 crafts turned most of their last block's output
onto one fixed direction within their first 1,145 steps, so the library's four bounds on the last block's write (`bank_last_fix`:
register, temperature, gain, and normwrite as the comparison arm) are screened against the bare crafts through the start of the diet
(4,960 steps; two bare seeds for the bar; Castor and the dual-hub Canon body), with the big direction's reads every 500 steps in the
same TensorBoard and the drift memory at the end; the records go to AbstractPhil/alephllm-nano-beatrix-training under `v1_lastfix/`:
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AbstractEyes/geolip-nano-beatrix-trainer/blob/main/colab/nano_v1_lastfix_screen.ipynb)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AbstractEyes/geolip-nano-beatrix-trainer/blob/main/colab/nano_v1_head_decay.ipynb)
(the second stage: the bounds moved the direction upstream, and its share climbs while the output layer grows under Muon, so three
arms move the output layer's readout to AdamW at decay 0, 0.1 and 1.2 from the plain Castor start; the first stage's plain runs are
read from the repo beside them)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AbstractEyes/geolip-nano-beatrix-trainer/blob/main/colab/nano_v1_head_decay_lead_s4.ipynb)
(the same change on the lead body and on Castor's second seed, each against its own plain run)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AbstractEyes/geolip-nano-beatrix-trainer/blob/main/colab/nano_v1_lastfix_resume_pdc_bare.ipynb)
(the lead body's plain screen run resumed from its saved state near step 4,000 to its end, the first screen's notebook with that arm alone)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AbstractEyes/geolip-nano-beatrix-trainer/blob/main/colab/nano_head_dial.ipynb)
(the head's dial: the aleph read as the output layer's confidence dial, seven arms at the v1.1 recipe, each against the readout-on-AdamW
run of its own body and start; the pass rule in the notebook's head)

The nano v1.1 crafts, `colab/nano_v1_1.ipynb`: the v1 notebook with one change to the recipe. The output layer's two readout matrices
(`head.w_h`, `head.w_s`) train with AdamW at 0.0003 and decoupled weight decay 0.1 instead of Muon, the change that held the last
block's dominant direction at a quarter of its output in the screen. The same four bodies from the same starts on the same diet; the
records go to the same repo under the `.1` names, beside the v1 folders, which stay as they were:
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AbstractEyes/geolip-nano-beatrix-trainer/blob/main/colab/nano_v1_1.ipynb)

Before and after the frames, `colab/nano_v1_1_frames.ipynb`: the 2s diet holds none of the nano bed's framed rows, so each finished
v1.1 craft continues from its exact final state for one more phase of 1.0 billion bytes, framed three rows in four (`-fr`), beside the
same continuation with every row plain (`-ct`); the two arms read the same text row for row, and each starts with a read of the v1.1
craft's end (the before):
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AbstractEyes/geolip-nano-beatrix-trainer/blob/main/colab/nano_v1_1_frames.ipynb)

The deciding tests again with AdamW, `colab/nano_bed_adamw.ipynb`: the screen showed the output layer's growth drives the last block's
big direction, and every test that chose the twins' bodies and frames ran on the bed's pure Adam. The notebook runs them again with AdamW
(decoupled decay 0.1 on every weight that maps the stream; the rest at decay 0) in seven parts that share no run, so each card takes
its own part (`PART` in C1): the frame form, the twins' and Canon bodies' curriculum, the splat bodies' curriculum, the records a frame
and the scrutiny set, the registers, the third pass, and the pair; C2 prices the part on the card in hand, C4 reads every run beside its
pure-Adam twin:
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AbstractEyes/geolip-nano-beatrix-trainer/blob/main/colab/nano_bed_adamw.ipynb)

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
`--gauge_only=1 --init=<a stored craft's folder>` reads a stored craft with every gauge of the run's settings and trains nothing: no
training corpus, no weights written; the record's tag ends `_gauge` and the record carries `gauge_only`. A gauge read of a stored block
reproduces that block's record (448 of 451 values exactly, the rest within 5e-16).

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

The frame composer's tests run on the CPU without a model. The AutoModel load test needs transformers: installed, or a folder that holds
it named by `NANO_TF_PKGS` (`pip install --target <folder> transformers==5.18.0`); without either it is skipped.

## Status

0.1.37 (2026-10-17): THE SELF-WRITTEN FRAMES PROTOTYPE, first instance the CARRY FRAME: `frames_data.carry_row` (one early lookup frame,
carries on the 64-byte grid restating it, the question far; the generator `nano-frames-carry` and its mix at the frames dose, the same
text row for row as the frames arm and the control), the `-cf` continuation crafts beside `-fr` and `-ct` (`nano_v1.CRAFTS_CONT`),
`bed_train --carry_gauge=1` (reading with notes: the model opens its own carries at the grid; the value scored at the far question on
its own row; the record's `carry` block, `bed/carry/*` scalars, the tag piece `_carry`), `nano_v1.gauge_stored` (a stored craft's gauge
read from the Hub), `cont_table`'s carry column, `colab/nano_v1_1_carry.ipynb`. Nothing existing changes its rows or its names.

0.1.36 (2026-10-16): the package's own version string, left at 0.1.34 by the 0.1.35 release while pyproject said 0.1.35, now 0.1.36
in both; `colab/nano_v1_1_frames.ipynb` pins v0.1.36 (its C1 version assert read the stale string and would have refused to run
after installing v0.1.35). Nothing else changes.

0.1.35 (2026-10-15): the frames continuation (`frames_data.py`, `nano_v1.CRAFTS_CONT`, `colab/nano_v1_1_frames.ipynb`): each nano
v1.1 craft continued from its last boundary (the exact training state: fp32 weights and both optimizers, placed at step 61,422) through
one appended phase of 1.0 billion bytes, framed three rows in four with the nano bed's framed rows (the sliced frame at 32 bytes, eight
records a frame, the caption register, frames up to 1,024 bytes, nesting 0.6, a key defined twice in 0.3 of the rows), beside the same
continuation with every row plain (the control). The two arms draw every row from the same corpus position with the same seed, so they
read the same text row for row and differ only by the frames; a row is exactly 4,096 bytes, one packed block with its document byte,
and a resumed stream skips its consumed rows without composing them. The crafts `nano-<body>-v1.1-fr` and `nano-<body>-v1.1-ct`;
each one's first boundary row is the source craft's end, read by the same gauges (the "before"). Tests: the framed rows equal the bed's
own (`bed_train.draw_plan` / `frame_row` lifted from the file, 60 seeds), one row a block, the paired text, the resume without
composition, the fineweb-edu holdout, registering only adds names, the eight crafts and their plans, the exact start (the real trainer
on the CPU: weights and both optimizers bit for bit). The library at rc12 8904037 computes one v1.1 step of every nano body bit for bit
as rc10 186434b does (loss, gradients, the weights after the step), so the continuation of a v1.1 run trained under rc10 is exact.

0.1.34 (2026-10-15): the head's dial screen (`lastfix_screen.py`'s head arms, `colab/nano_head_dial.ipynb`; the library at rc12
8904037, `AlephLMConfig.head_mode`). With the readout on AdamW the output layer's aleph read stays at its birth (its toggle +0.0003 /
+0.0006 bits a byte): a read born at zero beside a complete linear readout earns nothing the readout cannot write from the state; the
one function the readout cannot write after the final norm is scale, which the last block's big direction carries at the cost of the
content's room. The dial gives that function to the aleph read: the readout's logits times exp(ln4 tanh(w_t . s(h) - a_bar)), w_t born
zero (the plain craft at birth, bit for bit), the carrying read frozen at zero; the forms dial / dialraw (no centering) / dialplus (the
carrying read kept) / pergain (a per-byte gain, on pure Adam like the dial's weights) and the frozen-address control; seven arms at
the v1.1 recipe, each read against the readout-on-AdamW run of its own body and start (`ref`); the gauge adds the head's toggle in bits
(the dial held at 1) split into its per-byte part (the dial held at its own mean) and one overall temperature, the dial's spread,
saturation and correlation with the entropy of the prediction before the dial and with the byte's share on the direction, and the head
address's health; the dial computed in fp32.
The bed takes `--head_mode=<form>` (the tag piece `_hd<form>`; the runner's grid key).

0.1.33 (2026-10-15): the nano v1.1 crafts (`nano_v1.CRAFTS_V11`, `colab/nano_v1_1.ipynb`): the v1 crafts again with the output
layer's two readout matrices on AdamW (lr 0.0003, decoupled weight decay 0.1; their biases at 0) instead of Muon, set per craft
(`TrainConfig.decay_params`); the same bodies, births, diet and gauges; the gauge folders and records carry the version (`v11<code>`),
so the two versions never share a name; the repo's two cards name their version. The v1 crafts and their notebook are unchanged.
Tests: the v1.1 names and births, only v1.1 moves the readout (exactly the two matrices, off Muon, at 0.1), the gauge names, the
cards. The 4090 toy run: both v1.1 crafts through every phase of the diet at 2e-5 of its bytes, a stop and a resume. The train
driver (v1 and v1.1): a session that ends inside a boundary's handling (after the trainer's boundary, before the eval, the gauges and
the table row) left that boundary without its row; the next `run()` writes it from the craft's own weights when the craft still
stands at that boundary (`unrecorded_boundary`), and says so when it has trained past it (checked on the 4090: the row written again
at the same step with the same eval).

0.1.32 (2026-10-15): the registers part (`frames-registers`) at the first pass's diet: every row framed (its pure-Adam records
carry no plant piece), where 0.1.31 framed three rows in four; its twelve AdamW runs now land beside their pure-Adam twins. The 0.1.31
runs stay on the Hub under their own names (`_p75`). The read judges the copy's formation by one rule on both sides of a pair (the
current verdict recomputed from each record's fetch; records before 0.1.19 stored an older verdict). The notebook pins v0.1.32.

0.1.31 (2026-10-15): AdamW for the bed (`bed_optim.py`) and the deciding tests again under it (`bed_rerun.py`,
`colab/nano_bed_adamw.ipynb`). The v1 screen's second stage moved the output layer's readout from Muon to Adam or AdamW: the last
block's share on its big direction held at 0.23-0.32 against the plain runs' 0.92-0.93, the readout stayed 9-15 in size against 61, and
the bits at 4,960 steps came out within the seeds' spread or better (Castor seeds 3 and 4 at decay 0.1: +0.0017, -0.0065; at 1.2:
-0.0012, -0.0081; the lead body at 4,000 steps: -0.0044, -0.0013). `bed_train --wd=X` trains with AdamW, the decay set at X and every
other parameter at 0; without it the trainer is unchanged (a CPU smoke of Castor's stage-A block: the weights byte for byte those of the
0.1.24 golden run, every record field the same; the config adds only the library's Canon flag, off). The run's name carries the decay
(`_wd0.1`), so every AdamW run lands beside the pure-Adam run of the same settings; the runner takes `wd` and `wd_set` as grid keys.
`bed_rerun` holds every deciding test's settings verbatim from its notebook (the four frame passes, the twins' and Canon bodies'
curriculum, the splat bodies', the pair), with the stage-A crafts the third pass continued; a new session fetches the part's shipped
records, so a part resumes across sessions. The 4090 smoke ran the frame-form part and a Canon body's curriculum chain at toy scale end
to end (every record AdamW, every continuation from its own stage-A craft, the second call ran nothing).
`colab/nano_v1_lastfix_resume_pdc_bare.ipynb` resumes the lead body's plain screen run.

0.1.30 (2026-10-14): the second stage's carry-over (`colab/nano_v1_head_decay_lead_s4.ipynb`): the output layer's readout on AdamW
at decay 0.1 and 1.2 on the lead body (dual hubs + Canon, from `pdc-bare`'s start) and on Castor's second seed (from `c-bare-s4`'s
start); `table()` reads each arm against the plain run of its own body and start (`PLAIN`).

0.1.29 (2026-10-14): the screen's second stage, the output layer (`colab/nano_v1_head_decay.ipynb`). On Castor the bounds on the
last write held the last block's share on its direction only a little under the plain run (0.873-0.921 against 0.928 after 4,960
steps; the rank of the spread around it 15-46 against 11) and moved the direction upstream: the earlier blocks carried it instead
(block 5 at 0.80 against 0.58). The saved checkpoints place its use at the final norm: the norm rescales every byte to one size, so the
direction's size per byte sets how much room the content keeps and with it the prediction's confidence (removed, the predictions turn
far too sharp: +4.97 bits a byte; held at its mean size for every byte, +0.10; the bytes that carry more of it predict with more
uncertainty, rank correlation 0.79). Its share climbed while the output layer grew: the readout doubled in 3,800 steps under Muon, so
the content's net loudness rose only a tenth. Three arms move the output layer's two readout matrices from Muon to AdamW (the library's
`TrainConfig.decay_params`, alephllm `186434b`, `feat/canon-layers` rc10; off by default, the 2s split exactly) at decay 0, 0.1 and 1.2,
from the plain Castor start on the same data; the gauge also reads the output layer's size and loudness.

0.1.28 (2026-10-14): the last-write screen (`lastfix_screen.py`, `colab/nano_v1_lastfix_screen.ipynb`). Read on 32 held-out
documents, the v1 crafts' last block put 0.81-0.82 of its output's energy on one fixed direction after 1,145 steps of the 2s recipe
(0.19-0.24 at birth; 0.43-0.50 at the end of the same bodies' bed runs), its bank write 25-36x larger and the rank of its spread 13-16
of 256. Nine arms (Castor bare on two seeds, each of the four forms on Castor, the dual-hub Canon body bare, with the register and with
the temperature) train 4,960 steps each from the v1 starts on the same data; a fix arm starts from its craft's v1 start plus the
fix's own tensors (the register, the temperature and the gain give the plain craft's logits exactly at birth). The pass rule is
written in the notebook before any arm runs.

0.1.27 (2026-10-14): the AutoModel packages of the nano v1 crafts (`automodel.py`, the notebook's C5). transformers looks for remote
code at the top of a repo even when the config and the weights come from a subfolder, so the model code sits at the repo's top and,
byte for byte, in each craft's folder (a folder downloaded alone loads too). The wrapper refills the buffers a weight file does not
store (the byte-class table): transformers 5 re-creates them as empty memory and asks the model to fill them; each is copied from a
fresh build of the same craft (a control test shows the load gate failing a wrapper that leaves them empty). The config loads the
bf16 file into fp32 parameters, exactly. On the smoke's two finished crafts every gate passes: the vendored code gives the library's
logits exactly, and transformers 5.18 loads each craft, by both routes and from both weight files, with the same logits.

0.1.26 (2026-10-14): the nano v1 crafts. `nano_v1` trains four bodies from their bed births through Beatrix 2s's diet with the
library's trainer (`colab/nano_v1_2s.ipynb`): the 2s recipe (Muon 0.02 on the trunk's matrices, Adam 0.0003 on the rest, 200 warmup
steps then flat, clip 1.0, 262,144 bytes a step, 61,422 steps), round-robin by phase, the bed's gauges at intervals into the same
TensorBoard. Each craft's start equals its bed run's start bit for bit (the seed, then the craft built at 1,024, the position rows past
1,024 zero), checked against the stored configs and parameter counts. The arm table moved to `bodies.py` (the old and the new
`craft_cfg` agree on all 432 configs: every attention form, bank form and extra at both contexts). `bed_train --gauge_only=1` reads a
stored craft without training. The library pin moves to alephllm `b024c93` (`feat/canon-layers`, rc9: the model census reads dual
hubs and passes the byte classes to a bank with the byte table, so the library trainer's health readout and full eval run on these
bodies; every older form reads as before).

0.1.25 (2026-10-14): the Canon bodies' notebook, and the arms added since 0.1.24. `colab/nano_canon_train.ipynb` runs the v4
train's curriculum in phases (each phase a curriculum over its own chains, run to its end before the next): the twins on seed 4, both
Canon bodies on seed 3 (continued from their stage-A blocks on the Hub), the twins on seed 3; its preflight prices every arm at each
stage's rows on the card in hand and prints each chain's next block and the training time it has left. Released here from main: the
dual-hub bodies (`hubdb`, `altd`, `d5s1`, `d4s2`) and the Canon bodies (`altdc`, `hubdc`, `polluxc`), with the library pinned to
alephllm `d26b40d` (its `feat/canon-layers` branch: the Canon layers, born at zero; every flag off is the previous pin bit for bit).
The training code is otherwise 0.1.24's.

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
