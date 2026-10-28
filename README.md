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
| `geolip/nano_beatrix/automodel.py` | the finished crafts (the four bodies and, since 0.1.43, every continuation arm; `crafts.json` at the repo's top indexes them and each config says what its model is) as transformers AutoModel packages in their repo: the library's model files vendored with the wrapper (`hf_template/modeling_nano_beatrix.py`) at the repo's top and in each craft's folder, each craft's config and final weights (bf16, and the fp32 variant), and the gates every package passes before anything is pushed (exact parity with the library; a transformers load in a process where the library cannot be imported); `release(out, repo, crafts, token)` |

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

The framed pretrain, v1.2: the v1.1 recipe, phases and bytes with half of every phase's rows framed from the first step (carry rows, system
frames, caption windows, note rows), each craft read at every boundary against its plain v1.1 twin by the pre-registered marks, with the
copy read at every frame size in the table. Pollux dual, `colab/nano_v1_2.ipynb`:
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AbstractEyes/geolip-nano-beatrix-trainer/blob/main/colab/nano_v1_2.ipynb)
Castor, the lead body, `colab/nano_v1_2_castor.ipynb`:
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AbstractEyes/geolip-nano-beatrix-trainer/blob/main/colab/nano_v1_2_castor.ipynb)

The thought phase on the framed Castor (0.1.69): the art thought dataset's documents as one row in ten of the framed diet, the thought reads before and after, the control without thoughts beside; `colab/nano_thought_castor.ipynb`:
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AbstractEyes/geolip-nano-beatrix-trainer/blob/main/colab/nano_thought_castor.ipynb)

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

The scale test's cores (0.1.70), `colab/nano_scale_cores.ipynb`: **Alhena**, the third of the triplets (gamma Geminorum, the
third-brightest star of the Twins; al-han'a, the mark), is Castor's layout whose softmax blocks turn by the anchored multi-position
rotary (`anchored_rope.py`): each head's 16 rotate-half pairs split by job, six turned by the relative position (NTK-aware x4), two
carrying content unturned, eight turned by the two digits of the address a token sits in or, at a question's answer positions, asks for.
Half of each batch's rows are addressed grid rows (`grid_rows.py`: 24 compartments of 128 bytes at fixed places, each with one
addressed lookup frame, an index the model writes and 24 questions), the other half the bed's framed rows place for place, and a
per-digit address head trains with the core (`--grid=ag --head=digit`; Alhena is `--arm=alhena-both`). Beside it: Castor on the same
rows without the rotation, and the stored 4,096 framed cell trained again verbatim (seed 1 twice, the hardware repeat); two seeds:
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AbstractEyes/geolip-nano-beatrix-trainer/blob/main/colab/nano_scale_cores.ipynb)

The scale test's members (0.1.71), `colab/nano_scale_members.ipynb` (`rotary_scale.py`): on each grid core, 48 real arms (the arm
library's RelayPatchwork at 16 slots x 8 through an aleph read of 16 anchors, born inert; amoe-lora 0.2.12) and 48 toy members, each
owning one address of the window and answering its record's questions in its own letter code, trained alone on the frozen core after its
last block; 16 real arms again after block 3 of 6 on rows that each belong to one member. The read merges 16 / 32 / 48 of them under the
product gate with the rest discount, the request from the layout, the core's written index, a corrupted index or its address head, and
grades whether each merged member keeps its solo answers and the unowned questions keep the bare core's:
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AbstractEyes/geolip-nano-beatrix-trainer/blob/main/colab/nano_scale_members.ipynb)

The find screen (0.1.72): a request that must find its record. `--head_at=both` trains the same address head at each index entry's SEP
as well (`--head_lam_i`, weight 1): both address streams carry the null anchor there and nothing earlier in the row names the entry key's
address but its record, so the head must find the record by its key. Each term has its own novelty gate and descent record; every grid
core's end read adds the free-running index (the core's own index bytes in the row, at the fixed point of re-feeding them:
`grid_rows.free_running`). The read's two-pass form (`rotary_scale.FIND_SOURCES`): pass 1 reads the head at the index entries with every
question at the null anchor, a digit enters the question's stream when its own score passes .5, pass 2 reads the answers with that address
in the stream and the head's scores in the gate; every request source carries its own address. `--head_at=question` (the default) is
0.1.71 bit for bit. The cores (the scale test's stored cores read with no training, Alhena-F on two seeds, the replication core, the
retrain rule), `colab/nano_find_cores.ipynb`:
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AbstractEyes/geolip-nano-beatrix-trainer/blob/main/colab/nano_find_cores.ipynb)
The members on them (the scale test's recipe verbatim; the two-pass read), `colab/nano_find_members.ipynb`:
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AbstractEyes/geolip-nano-beatrix-trainer/blob/main/colab/nano_find_members.ipynb)

The wide-band screen (0.1.73): the anchored split as a parameter. `--rotary_content=N` gives each head N content pairs (unturned) and
R = 8 - N relative pairs, NTK-aware x4 over their 2R dims (w_j = (10000 * 4^(2R / (2R - 2)))^(-j / R), j = 0..R-1); the eight address
pairs are unchanged. The default (N = 2) is 0.1.72 bit for bit (a 3-step run's record, weights and side files, with and without the head
at the index entries); a run with another N carries `_cp<N>` in its name, its record and scale.json carry the split, and
`rotary_scale.Core` builds the rotation from scale.json. Alhena-W (N = 4: relative wavelengths 6.3 / 100 / 1,583 / 25,100 positions) on
four seeds, Alhena verbatim on two new seeds, Alhena-W with the head at the index entries on two, and the members on the Alhena-W cores
whose own written index is right, `colab/nano_wide.ipynb`:
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AbstractEyes/geolip-nano-beatrix-trainer/blob/main/colab/nano_wide.ipynb)

The posing screen (0.1.74): the grid rows' address as one byte. `--grid_abytes=1` writes each record's address as ONE byte 0x80 + a
(a = 0..63) in the record's frame (`DATA A CUE key SEP value END`) and in the index entry (`CUE key SEP A`), unique in its row; the
rotary's address streams and the address head keep the two digits, and the request read from the model's own index is the digits of
A' - 0x80 when 0x80 <= A' < 0xC0, else the null anchor at both levels. The default (two digit bytes) is 0.1.73 bit for bit (the rows on
fixed draws; a 3-step run's record, weights and side files on Alhena's and Alhena-W's grid recipes); a run in the one-byte form carries
`_ab1` in its name, its record and scale.json carry the form, and `rotary_scale.Core` composes the members' rows in it. Alhena-U on
Alhena-W's split on four seeds, on Alhena's split on two, Alhena-W's seed 2 again, and the members on the first two Alhena-U cores whose
own written index is right, `colab/nano_posing.ipynb`:
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AbstractEyes/geolip-nano-beatrix-trainer/blob/main/colab/nano_posing.ipynb)

The key-hard screen (0.1.75): training rows where only the whole key finds the record. A grid key is a word and a three-digit number;
`grid_rows.hard_keys` draws a row's 24 keys as all the combinations of 2 words x 2 hundreds digits x 2 tens digits x 3 units digits, so
every key has a neighbour in its row that differs from it in exactly one part, for each part (a match on the last two digits is right on
a quarter of such a row's entries, on the number alone on half; only the whole key on all). `--grid_hard=KH` makes the batch's last
round(N x KH) of its N grid rows key-hard rows; the rest of each row (addresses, values, questions, index order, streams) is the standard
draw. The default (KH = 0) is 0.1.74 bit for bit (the batch on fixed steps; a 3-step run's record, weights and side files on Alhena-W's
grid recipe); a run with KH > 0 carries `_kh<100 KH>` in its name and its record and scale.json carry the share. The end read adds the
index on 64 key-hard rows, teacher-forced and free-running, and the trace a key-hard point, when KH > 0 or `--hard_read=1` (a read only:
the training is unchanged bit for bit). Alhena-K (Alhena-W with half the grid rows key-hard) on six seeds, Alhena-W on two new seeds with
the key-hard read, and the members on the first two Alhena-K cores whose own written index is right, `colab/nano_keys.ipynb`:
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AbstractEyes/geolip-nano-beatrix-trainer/blob/main/colab/nano_keys.ipynb)

The value-path screen (0.1.76): the record's address on its every token. The anchored rotary puts a record's address into the attention
scores only; a look that finds a record by its key returns the record's tokens, and the address is just two of them, stored before the key.
`--value_addr=1` adds two level tables of 8 vectors (`anchored_rope.ValueAddress`: one table per address digit, born zero, 2 x 8 x d
parameters, trained with the craft under the same Adam and clip) to the input of every token: x_n = embed(t_n) + E_1[d_1(n)] + E_2[d_2(n)],
the digits read from the key stream the rotation reads. Inside a grid row's compartments that stream carries the compartment's address; at
the index entries, the questions and the plain rows it is the null anchor, which adds nothing, so the tables never see an answer. The
default is 0.1.75 bit for bit (a 3-step run's record, weights and side files on Alhena-W's grid recipe); born zero, the first step's
forward is the plain craft's to the last bit. A run with the tables carries `_va1` in its name; the tables ship beside the weights as
`value_addr.safetensors` (scale.json names them; the record carries their norms by level and by digit), and `rotary_scale.Core` applies
them in every forward of the frozen core. 0.1.77 makes the lookup cheap: each level is an embedding lookup into its 8 rows and a zero
null row (`padding_idx`), the same values as 0.1.76's indexed gather to the last bit, whose backward summed the ~24,000 null positions of
a batch into one row in series (+15% of a training step on an RTX 4090, about a third on an RTX PRO 6000; the embedding form +0.5%).
Alhena-V (Alhena-W with the tables) on six seeds, Alhena-W on two new seeds, and the members on the first two Alhena-V cores whose own
written index is right, `colab/nano_value.ipynb`:
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AbstractEyes/geolip-nano-beatrix-trainer/blob/main/colab/nano_value.ipynb)

The pointer screen (0.1.78): one attention head told where the record is. The only signal that trains the lookup arrives through the
two address bytes each index entry writes; on the runs that stall, the lookup head splits its attention between the right record and one
whose key ends the same way, or no head looks back at the records at all. `--ptr_lam=L` gives one head of one softmax block (`--ptr_block`,
numbered from 1, default 4; `--ptr_head`, default 0) a target at each index entry's SEP: the negative log of its causal softmax mass on the
frame of the record whose key the entry carries,

    L_ptr = -(1/|E|) sum_e log sum_{n in frame(e)} softmax_{n' <= i_e}( q(i_e) . k(n') / sqrt(d_head) )_n

computed from the block's own rotated q and k (`pointer.HeadCapture` wraps the block's rotation as an instance attribute: no parameter, the
forward unchanged) and added to the loss as L x L_ptr (the attention-sum form of Kadlec et al. 2016, arXiv 1603.01547, as an auxiliary;
one supervised head in a chosen layer as in Garg et al. 2019, arXiv 1909.02074). The term has its own collinearity gate at step 0 (refused
under .05, as the address head's) and its own descent record; the formation trace and the end read carry the head's mass on the true frame
(`pointer.log_mass`: the same quantity the band read measures). The default (L = 0) is 0.1.77 bit for bit (the batch on fixed steps; a
3-step run's record, weights and side files on the value-path recipe); a run with the term carries `_pt<100 L>b<block>h<head>` in its name,
its record a `ptr` block, its scale.json the form. Alhena-P (Alhena-V with the term on block 4's head 0) at two doses on seed 1, the dose
the plan's rule picks on five more seeds, Alhena-V on two new seeds, and the members on the first two Alhena-P cores whose own written index
is right, `colab/nano_pointer.ipynb`:
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AbstractEyes/geolip-nano-beatrix-trainer/blob/main/colab/nano_pointer.ipynb)

The local two-position form (0.1.79). An index entry writes its address in two bytes: the first is predicted at the entry's SEP, the
second at the first address byte. `--ptr_pos=both` asks the same mass at both positions (the term the mean over the 2|E| positions):

    L_ptr = -(1/(2|E|)) sum_e [ log m_e(i_e) + log m_e(i_e + 1) ]

`--ptr_stop=1` forms the term's query and key from the block's attention input detached, through the block's own rotation
(`pointer.HeadCapture(stop=True)`): the forward's values, and a gradient that reaches the supervised head's query and key rows of the
block's qkv projection and nothing else. With the gradient through the trunk the term can be met by an early look on the key's last two
bytes, which the SEP's own byte-trigram input already carries; kept in the head, the term can only point the head with the features the
cross-entropy builds. The defaults (`sep`, `0`) are 0.1.78 bit for bit; a run with either carries `p2` / `sg` after its `_pt` piece, its
record's `ptr` block the form and the mass at the first address byte (`mass_read2`, the trace's `ptr_mass2`). Alhena-P2 (Alhena-P with
both) on six seeds, Alhena-P again on two new seeds, and the members on the first two Alhena-P2 cores whose own written index is right,
`colab/nano_pointer2.ipynb`:
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AbstractEyes/geolip-nano-beatrix-trainer/blob/main/colab/nano_pointer2.ipynb)

The readout tie (0.1.80). On the Alhena-P models the supervised head looks at the right record while the index the model writes is often
wrong: other paths write it, and its second digit is often copied from a neighbouring record. `rotary_scale` reads the request off the
look itself. In the read's first pass the designated head's own causal attention (its block's rotated q and k from the forward, captured by
`pointer.HeadCapture`; the forward unchanged) is summed over each record's frame:

    M(r) = sum_{n in frame(r)} softmax_{n' <= i}( q(i) . k(n') / sqrt(d_head) )_n,     request(i) = address(r) if M(r) > 1/2, else none

Each record is judged by its own mass against one fixed number (at most one record can hold more than half), both digits come from that
one record, and with no such record there is no request and every member abstains (the answer form of pointer networks, Vinyals et al.
2015, arXiv 1506.03134, and of the attention-sum reader, read through an absolute threshold). Three request sources: `LOOK` (at the asked
key's index entry, the row as composed), `LOOKF` (the same on the model's own free-running row), `LOOKQF` (at the question's own SEP).
`rotary_scale.Core(look=...)` takes the head from the model's own pointer term (`"ptr"`: its scale.json), a given `(block, head)`, or none;
with none every read is 0.1.79's bit for bit, and with the look on every 0.1.79 source's cells are unchanged. `look_read` reads the sources
without members (each one's right, the share with no request and with a wrong record, the answers through each source's address, the
split and duplicate looks) on the read rows and on key-hard rows; `read0` and `run_core` carry the look sources in the two-pass form. The
look read on the stored Alhena-P, Alhena-P2 and (at the same head, as a control) Alhena-V models, Alhena-P seed 2's stored members read
through it, and the members on up to five models whose look passes the rule, `colab/nano_readout.ipynb`:
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AbstractEyes/geolip-nano-beatrix-trainer/blob/main/colab/nano_readout.ipynb)

The key-hard pointer screen (0.1.80 as is). The pointer term (0.1.78) trained on the key-hard rows (0.1.75): with `--grid_hard=0.5` on the
Alhena-P recipe the last two of the batch's four grid rows carry 24 keys built from parts that recombine, so a match on any part of a key
finds two to four records and only the whole key finds one, and the term reads every grid row's index entries, the hard rows' included.
The request is read off the head's look (`rotary_scale.look_read`, above). Six such models and Alhena-P again on two new seeds, each read
through its look as it lands, and the members on up to three models whose look passes the rule, `colab/nano_keypointer.ipynb`:
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AbstractEyes/geolip-nano-beatrix-trainer/blob/main/colab/nano_keypointer.ipynb)

The longer run (0.1.80 as is). The key-hard pointer recipe at `--steps=48000` (the cosine schedule over the longer run, nothing else
changed) on the five seeds whose head stalled at 24,000 steps, Alhena-P at 48,000 steps on two of the same seeds (the length without the
key-hard rows), and the key-hard pointer recipe at 24,000 steps on two new seeds (the replication), each read through its look as it lands,
and the members on up to three 48,000-step models whose look passes the rule, `colab/nano_keypointer_long.ipynb`:
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AbstractEyes/geolip-nano-beatrix-trainer/blob/main/colab/nano_keypointer_long.ipynb)

The key code at the end-marker (0.1.81). The byte-trigram input gives each position its own byte and the two before it, so the SEP
after a key carries only the key's last two bytes, on the record's side and on the index entry's side alike; on the key-hard rows four
records share them, and a head comparing one SEP with another finds four equal matches. `--key_code=fixed|learned` adds the whole key
at each SEP whose CUE lies in the 10 positions before it (`key_code.py`), one table for each distance back:

    e'(SEP) = e(SEP) + sum_{k=1}^{n} T_k[x_{SEP-k}],     n = the key's length (6 to 9 bytes on the bed's rows)

so the inner product of two SEPs' codes adds over the places where their keys' bytes agree. `fixed`: the rows are N(0, 1), the byte
tables' own initial scale, drawn once from a generator seeded `crc32("KEYCODE|v1")` and never trained (the same rows for every seed);
`learned`: the tables are born zero and trained with the model (589,824 parameters). The code reads only the bytes between a CUE and its
SEP (the key), never a value or an address; every other position, and a SEP with no CUE in reach (the rows' heads), is unchanged. Off (the
default) every run is 0.1.80's bit for bit; on, the name carries `_kcf` or `_kcl`, the tables ship beside the weights
(`key_code.safetensors`, `scale.json`'s `key_code`), the record carries the tables' norm by distance and the code's share of the SEPs'
input, and `rotary_scale.Core` applies the code after the value-path tables. Both forms on the key-hard pointer recipe at 24,000 steps,
seeds 1-6 each, every model read through its look as it lands, and the members on up to three models whose look passes the rule,
`colab/nano_keycode.ipynb` (prior art: the hashed byte n-gram embeddings of the Byte Latent Transformer, Pagnoni et al. 2024,
arXiv 2412.09871):
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AbstractEyes/geolip-nano-beatrix-trainer/blob/main/colab/nano_keycode.ipynb)

The key in the told head's rotation (0.1.82). An added code is a sum: two SEPs' codes agree in proportion to the places where their keys
agree, so a record one character away still matches most of the asked key and the head has to learn the margin. `--key_rot=1` puts the
key into the rotation of the head the pointer term trains (`--ptr_block`, `--ptr_head`), the way the anchored rotation carries a record's
address on its anchor pairs (`key_rotary.py`): at each SEP the key code would code (the same rule), each of the head's eight pairs before
the anchor levels turns by

    phi_j(SEP) = sum_{k=1}^{n} P[j, k, x_{SEP-k}]   (mod 2 pi),     P fixed phases, uniform on [0, 2 pi)

in place of its anchored angle, for the query and the key alike; between two SEPs those pairs then score
sum_j Re[q_j conj(k_j) e^{i (phi_j(asking) - phi_j(record))}]: the same key turns every pair alike, one byte different anywhere turns
every pair by an unrelated angle - a binding of the whole key (the angle addition of FHRR) rather than a sum of its parts. The phases are
drawn once from a generator seeded `crc32("KEYROT|v1")` (8 x 9 x 256, no gradient, the same for every seed); nothing is added to any input
and every other head, pair and position keeps the anchored angles. Off (the default) every run is 0.1.81's bit for bit; on, the name
carries `_krt`, the phases ship beside the weights (`key_rot.safetensors`, `scale.json`'s `key_rot`), the record carries the told head's
scores at the index entries by pair group (`grid.key_rot.use`), and `rotary_scale.Core` applies the rotation before the look's capture.
The rotation on the key-hard pointer recipe and on the pointer recipe without the key-hard rows at 24,000 steps, seeds 1-6 each, every
model read through its look as it lands, and the members on up to three models whose look passes the rule, `colab/nano_keyrot.ipynb`
(prior art: rotary position embedding, Su et al. 2021, arXiv 2104.09864; binding as angle addition, Schlegel, Neubert and Protzel,
A comparison of Vector Symbolic Architectures, arXiv 2001.11797):
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AbstractEyes/geolip-nano-beatrix-trainer/blob/main/colab/nano_keyrot.ipynb)

The larger row (0.1.83). The grid row of 24 records in 128-byte compartments is one layout of several the window holds. `--grid_comps=48`
composes the rows in the layout `AG48` (`grid_rows.LAYOUTS`): 64 slots of 64 bytes - the head, 48 compartments (48 of the 64 addresses,
sorted), the index of the 24 asked records, their 24 questions - the largest row the 4,096-byte window and the two-digit address code
hold with 24 questions; the 24 records not asked are present and addressed, and compete at every end-marker. `--grid_keys=long` names
the records with long keys (`bed_frames.key_long`: a cue word + six digits, 9-12 bytes) and reaches 12 bytes back from each end-marker
(`key_code.key_offsets(idx, k)`), so the key rotation turns by 8 x 12 x 256 phases of their own seeded draw (`crc32("KEYROT|v1|k12")`).
The key-hard rows grow with the row: 2 words x 2 hundreds digits x 2 tens digits x 6 units digits at 48 (the long keys: the fourth to
sixth digits at 2 x 2 x 6, the first three one draw for the row), so every key still has a neighbour one part away in each part and each
ending is shared by four records. At a question's end-marker at most 92 other coded end-markers compete (68 at 24 records); with eight
pairs of magnitude s and unrelated phases the asked record's mean weight is about

    exp(8 s) / (exp(8 s) + (N - 1) I0(s)^8)

(.9947 -> .9927 at s = 2 from 24 to 48 records). 0.1.84: the members' rows at 48 records ask the first 24 of each row's 36 owned
compartments in the cycled order, so every member is asked equally often in a batch (4 a batch), as the members' trainer requires. The
defaults are 0.1.82's rows and rotation bit for bit; at the larger row the name
carries `_gc48` (and `_kl` for the long keys), `scale.json` carries `grid_layout` and `grid_keys`, and `rotary_scale` composes every read
and every member row at the core's layout (at 48: 36 owned and 12 never-owned compartments, 24 of the owned asked). The rotation at 48
records with and without the key-hard rows, seeds 1-6 each, the members on up to three models whose look passes the rule, then the long
keys on the key-hard recipe, seeds 1-6, `colab/nano_largerows.ipynb` (prior art: Ramsauer et al. 2020, Hopfield Networks is All You Need,
arXiv 2008.02217; Frady, Kleyko and Sommer 2018, A theory of sequence indexing and working memory in recurrent neural networks,
arXiv 1803.00412):
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AbstractEyes/geolip-nano-beatrix-trainer/blob/main/colab/nano_largerows.ipynb)

0.1.85: the binding's capacity read (`capacity.py`). A row holds at most ~92 competing keys; the read measures how many the told
head's eight bound pairs keep apart past that, with no training. A stored model reads 192 rows at its own layout and keys' form (half
key-hard), the trunk through the told block only, and keeps the head's rotated query and key at every keyed end-marker; each index
entry's true score is set against pools of other keys - REAL (the other rows' end-markers) and EMULATED (a pooled key's content
unrotated by its own phases and turned by the phases of a fresh key, to 10^6) - and the leak, one minus the true record's share of
the head's softmax, is recorded at N = 10^2 .. 10^6 (mean, median, 99th percentile, the look error and the argmax error) beside the
moment of random phases, N I0(s)^8 / (exp(8 s) + N I0(s)^8). The moment is an annealed average, dominated by rare near-aligned
competitors (Derrida 1981, the random-energy model); the median shows the typical leak. The thirty stored binding models,
`colab/nano_capacity.ipynb` on a CPU runtime:
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AbstractEyes/geolip-nano-beatrix-trainer/blob/main/colab/nano_capacity.ipynb)
The same read as three lanes on three CPU runtimes at once, `colab/nano_capacity_lane{1,2,3}.ipynb`: every lane walks the thirty in
the same order and claims a model on the Hub before reading it, by a small file written only if the repo has not moved since the
lane's read of it (a commit with its parent named), so no two lanes read one model; a record does not depend on its lane.
[![Lane 1](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AbstractEyes/geolip-nano-beatrix-trainer/blob/main/colab/nano_capacity_lane1.ipynb)
[![Lane 2](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AbstractEyes/geolip-nano-beatrix-trainer/blob/main/colab/nano_capacity_lane2.ipynb)
[![Lane 3](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AbstractEyes/geolip-nano-beatrix-trainer/blob/main/colab/nano_capacity_lane3.ipynb)

0.1.86: the tail read (`capacity.tail_core`). The capacity read's mean leak follows the moment of random phases, but its look error
runs far above what random phases at one fixed pair length give. A pair's score is |q_j| |k_j| cos(the phase difference) / sqrt(hd),
so two length tails can carry the look error: the competitors' (pooled pairs longer than the true record's) and the entries' (true
scores low in the margin's spread). On the read's own rows and bank (its base equals the read's EMULATED numbers bit for bit) the leak
is computed three ways on the same competitors - as read; with every competitor's pair lengths set to its entry's true record's, its
angles kept; with every entry's true score set to the model's median - at N = 10^3, 4,096 (a window's worth of keyed end-markers),
10^4, 10^5 and 10^6, beside the look errors' entries by their margin's percentile, the pooled pairs' lengths against the true
records', and each look error's strongest competitor. The eighteen models at 48 records a row, as three lanes,
`colab/nano_capacity_tail_lane{1,2,3}.ipynb`:
[![Tail lane 1](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AbstractEyes/geolip-nano-beatrix-trainer/blob/main/colab/nano_capacity_tail_lane1.ipynb)
[![Tail lane 2](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AbstractEyes/geolip-nano-beatrix-trainer/blob/main/colab/nano_capacity_tail_lane2.ipynb)
[![Tail lane 3](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AbstractEyes/geolip-nano-beatrix-trainer/blob/main/colab/nano_capacity_tail_lane3.ipynb)

0.1.87: two forms of the binding, read for capacity. The key rotation binds the whole key into the told head's eight shared pairs; the
capacity read found that eight pairs hold ten thousand keys and leak near a million, and that the wrong looks at a million come from
competitors whose key pairs run a few percent longer than the true record's. `--key_norm=1` holds the key's bound pairs at one length
(4.16) at the coded end-markers before the turn, the query untouched: every competitor's pair lengths equal every true record's by
construction. `--key_pairs=16` binds all sixteen of the told head's pairs at the coded end-markers (the anchor pairs too, there only),
with phases drawn from crc32("KEYROT|v1|j16"): at the same margin a pair, the random-phase leak at a million falls from about .2 to
about 5e-8. Neither adds a parameter; the defaults are 0.1.86 bit for bit; the names carry `_krtn` and `_krt16`. `rotary_scale.Core`
reads either form from `scale.json`, and the capacity read takes the bound pairs from the model. Twelve models at 48 records a row (six
seeds of each form) on the G4, `colab/nano_norm16.ipynb`, and their capacity on two CPU runtimes beside it,
`colab/nano_capacity_ns_lane{1,2}.ipynb`:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AbstractEyes/geolip-nano-beatrix-trainer/blob/main/colab/nano_norm16.ipynb)
[![Capacity lane 1](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AbstractEyes/geolip-nano-beatrix-trainer/blob/main/colab/nano_capacity_ns_lane1.ipynb)
[![Capacity lane 2](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AbstractEyes/geolip-nano-beatrix-trainer/blob/main/colab/nano_capacity_ns_lane2.ipynb)

0.1.88: the hold at the records only. Sixteen bound pairs keep a million keys apart on most models; their few wrong looks there are
other rows' records whose key pairs run longer than the true record's. The hold at every coded end-marker removes that kind of miss on
eight pairs, but it also lengthens the index entries' keys, which trained models keep short. `--key_norm=2` holds the key's bound pairs
at one length at the records' coded end-markers only, and `--key_rho` sets the length (default 4.16). A record's end-marker is found from
the bytes at or before it (`key_rotary.records`): in a grid row a record's key follows DATA and its address, and in a framed row it lies
inside an open frame. The hold never reads ahead, and the index entries' and the questions' keys keep their lengths. `--key_norm=1` is
0.1.87's hold at every coded end-marker, and the defaults are 0.1.87 bit for bit. The names carry `_krt..r` and the length's digits when
it is not 4.16 (`_krt16r218`). Six sixteen-pair models at 48 records a row with the records held at 2.18 on the G4,
`colab/nano_rec16.ipynb`, and their capacity on two CPU runtimes beside it, `colab/nano_capacity_r16_lane{1,2}.ipynb`:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AbstractEyes/geolip-nano-beatrix-trainer/blob/main/colab/nano_rec16.ipynb)
[![Capacity lane 1](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AbstractEyes/geolip-nano-beatrix-trainer/blob/main/colab/nano_capacity_r16_lane1.ipynb)
[![Capacity lane 2](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AbstractEyes/geolip-nano-beatrix-trainer/blob/main/colab/nano_capacity_r16_lane2.ipynb)

0.1.89: the far read. The capacity read holds its whole bank of competing keys and a block of scores in memory, so it stops near a
million keys. `capacity_far.far_core` computes the same numbers streamed over the bank, on a grid to 10^8 competitors: the bank is the
capacity read's draw key for key (the keys drawn in numpy and kept in the loop's order; their phases the same sums in the same order),
the scores of each block go into the grid's running sums and maxima, and the median-score condition comes from the base sum (every true
score at the median is the same sum with the true score moved out of the exponent). Models of one layout, key form and rotation share one
bank, drawn once. Each record carries the first N with a wrong look and the N where half the looks go wrong, and `compare_tail` sets a far
record against a stored million-key record of the same bank. The capacity read and the tail read are unchanged. Twelve sixteen-pair
models (six with the records held, six without) read to 10^8 on three TPU runtimes' host machines,
`colab/nano_capacity_far_lane{1,2,3}.ipynb`, each lane first reproducing a stored million-key read:
[![Far lane 1](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AbstractEyes/geolip-nano-beatrix-trainer/blob/main/colab/nano_capacity_far_lane1.ipynb)
[![Far lane 2](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AbstractEyes/geolip-nano-beatrix-trainer/blob/main/colab/nano_capacity_far_lane2.ipynb)
[![Far lane 3](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AbstractEyes/geolip-nano-beatrix-trainer/blob/main/colab/nano_capacity_far_lane3.ipynb)

0.1.90: plain rows opened by a document. The binding's recipe frames every row that is not a grid row (`--plant=1.0`), so a held-out
row that opens with DOC and then plain text never appears in training, and one seed of six read such text as frame content. Below a
share of 1 the rows the share leaves unframed are plain windows of the packed stream, which almost never start at a DOC. `--plain_doc=1`
starts each plain row at the first DOC at or after its drawn place, so the row is a document from its first byte, the held-out text's own
form. Nothing extra is drawn: at a given share the framed and grid rows are the same rows. The default is 0.1.89 bit for bit, and the
names carry `d` after the share (`_p75d`). Six sixteen-pair models with the records held and one row in four of the others a plain
document, on the G4, `colab/nano_plain16.ipynb`, and their capacity on two CPU runtimes beside it, `colab/nano_capacity_p16_lane{1,2}.ipynb`:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AbstractEyes/geolip-nano-beatrix-trainer/blob/main/colab/nano_plain16.ipynb)
[![Capacity lane 1](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AbstractEyes/geolip-nano-beatrix-trainer/blob/main/colab/nano_capacity_p16_lane1.ipynb)
[![Capacity lane 2](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AbstractEyes/geolip-nano-beatrix-trainer/blob/main/colab/nano_capacity_p16_lane2.ipynb)

0.1.91: a larger address code. A grid row's address was two digits of eight (64 addresses), so no row could hold more than 64 records.
`--grid_code=4x4` writes and streams each address as four digits of four (256 addresses) on the same eight rotation pairs: each level
gets two pairs at angles 2 pi k / 5, so a level's five symbols (four digits and the null) stay equally far apart, as the two-digit code's
nine do at four pairs a level. The head's other pairs, the address head's sixteen outputs and the value-path tables' size are unchanged.
`--grid_comps=96` lays a row out as 128 slots of 32 bytes with 96 records, 24 of them asked, which needs the larger code. The default is
0.1.90 bit for bit; the names carry `_ac44` and `_gc96`. Twelve models (the larger code at 48 records and at 96) on one or two G4
runtimes, `colab/nano_code44_lane{1,2}.ipynb`, and their capacity on a CPU runtime, `colab/nano_capacity_c44.ipynb`:
[![G4 lane 1](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AbstractEyes/geolip-nano-beatrix-trainer/blob/main/colab/nano_code44_lane1.ipynb)
[![G4 lane 2](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AbstractEyes/geolip-nano-beatrix-trainer/blob/main/colab/nano_code44_lane2.ipynb)
[![Capacity](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AbstractEyes/geolip-nano-beatrix-trainer/blob/main/colab/nano_capacity_c44.ipynb)

0.1.92: the path in one head's rotation (`path_rotary.py`). The key rotation binds a record's whole key into one head's pairs at the end
of the key; this binds every byte's unit path into one head's pairs at every position. The units are the fusion stage's (the library's
`model/fusion.py`): a byte starts a unit when the next-byte entropy after the three bytes before it, read from an atlas table of byte
trigrams, is at least 3.0 bits (or the cell was seen fewer than eight times, or a special byte or a newline is at hand), so a unit runs
from one choice point to the next, 3.6 bytes on average on the bed's held-out text. `--path_bind=8 --path_form=same|next
[--path_head=7 --path_k=16]` turns head 7's eight address pairs in every block, at every position, by the phases of the bytes since its
unit's start (the last 16 of them): `phi_j(t) = sum_k P[j, k, x_(t-k)]`, P fixed random phases drawn once from a seeded generator. With
`same` the query and the key at a position carry the same path, so two positions with equal paths meet as if unrotated on those pairs;
with `next` the key at a position carries the path that ended one byte before it, so a query meets the bytes that followed its own path
earlier in the row (a continuation lookup whose context is the unit's path, a different length at every position). No parameter is
added and nothing enters the input; every other head and pair keeps its angles, the key rotation's head included. The end read adds
`heldout_bpb_path_off` (the same rows with the path off) and `heldout_bpb_by_repeat` (the held-out bits at the positions whose path
occurred earlier in the row, against the others); `--repeat_read=1` adds the split to a run without the path, and
`rotary_scale.text_read` reads the same numbers from any stored model. The atlas table (240 KB) is fetched without a token from the
rotary repo's `pillar/` folder and checked against its hash (`--path_table` or `$GEOLIP_ATLAS_TABLE` name a local copy). The default is
0.1.91 bit for bit; the names carry `_pbs8h7` or `_pbn8h7`. Twelve models (each form on six seeds, the plain-rows recipe otherwise) on one
or two G4 runtimes, `colab/nano_pillar1_lane{1,2}.ipynb`:
[![G4 lane 1](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AbstractEyes/geolip-nano-beatrix-trainer/blob/main/colab/nano_pillar1_lane1.ipynb)
[![G4 lane 2](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AbstractEyes/geolip-nano-beatrix-trainer/blob/main/colab/nano_pillar1_lane2.ipynb)

0.1.93: the room read (`room_read.py`; reads stored models, trains nothing). For a head and a set of its rotation pairs, it removes the
pairs from the head's scores in every block (the head's query and key zeroed on those pairs after the block's own rotation, so they add
nothing to the scores) or removes the head's output (its values zeroed), and measures the held-out text's cost - overall, at the
positions whose unit path occurred earlier in the row and elsewhere, and by position in the row. The pair groups come from the model's
own split (the relative band, the content pairs, the address anchors, all sixteen). The look reads every head's attention, at the
positions whose path occurred earlier in the row, on the bytes that followed the earlier copies of that path, against the share a
uniform attention would give, by the path's length. Nothing in training changes. The eighteen models of the path stage on one G4
runtime, `colab/nano_pillar_room.ipynb`:
[![Room read](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AbstractEyes/geolip-nano-beatrix-trainer/blob/main/colab/nano_pillar_room.ipynb)

0.1.94: the path in a head of its own (`ninth_head.py`). `--ninth_head=path --path_form=next` adds a ninth softmax head to every
anchored block, beside the eight: 32 dims (sixteen rotation pairs) with its own query, key and value rows read from the block's attention
input and its own output rows added to the block's attention output (196,608 parameters over six blocks, under the same optimizer and
clip). Its weights are born as the block's heads are born (rows of fresh orthogonal draws), from a generator seeded by the head's own tag,
so nothing is drawn from the run's generators: the eight heads, the rest of the model and the data order start as the run without it.
With the path, its sixteen pairs are turned by the unit path's phases (0.1.92's, at sixteen pairs): the query by the path at its own
position, the key by the path that ended one byte before it. `--ninth_head=plain` puts the same head on the block's own rotation (the
angles every head of the block gets): the added head's capacity without the path. The ninth heads ship beside the weights
(`extra_head.safetensors`, scale.json's `ninth_head`) and `rotary_scale.Core` installs them; the end read adds `heldout_bpb_path_off` for
the path form (its ninth head on the block's rotation instead) and the repeat split for both; the room read's look reads the ninth head
as each block's last head, and its kind `look` takes the model's own bits and the look alone. The default is 0.1.93 bit for bit; the
names carry `_nhpn16` or `_nhpl`. Twelve models (each form on six seeds, the plain-rows recipe otherwise) on one to three G4 runtimes,
`colab/nano_pillar1b_lane{1,2,3}.ipynb`:
[![G4 lane 1](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AbstractEyes/geolip-nano-beatrix-trainer/blob/main/colab/nano_pillar1b_lane1.ipynb)
[![G4 lane 2](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AbstractEyes/geolip-nano-beatrix-trainer/blob/main/colab/nano_pillar1b_lane2.ipynb)
[![G4 lane 3](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AbstractEyes/geolip-nano-beatrix-trainer/blob/main/colab/nano_pillar1b_lane3.ipynb)

0.1.95: the path in the early blocks. `--path_blocks=1,2,3` (with `--ninth_head=path`) puts the path in the ninth heads of the named
blocks only (numbered from 1); the other blocks' ninth heads sit on the block's own rotation, as in `--ninth_head=plain`. The heads are
born the same whatever the set, so the model has the same size and the same start as both 0.1.94 forms, and differs from the plain
form only in the named blocks' rotation. The end read's path-off puts every named block on the block's rotation (the plain form's
forward on the same weights). Unset, every block carries the path (0.1.94 bit for bit); scale.json's `ninth_head` names the set as
`path_blocks` when it is given, and the names carry it after `_nhpn16` (`_nhpn16b123`). Six models (six seeds) on one to three G4
runtimes, `colab/nano_pillar1c_lane{1,2,3}.ipynb`:
[![G4 lane 1](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AbstractEyes/geolip-nano-beatrix-trainer/blob/main/colab/nano_pillar1c_lane1.ipynb)
[![G4 lane 2](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AbstractEyes/geolip-nano-beatrix-trainer/blob/main/colab/nano_pillar1c_lane2.ipynb)
[![G4 lane 3](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AbstractEyes/geolip-nano-beatrix-trainer/blob/main/colab/nano_pillar1c_lane3.ipynb)

The path in block 3 only (`--path_blocks=3`, the names `_nhpn16b3`; the same release): six models on one to three G4 runtimes,
`colab/nano_pillar1d_lane{1,2,3}.ipynb`:
[![G4 lane 1](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AbstractEyes/geolip-nano-beatrix-trainer/blob/main/colab/nano_pillar1d_lane1.ipynb)
[![G4 lane 2](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AbstractEyes/geolip-nano-beatrix-trainer/blob/main/colab/nano_pillar1d_lane2.ipynb)
[![G4 lane 3](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AbstractEyes/geolip-nano-beatrix-trainer/blob/main/colab/nano_pillar1d_lane3.ipynb)

0.1.96: the block read (`rotary_scale.block_read`). For a stored model whose ninth heads carry the path, the held-out text read with
the path switched off in chosen path blocks: those blocks' ninth heads on the block's own rotation, the other path blocks on the path,
the trained weights unchanged; by default no block, each path block alone and all of them. No block off is the text read, and every
path block off its path-off read, bit for bit (the text read's loop is shared and its numbers unchanged). Nothing trains: one G4
runtime reads the six models with the path in blocks 1-3 (every subset of the three) and the six with the path in every block (one
block at a time), `colab/nano_pillar1g.ipynb`:
[![G4](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AbstractEyes/geolip-nano-beatrix-trainer/blob/main/colab/nano_pillar1g.ipynb)

0.1.97: the random-phase read (`block_read(..., mode="random")`, `ninth_head.random_phases`, `NinthHeads.random_blocks`). Switching a
path head to the block's own rotation does two things: it removes the path's matches, and it gives the head a positional rotation it
was never trained with (away from its matches a path head meets its keys at unrelated phases). The random mode separates them: the
chosen path blocks' ninth heads turn by fresh uniform phases instead, the query and the key each by its own draw at every position
(fixed by the block and the row length, the same draw in every read), so no match is left and the head stays position-free. The
control is read-only and empty by default (the training forward unchanged); no block on random phases is the text read, bit for bit.
One G4 runtime reads the same twelve models (the path in blocks 1-3: no block, each alone, all three; the path in every block: one
block at a time), `colab/nano_pillar1h.ipynb`:
[![G4](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AbstractEyes/geolip-nano-beatrix-trainer/blob/main/colab/nano_pillar1h.ipynb)

0.1.98: the relabel read (`block_read(..., mode="relabel", depth=True)`, `ninth_head.relabel_table`, `NinthHeads.relabel_blocks`,
`path_rotary.DepthSplit`). Random phases take away two things at once: that a path's phases are the same wherever it stands (the exact
matches) and the particular values the phase table gave each path, which a model can learn. The relabel mode keeps the first and changes
the second: the chosen path blocks' ninth heads take the path's phases from a fresh table of the same law (fixed, its own seed), so
equal paths still meet at equal phases. The depth split adds each read's bits by the path's depth at the predicting position (1, 2, 3,
4, 5 or more), crossed with the repeated-path split; it is off by default and changes no other number. The control is read-only and
empty by default (the training forward unchanged); no block relabelled is the text read, bit for bit. One G4 runtime reads the same
twelve models on the fresh table and again on random phases (those reads equal the random-phase read's records exactly),
`colab/nano_pillar1j.ipynb`:
[![G4](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AbstractEyes/geolip-nano-beatrix-trainer/blob/main/colab/nano_pillar1j.ipynb)

0.1.99: the slot read (`block_read(..., mode="relabel", slots=...)`, `ninth_head.relabel_table(k, slots)`, `NinthHeads.relabel_slots`).
A byte's phase is a sum of one table term per byte of its path, the term in slot k belonging to the byte k back. The relabel control can
now change only chosen slots: the table is the original with the named slots taken from the fresh one (no slot named = the original,
every slot = 0.1.98's fresh table, bit for bit). Every exact match is kept whatever the slots. The record names its slots and the record's
name carries them (`_k0`, `_k1-15`). One G4 runtime reads the same twelve models with every slot, the current byte's slot, the deeper
slots and the previous byte's slot changed (every slot reproduces the relabel read's records exactly), `colab/nano_pillar1k.ipynb`:
[![G4](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AbstractEyes/geolip-nano-beatrix-trainer/blob/main/colab/nano_pillar1k.ipynb)

The longer run at the four-digit code (0.1.91 as is). The four-digit address at 96 records a row and the plain-rows recipe at 48 records,
each at `--steps=48000` (the cosine schedule over the longer run, nothing else changed) on the six seeds of their stored 24,000-step
models: the batches are drawn from the seed and the step, so each long model trains on the same rows at the same steps as its stored one.
Beside them the four-digit address at 96 records at 24,000 steps on six new seeds (the replication). Eighteen models on up to three G4
runtimes, each claimed on the Hub before it trains and read through its look as it lands, `colab/nano_code44_long_lane{1,2,3}.ipynb`:
[![G4 lane 1](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AbstractEyes/geolip-nano-beatrix-trainer/blob/main/colab/nano_code44_long_lane1.ipynb)
[![G4 lane 2](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AbstractEyes/geolip-nano-beatrix-trainer/blob/main/colab/nano_code44_long_lane2.ipynb)
[![G4 lane 3](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AbstractEyes/geolip-nano-beatrix-trainer/blob/main/colab/nano_code44_long_lane3.ipynb)

The far read at the four-digit code (0.1.91 as is). The four-digit address at 96 records a row and the two-digit address at 48, each
trained for 48,000 and for 24,000 steps (six seeds each, 24 models), read to 10^8 competing keys by `capacity_far.far_core` at each
model's own row and address. The models of one row layout share one bank, so a lane draws at most two; every lane first reproduces one
stored million-key read of each layout. Three TPU runtimes' host machines, `colab/nano_capacity_far44_lane{1,2,3}.ipynb`:
[![Far lane 1](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AbstractEyes/geolip-nano-beatrix-trainer/blob/main/colab/nano_capacity_far44_lane1.ipynb)
[![Far lane 2](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AbstractEyes/geolip-nano-beatrix-trainer/blob/main/colab/nano_capacity_far44_lane2.ipynb)
[![Far lane 3](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AbstractEyes/geolip-nano-beatrix-trainer/blob/main/colab/nano_capacity_far44_lane3.ipynb)

0.1.100: the schedule's floor and a flat phase (`bed_optim.lr_schedule`). The rate warms up over the first W steps (100), then follows a
cosine from its peak down to a tenth of it at the last step S. `--lr_floor=m` ends the cosine at m times the peak instead of a tenth;
`--lr_flat=f` holds the peak for the share f of the run after the warm-up and runs the cosine over the rest:

    lr(step) = lr x (step + 1) / W                                                   for step < W
    lr(step) = lr                                                                    for W <= step < E,  E = W + f (S - W)
    lr(step) = lr x (m + (1 - m) x (1/2)(1 + cos(pi (step - E) / (S - E))))          for step >= E

The defaults (m = 0.1, f = 0) are every earlier run's schedule bit for bit (a default run's weights and record unchanged: three-step
models at the defaults equal 0.1.91's byte for byte); when not the defaults the record names both and the run's name carries `_lf<m>` and
`_fl<f>` after its length. Twelve models at 24,000 steps - the four-digit address at 96 records a row on seeds 4 and 1 and the plain-rows
recipe on seeds 2, 3, 6 and 1, each under `--lr_floor=0.5` (the cosine ends at half the peak) and under `--lr_flat=0.8` (the peak held
to step 19,220) - on the same rows at the same steps as the stored 24,000-step models of those seeds, on up to three G4 runtimes, each
claimed on the Hub before it trains and read through its look as it lands, `colab/nano_slow_anneal_lane{1,2,3}.ipynb`:
[![G4 lane 1](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AbstractEyes/geolip-nano-beatrix-trainer/blob/main/colab/nano_slow_anneal_lane1.ipynb)
[![G4 lane 2](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AbstractEyes/geolip-nano-beatrix-trainer/blob/main/colab/nano_slow_anneal_lane2.ipynb)
[![G4 lane 3](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AbstractEyes/geolip-nano-beatrix-trainer/blob/main/colab/nano_slow_anneal_lane3.ipynb)

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

0.1.69 (2026-10-20): THE GENERATOR STOPS AT THE WINDOW. The control arm's live thought read at its boundary crashed: an answer that never said END grew a row at the window's end to 4,097 bytes against a position table of 4,096 (the thought arm's read had ended every answer before the window). `caption_screen.greedy` takes the window (`ctx`) and stops a row that has reached it; `thought_screen` passes it. Matching answers are untouched: a value that fits ends before the window does, so only run-on wrong answers are cut. The control's end read is taken by `thought_stored` on its shipped end checkpoint. The thought notebook pins v0.1.69. 324 tests.

0.1.68 (2026-10-20): THE SPEC FRAME'S SIZE IS ALWAYS GENERIC. A thought row whose records all belong to one system rendered its spec frame at 32 bytes a slice (three slices = 96 bytes), a size no letter names; the sized open tag asserted, and the thought reads on the real rows failed at the first such row (the notebook's baseline read on the source caught it before any training). The single-system frame now rounds up to the next generic size (32, 64, 128, 256, 512), its spare slices filler, every record written; multi-system rows render byte-identical to 0.1.67. The thought notebook pins v0.1.68. 323 tests.

0.1.67 (2026-10-20): THE THOUGHT PHASE AND THE THOUGHT READS. `nano-castor-v1.2-th` (and the Pollux twin): the framed body's end state continued through one phase of a billion bytes of the plain continuation stream under the framed diet WITH the thought kind (`framed_diet.FORMS['p50t']`: the art thought dataset's documents in the frame grammar, one row in ten); `-tc` its control, the same phase under the diet without thoughts. The thought pack comes from the Hub (`thought_frames.ensure`: the dataset's shards); the loader holds one series in ten out of training when the dataset marks no row held (`thought_frames.is_held`, the same rule in the stream children and the reads). `thought_screen.py`: THE THOUGHT READS on held rows in three layouts (record, chain against the frameless layout, unknown, the sealed layout's travel read), the prior and the trained rows' reads beside them, every answer generated to END; `nano_v1.thought_live` at every boundary of a thought-phase craft (both arms), `thought_stored` for the baseline on the source; `thought_table`, `thought_marks`, `thought_marks_table`. THE EXPERIMENTS REPO: `nano_v1.REPO_EXPERIMENTS` (AbstractPhil/alephllm-nano-beatrix-training) takes the nano training experiments (`repo_of`); a continuation may start from another repo (`source_repo`); `control_of` honours a craft's named control. `colab/nano_thought_castor.ipynb` (the G4 named in its metadata). 322 tests.

0.1.66 (2026-10-19): THE FRAMED CASTOR, and every frame size in the table. `nano-castor-v1.2` (and its `-p25` / `-p75` doses): the Castor body from the same birth as `nano-castor-v1` and v1.1 (castor-both seed 3), the v1.1 recipe, the framed diet at the same dose as Pollux's v1.2, so the pair compares bodies and nothing else; `nano_v1.control_of(name)` names a framed craft's plain twin (`nano-castor-v1.1`, `nano-pollux-dual-v1.1`). The boundary row's reads now carry the copy whole at every frame size (32, 128, 256, 512 beside 1,024) and `v12_table` shows them, after the v1.2 run's table collapsed the sizes into one number (the 1,024-byte frame alone fell over the anneals while 32 to 512 held). `colab/nano_v1_2_castor.ipynb`: the v1.2 notebook for the Castor twin. 317 tests.

0.1.65 (2026-10-19): THE DATASET'S TOOLS LEAVE THE TRAINER. The art thought dataset's own tooling (the vocabulary and its seals, the row validator, the series check, the sampler, the generation prompt and its document, the seed contract and the license check: `thought_pack.py`, `docs/thought_generation_prompt.md`, `tests/test_thought_pack.py`, 0.1.59 to 0.1.64) moves to the dataset's own repo as its tools folder (`tools/thought_tools.py`, standing alone; the lists it shares with the caption pack inlined with the same members, so the seals are the same tenth), because a dataset's licenses and generation rules are not the experimental model's system. The trainer keeps what training needs: `thought_frames.py` (the renderer, the window, the loader, the framed diet's `thought` kind) with its own two schema facts, a spec's `records` and the `chain_slices` a chain of thoughts takes, which the dataset's validator shares; `ThoughtPack.check` (the validator call) becomes `depths`. Nothing a model sees changed. 317 tests.

0.1.64 (2026-10-19): THE SEED CONTRACT AND THE LICENSE of the art thought dataset. The dataset is a reusable public dataset under Apache-2.0 (the example row's license, the prompt and the document say so); data represented from other datasets enters only under permissive licenses with attribution and citation. `thought_pack.py`: a SEED is the facts of a hidden image drawn from a licensed source for the generating model to think from: the spec the prompt takes, with `seed_id` (<source>:<id>), `domain`, `genre`, an optional `hint` (at most 600 bytes of source text where the license allows), `source` (dataset, id, license, url) and `held` (true exactly when the spec uses a sealed value); `validate_seed` is its gate (the record rules shared with the row validator through `record_problems`; facts only, no mood, intent or story; one to thirty records; the license permissive by `permissive` / `license_key`; the held flag against `sealed_used`), `check_seed_file` and `--check-seeds=<file.jsonl>` the file checker, `EXAMPLE_SEED` the reference; the document gains the seeds' section (the fields, the layout seeds/<source>/, the example) and the license section. 324 tests.

0.1.63 (2026-10-19): THE IN-TRAINING CARD KEEPS A RELEASE CARD. v1.2's launch wrote the one-craft 'training now' card over the repo's v1.1 release card (the four models, their arms, the use and training sections), which had to be restored by hand. `card_in_training(names, existing)` now takes the repo's current README: when it is a release card (it has a '## Use' section) the training section (`training_block`: the version's note, the crafts, where their folders fill, between `<!-- training now -->` marks) is inserted before '## Use', or replaced in place when a launch wrote one before, and the rest of the card stays byte for byte; without a release card the in-training card stands alone as before (`_card_fresh`). `colab/nano_v1_2.ipynb` passes the current README and writes only when the card changes. 323 tests.

0.1.62 (2026-10-19): THE ART THOUGHT DATASET IN THE FRAME GRAMMAR, and its loader. `thought_frames.py`: a thought row becomes one document in the caption register (the head `DOC MODE c SEP t SEP`): THE CHAIN FRAME, one sized, sliced thinking frame whose size letter is the depth (simple c/d, medium e, deep f/g), every thought a child thinking frame (a run of 32-byte slices) whose first record is its kind (`CUE step SEP see`, a newline, then the text across the run), the children siblings under the one parent with no gaps; THE SPEC FRAME, the records as the caption pack's symbolic trie (a child run a system, a relation its own, the parent the smallest that holds them); THE WORDINGS, each introduced by `CUE say SEP caption|prompt|tags|critique` and a newline; THE QUESTION as the caption documents ask it; in four LAYOUTS (A thought-first, B caption-first without the chain, C frameless, D question-first with the question answered at the end), one sealed from training (D). `thought_window` packs documents to the context as the caption windows do; `ThoughtPack` / `load` read the Hub repo's JSON Lines layout (data/<depth>/<domain>/*.jsonl for train, held/*.jsonl for the reads; a train row with sealed values or a sealed layout dropped; `NANO_THOUGHT_PACK` for the stream children); `records_in` and `chain_text` read a rendered document back (the tests, the reads). The framed diet gains the `thought` kind in a form of its own, `FORMS['p50t']` (carry .30, frames .20, caption .20, note .10, thought .20; the live p50 form untouched). To make every row renderable without loss, `thought_pack.py` now bounds a record's key and value together to 27 bytes (one 32-byte slice; the caption pack's own budget) and the thoughts' text to the depth's thinking frame (`chain_slices`: simple 256, medium 512, deep 2,048 bytes of slices), both as validator rules; three attributes took shorter names (`form compose`, `intent for`, `mood from`) and a few list values shorter words; the example row's thoughts fit a 512-byte frame. 321 tests.

0.1.61 (2026-10-19): 0.1.60's adults-only rule WITHDRAWN from the dataset tooling: that content rule governs pictures generated from the diffusion models, not the text dataset, so `thought_pack.py` returns to the plan's eight rules with the 0.1.59 vocabulary (the gender, age, role, setting, purpose and audience lists as they were; the example row's booru tags as they were; the seals redrawn, identical to 0.1.59's). Kept from 0.1.60: the sampler draws hues only (no booru colour modifiers) for clothing, things, eyes and palettes. 317 tests.

0.1.60 (2026-10-19): THE ADULTS-ONLY RULE in the art thought tooling (a content rule set on 2026-10-19: no children, and no school or classroom defaults, in any row). `thought_pack.py`: the vocabulary carries no child, kid, boy, girl, baby, toddler, teen, student, pupil, school, classroom, playground or kindergarten (the gender list is woman / man / person / nonbinary person; the age list the adult decades 20s to 80s and elderly; the role, setting, purpose and audience lists cleaned); the sampler draws from those lists alone, and hues only (no booru colour modifiers) for clothing, things, eyes and palettes; the validator's rule 9 rejects a row holding any such word anywhere (the spec, the genre, the thoughts, the wordings, the tags including the booru count tags such as 1girl, the questions) by whole-word match with the school compounds; the prompt and `docs/thought_generation_prompt.md` state the rule first; the example row's tags read `solo, woman`. The seals are redrawn from the cleaned lists. 317 tests.

0.1.59 (2026-10-19): THE ART THOUGHT DATASET'S TOOLING, the generating side's instruments before any row is written (the dataset AbstractPhil/thought-symbolic-image-captions: text only, one act of thinking about an image nobody sees, at three depths). `thought_pack.py`: THE SYSTEMS TRIE (the caption pack's seven systems and five added: form, relation, mood, intent, story; 58 keys, 41 of them closed lists of 718 values in all, the pack's colours and expressions reused), THE SEALED HELD-OUT VALUES (`seal`: a tenth of every closed list, 73 values, drawn once from one seed, the schema example's values kept open; `lists_for` gives the training lists without them or the sealed values alone; `write_vocab` writes `vocab/trie.json` and `vocab/held.json`), THE ROW VALIDATOR (`validate`: the plan's eight rules and the depth counts: a record value plain, lower-case, at most 20 bytes, from its closed list; the depth's counts of records, thoughts and questions; every question answered by a record, by the chain under a key the spec lacks, or 'unknown'; the deep row's unknown and chain questions; the wordings, three for deep; a mood whose evidence is the key of a record; no pixels and no seeing the file; a training row free of sealed values and a held row using one; the source's license), THE SERIES CHECK (`validate_series`: simple within medium within deep), THE PILOT CHECKER (`check_file`: a JSON Lines file to the problems by rule and the depth counts), THE SPEC SAMPLER (`sample_spec` / `sample_series`: the facts drawn from the lists in one fill order, consistent and nested by construction; the model adds mood with its evidence, intent and story), THE PROMPT (`prompt`: the task, the schema with the example row, the rules, the lists the model may draw from, the series form, a held split for the reads) and `docs/thought_generation_prompt.md` (written by `--doc`, kept equal by a test). The renderer into the frame grammar (the depth as the thinking frame's size letter, the kind as its first record, parallel thoughts as sibling frames) and the framed diet's `thought` kind follow in 0.1.60. 317 tests.

0.1.58 (2026-10-19): THE FRAMED DIET, the frame-pretrained nano v1.2. Every nano so far met frames in a continuation of 1.0 to 1.5 billion bytes after the 16.1 billion byte plain diet, and a later phase displaced an earlier form; `framed_diet.py` puts the frames into the diet itself: `FramedStream` wraps any library training stream (opened at micro-batch 1) so that a fraction p of its rows is FRAMED from the first step, in four forms by a seeded draw in the three registers (carry rows, frames rows, the caption pack's trie windows with the frameless questions, note rows; a framed row takes two of the stream's rows as its text, a plain row is the stream's row verbatim; the forms `FORMS['p50' | 'p25' | 'p75']`), exact under resume (the state: the row counter, the kinds' counts, the inner stream's own state); built in the stream child (`ProcessStream(..., framed=)`), the phase's dataset name unchanged, so the manifest, the validation stream and the boundary names match the control's. The crafts `nano-pollux-dual-v1.2` (half the rows framed; `-p25`, `-p75`): Pollux dual from the v1 birth, the v1.1 recipe, the 2s diet; the plain v1.1 craft is the control. `run()` adds the caption read against the prior at every boundary of a framed craft (`caption_live`), the carry and note gauges, the framed diet in the manifest; `v12_marks` / `v12_table` / `v12_marks_table`: the loss rule (every frame read's end within 0.10 of its peak), the text bars against the control (0.02 at every boundary, 0.01 at the end), the meaning bar (+0.20 over the prior). `colab/nano_v1_2.ipynb`: the preflight measures the framed stream's pace on the card and prints a row of each kind; the train; the table beside the control and the marks. 311 tests.

0.1.57 (2026-10-19): THE MICRO TIER, a size step above the nano. `bodies.SIZES`: the nano body widened and deepened by one rule (the hub's K = d/8 at D = d/16, the head's K = d/4 at D = d/16, the bank's ff = d, seven experts, 64 dims a head; an arm's layout written for the size's depth), `craft_cfg(arm, ctx, args, size)` with the nano bit for bit (the six-block layouts and the bigger address stay the nano's); the crafts `micro26 / micro45 / micro66-pollux-dual-v1` (d384 x 8 = 25,787,841, d512 x 8 = 44,944,065, d512 x 12 = 66,077,601 parameters at context 4,096; the nano 9,267,281) and their `.1` crafts (the v1.1 recipe); the size flows to the births (`write_birth`, the file's metadata), the presets, the gauge child (`--size`) and the run tag (`_micro26`). `micro.py`: THE THROUGHPUT SCAN (the model step at three batch shapes on random bytes, the shape picked as the v1 runs pick theirs, then 300 steps on the fineweb-edu stream in the real trainer timed from a warm step; bytes a second, the card's memory, the bytes a day and a week, the bytes a parameter a week, the hours of the 2s diet; the record to `preflight/micro_scan_<utc>.json`, the weights never) and `colab/micro_scan.ipynb` (the nano reference row first). No micro body trains yet: the frame-pretrained nano cycle (v1.2, 0.1.58) runs before any micro. 305 tests.

0.1.56 (2026-10-19): THE FRAMELESS QUESTION, the caption phase's repair. The first caption phase (Pollux dual, 1.5 billion bytes)
taught the format and not the meaning: the body copies a record out of the nested system frames whole in 0.961 of 256 held items,
but asked from the caption alone it answers 0.168, and the control, whose frames never matched their captions, answers 0.172 with the
same value bits: each key's commonest value, not the caption. The cause is in the data form: the frame holding the answer stood before
every question, so copying the frame answered it and the caption was never needed. `frames_data.CAPTION2` withholds the frame in half
the caption documents (`p_frame` 0.5; a frameless document always asks its question, a record of its own, so the caption has to answer);
its control `CAPTION2_CTRL` keeps the swap, so in its frameless documents the truth is unrelated to the caption and only the prior can
be learned. The crafts `nano-<body>-v1.1-mx-cp2` and the control `nano-pollux-dual-v1.1-mx-cpx2` run the phase from the mixed arm's
end. `caption_screen.read` adds the prior-only baseline on the same held items (each key's commonest training value over a sample of
50,000 training documents; `prior_of`) and `extraction_over_prior`; `caption_table` takes the arm. The caption notebook runs the
repaired pair, prints the prior beside every extraction number and marks NEW WORDINGS against both the control and the prior; it pins
v0.1.56. The first phase's documents are unchanged (`p_frame` absent draws nothing).

0.1.55 (2026-10-19): THE CAPTION PHASE on the caption pack. `caption_pack` reads the Beatrix caption arm's pack from the Hub (the
photograph, illustration, web-image and COCO rows of `AbstractPhil/alephllm-mini-beatrix-training`, the first shards of each source;
`NANO_CAPTION_PACK` names the cache) into a byte blob and an index: a row's attributes become records whose keys are PATHS of a
closed vocabulary, the system then the attribute (`look hair`, `person age`, `wear clothing`, `scene mood`, `shot light`,
`thing <name>`), a row's several captions are its wordings, the pack's held split is the gauge. `frames_data.caption_frame` writes a
document's records as one sized frame: the records of one system in a child frame of one or two 32-byte slices, the children nested
in the smallest frame that holds them, their places shuffled (a symbolic trie: a system of systems); one system alone gives the flat
frame. A caption document is the caption head, the frame (a shuffled subset of the records), a newline, one caption and, in half the
documents, the question of a framed record; a caption window is documents back to back. The phase `cp` (`nano-<body>-v1.1-mx-cp`,
1.5 billion bytes from the mixed arm's end) keeps the mixed arm's coin, registers and plain rows and draws, of its framed rows, a
carry row one time in three and a caption window two; the control `nano-pollux-dual-v1.1-mx-cpx` takes every document's frame from
another document. `caption_screen` reads a stored craft on the held split: the extraction read (the caption alone, then the path
question: the value) and the copy read (the frame, the caption, the question); `nano_v1.caption_stored` runs it as a child and
`caption_table` lays the boundary reads beside the mixed arm's end; the caption notebook (`colab/nano_v1_1_caption.ipynb`) runs the
phase, the control and the reads and pins v0.1.55.

0.1.54 (2026-10-18): `gate_stored` writes the held-out head beside its records before the screen's child runs. The gate notebook's
preflight smoke runs in its own folder, which had no `heldout_fineweb_edu_head2048.jsonl`: the child asserted on its first run on
a rented card (the training entry point and the gauge path write that file; the gate runner did not). The preflight also copies
the held-out it already fetched into the smoke folder, so nothing is downloaded twice; the gate notebook pins v0.1.54.

0.1.53 (2026-10-18): the gate screen's protocol runs under no_grad. The first 192-row read on a 24 GB card stopped at the 10 GB
budget `gate_stored` gives the child: the grid reads and the frame generation kept an autograd graph the scoring reads never
needed (the two-row smoke on a CPU had the memory for it). `gate_screen.screen` is decorated `torch.no_grad`; the record and the
notebook are unchanged otherwise; the gate notebook pins v0.1.53.

0.1.52 (2026-10-18): the second seed of the mixed arm, and the mixed notebook reads any mixed craft. `nano_v1.CRAFTS_MIX2` adds
`nano-pollux-dual-v1.1-mx2`: the mixed phase from the same diet-end state, dose and mix as `-mx`, at a second data seed
(`TrainConfig.seed` 2 against the default; the phase's stream and its framed rows draw from it), so the mixed phase's marks get a
seed pair on one body. `mixed_table` lists every mixed craft of a body beside its frames, carry and control arms. The mixed
notebook's read cell takes the mixed craft named for each body (`-mx` or `-mx2`) and the arm code from the craft table, never from
the name's tail; it pins v0.1.52. Tests.

0.1.51 (2026-10-18): THE GATE SCREEN. The body's own loss decides which of its notes to keep. `gate_screen.py` keeps the
reading-with-notes protocol as it is (a held-out row read in 64-byte grid steps; at each grid point the greedy next byte decides
whether a frame opens, and an opened frame is written to END and inserted) and adds, for every note the body writes, three
teacher-forced reads of the bytes that follow it: with the note, without it, and with its value replaced by spaces. A note is kept
at a horizon and a margin when the text after it costs at least the margin fewer bits than both controls (Toolformer's two-control
filter); horizons 64 to 512 bytes and margins 0.25 to 4 bits are scanned and reported as a table, never set. A random-note control
(a one-record frame with a key and a value from the bed's vocabulary at the same positions) runs beside every read and must fail.
Two modes: `self` reads the note gauge's rows (held-out documents holding a fact), `plain` the carry gauge's plain rows.
`nano_v1.gate_stored` runs it on a stored craft as `gauge_stored` runs the gauges. The record carries the notes a row, the kept
share at every setting, the gain distributions, the share of kept notes whose value stands in the preceding 1,024 bytes, the share
whose key the extractor also finds, examples kept and dropped, and the wall time. The notebook `colab/nano_v1_1_gate.ipynb` reads
the premise (does the body propose at all) and the screen on the candidates and ships the records to the Hub as JSON. Tests.

0.1.50 (2026-10-18): document-start plain rows in the mixed phase, and a preflight fix. A read of every final's cost over the
first 64 bytes of a plain held-out document found the framed streams' head tax to be the opening: their plain rows are slices
taken mid-document and never begin with DOC, so a window-initial DOC announced a frame with probability 1.0 and a plain document's
first byte cost 17 to 22 bits against 5 to 6 for the controls (a third to a half of a framed phase's text cost).
`frames_data.MIXED` gains `p_doc_start = 0.5`: half of the mixed phase's plain rows begin at a document boundary (`doc_start_row`:
DOC and the document from its first byte), the other half keep the mid-document start the diet's windows have; the frames, carry
and note streams are unchanged byte for byte. The arm notebooks' preflight smoke reads the first craft's own start step (a list
mixing `-cf-nt` and `-nt` arms took the last arm's step and asked the Hub for a checkpoint that does not exist). The notebooks pin
v0.1.50; tests.

0.1.49 (2026-10-18): THE MIXED-PHASE ARM. The frames-then-carry arm lost the near read in the carry phase's first 0.2 billion
bytes (the data form, not the habit), and the register fade showed a register's lookup survives a framed phase only if that
register is in the mix. `frames_data.MIXED`: a framed row (the bed's dose; the same coin and text position as every other arm's
row) is a carry row two times in three and a frames row once, its register drawn caption one time in two, math and code one in
four each, from a third generator so a caption carry row is the carry arm's row byte for byte (`mixed_draw`, `row_bytes(...,
carry='mixed')`, the stream `nano-frames-mixed`). `nano_v1.CRAFTS_MIX` (`nano-<body>-v1.1-mx`, from the diet's end),
`CONT_TOKENS_OF` (the mixed phase 1.5 billion bytes: its carry rows total the carry arm's 0.75 billion while the frames rows add
0.375), `mixed_table`; the carry gauge on the mixed arm's reads; the card names the arm. `colab/nano_v1_1_mixed.ipynb` (the marks
before the run: NEAR KEPT, FAR GAINED and FAR MATCHED, TEXT, REGISTERS KEPT; C4 reads the math and code registers on the mixed,
carry and control ends). `colab/nano_v1_1_release.ipynb` pins v0.1.49; tests.

0.1.48 (2026-10-18): the v1.1 card's note on the v1 run points at the archive. The v1 folders moved to
`AbstractPhil/alephllm-nano-beatrix-training` under `archive/castor-pollux-v1/` on 2026-10-16 while the card, rendered from
`nano_v1.V11_NOTE`, still said they stayed in place; the note now names the archive with a link, and the twins card is regenerated
from it (the card is generated: a wording change goes into the generator, never into the Hub file).
`colab/nano_v1_1_release.ipynb` pins v0.1.48; tests.

0.1.47 (2026-10-18): `automodel.complete_config`: a manifest written by an older library lacks the model-config fields added since
(the v1.1 bodies, trained before the head's mode and dial existed, have no `head_mode`; the first local release stopped at the
parity stage with a KeyError). For a craft in nano_v1's table the missing fields come from the bodies table (`bodies.craft_cfg` at
the v1 context, the config the trainer and the gauges build the model from), which must agree with the manifest on every field the
manifest has (a disagreement fails the config gate); for a craft outside the table from the library's defaults. The config gate's
record names the fields filled and their source. `colab/nano_v1_1_release.ipynb` pins v0.1.47; tests.

0.1.46 (2026-10-18): the parity gate's record carries the device as its own field (0.1.45 put it among the weight files' entries,
where readers expect one entry a file; `test_a_tiny_craft_passes_config_weights_and_parity` failed and 0.1.45 was tagged over it);
`colab/nano_v1_1_release.ipynb` pins v0.1.46.

0.1.45 (2026-10-18): the AutoModel gates run on the card when one is visible (`automodel.device`): the parity stage and the load
gate build and run their models on the same device and compare the logits on the CPU, so the exactness bars (vendored equals the
library exactly; the loaded model within 1e-5 of the parity logits) are read between runs of the same kernels; the gate records
carry the device. `colab/nano_v1_1_release.ipynb` pins v0.1.45.

0.1.44 (2026-10-18): the load gate's child interpreter runs with UTF-8 standard streams (`-X utf8`, `PYTHONIOENCODING`, the
streams reconfigured to backslash-replace): on a Windows console (cp1252) the gate's log line, which carries the model's greedy
bytes, ended the child with a UnicodeEncodeError after the load had passed, so the gate read FAIL for a reason that was not the
model's. Found running the gate locally under transformers 5.20.1; `colab/nano_v1_1_release.ipynb` pins v0.1.44.

0.1.43 (2026-10-18): THE AUTOMODEL COVERS EVERY CRAFT FOLDER. `automodel.release(out, repo, crafts=None, token)` packages every
craft folder the repo holds (the four bodies and their continuation arms; `final_files` takes the resume state of the last phase
done); each `config.json` carries `craft` (the model's name, its body, its arm in plain words, what it continued from, the step
its own training started at, the phase and step its weights ended on, the bytes seen) and `activated` (the mechanisms its body
switches on: the hub blocks, dual hubs, rotary codes, the softmax blocks, the Canon layers, the expert bank and its byte table,
the last block's bound, the head's mode, the position table), so `model.config.craft` says what a loaded model is; `crafts.json`
at the repo's top indexes every packaged folder with its end reads (the last boundary row, or a later protocol's re-read of the
end checkpoint from a `regauge_*/` folder, which wins; `end_reads`); the card renders the continuation arms body by body from the
same facts (what each arm trains on, the steps, the bed's reads, the note arm's own columns) and the bodies' table stays the
bodies'. `nano_v1.body_of`, `arm_text` ('frames, then carry'), `start_step`; `version_of` knows every arm family.
`colab/nano_v1_1_release.ipynb` runs the release (the gates run on the CPU; a write token); tests.

0.1.42 (2026-10-18): the note row's question takes its room from the real record's own length (0.1.41 drew it from the bed's
synthetic record length, so a key and value up to 10 bytes longer overran the window by up to 8 bytes in the first preflight); a
test forces the longest record over 300 draws. `colab/nano_v1_1_notes.ipynb` pins v0.1.42.

0.1.41 (2026-10-18): THE NOTE ROW, the second instance of the self-written frames: the frame's content is a fact the text itself
states. `frames_data.extract_facts` finds definitional facts in a row's own text by fixed patterns (a definition, an abbreviation,
a quantity, a capital, a place, the founding year of an institution; no pattern reads a person's name with a birth, an age or an
address), `frames_data.note_row` writes one 32-byte lookup frame holding the fact at the first 64-byte grid point after the
sentence that states it, carries restating it on the grid every 384 to 640 bytes, and the question in the last 512 bytes asking
the key; the question's distance to the fact follows where the fact falls, so one data form trains the near and the far read; a
text with no usable fact stays plain. The stream `nano-frames-note` and its mix `nano-frames-note-mix`; the crafts
`nano-<body>-v1.1-nt` (from the diet's end) and `nano-<body>-v1.1-cf-nt` (from the carry arm's end state), one note phase of 1.0
billion bytes. The reading-with-notes gauge gains two note modes (`--note_gauge=1`, the record's `note_planted` and `note_self`
blocks, the tag's `_note`): the note given with the carries the body's own, and the note absent so the body must notice the fact
and write it; each reports the whole value by the question's distance to the fact, the rows whose own note held the key, and the
per-row dump. `colab/nano_v1_1_notes.ipynb` (one body by default; its preflight reads the start state in both modes before any
training) pins v0.1.41; tests.

0.1.40 (2026-10-17): THE FRAMES-THEN-CARRY ARM: a second-stage continuation craft a body, `nano-<body>-v1.1-fr-cf`
(`nano_v1.CRAFTS_CONT2`), starts from the frames arm's end state (the exact state: fp32 weights and both optimizers at step
65,237) and runs the carry phase (1.0 billion bytes, 3,815 steps, to 69,052), so a body that already reads a frame within a
thousand bytes of its question learns to write carries; `nano_v1.diet_of` follows a craft's source (a second-stage plan = the
source's plan plus the carry phase), `cont_init` reads the source's own last boundary, `plan_steps` gives a craft's whole plan in
steps, `cont2_table` the four-column read (frames end, after, the carry arm, the control). The carry phase sits one slot later in
the plan than the -cf arm's, so it reads the same mix and dose in a different shuffle. `colab/nano_v1_1_frames_carry.ipynb` (one
body by default) and `colab/nano_v1_1_carry.ipynb` pin v0.1.40; the carry notebook's end step now comes from `plan_steps` (the
floor division before it read 65,236 where the stored final is 65,237). The first-stage arms read exactly as before; tests.

0.1.39 (2026-10-17): the reading-with-notes gauge gains a per-row dump (`--carry_dump=1`: beside its record,
`bed_<tag>_rows.jsonl`, one line a row with the early records, the asked key and value, every grid read with the open byte's
probability, every carry as generated and as inserted, the latest positions holding the key and any carry, the answer as generated
and the value's per-byte bits; and `bed_<tag>_rows.npy`, the rows as the model left them), a plain-row read (`--carry_plain=1`:
held-out text with no early frame and no question, read on the same grid over the same span; the record's `carry_plain` block and
its own `_plain` dump), and a row count (`--carry_rows=N`, default 192; the first 192 rows of any N are the 192-row read's rows).
`nano_v1.gauge_stored(..., extra={...})` passes such arguments to the child; the run tag carries `_rowsN` and `_plain` when set.
`frames_data.carry_row`'s meta adds `early_vals`. Defaults unchanged: a 192-row record reads as before.
`colab/nano_v1_1_carry.ipynb` pins v0.1.39.

0.1.38 (2026-10-17): the reading-with-notes gauge reads the 64-byte grid of the row AS IT STANDS after writing its own carry
(`frames_data.next_grid`): every training carry sits on that grid, and 0.1.37 stepped off it by the carry's length after a row's
first carry, so the first -cf arm's mid-run gauges opened exactly one carry in every opening row and never a second; gauges read
with 0.1.37 (`carry` blocks dated before this release) are re-read with this one. `colab/nano_v1_1_carry.ipynb` pins v0.1.38.
Nothing else changes.

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
