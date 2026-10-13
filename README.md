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
| `geolip/nano_beatrix/bed_runner.py` | runs a grid of arms in sequence on one card (the launch point for a rented card): the grid expands from comma lists, finished runs are remembered so a restart resumes, a stop file ends the grid between runs, each finished run can ship its record and weights to the Hub repo |

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

The bed reads its corpus (`<out>/bed/bed_corpus.bin` + offsets) and a held-out file (`<out>/heldout_fineweb_edu_head2048.jsonl`)
from an output folder (`bed_corpus.py` builds them), then:

```bash
python -m geolip.nano_beatrix.bed_train --out=<out> --arm=smr-fixed --seed=1 --ctx=4096 --mb=8 --steps=12000 --plant=1.0 --records=8 --dists=64,512,1024,2048,3072
```

Framed rows: add `--frames=dyn` (dynamic sectors of generic sizes; `--sizes=32,128,256 --frame_records=4 --nest=0.3 --overlap=0.0
--ask=any|nested|overlap|weights --facts=M --register=caption|math|code --tags=mode|esc`) or `--frames=A` (the fixed reference
layout; `--n_lookup --n_bulk --ask=lookup|bulk|weights`). Without `--frames` the rows are the planted rows, bit for bit.

Every run prints its job size, the time spent and the time left, and ends with the gauges. The record lands at `<out>/bed/bed_<tag>.json`
and the weights at `<out>/bed/<tag>/`. `python -m geolip.nano_beatrix.bed_table <out>/bed --pairs` builds the results index.

## Tests

```bash
pytest -q
```

The frame composer's tests run on the CPU without a model.

## Status

0.1.2 (2026-10-12): the bed as it ran the first night, moved here from the research tree, with the frame composer and its wiring;
the framed trainer smoked on the card from this repo (the dynamic frames, the weights source, the fixed layout).

## Licence

MIT (LICENSE). The weights on the Hub carry their own licence (CC BY-NC-SA 4.0, stated there).
