# -*- coding: utf-8 -*-
"""nano_v1.py - THE NANO V1 CRAFTS (0.1.26): four bodies of the nano bed, each from the birth of its strongest bed run, trained through
Beatrix 2s's diet by the library's own trainer (geolip.alephllm Trainer), with the bed's gauges read into the same TensorBoard at
intervals.

The crafts (CRAFTS; the strongest seed of each body by the v4 bed's whole trains, 2026-10-14):
  nano-pollux-dual-canon-v1   altdc-both seed 1   Pollux's layout (hubs in blocks 0/2/4, rotary softmax in 1/3/5), dual hubs, Canon layers
  nano-pollux-canon-v1        polluxc-both seed 2 Pollux (rotary hubs and rotary softmax alternating) with Canon layers
  nano-pollux-dual-v1         altd-both seed 1    Pollux's layout with dual hubs
  nano-castor-v1              castor-both seed 3  Castor: rotary softmax in every block (the baseline)
The body (bodies.py): d256, six blocks, eight heads, hub address 32 @ D16, bank seven experts x ff256 with the byte table added, head
64 @ D16; context 4,096.

The start (birth_state): the bed built each craft at context 1,024 right after torch.manual_seed(seed) (a CPU build) and carried the
position table to 4,096 with the new rows zero; the v1 start is exactly that, written once as a safetensors file and taken by the
library's weights-only start at step 0 (Preset.init_from).

The diet (diet): the library's chronological form at scale 1 (make_v3_preset's phases) = Beatrix 2s's: warmup_wikitext 0.3B,
fineweb_main 5.0B, curriculum_s0..s8 8.8B, anneal_nochat 1.0B, anneal_mix 1.0B = 16.1B bytes; every phase's stream at its own seed
offset (phase_seed_offset, so no stage replays another's shuffle head).

The recipe (preset): the 2s recipe: Muon 2e-2 on the 2D/3D trunk tensors + pure Adam 3e-4 on the rest, warmup 200 steps then flat,
clip 1.0, bf16 autocast, eager, 262,144 tokens a step (micro_batch x grad_accum x 4,096); the head address trains (the birth recipe);
no anchor governor (the bed's bodies were never governed).

The curves: the trainer's own (train/*, the health census every 500 steps, the full eval every 2,000: the val bits a byte on the
fineweb-edu holdout = the 2s's gauge, the toggle ledger, canaries, the special-token gauge, bits by position band) and the bed's gauges
(bed_train --gauge_only on the craft's weights, a child process) every GAUGE_EVERY steps and at every phase boundary, under bed/*: the
keyed record by frame size 32..1,024 (bed/scaling), the copy, the nested and the redefined-key questions (bed/frame), the planted record
at distances to 3,584 bytes and the held-out bits by position (bed/window), the canary rows, the bits by byte class and with the bank
off (bed/text). The 2s diet holds no framed rows: the frame reads are transfer reads (what the diet teaches about lookup in context).

The data: every training stream is composed in a child process (ProcessStream) that runs ahead of the card by a few batches; the
batches and the resume position are exactly the library stream's (each batch travels with the stream state taken right after it).

The order (run): round-robin by phase: every craft through a phase before any craft goes on, so the four curves stay comparable at
any stop; every craft resumes from its hub record (the trainer's resume-first contract) and every train() call returns at the phase
boundary (stop_at_boundary).

THE V1.1 CRAFTS (0.1.33; CRAFTS_V11, the same names with a .1): the same four bodies from the same births on the same diet, with one
change to the recipe: the output layer's two readout matrices (head.w_h, head.w_s; their biases stay undecayed) train with AdamW at the
Adam rate (3e-4) and decoupled weight decay 0.1 instead of Muon (TrainConfig.decay_params). Under Muon the readout takes a full-size step
every update with no decay and keeps growing, and the last block's output gathered onto one direction (0.92-0.93 of its energy within
4,960 steps); with the readout on AdamW the same crafts held it at 0.23-0.32 at no cost in bits (lastfix_screen.py, its second stage).
The v1 crafts keep their names and their recipe; ALL holds both sets."""
from __future__ import annotations

import json
import math
import os
import queue
import shutil
import subprocess
import sys
import time

import numpy as np
import torch

from .bodies import craft_cfg
from .frames_data import CAPTION_CTRL_MIX, CAPTION_MIX, CARRY_MIX, CONTROL_MIX, CORPUS_ENV, FRAMES_MIX, MIXED_MIX, NOTE_MIX  # noqa: F401  (importing registers the frames streams, here and in every stream child)
from .names import run_tag

REPO = "AbstractPhil/alephllm-nano-castor-pollux"
CTX, BIRTH_CTX = 4096, 1024
TOKENS_PER_STEP = 262_144
CRAFTS = {
    "nano-pollux-dual-canon-v1": dict(arm="altdc-both", seed=1, code="pdc",
                                      body="Pollux's layout (hubs in blocks 0, 2, 4; rotary softmax in 1, 3, 5) with dual hubs and Canon layers"),
    "nano-pollux-canon-v1": dict(arm="polluxc-both", seed=2, code="pc",
                                 body="Pollux (rotary hubs and rotary softmax alternating) with Canon layers"),
    "nano-pollux-dual-v1": dict(arm="altd-both", seed=1, code="pd", body="Pollux's layout with dual hubs"),
    "nano-castor-v1": dict(arm="castor-both", seed=3, code="c", body="Castor: rotary softmax in every block (the baseline)"),
}
# the bed's long-window gauge settings (the v4 train's anneal block: frame sizes to 1,024, four records a frame, the strict questions read
# on their own rows) with the planted record carried to the far end of the 4,096-byte window; one gauge seed for every craft, so the four
# curves are read on the very same rows
GAUGE_ARGS = dict(ctx="4096", frames="dyn", register="caption", tags="mode", sized="1", sliced="32", sizes="32,128,256,512,1024",
                  frame_records="4", nest="0.3", overlap="0.0", ask="any", plant="0.75", gauge_asks="nested,overlap",
                  dists="64,256,512,896,2048,3584")
GAUGE_SEED = 1
GAUGE_EVERY = 4000              # steps (two full evals); and every phase boundary

READOUT = ("head.w_h.", "head.w_s.")                     # the output layer's two readout matrices (their biases stay undecayed)
V11_TRAIN = dict(decay_params=READOUT, decay_wd=0.1)      # the readout on AdamW at the Adam rate, decoupled decay 0.1, instead of Muon
CRAFTS_V11 = {f"{n}.1": dict(c, train=V11_TRAIN, tag="v11") for n, c in CRAFTS.items()}
# 0.1.35: the frames continuation (before and after the frames, one seed). Each v1.1 craft continues from its last boundary (the exact
# state: fp32 weights and both optimizers) through one appended phase of CONT_TOKENS: framed three rows in four (-fr), or every row
# plain (-ct, the control). The two arms read the same text row for row (frames_data.py), so they differ only by the frames.
CONT_TOKENS = 1_000_000_000
CONT_PHASE = {"fr": ("frames_continuation", FRAMES_MIX), "ct": ("control_continuation", CONTROL_MIX),
              "cf": ("carry_continuation", CARRY_MIX),       # 0.1.37: the carry rows (the self-written frames prototype)
              "nt": ("note_continuation", NOTE_MIX),         # 0.1.41: the note rows (form A, second instance: a fact of the text framed)
              "mx": ("mixed_continuation", MIXED_MIX),       # 0.1.49: the mixed phase (carry rows and frames rows, three registers)
              "cp": ("caption_continuation", CAPTION_MIX),   # 0.1.55: the caption phase (caption windows from the pack and carry rows)
              "cpx": ("caption_control_continuation", CAPTION_CTRL_MIX)}   # 0.1.55: its control (every frame from another document)
# 0.1.49: a continuation phase's bytes when they differ from CONT_TOKENS: the mixed phase runs 1.5 billion so that its carry rows (two
# framed rows in three) total the carry arm's 0.75 billion, the far habit's dose kept, while its frames rows add 0.375 billion.
# 0.1.55: the caption phase runs the mixed phase's 1.5 billion (the same dose on the mixed arm's end: about half the bytes caption windows).
CONT_TOKENS_OF = {"mx": 1_500_000_000, "cp": 1_500_000_000, "cpx": 1_500_000_000}
CRAFTS_CONT = {f"{n}.1-{arm}": dict(c, train=V11_TRAIN, tag=f"v11{arm}", source=f"{n}.1", cont=arm)
               for n, c in CRAFTS.items() for arm in ("fr", "ct", "cf")}
# 0.1.40: the frames-then-carry arm (one seed): the frames craft's end state (fp32 weights and both optimizers at step 65,237 at the 2s
# step) continued through the carry phase, so a body that already reads a frame within a thousand bytes of its question learns to write
# carries. Its plan is its source's whole plan plus the carry phase (69,052 steps); that phase sits one slot later than the -cf arm's, so
# it reads the same mix and dose in a different shuffle (the library offsets a phase's stream seed by its position in the plan).
CRAFTS_CONT2 = {f"{n}.1-fr-cf": dict(c, train=V11_TRAIN, tag="v11frcf", source=f"{n}.1-fr", cont="cf") for n, c in CRAFTS.items()}
# 0.1.41: the note arms (one seed): -nt from the diet's end (the single-phase form), -cf-nt from the carry arm's end state (the body that
# already writes carries learns WHAT to write: the fact the text states); both run the note phase (NOTE_MIX, CONT_TOKENS).
CRAFTS_NOTE = {f"{n}.1-nt": dict(c, train=V11_TRAIN, tag="v11nt", source=f"{n}.1", cont="nt") for n, c in CRAFTS.items()}
CRAFTS_NOTE2 = {f"{n}.1-cf-nt": dict(c, train=V11_TRAIN, tag="v11cfnt", source=f"{n}.1-cf", cont="nt") for n, c in CRAFTS.items()}
# 0.1.49: the mixed-phase arm (one seed): -mx from the diet's end through one phase of carry rows and frames rows across the three
# registers (frames_data.MIXED; plans/2026-10-18_mixed_phase_arm_plan.md): the near read and the far habit in one body, the lookup kept
# in every register.
CRAFTS_MIX = {f"{n}.1-mx": dict(c, train=V11_TRAIN, tag="v11mx", source=f"{n}.1", cont="mx") for n, c in CRAFTS.items()}
# 0.1.52: the second seed of the mixed arm on the best carry body: the same diet-end start, dose and mix at a second data seed (TrainConfig.seed 2
# against the default 1337; the phase stream and its rows draw from it), so the mixed phase's marks get a seed pair on one body.
CRAFTS_MIX2 = {"nano-pollux-dual-v1.1-mx2": dict(CRAFTS["nano-pollux-dual-v1"], train={**V11_TRAIN, "seed": 2}, tag="v11mx2",
                                                source="nano-pollux-dual-v1.1", cont="mx")}
# 0.1.55: the caption arms (plans/2026-10-19_caption_finetune_plan.md): -mx-cp from the mixed arm's end (the dominant form, the release
# repo's resident arm) through one phase of caption windows from the caption pack and carry rows (frames_data.CAPTION); the control
# -mx-cpx on Pollux dual, the same phase with every frame taken from another document (the frame says nothing about the caption).
CRAFTS_CAPTION = {f"{n}.1-mx-cp": dict(c, train=V11_TRAIN, tag="v11mxcp", source=f"{n}.1-mx", cont="cp") for n, c in CRAFTS.items()}
CRAFTS_CAPTION_CTRL = {"nano-pollux-dual-v1.1-mx-cpx": dict(CRAFTS["nano-pollux-dual-v1"], train=V11_TRAIN, tag="v11mxcpx",
                                                           source="nano-pollux-dual-v1.1-mx", cont="cpx")}
ALL = {**CRAFTS, **CRAFTS_V11, **CRAFTS_CONT, **CRAFTS_CONT2, **CRAFTS_NOTE, **CRAFTS_NOTE2, **CRAFTS_MIX, **CRAFTS_MIX2, **CRAFTS_CAPTION,
       **CRAFTS_CAPTION_CTRL}


def train_of(name: str) -> dict:
    """The TrainConfig fields a craft's version sets: none for a v1 craft, the readout's AdamW group for a v1.1 craft."""
    return dict(ALL[name].get("train") or {})


ARM_TEXT = {"fr": "frames", "ct": "control", "cf": "carry", "nt": "notes", "mx": "mixed", "cp": "caption", "cpx": "caption control"}   # 0.1.43: a continuation kind in plain words; 0.1.49 mixed; 0.1.55 caption


def body_of(name: str) -> str:
    """0.1.43: the body a craft is a continuation of: the craft itself for a body, else its source's body (the chain followed to its end)."""
    c = ALL[name]
    return body_of(c["source"]) if c.get("source") else name


def arm_text(name: str) -> str | None:
    """0.1.43: a craft's continuation chain in plain words ('carry'; 'frames, then carry'; 'carry, then notes'); None for a body."""
    parts, n = [], name
    while ALL[n].get("source"):
        parts.append(ARM_TEXT[ALL[n]["cont"]])
        n = ALL[n]["source"]
    return ", then ".join(reversed(parts)) or None


def start_step(name: str, tokens_per_step: int = TOKENS_PER_STEP) -> int:
    """0.1.43: the step a craft's own training started at: its source's planned end (plan_steps), 0 for a body born in the bed."""
    c = ALL[name]
    return plan_steps(c["source"], tokens_per_step) if c.get("source") else 0


def gauge_init(name: str, step: int) -> str:
    """The folder the bed's gauge reads a craft's weights from, also its record's starting-craft piece: v1<code>_<step> for a v1 craft,
    v11<code>_<step> for a v1.1 craft (so the two versions' records never share a name)."""
    c = ALL[name]
    return f"{c.get('tag', 'v1')}{c['code']}_{step}"


# ------------------------------------------------------------------ the start
def birth_state(arm: str, seed: int) -> dict:
    """The bed's birth of `arm` at `seed`, carried to the v1 context: torch.manual_seed(seed) then the craft built at context 1,024 on
    the CPU (bed_train.Craft's sequence), the position table's rows 1,024..4,095 zero (bed_train's continuation to a longer window).
    -> the state dict (fp32, CPU)."""
    from geolip.alephllm.model.alephlm import AlephLM
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)         # the bed seeds both generators; the build itself draws from the CPU's
    model = AlephLM(craft_cfg(arm, BIRTH_CTX))
    sd = {k: v.detach().clone().contiguous() for k, v in model.state_dict().items()}
    pos = sd["embed.pos"]
    new = torch.zeros(pos.shape[0], CTX, pos.shape[2], dtype=pos.dtype)
    new[:, :BIRTH_CTX] = pos
    sd["embed.pos"] = new
    return sd


def write_birth(name: str, folder: str) -> str:
    """The craft's start file <folder>/<name>_birth.safetensors (written once; a present file is checked against a fresh build).
    -> the path."""
    from safetensors.torch import load_file, save_file
    c = ALL[name]
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"{name}_birth.safetensors")
    sd = birth_state(c["arm"], c["seed"])
    if os.path.exists(path):
        old = load_file(path)
        assert set(old) == set(sd) and all(torch.equal(old[k], sd[k]) for k in sd), f"{path} is not the birth of {c['arm']} seed {c['seed']}"
        return path
    save_file(sd, path + ".part", metadata={"arm": c["arm"], "seed": str(c["seed"]), "birth_context": str(BIRTH_CTX), "context": str(CTX)})
    os.replace(path + ".part", path)
    return path


# ------------------------------------------------------------------ the recipe
def diet(scale: float = 1.0) -> list:
    """Beatrix 2s's diet as the library's chronological phases (make_v3_preset at data scale 1). scale < 1 shrinks every phase's
    planned tokens (a smoke run only: the mixes and their streams are untouched)."""
    from geolip.alephllm.presets import make_v3_preset
    phases = [dict(ph) for ph in make_v3_preset(data_scale=1.0).curriculum]
    if scale != 1.0:
        for ph in phases:
            ph["planned_tokens"] = max(1, int(ph["planned_tokens"] * scale))
    return phases


def diet_of(name: str, scale: float = 1.0) -> list:
    """A craft's plan: the 2s diet, and for a continuation craft its source's whole plan (0.1.40: a second-stage craft inherits its
    source's appended phase) plus its own appended phase (CONT_TOKENS, scaled with the diet)."""
    arm = ALL[name].get("cont")
    if not arm:
        return diet(scale)
    phases = diet_of(ALL[name]["source"], scale)
    pname, ds = CONT_PHASE[arm]
    phases.append(dict(name=pname, dataset=ds, planned_tokens=max(1, int(CONT_TOKENS_OF.get(arm, CONT_TOKENS) * scale)), status="planned"))
    return phases


def plan_steps(name: str, tokens_per_step: int, scale: float = 1.0) -> int:
    """0.1.40: the steps of a craft's whole plan (the diet and every appended phase) at this scale and step size: 61,422 for a v1.1
    craft, 65,237 for a continuation craft and 69,052 for a frames-then-carry craft at the 2s step."""
    return sum(math.ceil(ph["planned_tokens"] / tokens_per_step) for ph in diet_of(name, scale))


def diet_steps(scale: float, tokens_per_step: int) -> int:
    """The steps of the 2s diet at this scale and step size (the trainer closes a phase on whole steps)."""
    return sum(math.ceil(ph["planned_tokens"] / tokens_per_step) for ph in diet(scale))


def cont_init(name: str, tokens_per_step: int, scale: float = 1.0, repo: str = REPO, cont_from: str | None = None) -> dict:
    """A continuation craft's start: its source craft's last boundary state (resume/boundary_<last phase>_step<steps>.pt) in the hub
    repo, or under a local runs folder (cont_from: <cont_from>/<source>/resume/...; a toy run), at the step the source ended."""
    c = ALL[name]
    steps = plan_steps(c["source"], tokens_per_step, scale)                  # 0.1.40: the source's whole plan (its own phase included)
    rel = f"resume/boundary_{diet_of(c['source'], scale)[-1]['name']}_step{steps}.pt"
    spec = dict(step=steps, tokens_per_step=tokens_per_step, source=c["source"])
    if cont_from:
        return dict(spec, payload=os.path.join(cont_from, c["source"], *rel.split("/")))
    return dict(spec, payload=rel, repo=repo, prefix=c["source"])


def preset(name: str, birth_file: str, micro_batch: int = 16, grad_accum: int = 4, repo: str = REPO, diet_scale: float = 1.0,
           cont_from: str | None = None, **train_over):
    """The library preset of a v1 craft: the body at context 4,096 named for the craft (its prefix in the repo), the 2s recipe, the 2s
    diet, the weights-only start from the birth file at step 0. The craft's version sets its recipe fields (train_of: a v1.1 craft's
    readout on AdamW); train_over: TrainConfig fields to override (a smoke run's cadences)."""
    from geolip.alephllm.presets import Preset, TrainConfig
    cfg = craft_cfg(ALL[name]["arm"], CTX)
    cfg.name = name
    if diet_scale == 1.0:
        assert micro_batch * grad_accum * CTX == TOKENS_PER_STEP, f"the 2s step is {TOKENS_PER_STEP:,} tokens"
    tc = TrainConfig(micro_batch=micro_batch, grad_accum=grad_accum, governor="", head_addr_frozen=False, phase_seed_offset=True,
                     **{**train_of(name), **train_over})
    init = (cont_init(name, micro_batch * grad_accum * CTX, diet_scale, repo, cont_from) if ALL[name].get("cont")
            else {"file": birth_file, "step": 0})
    return Preset(model=cfg, train=tc, hf_repo=repo, curriculum=diet_of(name, diet_scale), data_scale=1.0, init_from=init)


def bench(name: str, birth_file: str, micro_batch: int, grad_accum: int, steps: int = 15, warm: int = 5, device: str = "cuda") -> dict:
    """The 2s step of a v1 craft on this card without the data plane: the library trainer's step written out (bf16 autocast, the loss
    per micro-batch read back, the clip on the whole craft, Muon + pure Adam at the 2s rates) on random bytes at the given shape, from the
    craft's birth. A shape that does not fit the card reads tok_s 0 and peak_gb inf. -> dict(step_s, tok_s, peak_gb)."""
    try:
        return _bench(name, birth_file, micro_batch, grad_accum, steps, warm, device)
    except torch.OutOfMemoryError:
        torch.cuda.empty_cache()
        return dict(name=name, micro_batch=micro_batch, grad_accum=grad_accum, step_s=float("inf"), tok_s=0.0, peak_gb=float("inf"))


def _bench(name, birth_file, micro_batch, grad_accum, steps, warm, device) -> dict:
    from safetensors.torch import load_file
    from geolip.alephllm.model.alephlm import AlephLM
    from geolip.alephllm.train.optim import build_optimizers
    from geolip.alephllm.train.precision import autocast
    model = AlephLM(craft_cfg(ALL[name]["arm"], CTX)).to(device)
    model.load_state_dict({k: v.to(device) for k, v in load_file(birth_file).items()}, strict=True)
    model.train()
    tr = train_of(name)
    opts = build_optimizers(model, 2e-2, 0.95, 3e-4, decay_prefixes=tuple(tr.get("decay_params", ())), decay_wd=tr.get("decay_wd", 0.0))
    params = list(model.parameters())
    xb = torch.randint(0, 256, (micro_batch, CTX + 1), generator=torch.Generator().manual_seed(0)).to(device)
    torch.cuda.reset_peak_memory_stats()
    t0 = None
    for i in range(warm + steps):
        if i == warm:
            torch.cuda.synchronize()
            t0 = time.time()
        for o in opts:
            o.zero_grad(set_to_none=True)
        for _ in range(grad_accum):
            with autocast(device, "bf16"):
                _, loss = model(xb[:, :-1], targets=xb[:, 1:])
            (loss / grad_accum).backward()
            float(loss.item())
        gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
        assert bool(torch.isfinite(gn))
        for o in opts:
            o.step()
    torch.cuda.synchronize()
    dt = (time.time() - t0) / steps
    out = dict(name=name, micro_batch=micro_batch, grad_accum=grad_accum, step_s=dt, tok_s=micro_batch * grad_accum * CTX / dt,
               peak_gb=torch.cuda.max_memory_allocated() / 2 ** 30)
    del model, opts, params, xb
    torch.cuda.empty_cache()
    return out


def pick_shape(results: list, card_gb: float, prefer=(16, 4), gain: float = 1.10, room: float = 0.8):
    """The micro-batch shape for every craft from the bench: the 2s shape (16 x 4) unless another shape that fits (peak under `room` of
    the card for every craft) is faster by `gain` or more on the mean of the crafts' rates. -> ((micro_batch, grad_accum), the reason)."""
    shapes = sorted({(r["micro_batch"], r["grad_accum"]) for r in results})
    fits = {s: all(r["peak_gb"] < room * card_gb for r in results if (r["micro_batch"], r["grad_accum"]) == s) for s in shapes}
    rate = {s: sum(r["tok_s"] for r in results if (r["micro_batch"], r["grad_accum"]) == s)
            / max(1, sum(1 for r in results if (r["micro_batch"], r["grad_accum"]) == s)) for s in shapes}
    ok = [s for s in shapes if fits[s]]
    assert ok, "no shape fits the card"
    best = max(ok, key=lambda s: rate[s])
    if prefer in ok and rate[best] < gain * rate[prefer]:
        return prefer, f"the 2s shape {prefer[0]} x {prefer[1]} (the fastest, {best[0]} x {best[1]}, is within {gain - 1:.0%})"
    return best, (f"{best[0]} x {best[1]}: {rate[best] / 1e3:.0f}k tokens a second on the crafts' mean"
                  + (f" against {rate[prefer] / 1e3:.0f}k for the 2s shape" if prefer in rate else ""))


# ------------------------------------------------------------------ the data
def _stream_worker(spec: dict, q, stop) -> None:
    """The child of a ProcessStream: build the library stream from its recipe, take the starting state, then send (batch, the state
    after it) ahead through the bounded queue until told to stop. A failure goes to the parent as text (it raises there)."""
    try:
        import copy
        from geolip.alephllm.data import curriculum  # noqa: F401  (registers the curriculum's corpora and mixes)
        from geolip.alephllm.data.streams import build_stream
        from geolip.alephllm.data.tokenizers import build_tokenizer
        tok = build_tokenizer(spec["tokenizer"])
        s = build_stream(spec["dataset"], tok, spec["context"], spec["micro_batch"], seed=spec["seed"], shard=spec.get("shard"))
        if spec.get("state") is not None:
            s.load_state_dict(spec["state"])
        byte = int(getattr(tok, "vocab_size", 256) or 256) <= 256
        while not stop.is_set():
            b = s.next_batch().numpy()
            if byte:
                assert int(b.min()) >= 0 and int(b.max()) < 256
                b = b.astype(np.uint8)
            item = ("batch", b, copy.deepcopy(s.state_dict()), int(s.rows_consumed), int(s.epoch))
            while not stop.is_set():
                try:
                    q.put(item, timeout=0.5)
                    break
                except queue.Full:
                    continue
    except BaseException as e:  # noqa: BLE001  (the parent decides)
        import traceback
        try:
            q.put(("error", f"{type(e).__name__}: {e}\n{traceback.format_exc()}"), timeout=30)
        except Exception:  # noqa: BLE001
            pass


class ProcessStream:
    """A library training stream composed in a child process (spawned), a few batches ahead of the card. The library's streams compose
    their rows in the calling thread, between the card's steps; a 9M craft's step is short enough that the composition shows (and a
    thread would fight the trainer for the interpreter lock). Exact: the child builds the stream the trainer would build (the same
    dataset, tokenizer, context, micro-batch, seed and starting state), the batches come in its order, and state_dict() is the stream's
    state right after the last batch handed out, so a checkpoint resumes where the trainer stands, not where the child ran ahead."""

    def __init__(self, dataset: str, tokenizer: str, context: int, micro_batch: int, seed: int, state=None, shard=None,
                 depth: int = 6, timeout: float = 1800.0):
        import multiprocessing as mp
        self.dataset, self.timeout = dataset, timeout
        ctx = mp.get_context("spawn")
        self._q, self._stop = ctx.Queue(maxsize=depth), ctx.Event()
        spec = dict(dataset=dataset, tokenizer=tokenizer, context=context, micro_batch=micro_batch, seed=seed, state=state, shard=shard)
        self._p = ctx.Process(target=_stream_worker, args=(spec, self._q, self._stop), daemon=True, name=f"stream-{dataset}")
        self._p.start()
        self._state, self._rows, self._epoch = state, int((state or {}).get("rows_consumed", 0) or 0), int((state or {}).get("epoch", 0) or 0)

    def next_batch(self) -> torch.Tensor:
        t0 = time.time()
        while True:
            try:
                item = self._q.get(timeout=5.0)
                break
            except queue.Empty:
                if not self._p.is_alive():
                    raise RuntimeError(f"the stream process for '{self.dataset}' died (exit code {self._p.exitcode})")
                if time.time() - t0 > self.timeout:
                    raise RuntimeError(f"the stream process for '{self.dataset}' sent nothing in {self.timeout:.0f} s")
        if item[0] == "error":
            raise RuntimeError(f"the stream process for '{self.dataset}' failed: {item[1]}")
        _, b, self._state, self._rows, self._epoch = item
        return torch.from_numpy(b.astype(np.int64))

    def state_dict(self) -> dict:
        return self._state

    def load_state_dict(self, st):
        raise RuntimeError("a ProcessStream takes its starting state at construction")

    @property
    def rows_consumed(self) -> int:
        return self._rows

    @property
    def epoch(self) -> int:
        return self._epoch

    def close(self):
        self._stop.set()
        try:
            while True:
                self._q.get_nowait()
        except Exception:  # noqa: BLE001
            pass
        self._p.join(timeout=10)
        if self._p.is_alive():
            self._p.terminate()
            self._p.join(timeout=5)


def make_trainer(preset_obj, hf_token: str | None, out_dir: str, process_streams: bool = True, after_eval=None, device=None):
    """The library's Trainer for a v1 craft with two additions: the training streams composed in child processes (process_streams), and
    a hook after every full eval (after_eval(trainer): the interval gauges)."""
    from geolip.alephllm.train.trainer import Trainer

    class V1Trainer(Trainer):
        def _init_from_checkpoint(self, spec):
            if "payload" not in spec:
                return super()._init_from_checkpoint(spec)
            # 0.1.35: an exact continuation: the source run's boundary state (fp32 weights and both optimizers) placed at its step;
            # the library's weights-only start does the placement (the fresh manifest, the cursor), then the optimizers are restored
            from safetensors.torch import save_file
            path = spec["payload"] if "repo" not in spec else self.hub.fetch(spec["repo"], f"{spec['prefix']}/{spec['payload']}")
            payload = torch.load(path, map_location="cpu", weights_only=False)
            assert int(payload["step"]) == int(spec["step"]), (f"the source state is at step {payload['step']:,}, "
                                                                f"not {spec['step']:,}")
            src = (payload.get("manifest") or {}).get("phases") or []
            assert src and all(ph.get("status") == "done" for ph in src), "the source run has not finished its diet"
            os.makedirs(out_dir, exist_ok=True)
            tmp = os.path.join(out_dir, f"{self.cfg.name}_cont_start.safetensors")
            save_file({k: v.detach().float().contiguous() for k, v in payload["model"].items()}, tmp)
            super()._init_from_checkpoint(dict(file=tmp, step=int(spec["step"]), tokens_per_step=int(spec["tokens_per_step"])))
            os.remove(tmp)
            for opt, st in zip(self.optimizers, payload["optimizers"]):
                opt.load_state_dict(st)
            where = path if "repo" not in spec else f"{spec['repo']}/{spec['prefix']}/{spec['payload']}"
            self.manifest.init_from.update(source=where, weights="the fp32 masters of the source's boundary state",
                                           optimizers="carried from the source's boundary state (Muon momentum, the Adam/AdamW moments)")
            self.manifest.note(f"CONTINUATION at step {self.step:,} from {where}: the exact training state (fp32 weights, both "
                               f"optimizers); the appended phase '{self.manifest.phases[-1]['name']}' opens here")
            print(f"[v1] {self.cfg.name}: continued from {where} at step {self.step:,} (weights and both optimizers)", flush=True)

        def _open_stream(self):
            if not process_streams:
                return super()._open_stream()
            assert self.world == 1, "the v1 runs are single-card"
            ph = self.manifest.advance_phases()
            if ph is None:
                return None
            if self.stream is None or self.stream.dataset != ph["dataset"]:      # the library's _open_stream, the stream in a child
                st = getattr(self, "_stream_state", None)
                state = st["state"] if st and st.get("dataset") == ph["dataset"] else None
                self._stream_state = None
                close_stream(self)
                self.stream = ProcessStream(ph["dataset"], self.cfg.tokenizer, self.cfg.context, self.tc.micro_batch,
                                            seed=self._phase_seed(ph), state=state)
            return ph

        def _full_eval(self):
            out = super()._full_eval()
            if after_eval is not None:
                after_eval(self)
            return out

    return V1Trainer(preset_obj, hf_token=hf_token, out_dir=out_dir, device=device)


def close_stream(trainer) -> None:
    """Stop a trainer's stream child (a finished phase's stream holds its corpora open) and drop it."""
    s = getattr(trainer, "stream", None)
    if s is not None and hasattr(s, "close"):
        s.close()
    trainer.stream = None


# ------------------------------------------------------------------ the bed's gauges
def gauge_scalars(rec: dict) -> dict:
    """A gauge record's TensorBoard scalars: bed/text (held-out bits, by byte class, the bank off), bed/window (bits by position in the
    window, the planted record by distance), bed/scaling (the keyed record by frame size), bed/frame (the copy pooled over the frames, the
    nested and the redefined-key questions, the copy with the bank off), bed/canary."""
    from .bed_frames import pooled_fetch
    out = {}

    def put(k, v):
        if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v):
            out[k] = float(v)
    put("bed/text/heldout_bits", rec.get("heldout_bpb"))
    put("bed/text/bank_off_bits", rec.get("dispatch_off_bpb"))
    for c, v in (rec.get("heldout_bpb_by_class") or {}).items():
        put(f"bed/text/class_{c}", v)
    for band, v in ((rec.get("heldout_bpb_by_position") or {}).get("row") or {}).items():
        put(f"bed/window/position_{band}", v)
    for d, f in (rec.get("fetch") or {}).items():
        put(f"bed/window/planted_{d}_bits", f.get("value_bits"))
        put(f"bed/window/planted_{d}_first", f.get("first_byte"))
        put(f"bed/window/planted_{d}_all5", f.get("all_five"))
    fb = rec.get("fetch_by_frame") or {}
    for k, g in fb.items():
        if k.startswith("data/") and k.endswith("/depth0"):
            size = k.split("/")[1]
            put(f"bed/scaling/frame_{size}_first", g.get("first_byte"))
            put(f"bed/scaling/frame_{size}_whole", g.get("full_value"))
            put(f"bed/scaling/frame_{size}_bits", g.get("value_bits"))
    for key, groups in (("copy", fb), ("copy_bank_off", rec.get("fetch_by_frame_bank_off")),
                        ("nested", (rec.get("fetch_by_frame_asks") or {}).get("nested")),
                        ("redefined_key", (rec.get("fetch_by_frame_asks") or {}).get("overlap"))):
        p = pooled_fetch(groups)
        if p:
            put(f"bed/frame/{key}_whole", p["full_value"])
            put(f"bed/frame/{key}_first", p["first_byte"])
            put(f"bed/frame/{key}_bits", p["value_bits"])
    for kind, c in (rec.get("canary") or {}).items():
        if isinstance(c, dict):
            put(f"bed/canary/{kind}_bits", c.get("bits"))
    for k, v in (rec.get("carry") or {}).items():          # 0.1.37: reading with notes (bed_train.carry_gauge)
        if isinstance(v, (int, float)):                      # 0.1.39: the block also names its dump files
            put(f"bed/carry/{k}", v)
    for k, v in (rec.get("carry_plain") or {}).items():    # 0.1.39: the plain-row read (no question: its counts and the open probability)
        if isinstance(v, (int, float)):
            put(f"bed/carry_plain/{k}", v)
    for blk in ("note_self", "note_planted"):              # 0.1.41: the note rows (bed_train.carry_gauge(note=...)), with the whole value by distance
        b = rec.get(blk) or {}
        for k, v in b.items():
            if isinstance(v, (int, float)):
                put(f"bed/{blk}/{k}", v)
        for bin_, d in (b.get("by_dist") or {}).items():
            for k in ("full_value", "first_byte", "value_bits", "n"):
                put(f"bed/{blk}/{bin_}_{k}", (d or {}).get(k))
    put("bed/gauge_seconds", rec.get("wall_s"))
    return out


def gauge_args(name: str, init: str, out: str) -> dict:
    c = ALL[name]
    args = dict(GAUGE_ARGS, out=out, arm=c["arm"], seed=str(GAUGE_SEED), init=init, gauge_only="1")
    if c.get("cont") in ("cf", "nt", "mx", "cp", "cpx"):
        args["carry_gauge"] = "1"                        # 0.1.37: the carry arm's reads add reading with notes (its record's tag ends _carry_gauge); 0.1.49 the mixed arm; 0.1.55 the caption arms
    if c.get("cont") == "nt":
        args["note_gauge"] = "1"                         # 0.1.41: the note arm's reads add the note rows, given and self-written (the tag ends _carry_note_gauge)
    return args


def bed_gauge(trainer, name: str, out: str, gauge_cmd=None, gauge_gb: float = 10.0, keep_weights: bool = False) -> dict | None:
    """The bed's gauges on the craft's weights now: the weights to <out>/bed/v1<code>_<step> (the bed's folder form: config.json +
    model.safetensors), bed_train --gauge_only in a child process (gauge_cmd: the command prefix; default this interpreter with
    -m geolip.nano_beatrix.bed_train; its card budget gauge_gb, the frame gauges at 4,096 x 16 rows need about 5 GB), the record's
    scalars into the trainer's TensorBoard at its step, the record to the craft's prefix on the hub (bed_gauges/). A failed read is
    logged and skipped: a diagnostic never stops the train (it runs inside the trainer's eval). -> the record or None."""
    try:
        return _bed_gauge(trainer, name, out, gauge_cmd, gauge_gb)
    except Exception as e:  # noqa: BLE001
        print(f"[v1] {name}: the bed gauge at step {trainer.step:,} raised {type(e).__name__}: {e}; skipped", flush=True)
        return None
    finally:
        if not keep_weights:                              # the craft's copy for the child goes, read or not
            shutil.rmtree(os.path.join(out, "bed", gauge_init(name, trainer.step)), ignore_errors=True)


def _bed_gauge(trainer, name, out, gauge_cmd, gauge_gb):
    total = torch.cuda.get_device_properties(0).total_memory / 2 ** 30 if torch.cuda.is_available() else gauge_gb
    gpu_frac = min(0.6, max(0.05, gauge_gb / total))
    from safetensors.torch import save_file
    c = ALL[name]
    init = gauge_init(name, trainer.step)
    bed = os.path.join(out, "bed")
    wdir = os.path.join(bed, init)
    args = gauge_args(name, init, out)
    rec_path = os.path.join(bed, f"bed_{run_tag(args)}.json")
    t0 = time.time()
    if not os.path.exists(rec_path):
        os.makedirs(wdir, exist_ok=True)
        save_file({k: v.detach().float().cpu().contiguous() for k, v in trainer.raw_model.state_dict().items()},
                  os.path.join(wdir, "model.safetensors"))
        json.dump(craft_cfg(c["arm"], CTX).to_dict(), open(os.path.join(wdir, "config.json"), "w", encoding="utf-8"), indent=1)
        cmd = list(gauge_cmd or [sys.executable, "-X", "utf8", "-m", "geolip.nano_beatrix.bed_train"]) + [f"--{k}={v}" for k, v in args.items()]
        env = dict(os.environ, BATTERY_GPU_FRAC=str(gpu_frac), PYTHONIOENCODING="utf-8")
        env.pop("HF_TOKEN", None)                         # the gauge reads local files only
        os.makedirs(os.path.join(bed, "gauge_logs"), exist_ok=True)
        log_path = os.path.join(bed, "gauge_logs", f"{init}.log")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        with open(log_path, "w", encoding="utf-8") as lf:
            rc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env, timeout=3600).returncode
        if rc != 0 or not os.path.exists(rec_path):
            tail = open(log_path, encoding="utf-8", errors="replace").read()[-1500:]
            print(f"[v1] {name}: the bed gauge at step {trainer.step:,} failed (exit {rc}); its log ends:\n{tail}", flush=True)
            return None
    rec = json.load(open(rec_path, encoding="utf-8"))
    sc = gauge_scalars(rec)
    for k, v in sc.items():
        trainer.writer.add_scalar(k, v, trainer.step)
    trainer.writer.flush()
    trainer.hub._up(rec_path, f"bed_gauges/{os.path.basename(rec_path)}")
    from .bed_frames import pooled_fetch
    p = pooled_fetch(rec.get("fetch_by_frame")) or {}
    print(f"[v1] {name} step {trainer.step:,}: bed gauges in {time.time() - t0:.0f} s: held-out {rec.get('heldout_bpb', float('nan')):.4f} "
          f"bits a byte; the copy (whole value, pooled over the frames) {p.get('full_value', float('nan')):.2f}; "
          f"{len(sc)} scalars to TensorBoard", flush=True)
    return rec


def gauge_stored(name: str, step: int, out: str, repo: str = REPO, carry: bool = True, gauge_cmd=None, gauge_gb: float = 10.0,
                 token=None, extra: dict | None = None) -> dict:
    """0.1.37: the bed's gauge read of a STORED craft (its checkpoint at `step` in the hub repo, the bf16 file; the public repos read
    tokenless), with the carry gauge (reading with notes) when `carry`: the weights to <out>/bed/<init> in the bed's folder form, the
    child command as bed_gauge's, the record read back. A record already on disk is returned as it is. -> the record."""
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file, save_file
    c = ALL[name]
    init = gauge_init(name, step)
    bed = os.path.join(out, "bed")
    wdir = os.path.join(bed, init)
    args = gauge_args(name, init, out)
    if carry:
        args["carry_gauge"] = "1"
    if extra:                                            # 0.1.39: the child's extra arguments (carry_rows, carry_dump, carry_plain, ...)
        args.update({k: str(v) for k, v in extra.items()})
    rec_path = os.path.join(bed, f"bed_{run_tag(args)}.json")
    if os.path.exists(rec_path):
        return json.load(open(rec_path, encoding="utf-8"))
    os.makedirs(wdir, exist_ok=True)
    if not os.path.exists(os.path.join(wdir, "model.safetensors")):
        ck = hf_hub_download(repo, f"{name}/checkpoints/step_{step:08d}.safetensors", token=token if token else False)
        save_file({k: v.float().contiguous() for k, v in load_file(ck).items()}, os.path.join(wdir, "model.safetensors"))
        json.dump(craft_cfg(c["arm"], CTX).to_dict(), open(os.path.join(wdir, "config.json"), "w", encoding="utf-8"), indent=1)
    total = torch.cuda.get_device_properties(0).total_memory / 2 ** 30 if torch.cuda.is_available() else gauge_gb
    cmd = list(gauge_cmd or [sys.executable, "-X", "utf8", "-m", "geolip.nano_beatrix.bed_train"]) + [f"--{k}={v}" for k, v in args.items()]
    env = dict(os.environ, BATTERY_GPU_FRAC=str(min(0.6, max(0.05, gauge_gb / total))), PYTHONIOENCODING="utf-8")
    env.pop("HF_TOKEN", None)
    os.makedirs(os.path.join(bed, "gauge_logs"), exist_ok=True)
    log_path = os.path.join(bed, "gauge_logs", f"{init}{'_carry' if carry else ''}.log")
    with open(log_path, "w", encoding="utf-8") as lf:
        rc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env, timeout=7200).returncode
    assert rc == 0 and os.path.exists(rec_path), f"the gauge read of {name} at {step:,} failed (exit {rc}); its log: {log_path}"
    return json.load(open(rec_path, encoding="utf-8"))


def caption_stored(name: str, step: int, out: str, repo: str = REPO, token=None, rows: int = 256, gauge_gb: float = 10.0,
                   pack: str | None = None, pack_rows: int | None = None, caption_cmd=None) -> dict:
    """0.1.55: THE CAPTION READS on a stored craft (caption_screen.py as a child; the weights pulled to <out>/bed/<init> as gauge_stored
    pulls them; the held split of the caption pack at `pack` or <out>/caption_pack, built when absent). A record already on disk is
    returned as it is. -> the record (caption_<init>.json)."""
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file, save_file
    from . import caption_pack as cp
    c = ALL[name]
    init = gauge_init(name, step)
    bed = os.path.join(out, "bed")
    wdir = os.path.join(bed, init)
    rec_path = os.path.join(bed, f"caption_{init}.json")
    if os.path.exists(rec_path):
        return json.load(open(rec_path, encoding="utf-8"))
    folder = pack or os.path.join(out, "caption_pack")
    cp.ensure(folder, token, pack_rows)
    os.makedirs(wdir, exist_ok=True)
    if not os.path.exists(os.path.join(wdir, "model.safetensors")):
        ck = hf_hub_download(repo, f"{name}/checkpoints/step_{step:08d}.safetensors", token=token if token else False)
        save_file({k: v.float().contiguous() for k, v in load_file(ck).items()}, os.path.join(wdir, "model.safetensors"))
        json.dump(craft_cfg(c["arm"], CTX).to_dict(), open(os.path.join(wdir, "config.json"), "w", encoding="utf-8"), indent=1)
    args = dict(out=out, init=init, rows=str(rows), ctx=str(CTX), seed=str(GAUGE_SEED), pack=folder)
    total = torch.cuda.get_device_properties(0).total_memory / 2 ** 30 if torch.cuda.is_available() else gauge_gb
    cmd = list(caption_cmd or [sys.executable, "-X", "utf8", "-m", "geolip.nano_beatrix.caption_screen"]) + [f"--{k}={v}" for k, v in args.items()]
    env = dict(os.environ, BATTERY_GPU_FRAC=str(min(0.6, max(0.05, gauge_gb / total))), PYTHONIOENCODING="utf-8")
    env.pop("HF_TOKEN", None)
    os.makedirs(os.path.join(bed, "gauge_logs"), exist_ok=True)
    log_path = os.path.join(bed, "gauge_logs", f"{init}_caption.log")
    with open(log_path, "w", encoding="utf-8") as lf:
        rc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env, timeout=7200).returncode
    assert rc == 0 and os.path.exists(rec_path), f"the caption read of {name} at {step:,} failed (exit {rc}); its log: {log_path}"
    return json.load(open(rec_path, encoding="utf-8"))


def gate_stored(name: str, step: int, out: str, repo: str = REPO, mode: str = "self", token=None, rows: int = 192, gauge_gb: float = 10.0,
                gate_cmd=None, extra: dict | None = None) -> dict:
    """0.1.51: THE GATE SCREEN on a STORED craft (gate_screen.py as a child, the weights pulled to <out>/bed/<init> as gauge_stored pulls
    them): the body proposes notes on the grid protocol, every note scored with and without it and with its value withheld over the
    bytes that follow; mode self (the note gauge's fact rows) or plain (the carry gauge's plain rows). A record already on disk is
    returned as it is; the held-out head is written to <out>/bed when absent (0.1.54). -> the record (gate_<init>_<mode>.json)."""
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file, save_file
    c = ALL[name]
    init = gauge_init(name, step)
    bed = os.path.join(out, "bed")
    wdir = os.path.join(bed, init)
    rec_path = os.path.join(bed, f"gate_{init}_{mode}.json")
    if os.path.exists(rec_path):
        return json.load(open(rec_path, encoding="utf-8"))
    os.makedirs(wdir, exist_ok=True)
    from .bed_corpus import write_heldout
    write_heldout(bed)                                   # 0.1.54: the child reads the held-out head from <out>/bed; a fresh folder gets it here
    if not os.path.exists(os.path.join(wdir, "model.safetensors")):
        ck = hf_hub_download(repo, f"{name}/checkpoints/step_{step:08d}.safetensors", token=token if token else False)
        save_file({k: v.float().contiguous() for k, v in load_file(ck).items()}, os.path.join(wdir, "model.safetensors"))
        json.dump(craft_cfg(c["arm"], CTX).to_dict(), open(os.path.join(wdir, "config.json"), "w", encoding="utf-8"), indent=1)
    args = dict(out=out, init=init, mode=mode, rows=str(rows), ctx=str(CTX), seed=str(GAUGE_SEED))
    if extra:
        args.update({k: str(v) for k, v in extra.items()})
    total = torch.cuda.get_device_properties(0).total_memory / 2 ** 30 if torch.cuda.is_available() else gauge_gb
    cmd = list(gate_cmd or [sys.executable, "-X", "utf8", "-m", "geolip.nano_beatrix.gate_screen"]) + [f"--{k}={v}" for k, v in args.items()]
    env = dict(os.environ, BATTERY_GPU_FRAC=str(min(0.6, max(0.05, gauge_gb / total))), PYTHONIOENCODING="utf-8")
    env.pop("HF_TOKEN", None)
    os.makedirs(os.path.join(bed, "gauge_logs"), exist_ok=True)
    log_path = os.path.join(bed, "gauge_logs", f"{init}_gate_{mode}.log")
    with open(log_path, "w", encoding="utf-8") as lf:
        rc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env, timeout=7200).returncode
    assert rc == 0 and os.path.exists(rec_path), f"the gate screen of {name} at {step:,} ({mode}) failed (exit {rc}); its log: {log_path}"
    return json.load(open(rec_path, encoding="utf-8"))


# ------------------------------------------------------------------ the run
def phase_index(trainer) -> int | None:
    """The index of the craft's current phase (active, else the first planned), None when its diet is done; reads only."""
    phs = trainer.manifest.phases
    for status in ("active", "planned"):
        for i, ph in enumerate(phs):
            if ph["status"] == status:
                return i
    return None


def unrecorded_boundary(trainer, rows: list) -> tuple:
    """The craft's last finished phase when the boundary table has no row for it (a session that ended inside the boundary's handling:
    after the trainer's boundary, before the driver's eval, gauge and row). -> (phase name, exact): exact when the craft still stands at
    that boundary (the next phase has trained nothing, or the diet is done), so the row can be written from these very weights;
    (None, False) when every finished phase has its row; reads only."""
    phs = trainer.manifest.phases
    done = [i for i, ph in enumerate(phs) if ph["status"] == "done"]
    if not done:
        return None, False
    i = done[-1]
    name = phs[i]["name"]
    if any(r.get("phase") == name for r in rows):
        return None, False
    nxt = phs[i + 1] if i + 1 < len(phs) else None
    return name, nxt is None or not int(nxt.get("tokens_done") or 0)


def run(out: str, hf_token: str | None = None, names=None, micro_batch: int = 16, grad_accum: int = 4, repo: str = REPO,
        gauge_every: int = GAUGE_EVERY, gauge_cmd=None, max_hours: float | None = None, process_streams: bool = True,
        diet_scale: float = 1.0, device=None, gauge_gb: float = 10.0, cont_from: str | None = None, frames_corpus: str | None = None,
        frames_corpus_bytes: float = 1.0e9, caption_pack: str | None = None, caption_pack_rows: int | None = None, **train_over) -> dict:
    """Train the v1 crafts through the diet, round-robin by phase. out: the local folder (births, runs, the bed's gauge folder);
    hf_token: None = local only (no hub record, no uploads); max_hours: this session's wall budget (a stop between or inside phases is
    a pause: the next call resumes from the hub); gauge_gb: the bed gauge child's card budget. -> {craft: [one entry a boundary: phase,
    step, tokens, val bits a byte, the gauge's headline reads]} (also written to <out>/v1_boundaries.json)."""
    from .bed_corpus import write_heldout
    names = list(names or CRAFTS)
    t_end = None if max_hours is None else time.time() + max_hours * 3600
    births = {n: (None if ALL[n].get("cont") else write_birth(n, os.path.join(out, "births"))) for n in names}
    write_heldout(os.path.join(out, "bed"))
    if any(ALL[n].get("cont") for n in names):          # 0.1.35: the frames corpus, reached by the stream children
        from .bed_corpus import build as build_corpus
        folder = frames_corpus or os.path.join(out, "frames_corpus")
        if not os.path.exists(os.path.join(folder, "bed_corpus.bin")):
            build_corpus(folder, int(frames_corpus_bytes))
        os.environ[CORPUS_ENV] = folder
    if any(ALL[n].get("cont") in ("cp", "cpx") for n in names):     # 0.1.55: the caption pack, reached by the stream children
        from . import caption_pack as cp
        pack_folder = caption_pack or os.path.join(out, "caption_pack")
        cp.ensure(pack_folder, hf_token, caption_pack_rows, log=lambda *a: print("[v1]", *a, flush=True))
        os.environ[cp.PACK_ENV] = pack_folder
    hist_p = os.path.join(out, "v1_boundaries.json")
    hist = json.load(open(hist_p, encoding="utf-8")) if os.path.exists(hist_p) else {n: [] for n in names}
    gauged = {n: set() for n in names}

    def hook_for(n):
        def hook(t):
            if gauge_every and t.step % gauge_every == 0 and t.step not in gauged[n]:
                gauged[n].add(t.step)
                bed_gauge(t, n, out, gauge_cmd, gauge_gb)
        return hook

    def record(n, t, done, eval_now):
        """A boundary's eval, the bed's gauges, its row in the table (local and the craft's hub record) and the TensorBoard upload."""
        if eval_now:
            t._full_eval()                                # the boundary's own eval (the hook gauges at its cadence)
        if t.step not in gauged[n]:
            gauged[n].add(t.step)
            rec = bed_gauge(t, n, out, gauge_cmd, gauge_gb)
        else:
            rec = None
        ledger = (getattr(t, "_last_eval", {}) or {}).get("ledger") or {}
        hist.setdefault(n, []).append(dict(phase=done, step=t.step, tokens=t.manifest.tokens_seen, val_bpb=ledger.get("bpb_full"),
                                           bed=None if rec is None else {k: v for k, v in gauge_scalars(rec).items()
                                                                         if k in HEADLINE}))
        json.dump(hist, open(hist_p, "w", encoding="utf-8"), indent=1)
        t.hub.upload_bytes(json.dumps(hist[n], indent=1).encode(), "v1_boundaries.json")
        t.writer.flush()
        t.hub.upload_tensorboard(t.tb_dir)
        print(f"[v1] {n}: '{done}' done at step {t.step:,} ({t.manifest.tokens_seen / 1e9:.3f}B tokens); val "
              f"{ledger.get('bpb_full', float('nan')):.4f} bits a byte", flush=True)

    trainers, failed = {}, {}
    for n in names:
        p = preset(n, births[n], micro_batch, grad_accum, repo, diet_scale, cont_from, **train_over)
        t = make_trainer(p, hf_token, os.path.join(out, "runs"), process_streams, hook_for(n), device)
        if births[n] and t.step == 0 and t.manifest.init_from and not any("BIRTH:" in str(x) for x in (t.manifest.notes or [])):
            c = ALL[n]
            t.manifest.note(f"BIRTH: the nano bed's birth of {c['arm']} seed {c['seed']} (torch.manual_seed then the CPU build at context "
                            f"{BIRTH_CTX}, the position rows {BIRTH_CTX}..{CTX - 1} zero): the start of its strongest v4 bed run; {c['body']}")
        if not hist.get(n):                               # a fresh machine: the boundary table so far from the craft's hub record
            prev = t.hub.download("v1_boundaries.json") if t.hub.api else None
            hist[n] = json.load(open(prev, encoding="utf-8")) if prev else []
        trainers[n] = t
        print(f"[v1] {n}: {c_arm(n)} seed {ALL[n]['seed']}, {t.raw_model.param_count():,} parameters, step {t.step:,}, "
              f"phase {phase_index(t)}", flush=True)
        lost, exact = unrecorded_boundary(t, hist.get(n) or [])
        if lost and exact:                                # the last session ended inside this boundary's handling: write it now
            print(f"[v1] {n}: the boundary '{lost}' has no row (the last session ended inside its handling); written now at step "
                  f"{t.step:,}", flush=True)
            record(n, t, lost, eval_now=True)
        elif lost:
            print(f"[v1] {n}: the boundary '{lost}' has no row and the craft has trained past it; its reads are lost", flush=True)

    stopped = None
    while stopped is None:
        live = {n: phase_index(t) for n, t in trainers.items() if n not in failed}
        live = {n: i for n, i in live.items() if i is not None}
        if not live:
            break
        lo = min(live.values())
        for n in [n for n in names if live.get(n) == lo]:
            t = trainers[n]
            left = None if t_end is None else (t_end - time.time()) / 3600
            if left is not None and left <= 0:
                stopped = "the session's wall budget"
                break
            try:
                t.train(stop_at_boundary=True, max_hours=left)
            except FloatingPointError as e:               # one craft's divergence is that craft's: its hub record stays at its last checkpoint
                failed[n] = f"step {t.step:,}: {e}"
                close_stream(t)
                print(f"[v1] {n} STOPPED (non-finite): {e}; the other crafts continue", flush=True)
                continue
            if t._interrupted:
                stopped = "a manual stop"
                break
            if t._guard_halt:
                stopped = f"a guard halt on {n}"
                break
            done = t._last_boundary
            if done is None:                              # a cap stop inside the phase: the session is over
                stopped = "the session's wall budget"
                break
            close_stream(t)
            record(n, t, done, eval_now=t.step % t.tc.eval_every != 0)
    for t in trainers.values():
        close_stream(t)
    for n, why in failed.items():
        print(f"[v1] {n} stopped at a non-finite step ({why})", flush=True)
    print(f"[v1] {'stopped: ' + stopped if stopped else 'every craft finished the diet' if not failed else 'the live crafts finished the diet'}",
          flush=True)
    return hist


def version_of(names) -> str:
    """The version a set of crafts belongs to: v1.1 when any is a v1.1 craft (or a continuation of one), else v1."""
    return "v1.1" if any(n in CRAFTS_V11 or n in CRAFTS_CONT or n in CRAFTS_CONT2 or n in CRAFTS_NOTE or n in CRAFTS_NOTE2 or n in CRAFTS_MIX or n in CRAFTS_MIX2 or n in CRAFTS_CAPTION or n in CRAFTS_CAPTION_CTRL
                         for n in names) else "v1"


def card_in_training(names=None) -> str:
    """The repo's card while the crafts train (the AutoModel card replaces it when they finish); names: the crafts training (the v1
    crafts by default, the v1 text unchanged)."""
    names = list(names or CRAFTS)
    ver = version_of(names)
    rows = "\n".join(f"| `{n}` | {ALL[n]['body']} | `{ALL[n]['arm']}`, seed {ALL[n]['seed']} |" for n in names)
    note = V11_NOTE if ver == "v1.1" else ""
    recipe = (" The output layer's two readout matrices train with AdamW (0.0003, decoupled weight decay 0.1) instead of Muon."
              if ver == "v1.1" else "")
    nb = "colab/nano_v1_1.ipynb" if ver == "v1.1" else "colab/nano_v1_2s.ipynb"
    return f"""---
license: cc-by-nc-sa-4.0
language: en
tags:
- byte-level
- aleph
- splat-attention
- custom_code
---

# AlephLLM nano: Castor and Pollux, {ver}

Four small byte-level AlephLLM crafts (about 9.2 million parameters, a 4,096-byte window) training from birth on Beatrix 2s's diet:
16.1 billion bytes in thirteen phases (wikitext-103, fineweb-edu, nine curriculum stages, two anneals). **Training now**; when they
finish, each craft's folder becomes a transformers AutoModel package (`trust_remote_code=True`, raw UTF-8 bytes in and out).{note}

| craft | the body | started from the start of its bed run |
|---|---|---|
{rows}

Each folder holds the run's record: `checkpoints/` (bf16 every 2,000 steps, fp8 every fifth), `resume/` (the full training state, the
phase boundaries kept), `manifest.json` (the plan, the phases, every checkpoint), `runs/` (TensorBoard: the training curves, the
health census, the evals on the fineweb-edu holdout with the bits by position band, and `bed/*`, the nano bed's gauges of lookup by
frame size, the frame questions and the far reach), `bed_gauges/` (the gauge records) and `v1_boundaries.json` (one row a phase).

The recipe is Beatrix 2s's: Muon at 0.02 on the trunk's matrices, Adam at 0.0003 on the rest, 200 warmup steps then a flat rate,
gradient clip 1.0, 262,144 bytes a step (61,422 steps a craft).{recipe} The trainer: [geolip-nano-beatrix-trainer](https://github.com/AbstractEyes/geolip-nano-beatrix-trainer)
(`nano_v1.py`, `{nb}`) on [AlephLLM](https://github.com/AbstractEyes/alephllm). The license follows the training
data's share-alike and non-commercial terms, as Beatrix 2s's does.
"""


V11_NOTE = ("\n\nThis is v1.1: the same four crafts from the same starts on the same diet, with the output layer's two readout matrices trained by AdamW\n(rate 0.0003, decoupled weight decay 0.1) instead of Muon. In the v1 run the readout kept growing under Muon and\nthe last block's output gathered onto a single direction within its first thousand steps; that run was stopped, and its folders\nare archived in [alephllm-nano-beatrix-training](https://huggingface.co/AbstractPhil/alephllm-nano-beatrix-training/tree/main/archive/castor-pollux-v1) under `archive/castor-pollux-v1/`.")   # 0.1.48: the card re-point (the v1 folders moved 10-16)

HEADLINE = ("bed/note_self/full_value", "bed/note_self/far_ge2048_full_value", "bed/note_self/near_lt1024_full_value", "bed/note_self/carries_per_row",
            "bed/note_self/rows_with_key", "bed/note_self/carry_exact", "bed/note_planted/full_value", "bed/note_planted/far_ge2048_full_value",   # 0.1.41: the note rows
            "bed/carry/full_value", "bed/carry/first_byte", "bed/carry/value_bits", "bed/carry/carries_per_row", "bed/carry/rows_with_carries",
            "bed/carry/carry_exact", "bed/carry/open_prob", "bed/carry/dist_latest",            # 0.1.37: reading with notes, when the record carries it
            "bed/text/heldout_bits", "bed/frame/copy_whole", "bed/frame/nested_whole", "bed/frame/redefined_key_whole",
            "bed/scaling/frame_1024_whole", "bed/window/planted_2048_bits", "bed/window/planted_3584_bits")


def c_arm(n: str) -> str:
    return ALL[n]["arm"]


CONT_READS = (("val bits a byte", "val_bpb", 4), ("copy", "bed/frame/copy_whole", 2), ("nested", "bed/frame/nested_whole", 2),
              ("redefined key", "bed/frame/redefined_key_whole", 2), ("1,024-byte frame", "bed/scaling/frame_1024_whole", 2),
              ("planted at 3,584 bytes (bits)", "bed/window/planted_3584_bits", 1),
              ("far question, own carries (whole value)", "bed/carry/full_value", 2), ("carries written a row", "bed/carry/carries_per_row", 1))


def cont_table(hist: dict) -> str:
    """The continuation reads before and after (markdown): per body and read, the source craft's end (the first row of its
    continuation crafts: the same weights), the end of the frames arm, of the carry arm (0.1.37) and of the control, and each
    arm minus the control; a read no arm carries is left out."""
    rows = ["| body | read | before (v1.1's end) | after, frames | after, carry | after, control | frames minus control | carry minus control |",
            "|---|---|---|---|---|---|---|---|"]

    def at(entries, phase):
        return next((e for e in entries if e.get("phase") == phase), None)

    def get(e, k):
        return None if e is None else (e.get(k) if k == "val_bpb" else (e.get("bed") or {}).get(k))
    last = diet()[-1]["name"]
    for n in CRAFTS:
        fr, ct, cf = hist.get(f"{n}.1-fr") or [], hist.get(f"{n}.1-ct") or [], hist.get(f"{n}.1-cf") or []
        before = at(fr, last) or at(ct, last) or at(cf, last)
        a_fr, a_ct, a_cf = at(fr, CONT_PHASE["fr"][0]), at(ct, CONT_PHASE["ct"][0]), at(cf, CONT_PHASE["cf"][0])
        for label, k, nd in CONT_READS:
            b, x, z, y = get(before, k), get(a_fr, k), get(a_cf, k), get(a_ct, k)
            if b is None and x is None and z is None and y is None:
                continue
            f = ["-" if v is None else f"{v:.{nd}f}" for v in (b, x, z, y)]
            d1 = "-" if x is None or y is None else f"{x - y:+.{nd}f}"
            d2 = "-" if z is None or y is None else f"{z - y:+.{nd}f}"
            rows.append(f"| {n} | {label} | {f[0]} | {f[1]} | {f[2]} | {f[3]} | {d1} | {d2} |")
    return "\n".join(rows)


def cont2_table(hist: dict) -> str:
    """0.1.40: the frames-then-carry reads (markdown): per body and read, the frames arm's end (the -fr-cf craft's first row: the same
    weights), the end of the frames-then-carry arm, of the carry arm and of the control, then frames-then-carry minus the frames end
    (what the carry phase did to a frames body) and minus the carry arm (what a frames start did for the carry skill); a read no arm
    carries is left out."""
    rows = ["| body | read | frames end (the start) | after frames-then-carry | carry arm | control | minus the frames end | minus the carry arm |",
            "|---|---|---|---|---|---|---|---|"]

    def at(entries, phase):
        return next((e for e in entries if e.get("phase") == phase), None)

    def get(e, k):
        return None if e is None else (e.get(k) if k == "val_bpb" else (e.get("bed") or {}).get(k))
    fr_end, cf_end, ct_end = CONT_PHASE["fr"][0], CONT_PHASE["cf"][0], CONT_PHASE["ct"][0]
    for n in CRAFTS:
        fc, fr = hist.get(f"{n}.1-fr-cf") or [], hist.get(f"{n}.1-fr") or []
        cf, ct = hist.get(f"{n}.1-cf") or [], hist.get(f"{n}.1-ct") or []
        start, after, a_cf, a_ct = at(fc, fr_end) or at(fr, fr_end), at(fc, cf_end), at(cf, cf_end), at(ct, ct_end)
        for label, k, nd in CONT_READS:
            s, a, z, y = get(start, k), get(after, k), get(a_cf, k), get(a_ct, k)
            if s is None and a is None and z is None and y is None:
                continue
            f = ["-" if v is None else f"{v:.{nd}f}" for v in (s, a, z, y)]
            d1 = "-" if a is None or s is None else f"{a - s:+.{nd}f}"
            d2 = "-" if a is None or z is None else f"{a - z:+.{nd}f}"
            rows.append(f"| {n} | {label} | {f[0]} | {f[1]} | {f[2]} | {f[3]} | {d1} | {d2} |")
    return "\n".join(rows)


def mixed_table(hist: dict) -> str:
    """0.1.49: the mixed-phase reads (markdown): per body and read, the diet's end (the -mx craft's first row: the same weights), the end
    of the mixed arm, of the frames arm, of the carry arm and of the control, then mixed minus frames (the near read kept or lost) and
    mixed minus carry (the far habit kept or lost); a read no arm carries is left out."""
    rows = ["| body | read | diet's end (the start) | after mixed | frames arm | carry arm | control | minus the frames arm | minus the carry arm |",
            "|---|---|---|---|---|---|---|---|---|"]

    def at(entries, phase):
        return next((e for e in entries if e.get("phase") == phase), None)

    def get(e, k):
        return None if e is None else (e.get(k) if k == "val_bpb" else (e.get("bed") or {}).get(k))
    last = diet()[-1]["name"]
    mx_end, fr_end, cf_end, ct_end = (CONT_PHASE[a][0] for a in ("mx", "fr", "cf", "ct"))
    pairs = [(n, f"{n}.1-mx") for n in CRAFTS]                                 # 0.1.52: every mixed craft of a body, the -mx first
    pairs += [(k, k) for k in sorted(hist) if k in ALL and ALL[k].get("cont") == "mx" and k not in {m for _, m in pairs}]
    for body_label, mxn in pairs:
        n = body_label if body_label in CRAFTS else body_of(mxn)[:-2]
        mx, fr = hist.get(mxn) or [], hist.get(f"{n}.1-fr") or []
        cf, ct = hist.get(f"{n}.1-cf") or [], hist.get(f"{n}.1-ct") or []
        start = at(mx, last) or at(fr, last) or at(cf, last) or at(ct, last)
        after, a_fr, a_cf, a_ct = at(mx, mx_end), at(fr, fr_end), at(cf, cf_end), at(ct, ct_end)
        for label, k, nd in CONT_READS:
            s, a, x, z, y = get(start, k), get(after, k), get(a_fr, k), get(a_cf, k), get(a_ct, k)
            if s is None and a is None and x is None and z is None and y is None:
                continue
            f = ["-" if v is None else f"{v:.{nd}f}" for v in (s, a, x, z, y)]
            d1 = "-" if a is None or x is None else f"{a - x:+.{nd}f}"
            d2 = "-" if a is None or z is None else f"{a - z:+.{nd}f}"
            rows.append(f"| {body_label} | {label} | {f[0]} | {f[1]} | {f[2]} | {f[3]} | {f[4]} | {d1} | {d2} |")
    return "\n".join(rows)


def caption_table(hist: dict) -> str:
    """0.1.55: the caption-phase reads (markdown): per body and read, the mixed arm's end (the -mx-cp craft's first row: the same
    weights), the end of the caption phase, the end of the control (every document's frame from another document) and each minus the
    start; a read no craft carries is left out."""
    rows = ["| body | read | mixed arm's end (the start) | after the caption phase | the control's end | caption minus the start | control minus the start |",
            "|---|---|---|---|---|---|---|"]

    def at(entries, phase):
        return next((e for e in entries if e.get("phase") == phase), None)

    def get(e, k):
        return None if e is None else (e.get(k) if k == "val_bpb" else (e.get("bed") or {}).get(k))
    mx_end, cp_end, cpx_end = (CONT_PHASE[a][0] for a in ("mx", "cp", "cpx"))
    for n in CRAFTS:
        cpe, cpx, mx = hist.get(f"{n}.1-mx-cp") or [], hist.get(f"{n}.1-mx-cpx") or [], hist.get(f"{n}.1-mx") or []
        start = at(cpe, mx_end) or at(cpx, mx_end) or at(mx, mx_end)
        after, ctrl = at(cpe, cp_end), at(cpx, cpx_end)
        for label, k, nd in CONT_READS:
            s, a, x = get(start, k), get(after, k), get(ctrl, k)
            if s is None and a is None and x is None:
                continue
            f = ["-" if v is None else f"{v:.{nd}f}" for v in (s, a, x)]
            d1 = "-" if a is None or s is None else f"{a - s:+.{nd}f}"
            d2 = "-" if x is None or s is None else f"{x - s:+.{nd}f}"
            rows.append(f"| {n} | {label} | {f[0]} | {f[1]} | {f[2]} | {d1} | {d2} |")
    return "\n".join(rows)


def table(hist: dict) -> str:
    """The boundary table (markdown): per craft and phase, the val bits a byte and the bed's headline reads."""
    rows = ["| craft | phase | step | tokens (B) | val bits a byte | bed held-out | copy | nested | redefined key | 1,024-byte frame |",
            "|---|---|---|---|---|---|---|---|---|---|"]

    def f(x, nd=3):
        return "-" if x is None else f"{x:.{nd}f}"
    for n, entries in hist.items():
        for e in entries:
            b = e.get("bed") or {}
            rows.append(f"| {n} | {e['phase']} | {e['step']:,} | {e['tokens'] / 1e9:.2f} | {f(e.get('val_bpb'), 4)} | "
                        f"{f(b.get('bed/text/heldout_bits'), 4)} | {f(b.get('bed/frame/copy_whole'), 2)} | {f(b.get('bed/frame/nested_whole'), 2)} | "
                        f"{f(b.get('bed/frame/redefined_key_whole'), 2)} | {f(b.get('bed/scaling/frame_1024_whole'), 2)} |")
    return "\n".join(rows)
