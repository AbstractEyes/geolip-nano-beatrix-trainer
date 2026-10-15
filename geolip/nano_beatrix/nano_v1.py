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
ALL = {**CRAFTS, **CRAFTS_V11}


def train_of(name: str) -> dict:
    """The TrainConfig fields a craft's version sets: none for a v1 craft, the readout's AdamW group for a v1.1 craft."""
    return dict(ALL[name].get("train") or {})


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


def preset(name: str, birth_file: str, micro_batch: int = 16, grad_accum: int = 4, repo: str = REPO, diet_scale: float = 1.0,
           **train_over):
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
    return Preset(model=cfg, train=tc, hf_repo=repo, curriculum=diet(diet_scale), data_scale=1.0,
                  init_from={"file": birth_file, "step": 0})


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
    put("bed/gauge_seconds", rec.get("wall_s"))
    return out


def gauge_args(name: str, init: str, out: str) -> dict:
    c = ALL[name]
    return dict(GAUGE_ARGS, out=out, arm=c["arm"], seed=str(GAUGE_SEED), init=init, gauge_only="1")


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
        diet_scale: float = 1.0, device=None, gauge_gb: float = 10.0, **train_over) -> dict:
    """Train the v1 crafts through the diet, round-robin by phase. out: the local folder (births, runs, the bed's gauge folder);
    hf_token: None = local only (no hub record, no uploads); max_hours: this session's wall budget (a stop between or inside phases is
    a pause: the next call resumes from the hub); gauge_gb: the bed gauge child's card budget. -> {craft: [one entry a boundary: phase,
    step, tokens, val bits a byte, the gauge's headline reads]} (also written to <out>/v1_boundaries.json)."""
    from .bed_corpus import write_heldout
    names = list(names or CRAFTS)
    t_end = None if max_hours is None else time.time() + max_hours * 3600
    births = {n: write_birth(n, os.path.join(out, "births")) for n in names}
    write_heldout(os.path.join(out, "bed"))
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
        p = preset(n, births[n], micro_batch, grad_accum, repo, diet_scale, **train_over)
        t = make_trainer(p, hf_token, os.path.join(out, "runs"), process_streams, hook_for(n), device)
        if t.step == 0 and t.manifest.init_from and not any("BIRTH:" in str(x) for x in (t.manifest.notes or [])):
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
    """The version a set of crafts belongs to: v1.1 when any is a v1.1 craft, else v1."""
    return "v1.1" if any(n in CRAFTS_V11 for n in names) else "v1"


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


V11_NOTE = ("\n\nThis is v1.1: the same four crafts from the same starts on the same diet, with the output layer's two readout matrices trained by AdamW\n(rate 0.0003, decoupled weight decay 0.1) instead of Muon. In the v1 run (the folders without the .1) the readout kept growing under\nMuon and the last block's output gathered onto a single direction within its first thousand steps; that run was stopped and\nits folders stay as they were.")

HEADLINE = ("bed/text/heldout_bits", "bed/frame/copy_whole", "bed/frame/nested_whole", "bed/frame/redefined_key_whole",
            "bed/scaling/frame_1024_whole", "bed/window/planted_2048_bits", "bed/window/planted_3584_bits")


def c_arm(n: str) -> str:
    return ALL[n]["arm"]


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
