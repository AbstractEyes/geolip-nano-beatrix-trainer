# -*- coding: utf-8 -*-
"""bed_rerun.py - the bed's deciding tests again with AdamW in place of the pure Adam they ran on (0.1.31). Every test keeps its
notebook's settings verbatim; each run adds the optimizer (wd, wd_set: bed_optim.py), so its record lands under a new name beside the
pure-Adam record of the same settings, and the read puts the two side by side.

The tests come in PARTS, each a list of steps (a grid of runs, a curriculum train, or the twins' pair runs). No two parts run the same
run, so several cards can each take a part at once; a part on one card resumes from its records (on disk, or on the Hub in a new
session). What each part decided under the pure Adam:

  frames-form       the frame form (plain, sized or sliced): the fourth pass (colab/nano_pass4.ipynb): sliced, every close free; Pollux's
                    copy and long window best sliced on both seeds, Castor the same in every form. The bodies' trains and the pairs
                    run on this pick.
  bodies-twins      the curriculum train of the twins and the Canon bodies (colab/nano_v4_train.ipynb, colab/nano_canon_train.ipynb):
                    Castor, Pollux, Pollux with Canon layers, Pollux's layout with dual hubs and Canon layers, seeds 1 and 2.
  bodies-splat      the splat bodies' train (the 4090's splat night, 2026-10-14): Pollux's layout with dual hubs, four dual hubs and
                    two softmax blocks, six dual hubs with Canon layers, five dual hubs and one softmax block, six dual hubs; seeds 1, 2.
  frames-registers  the first pass (colab/nano_frame_grid.ipynb): letters, digits and hex values under their own register tags.
  frames-capacity   the second pass (colab/nano_frame_pass2.ipynb): four, eight or sixteen records a lookup frame, and the scrutiny
                    set (nesting 0.6, a key defined twice in 0.3 of the rows, the question on a nested frame or a redefined key).
  frames-pass3      the third pass (colab/nano_pass3.ipynb) with the stage-A crafts it continued (the 2026-10-13 night's): the scrutiny
                    reads by continuation, sixteen records a frame, Pollux's hub address at 64 x 32 and 128 x 64; and the plain
                    series' Pollux seed 2 continued on eight-record frames (the fourth pass's last grid, which needs this part's craft).
  pair              the twins trained together (colab/nano_pair.ipynb): three forms of the cross term, seeds 1 and 2; its solo
                    controls are bodies-twins' first blocks of Castor and Pollux.

`parts(wd, wd_set)` -> {part: [steps]}; `plan(part, ...)` prints every run's name; `price(part, rate)` the training hours;
`run(part, out, push, ...)` trains the part (resume-first); `read(part, out, repo, ...)` the side-by-side tables (pure Adam -> AdamW)."""
from __future__ import annotations

import os

from . import bed_curriculum as bc
from . import bed_frames as bf
from . import bed_runner as br
from .names import partner_args, run_tag

REPO = br.DEFAULT_REPO
CORPUS_BYTES = 160e6                                    # every pass used these 160 MB of fineweb-edu, in the same order
TWINS = "castor-both,pollux-both"
FORMS = dict(plain={}, sized=dict(sized=1), sliced=dict(sized=1, sliced=32))
FORM = "sliced"                                         # the fourth pass's pick under the pure Adam; frames-form reads it again

# ------------------------------------------------------------------ the frame passes, verbatim from their notebooks
P1 = dict(arms=TWINS, seeds="1,2", register="caption,math,code", ctx=4096, mb=8, steps=12000, frames="dyn", plant=0.75,
          sizes="32,128,256,512", frame_records=4, nest=0.3, overlap=0.0, ask="any", tags="mode", log_every=100)
P2 = dict(arms=TWINS, seeds="1,2", register="caption", ctx=4096, mb=8, steps=12000, frames="dyn", plant=0.75,
          frame_records=4, nest=0.3, overlap=0.0, ask="any", tags="mode", log_every=100)
P3_SHORT = dict(P2, ctx=1024, mb=32)                    # stage A: the short window at 32 rows a step
P3_CONT = dict(P2, steps=4000)                          # a continuation: 4,000 steps at 4,096 x 8 from a stage-A craft
P4 = dict(arms=TWINS, seeds="3,4", register="caption", frames="dyn", plant=0.75, frame_records=8, nest=0.3, overlap=0.0, ask="any",
          tags="mode", gauge_asks="nested", log_every=100)
P4_A = dict(P4, ctx=1024, mb=32, steps=12000, sizes="32,128,256")
P4_CONT = dict(P4, ctx=4096, mb=8, steps=4000, sizes="32,128,256,512,1024")
_R8 = "_frdyn_cap_mode_sz32-128-256_r8_n30_o0_askany_p75"     # the stage-A crafts' tags after their arm, seed and steps

# ------------------------------------------------------------------ the curriculum (the v4 train's, unchanged)
CURR_BASE = dict(register="caption", frames="dyn", plant=0.75, ask="any", tags="mode", gauge_asks="nested,overlap", log_every=100)
STAGES = [
    dict(name="A", grid=dict(ctx=1024, mb=32, steps=12000, sizes="32,128,256", frame_records=8, nest=0.6, overlap=0.3),
         rule="formed", max_blocks=2),
    dict(name="B", grid=dict(ctx=4096, mb=8, steps=4000, sizes="32,128,256,512,1024", frame_records=16, nest=0.6, overlap=0.3),
         rule="far_fall", fall=0.1, window=800, band="512", max_blocks=5),
    dict(name="anneal", grid=dict(ctx=4096, mb=8, steps=4000, sizes="32,128,256,512,1024", frame_records=4, nest=0.3, overlap=0.0),
         rule=None, max_blocks=1),
]
BODIES_TWINS = "castor-both,pollux-both,polluxc-both,altdc-both"
BODIES_SPLAT = "altd-both,d4s2-both,hubdc-both,d5s1-both,hubd-both"
SEEDS = "1,2"

# ------------------------------------------------------------------ the pairs (colab/nano_pair.ipynb)
PAIR_ARM, PAIR_PARTNER = "castor-both", "pollux-both"
PAIR_FORMS = [dict(xc="one", xc_pos="value"), dict(xc="one", xc_pos="all"), dict(xc="mutual", xc_pos="all")]
PAIR_A = dict(ctx=1024, mb=32, steps=12000, sizes="32,128,256", frame_records=8, nest=0.6, overlap=0.3, stage="A1")
PAIR_B = dict(ctx=4096, mb=8, steps=4000, sizes="32,128,256,512,1024", frame_records=16, nest=0.6, overlap=0.3, stage="B1")

ORDER = ("frames-form", "bodies-twins", "bodies-splat", "frames-registers", "frames-capacity", "frames-pass3", "pair")


def opt_keys(wd: float, wd_set: str = "mat") -> dict:
    """The run keys of the optimizer: none for the pure Adam (the earlier runs' names), wd (and the set) for AdamW."""
    if not wd:
        return {}
    return dict(wd=wd) if wd_set == "mat" else dict(wd=wd, wd_set=wd_set)


def _with(init: str, wd: float, wd_set: str) -> str:
    """A continuation's starting-craft template under the optimizer: the tag piece names.run_tag puts before the frame piece."""
    if not wd:
        return init
    piece = f"_wd{wd:g}" + ("" if wd_set == "mat" else wd_set)
    assert "_frdyn" in init, init
    return init.replace("_frdyn", piece + "_frdyn", 1)


def tags_of(spec: dict) -> dict:
    """The records a one-arm spec writes, per run: {run_tag: the run's argument dict}; a pair spec writes two records a run."""
    out = {}
    for r in br.expand({k: v for k, v in spec.items() if k not in br.RUNNER_KEYS}):
        kv = dict(a.lstrip("-").split("=", 1) for a in r)
        out[run_tag(kv)] = kv
        if kv.get("pair"):
            pk = partner_args(kv)
            out[run_tag(pk)] = pk
    return out


def _pair_steps(wd: float, wd_set: str, over) -> list:
    base = dict(CURR_BASE, **FORMS[FORM], **opt_keys(wd, wd_set))
    steps = []
    for i, f in enumerate(PAIR_FORMS, 1):
        xc = dict(pair=PAIR_PARTNER, xc=f["xc"], xc_pos=f["xc_pos"], xc_w=1.0, xc_from=0)
        a = over(dict(base, **PAIR_A, **xc, arms=PAIR_ARM, seeds=SEEDS))
        steps.append(dict(kind="grid", name=f"pair form {i} ({f['xc']} at {f['xc_pos']}): the first stage", spec=a, form=i))
        for s in SEEDS.split(","):
            one = dict(a, seeds=s)
            kv = next(iter(tags_of(one).values()))
            t, p = run_tag(kv), run_tag(partner_args(kv))
            steps.append(dict(kind="grid", name=f"pair form {i}: the long-window block, seed {s}",
                              spec=over(dict(base, **PAIR_B, **xc, arms=PAIR_ARM, seeds=s, init=t, pair_init=p)), form=i))
    return steps


def smoke_over(spec: dict) -> dict:
    """The toy scale of a spec (a local smoke of the whole part): the bed's smoke run (--smoke=1, its short gauges) at 12 steps of 4 rows,
    the windows and every other setting as given. The tags then carry _smoke in place of the step count."""
    return dict(spec, smoke=1, steps=12, mb=4)


def parts(wd: float = 0.1, wd_set: str = "mat", over=None) -> dict:
    """Every part's steps under the optimizer (wd 0 gives the earlier runs' own specs and names). `over` rewrites every run's spec and
    every curriculum stage's grid (smoke_over: the toy scale); the continuations' starting crafts follow it."""
    over = over or (lambda s: s)
    o = opt_keys(wd, wd_set)
    head = "_smoke" if over({}).get("smoke") else "_L12000"                              # the stage-A crafts' step piece
    w = lambda init: _with(init.replace("_L12000", head, 1), wd, wd_set)                 # noqa: E731
    g = lambda name, spec: dict(kind="grid", name=name, spec=over(dict(spec, **o)))     # noqa: E731
    curr = lambda name, arms: dict(kind="curriculum", name=name,                       # noqa: E731
                                   stages=[dict(s, grid=over(dict(s["grid"]))) for s in STAGES],
                                   base=dict(CURR_BASE, **FORMS[FORM], **o), arms=arms, seeds=SEEDS)
    return {
        "frames-form": [
            g("stage A, plain frames", P4_A), g("stage A, sized frames", dict(P4_A, **FORMS["sized"])),
            g("stage A, sliced frames", dict(P4_A, **FORMS["sliced"])),
            g("continued at 4,096, plain", dict(P4_CONT, init=w("{arm}_s{seed}_L12000" + _R8))),
            g("continued at 4,096, sized", dict(P4_CONT, **FORMS["sized"], init=w("{arm}_s{seed}_L12000_frdyn_cap_mode_sized_sz32-128-256_r8_n30_o0_askany_p75"))),
            g("continued at 4,096, sliced", dict(P4_CONT, **FORMS["sliced"], init=w("{arm}_s{seed}_L12000_frdyn_cap_mode_sized_sl32_sz32-128-256_r8_n30_o0_askany_p75"))),
        ],
        "bodies-twins": [curr("the twins and the Canon bodies", BODIES_TWINS)],
        "bodies-splat": [curr("the splat bodies", BODIES_SPLAT)],
        "frames-registers": [g("the registers (caption, math, code)", P1)],
        "frames-capacity": [
            g("records a lookup frame (4, 8, 16)", dict(P2, sizes="32,128,256,512,1024", frame_records="4,8,16")),
            g("the scrutiny set", dict(P2, sizes="32,128,256,512", nest=0.6, overlap=0.3, ask="nested,overlap")),
        ],
        "frames-pass3": [
            g("stage A on the scrutiny rows (the night's crafts)", dict(P3_SHORT, nest=0.6, overlap=0.3, ask="nested,overlap")),
            g("stage A on eight-record frames (the night's crafts)", dict(P3_SHORT, frame_records=8)),
            g("the scrutiny reads by continuation", dict(P3_CONT, sizes="32,128,256,512", nest=0.6, overlap=0.3, ask="nested,overlap",
                                                         init=w("{arm}_s{seed}_L12000_frdyn_cap_mode_sz32-128-256_r4_n60_o30_ask{ask}_p75"))),
            g("sixteen records a frame by continuation", dict(P3_CONT, frame_records=16, sizes="32,128,256,512,1024",
                                                               init=w("{arm}_s{seed}_L12000" + _R8))),
            g("Pollux's address 64 x 32, stage A", dict(P3_SHORT, arms="pollux-both", frame_records=8, hub_K=64, hub_D=32)),
            g("Pollux's address 128 x 64, stage A", dict(P3_SHORT, arms="pollux-both", frame_records=8, hub_K=128, hub_D=64)),
            g("Pollux's address 64 x 32, continued", dict(P3_CONT, arms="pollux-both", frame_records="8,16", sizes="32,128,256,512,1024",
                                                          hub_K=64, hub_D=32, init=w("{arm}_s{seed}_L12000_K{hub_K}D{hub_D}c1" + _R8))),
            g("Pollux's address 128 x 64, continued", dict(P3_CONT, arms="pollux-both", frame_records="8,16", sizes="32,128,256,512,1024",
                                                           hub_K=128, hub_D=64, init=w("{arm}_s{seed}_L12000_K{hub_K}D{hub_D}c1" + _R8))),
            g("the plain series' Pollux seed 2 on eight-record frames, continued", dict(P4_CONT, arms="pollux-both", seeds="2",
                                                                                        init=w("{arm}_s{seed}_L12000" + _R8))),
        ],
        "pair": _pair_steps(wd, wd_set, over),
    }


def runs_of(step: dict) -> list:
    """A grid step's runs -> [(tag, args)] (a pair run: its arm's record only; the partner's lands beside it)."""
    out = []
    for r in br.expand({k: v for k, v in step["spec"].items() if k not in br.RUNNER_KEYS}):
        kv = dict(a.lstrip("-").split("=", 1) for a in r)
        out.append((run_tag(kv), kv))
    return out


def plan(part: str, wd: float = 0.1, wd_set: str = "mat", out: str = "", repo: str = "", log=print, over=None) -> dict:
    """Every run a part may train, by name -> dict(runs, curriculum_chains, tokens (a grid's exact; a chain's fewest and most)); with
    `repo`, the runs already on the Hub are marked."""
    on_hub = set()
    if repo:
        from huggingface_hub import HfApi
        on_hub = {f[len("bed/reports/bed_"):-5] for f in HfApi(token=False).list_repo_files(repo) if f.startswith("bed/reports/bed_")}
    n, tok, chains, lo, hi = 0, 0, 0, 0, 0
    for st in parts(wd, wd_set, over)[part]:
        if st["kind"] == "grid":
            log(f"== {st['name']}")
            for tag, kv in runs_of(st):
                n += 1
                t = int(kv.get("steps", 3000)) * int(kv.get("ctx", 1024)) * int(kv.get("mb", 32)) * (2 if kv.get("pair") else 1)
                tok += t
                log(f"   {'ON THE HUB ' if tag in on_hub else ''}{tag}" + (" (+ its partner)" if kv.get("pair") else ""))
        else:
            ch = bc.chains_of(st["arms"], st["seeds"])
            a, b = bc.bounds(st["stages"], st["base"])
            chains, lo, hi = chains + len(ch), lo + a * len(ch), hi + b * len(ch)
            log(f"== {st['name']}: {len(ch)} chains ({', '.join(f'{x} s{s}' for x, s in ch)})")
            for blk in bc.plan(st["stages"], st["base"], st["arms"], st["seeds"]):
                log(f"   may run {'ON THE HUB ' if blk['tag'] in on_hub else ''}{blk['tag']}")
    return dict(runs=n, grid_tokens=tok, chains=chains, chain_tokens=(lo, hi))


def price(part: str, rate, wd: float = 0.1, wd_set: str = "mat", over=None) -> tuple:
    """The part's training hours (fewest, most) at `rate(arm, ctx, mb)` tokens a second (a pair run trains two crafts: twice the
    tokens at the arm's single rate, an upper estimate). Gauges, starts and ships add about a minute a run."""
    lo = hi = 0.0
    for st in parts(wd, wd_set, over)[part]:
        if st["kind"] == "grid":
            for _, kv in runs_of(st):
                ctx, mb = int(kv.get("ctx", 1024)), int(kv.get("mb", 32))
                t = int(kv.get("steps", 3000)) * ctx * mb * (2 if kv.get("pair") else 1) / rate(kv["arm"], ctx, mb)
                lo, hi = lo + t, hi + t
        else:
            for arm, _ in bc.chains_of(st["arms"], st["seeds"]):
                for s in st["stages"]:
                    gk = {**st["base"], **s["grid"]}
                    ctx, mb = int(gk["ctx"]), int(gk["mb"])
                    t = int(gk["steps"]) * ctx * mb / rate(arm, ctx, mb)
                    lo, hi = lo + t, hi + t * int(s.get("max_blocks", 1))
    return lo / 3600, hi / 3600


def shapes(part: str, wd: float = 0.1, wd_set: str = "mat", over=None) -> list:
    """The (arm, ctx, mb) shapes a part trains: the price line measures each once."""
    out = []
    for st in parts(wd, wd_set, over)[part]:
        if st["kind"] == "grid":
            out += [(kv["arm"], int(kv.get("ctx", 1024)), int(kv.get("mb", 32))) for _, kv in runs_of(st)]
        else:
            out += [(a, int(s["grid"]["ctx"]), int(s["grid"]["mb"])) for a, _ in bc.chains_of(st["arms"], st["seeds"]) for s in st["stages"]]
    return list(dict.fromkeys(out))


def _on_hub(repo: str) -> set:
    """The run names whose records are on the Hub (tokenless; an unreachable Hub reads as none)."""
    try:
        from huggingface_hub import HfApi
        return {f[len("bed/reports/bed_"):-5] for f in HfApi(token=False).list_repo_files(repo) if f.startswith("bed/reports/bed_")}
    except Exception:  # noqa: BLE001
        return set()


def run(part: str, out: str, push: str = "", wd: float = 0.1, wd_set: str = "mat", parallel: int = 1, log=print, over=None) -> dict:
    """Train a part, step after step, resume-first (a finished run's record on disk or on the Hub is read, not rerun). A STOP file in
    `out` starts no new run. -> dict(steps (name -> the runner's or the curriculum's result), stopped, failed)."""
    res = dict(steps={}, stopped=False, failed=None)
    shipped = _on_hub(push or REPO)
    for st in parts(wd, wd_set, over)[part]:
        log(f"===== {part}: {st['name']} =====")
        if st["kind"] == "grid":
            # a new session: the grid's runs an earlier session shipped are fetched as records, so the runner reads them as done
            for tag in (t for t, kv in runs_of(st) for t in ([t, run_tag(partner_args(kv))] if kv.get("pair") else [t])):
                if tag in shipped and not os.path.exists(os.path.join(out, "bed", f"bed_{tag}.json")):
                    bc.fetch_record(push or REPO, out, tag, log)
            r = br.run(st["spec"], out=out, push=push, parallel=parallel)
        else:
            r = bc.run(st["stages"], st["base"], st["arms"], st["seeds"], out=out, push=push, name=f"bed_adamw_{part}")
        res["steps"][st["name"]] = {k: v for k, v in r.items() if k not in ("table", "ran")}
        if r.get("failed") is not None or r.get("stopped"):
            res["failed"], res["stopped"] = r.get("failed"), bool(r.get("stopped"))
            log(f"{part} paused at {st['name']}: {'a failure' if r.get('failed') is not None else 'STOP'}; rerun to continue")
            break
    return res


# ------------------------------------------------------------------ the read: pure Adam -> AdamW, side by side
def _f(x, nd=2) -> str:
    return "-" if x is None else f"{x:.{nd}f}"


def _formed(rec: dict, floor: float = 0.05, min_trials: int = 8) -> bool:
    """The copy formed, by the current verdict (bf.formation, 0.1.19) recomputed from the record's own fetch: some lookup group of
    `min_trials` trials or more writes the value whole above `floor`. Records before 0.1.19 stored a verdict under an older rule (and
    no groups_judged), so the stored field is not comparable across the two sides of a pair."""
    lookup = [a["full_value"] for g, a in (rec.get("fetch_by_frame") or {}).items() if g.startswith("data/") and a.get("n", 0) >= min_trials]
    return bool(lookup) and not all(v <= floor for v in lookup)


def _reads(rec: dict | None) -> dict:
    if rec is None:
        return {}
    whole = bf.pooled_fetch(rec.get("fetch_by_frame"))
    asks = rec.get("fetch_by_frame_asks") or {}
    nested = bf.pooled_fetch(asks.get("nested")) if asks.get("nested") else None
    redef = bf.pooled_fetch(asks.get("overlap")) if asks.get("overlap") else None
    return dict(text=rec.get("heldout_bpb"), whole=None if whole is None else whole["full_value"],
                by_size=bc._whole_by_size(rec), formed=_formed(rec),
                nested=None if nested is None else nested["full_value"], redef=None if redef is None else redef["full_value"])


def _pair_cell(a, b, nd=2) -> str:
    if a is None and b is None:
        return "-"
    return f"{_f(a, nd)} -> {_f(b, nd)}"


HEAD = ("| run | text bits a byte (Adam -> AdamW) | the copy's whole value, pooled | whole value by frame size, AdamW (32 / 128 / 256 / "
        "512 / 1,024) | copy formed | nested question | redefined key |")


def _row(label: str, old: dict, new: dict) -> str:
    fm = "-" if not old and not new else f"{'yes' if old.get('formed') else 'no' if old else '-'} -> {'yes' if new.get('formed') else 'no' if new else '-'}"
    return (f"| {label} | {_pair_cell(old.get('text'), new.get('text'), 4)} | {_pair_cell(old.get('whole'), new.get('whole'))} "
            f"| {new.get('by_size', '-')} | {fm} | {_pair_cell(old.get('nested'), new.get('nested'))} | {_pair_cell(old.get('redef'), new.get('redef'))} |")


def read(part: str, out: str, repo: str = REPO, wd: float = 0.1, wd_set: str = "mat", over=None) -> str:
    """The part's side-by-side tables (markdown): every AdamW run beside the pure-Adam run of the same settings (records from `out` or
    the Hub); a curriculum chain block by block, matched by block name; runs not trained yet read '-'."""
    if part == "pair":                                  # the pair notebook's own read: each pair beside its solo twins (bodies-twins)
        from . import bed_pair
        rows = []
        for st in parts(wd, wd_set, over)["pair"]:
            i = st["form"]
            f = PAIR_FORMS[i - 1]
            for tag, kv in runs_of(st):
                rows.append((f"{i}: {f['xc']} / {f['xc_pos']}", "A" if kv.get("stage") == "A1" else "B", int(kv["seed"]), tag,
                             run_tag(partner_args(kv))))
        return bed_pair.report(rows, out=out, repo=repo)
    records = bc.Records(out, repo, lambda m: None)
    new_p, old_p = parts(wd, wd_set, over)[part], parts(0.0, over=over)[part]
    md = [f"### {part}: pure Adam -> AdamW (decoupled decay {wd:g} on the {wd_set} set)\n"]
    for new, old in zip(new_p, old_p):
        md.append(f"#### {new['name']}\n")
        md.append(HEAD)
        md.append("|---|---|---|---|---|---|---|")
        if new["kind"] == "grid":
            for (tn, kv), (to, _) in zip(runs_of(new), runs_of(old)):
                lab = br.run_label([f"--{k}={v}" for k, v in kv.items() if k not in ("wd", "wd_set")])
                md.append(_row(lab, _reads(records(to)), _reads(records(tn))))
        else:
            for arm, seed in bc.chains_of(new["arms"], new["seeds"]):
                walks = []
                for st in (old, new):
                    try:
                        walks.append({b["block"]: b["record"] for b in bc.walk(st["stages"], st["base"], arm, seed, records)[0]})
                    except RuntimeError:                       # an earlier chain continued differently: its blocks are not matched
                        walks.append({})
                for blk in dict.fromkeys(list(walks[0]) + list(walks[1])):
                    md.append(_row(f"{arm} s{seed} {blk}", _reads(walks[0].get(blk)), _reads(walks[1].get(blk))))
        md.append("")
    return "\n".join(md)
