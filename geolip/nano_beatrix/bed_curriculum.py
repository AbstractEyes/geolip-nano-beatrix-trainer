# -*- coding: utf-8 -*-
"""bed_curriculum.py - a curriculum train as chains of continuation blocks (the nano v4 curriculum, 2026-10-13). A chain is one craft
(an arm and a seed) taken through the stages in order. A stage runs in blocks: each block is one bed run continued from the chain's
previous block, with its own warmup and cosine (the measured recipe), its tag carrying the block's name (--stage=B2) in place of the
starting craft's. The block's record decides what follows, by its stage's rule:
  "formed"    advance when the copy formed by the whole-value rule (the record's formation verdict); a craft that has not formed takes
              another block, up to max_blocks, then advances marked;
  "far_fall"  another block while the far band's online value bits (the records more than 512 bytes back) fell by more than `fall` bits
              over the block's last `window` steps against the `window` before them (bed_frames.band_fall, exact from the curve's
              counts), up to max_blocks;
  None        one block.
Every decision is a pure function of the records (walk), so a restart replays the decisions from the records on disk and runs only
what is missing; a block whose record is not on disk is fetched from the Hub when it is there (a new session resumes the train; the
runner fetches a starting craft's weights). The chains advance in rounds, one block each, one run on the card at a time: every chain
reaches a stage's boundary before any chain runs past it. A STOP file in <out> starts no new block. Each block ships as it lands
(one background shipper for the whole train); the decisions go to <out>/curriculum_<name>.json and the boundary table to
<out>/bed/CURRICULUM_<name>.md, both shipped to bed/curriculum/ at the end.

From Python (a notebook cell):
    from geolip.nano_beatrix import bed_curriculum as bc
    bc.run(STAGES, BASE, arms="castor-both,pollux-both", seeds="1,2", out="/content/out", push=REPO, name="v4_train")
STAGES: a list of dict(name="A", grid={the bed keys of the stage}, rule="formed", max_blocks=2[, fall=0.1, window=800, band="512"]),
in order; BASE: the bed keys every block shares (the arms and seeds come from the chains)."""
from __future__ import annotations

import json
import os
import shutil
import time

from . import bed_frames as bf
from . import bed_runner as br
from .names import run_tag

RULES = ("formed", "far_fall", None)
OWN_KEYS = ("arms", "seeds", "arm", "seed", "stage", "init")      # set by the chain, never by a stage or the shared keys
FRAME_SIZES = (32, 128, 256, 512, 1024)


def check(stages: list, base: dict) -> None:
    """The stages are well formed: unique names that end in a letter (a block is the name and its number: A1, B2, anneal1), a known
    rule, at least one block, and no key the chain sets itself."""
    names = [st["name"] for st in stages]
    assert stages and len(set(names)) == len(names), f"stage names must be unique: {names}"
    for st in stages:
        assert st["name"] and st["name"][-1].isalpha() and st["name"].isalnum(), f"a stage name is letters and digits ending in a letter: {st['name']!r}"
        assert st.get("rule") in RULES, f"stage {st['name']}: rule {st.get('rule')!r} not in {RULES}"
        assert int(st.get("max_blocks", 1)) >= 1, st
        assert not set(st.get("grid", {})) & set(OWN_KEYS), f"stage {st['name']}: the chain sets {OWN_KEYS}"
    assert not set(base) & set(OWN_KEYS), f"the shared keys may not set {OWN_KEYS}"


def block_name(stage: dict, k: int) -> str:
    return f"{stage['name']}{k}"


def block_spec(base: dict, stage: dict, arm: str, seed, k: int, init: str | None) -> dict:
    """The bed keys of a chain's k-th block of a stage: the shared keys, the stage's over them, the chain's arm and seed, the block's
    name, and the starting craft (the chain's previous block; none for the chain's first block)."""
    spec = {**base, **stage.get("grid", {}), "arms": arm, "seeds": str(seed), "stage": block_name(stage, k)}
    if init:
        spec["init"] = init
    return spec


def spec_tag(spec: dict) -> str:
    """A one-run spec's tag: names.run_tag over the runner's own expansion (the single source of the record's name)."""
    runs = br.expand({k: v for k, v in spec.items() if k not in br.RUNNER_KEYS})
    assert len(runs) == 1, f"a block is one run, not {len(runs)}: {spec}"
    return run_tag(dict(a.lstrip("-").split("=", 1) for a in runs[0]))


def decide(stage: dict, k: int, rec: dict) -> dict:
    """The boundary read of a chain's k-th block of `stage` from its record -> dict(action "again" | "advance", mark (a note the
    boundary table carries, or ""), reason (plain words), and the numbers the rule read)."""
    rule, cap = stage.get("rule"), int(stage.get("max_blocks", 1))
    if rule == "formed":
        f = rec.get("formation") or {}
        judged = int(f.get("groups_judged", 0))
        formed = bool(f.get("formed")) and judged > 0             # no lookup group read on enough trials: no verdict, not a formed copy
        whole = bf.pooled_fetch(rec.get("fetch_by_frame"))
        nums = dict(formed=formed, groups_judged=judged, whole_value=None if whole is None else round(whole["full_value"], 4))
        said = ("" if whole is None else f" (the whole value {whole['full_value']:.2f} over {whole['n']} trials)") + \
               ("" if judged else "; no lookup group was read on enough trials to judge")
        if formed:
            return dict(action="advance", mark="", reason=f"the copy formed{said}", **nums)
        if k < cap:
            return dict(action="again", mark="", reason=f"the copy has not formed{said}: another block", **nums)
        return dict(action="advance", mark=f"not formed after {k} blocks", reason=f"the copy has not formed after {k} blocks{said}: moving on, marked", **nums)
    if rule == "far_fall":
        thr, win, band = float(stage.get("fall", 0.1)), int(stage.get("window", 800)), str(stage.get("band", "512"))
        f = bf.band_fall(rec.get("curve") or [], band, win)
        if f is None:
            return dict(action="advance", mark="no far-band window", fall=None, last=None, before=None,
                        reason=f"the far band has no two {win}-step windows with values in this block: moving on, marked")
        nums = dict(fall=round(f["fall"], 4), last=round(f["last"], 4), before=round(f["before"], 4))
        said = f"the far band read {f['before']:.2f} then {f['last']:.2f} bits over the block's last two {win}-step windows (a fall of {f['fall']:.2f})"
        if f["fall"] > thr and k < cap:
            return dict(action="again", mark="", reason=f"{said}, more than {thr}: another block", **nums)
        if f["fall"] > thr:
            return dict(action="advance", mark=f"still improving after {k} blocks", reason=f"{said}, more than {thr}, after {k} blocks (the most): moving on, marked", **nums)
        return dict(action="advance", mark="", reason=f"{said}, at or under {thr}: settled", **nums)
    return dict(action="advance", mark="", reason="one block")


def walk(stages: list, base: dict, arm: str, seed, records) -> tuple:
    """A chain's blocks from its records -> (done, next): done = [dict(stage, k, block, tag, init, record, decision)] for every block
    whose record is known, in order; next = dict(si, k, block, spec, tag, init) of the block to run, or None when the chain has finished
    its last stage. `records(tag)` returns a block's record or None. A record continued from another craft than the chain's previous
    block (a rule changed since it ran) stops the walk: its decisions would not be this curriculum's."""
    done, si, k, init = [], 0, 1, None
    while si < len(stages):
        st = stages[si]
        spec = block_spec(base, st, arm, seed, k, init)
        tag = spec_tag(spec)
        rec = records(tag)
        if rec is None:
            return done, dict(si=si, k=k, block=block_name(st, k), spec=spec, tag=tag, init=init)
        if (rec.get("init") or None) != init:
            raise RuntimeError(f"{tag}: its record was continued from {rec.get('init') or 'birth'}, the chain's previous block is "
                               f"{init or 'birth'} (a rule changed since it ran): move the chain's records aside or use new seeds")
        d = decide(st, k, rec)
        done.append(dict(stage=st["name"], k=k, block=block_name(st, k), tag=tag, init=init, record=rec, decision=d))
        init = tag
        if d["action"] == "again":
            k += 1
        else:
            si, k = si + 1, 1
    return done, None


def plan(stages: list, base: dict, arms: str, seeds) -> list:
    """Every block a chain may run (each stage up to its max_blocks; a block's tag does not depend on its starting craft) ->
    [dict(arm, seed, block, tag, steps, tokens)]: the names to check before a launch and the bounds of the train's length."""
    out = []
    for arm, seed in chains_of(arms, seeds):
        for st in stages:
            for k in range(1, int(st.get("max_blocks", 1)) + 1):
                spec = block_spec(base, st, arm, seed, k, "x")
                g = {**base, **st.get("grid", {})}
                steps, ctx, mb = int(g.get("steps", 3000)), int(g.get("ctx", 1024)), int(g.get("mb", 32))
                out.append(dict(arm=arm, seed=int(seed), stage=st["name"], k=k, block=block_name(st, k), tag=spec_tag(spec),
                                steps=steps, tokens=steps * ctx * mb))
    return out


def chains_of(arms, seeds) -> list:
    return [(a.strip(), int(s)) for a in str(arms).split(",") if a.strip() for s in str(seeds).split(",") if str(s).strip()]


def bounds(stages: list, base: dict) -> tuple:
    """The fewest and the most tokens one chain trains (every stage one block; every stage its most blocks)."""
    lo = hi = 0
    for st in stages:
        g = {**base, **st.get("grid", {})}
        t = int(g.get("steps", 3000)) * int(g.get("ctx", 1024)) * int(g.get("mb", 32))
        lo, hi = lo + t, hi + t * int(st.get("max_blocks", 1))
    return lo, hi


def fetch_record(repo: str, out: str, tag: str, log) -> bool:
    """A block's record that is not in <out>/bed, from the Hub's bed/reports/ (tokenless; the repo is public): a new session resumes
    the train from what shipped. The weights stay on the Hub until a block continues from them (the runner fetches a starting craft).
    -> fetched or not (not on the Hub = the block runs; nothing is logged)."""
    try:
        from huggingface_hub import hf_hub_download
        p = hf_hub_download(repo, f"bed/reports/bed_{tag}.json", local_dir=os.path.join(out, "_hub"), token=False)
        os.makedirs(os.path.join(out, "bed"), exist_ok=True)
        shutil.copy2(p, os.path.join(out, "bed", f"bed_{tag}.json"))
        log(f"the record of {tag} fetched from {repo} (shipped by an earlier session)")
        return True
    except Exception:  # noqa: BLE001
        return False


class Records:
    """A chain's records by tag: from <out>/bed, else from the Hub once (`repo`; smokes never), cached once found."""

    def __init__(self, out: str, repo: str, log) -> None:
        self.out, self.repo, self.log, self.cache, self.tried = out, repo, log, {}, set()

    def __call__(self, tag: str):
        if tag in self.cache:
            return self.cache[tag]
        p = os.path.join(self.out, "bed", f"bed_{tag}.json")
        if not os.path.exists(p) and self.repo and "_smoke" not in tag and tag not in self.tried:
            self.tried.add(tag)
            fetch_record(self.repo, self.out, tag, self.log)
        if os.path.exists(p):
            self.cache[tag] = json.load(open(p, encoding="utf-8"))
            return self.cache[tag]
        return None


def _f(x, nd=2) -> str:
    return "-" if x is None else f"{x:.{nd}f}"


def _whole_by_size(rec: dict) -> str:
    g = rec.get("fetch_by_frame") or {}
    return " / ".join(_f((g.get(f"data/{s}/depth0") or {}).get("full_value")) for s in FRAME_SIZES)


def _pooled(rec: dict, ask: str) -> str:
    p = bf.pooled_fetch(((rec.get("fetch_by_frame_asks") or {}).get(ask)))
    return "-" if p is None else f"{p['full_value']:.2f}"


def table(summary: dict) -> str:
    """The boundary table (markdown): one row per block of every chain, in order, with what its record read and what followed."""
    head = ("| craft | block | steps x window x rows | from | text bits a byte | whole value by frame size (32 / 128 / 256 / 512 / 1,024) "
            "| nested question | redefined key | far reads, the window before -> the last (bits) | canary rows (plain / tag / frame) | flags "
            "| k tok/s | min | next |")
    lines = [head, "|" + "---|" * 14]
    for craft, blocks in summary.items():
        for b in blocks:
            r, d, f = b["record"], b["decision"], b.get("far")
            far = "-" if f is None else f"{f['before']:.2f} -> {f['last']:.2f}"
            can = r.get("canary") or {}
            canary = " / ".join(_f((can.get(k) or {}).get("bits"), 3) for k in ("plain", "tag", "frame"))
            flags = [x for x in (("copy not formed" if (r.get("formation") or {}).get("failure") else ""),
                                 ("plain collapse" if (r.get("plain_collapse") or {}).get("flag") else ""), d.get("mark", "")) if x]
            lines.append(f"| {craft} | {b['block']} | {r['steps']:,} x {r['ctx']:,} x {r['micro_batch']} | {b['prev'] or 'birth'} "
                         f"| {_f(r.get('heldout_bpb'), 4)} | {_whole_by_size(r)} | {_pooled(r, 'nested')} | {_pooled(r, 'overlap')} "
                         f"| {far} | {canary} | {'; '.join(flags) or '-'} | {r.get('tok_per_s', 0) / 1e3:.0f} "
                         f"| {r.get('train_minutes', 0):.1f} | {b['next']} |")
    return "\n".join(lines) + "\n"


def summarize(stages: list, base: dict, arms, seeds, records) -> tuple:
    """Every chain walked -> ({craft: [block rows]}, {craft: next block name or None}); a block row carries its record, its decision,
    the block it continued from and what followed it."""
    summary, nexts = {}, {}
    by_name = {st["name"]: st for st in stages}
    for arm, seed in chains_of(arms, seeds):
        done, nxt = walk(stages, base, arm, seed, records)
        rows = []
        for i, b in enumerate(done):
            after = done[i + 1]["block"] if i + 1 < len(done) else (nxt["block"] + " (to run)" if nxt else "done")
            st = by_name[b["stage"]]                                 # the far reads by the stage's own band and window (the B rule's)
            far = bf.band_fall(b["record"].get("curve") or [], str(st.get("band", "512")), int(st.get("window", 800)))
            rows.append(dict(b, prev=done[i - 1]["block"] if i else None, next=after, far=far))
        summary[f"{arm} s{seed}"] = rows
        nexts[f"{arm} s{seed}"] = nxt["block"] if nxt else None
    return summary, nexts


def report(stages: list, base: dict, arms, seeds, out: str, repo: str = "") -> str:
    """The boundary table from the records, running nothing: those in <out>/bed, and those on the Hub's bed/reports/ when `repo` names
    it (a new session reads what an earlier one shipped); the blocks still to run named under it."""
    check(stages, base)
    summary, nexts = summarize(stages, base, arms, seeds, Records(out, repo, print))
    todo = ", ".join(f"{c} {n}" for c, n in nexts.items() if n)
    return table(summary) + "\n" + (f"To run next: {todo}." if todo else "Every chain finished its last stage.") + "\n"


def _ship_summary(repo: str, paths: list, log, attempts: int = 3) -> bool:
    from huggingface_hub import HfApi
    for k in range(1, attempts + 1):
        try:
            api = HfApi(token=os.environ["HF_TOKEN"])
            for p in paths:
                api.upload_file(path_or_fileobj=p, path_in_repo=f"bed/curriculum/{os.path.basename(p)}", repo_id=repo,
                                commit_message=f"curriculum: {os.path.basename(p)}")
            log(f"the curriculum's decisions and table shipped to {repo} bed/curriculum/")
            return True
        except Exception as e:  # noqa: BLE001
            log(f"the curriculum summary's ship failed (attempt {k}/{attempts}): {type(e).__name__}: {str(e)[:120]}")
            time.sleep(min(15 * 2 ** (k - 1), 120))
    return False


def run(stages: list, base: dict, arms, seeds, out: str, push: str = "", name: str = "train", python: str | None = None,
        dry: bool = False, init_repo: str = "") -> dict:
    """Run the curriculum: every chain (arm x seed) through the stages, one block at a time on the card, in rounds. dry=True prints
    every chain's next block and the names of every block the train may run, and runs nothing. -> dict(chains, blocks_run [tags],
    failed (rc or None), stopped, done (every chain finished), next {craft: block or None}, unshipped [tags], table (markdown))."""
    check(stages, base)
    os.makedirs(out, exist_ok=True)
    logf, stop_f, t0 = os.path.join(out, "runner.log"), os.path.join(out, "STOP"), time.time()

    def log(msg: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')} +{time.time() - t0:6.0f}s] {msg}"
        print(line, flush=True)
        with open(logf, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    chains = chains_of(arms, seeds)
    records = Records(out, init_repo or push or br.DEFAULT_REPO, log)
    result = dict(chains=len(chains), blocks_run=[], failed=None, stopped=False, done=False, next={}, unshipped=[], table="")
    lo, hi = bounds(stages, base)
    log(f"curriculum {name}: {len(chains)} chains ({', '.join(f'{a} s{s}' for a, s in chains)}); a chain trains "
        f"{lo / 1e9:.2f}B to {hi / 1e9:.2f}B tokens by its boundary reads; out {out}; push {push or 'no'}")
    for st in stages:
        g = {**base, **st.get("grid", {})}
        log(f"  stage {st['name']}: {int(g.get('steps', 3000)):,} steps a block at {int(g.get('ctx', 1024)):,} x {int(g.get('mb', 32))}, "
            f"at most {int(st.get('max_blocks', 1))} block(s); rule {st.get('rule') or 'one block'}")
    if dry:
        for b in plan(stages, base, arms, seeds):
            log(f"  may run {b['arm']} s{b['seed']} {b['block']}: {b['tag']} ({len(b['tag'])} characters)")
    elif push and not os.environ.get("HF_TOKEN"):
        log("no HF_TOKEN in the environment: the blocks would not ship; set it or drop push")
        result["failed"] = 2
        return result
    shipper = br._Shipper(push, out, log) if (push and not dry) else None
    if shipper:
        for rec in br.pending_records(out):
            log(f"unshipped record from before: {os.path.basename(rec)}; shipping in the background")
            shipper.submit(rec)
    told = set()

    def tell(arm, seed):
        """Walk a chain and say every boundary not said yet (a block's read and what follows it) -> the walk."""
        done, nxt = walk(stages, base, arm, seed, records)
        for i, b in enumerate(done):
            if (arm, seed, b["block"]) not in told:
                told.add((arm, seed, b["block"]))
                after = done[i + 1]["block"] if i + 1 < len(done) else (nxt["block"] if nxt else "the chain is done")
                log(f"BOUNDARY {arm} s{seed} {b['block']}: {b['decision']['reason']} -> {after}")
        return done, nxt

    while True:
        moved = False
        for arm, seed in chains:
            _, nxt = tell(arm, seed)
            if nxt is None:
                continue
            if dry:
                log(f"next: {arm} s{seed} {nxt['block']} ({nxt['tag']}) from {nxt['init'] or 'birth'}")
                continue
            if os.path.exists(stop_f):
                log("STOP present: no new block starts; remove the file and rerun to continue")
                result["stopped"] = True
                break
            log(f"block {nxt['block']} of {arm} s{seed} starts from {nxt['init'] or 'birth'}")
            res = br.run(nxt["spec"], out, push=push, python=python, init_repo=init_repo, shipper=shipper)
            if res["failed"] is not None or res["stopped"]:
                result["failed"], result["stopped"] = res["failed"], bool(res["stopped"])
                log(f"block {nxt['block']} of {arm} s{seed} " + ("failed: no new block starts (fix, then rerun: the finished blocks are kept)"
                                                                if res["failed"] is not None else "stopped"))
                break
            if records(nxt["tag"]) is None:
                result["failed"] = 5
                log(f"block {nxt['block']} of {arm} s{seed} finished without its record {nxt['tag']}: no new block starts")
                break
            result["blocks_run"].append(nxt["tag"])
            tell(arm, seed)                                          # the block's boundary read, said as it lands
            moved = True
        if dry or result["failed"] is not None or result["stopped"] or not moved:
            break
    summary, result["next"] = summarize(stages, base, arms, seeds, records)
    result["done"] = all(v is None for v in result["next"].values())
    result["table"] = table(summary)
    if not dry:
        jpath, mpath = os.path.join(out, f"curriculum_{name}.json"), os.path.join(out, "bed", f"CURRICULUM_{name}.md")
        os.makedirs(os.path.dirname(mpath), exist_ok=True)
        keep = {c: [{k: v for k, v in b.items() if k != "record"} for b in rows] for c, rows in summary.items()}
        json.dump(dict(name=name, utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), stages=stages, base=base, chains=keep,
                       next=result["next"], done=result["done"]), open(jpath, "w", encoding="utf-8"), indent=1)
        with open(mpath, "w", encoding="utf-8") as f:
            f.write(f"# Curriculum {name}\n\n" + result["table"])
        if shipper:
            shipper.drain(log)
            result["unshipped"] = [os.path.basename(p)[4:-5] for p in br.pending_records(out)]
            if result["unshipped"]:
                result["unshipped"] = br.ship_pending(push, out, log, attempts=3)
            if result["unshipped"]:
                log(f"UNSHIPPED after the train: {result['unshipped']}; bed_runner.ship_pending(REPO, OUT) ships them by hand")
            _ship_summary(push, [jpath, mpath], log)
        log("curriculum " + ("done: every chain finished its last stage" if result["done"] else
                             "paused: " + ", ".join(f"{c} next {n}" for c, n in result["next"].items() if n)))
    return result
