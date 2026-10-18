# -*- coding: utf-8 -*-
"""0.1.40: the frames-then-carry arm: a second-stage continuation craft (nano-<body>-v1.1-fr-cf) whose source is the frames arm's end
state and whose plan is the source's whole plan plus the carry phase; the first-stage arms unchanged; the toy exact continuation from a
toy frames boundary; the frames-then-carry table."""
import os

import numpy as np
import pytest
import torch

from geolip.nano_beatrix import frames_data as fd
from geolip.nano_beatrix import nano_v1

WORDS = ("the river carried silt past the old mill while the town slept and the lamps burned low over quiet streets "
         "a teacher wrote numbers on the board and the class copied every line into worn notebooks before the bell").split()


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    """A small corpus in bed_corpus.py's format (the continuation tests' fixture), set for the stream children."""
    d = tmp_path_factory.mktemp("frames_corpus")
    rng = np.random.default_rng(0)
    docs, offs, total = [], [0], 0
    while total < 3_000_000:
        n = int(rng.integers(80, 1000))
        b = (" ".join(WORDS[int(i)] for i in rng.integers(len(WORDS), size=n)) + ".").encode()
        docs.append(b)
        total += len(b)
        offs.append(total)
    np.frombuffer(b"".join(docs), dtype=np.uint8).tofile(str(d / "bed_corpus.bin"))
    np.save(str(d / "bed_corpus_offsets.npy"), np.asarray(offs, dtype=np.int64))
    old = os.environ.get(fd.CORPUS_ENV)
    os.environ[fd.CORPUS_ENV] = str(d)
    yield str(d)
    if old is None:
        os.environ.pop(fd.CORPUS_ENV, None)
    else:
        os.environ[fd.CORPUS_ENV] = old


def test_the_frames_then_carry_crafts_and_their_plans():
    C2 = nano_v1.CRAFTS_CONT2
    assert list(C2) == [f"{n}.1-fr-cf" for n in nano_v1.CRAFTS] and all(n in nano_v1.ALL for n in C2)
    assert all(C2[n]["source"] == n[:-3] and C2[n]["source"] in nano_v1.CRAFTS_CONT and C2[n]["cont"] == "cf" for n in C2)
    assert len({c["tag"] for c in C2.values()}) == 1 and not (set(C2) & set(nano_v1.CRAFTS_CONT))
    inits = {nano_v1.gauge_init(n, 69052) for n in nano_v1.ALL}
    assert len(inits) == len(nano_v1.ALL)                                      # no two crafts share a gauge folder
    from geolip.alephllm.train.manifest import cursor_at_step
    for n in C2:
        ph = nano_v1.diet_of(n)
        assert len(ph) == 15 and ph[:13] == nano_v1.diet()
        assert [p["name"] for p in ph[13:]] == ["frames_continuation", "carry_continuation"]
        assert [p["dataset"] for p in ph[13:]] == [fd.FRAMES_MIX, fd.CARRY_MIX]
        assert all(p["planned_tokens"] == nano_v1.CONT_TOKENS for p in ph[13:])
        assert nano_v1.plan_steps(n, 262144) == 69052 and nano_v1.plan_steps(C2[n]["source"], 262144) == 65237
        placed = cursor_at_step(ph, 65237, 262144)
        assert all(p["status"] == "done" for p in placed[:14]) and placed[-1]["status"] == "active" and placed[-1]["tokens_done"] == 0
        spec = nano_v1.cont_init(n, 262144)
        assert spec["payload"] == "resume/boundary_frames_continuation_step65237.pt"
        assert spec["prefix"] == C2[n]["source"] and spec["step"] == 65237 and spec["repo"] == nano_v1.REPO
        p = nano_v1.preset(n, None)
        assert p.init_from == spec and p.train.decay_params == nano_v1.READOUT and p.train.decay_wd == 0.1
        assert [x["name"] for x in p.curriculum] == [x["name"] for x in ph]
    # the first-stage arms read exactly as before
    for n in nano_v1.CRAFTS_CONT:
        spec = nano_v1.cont_init(n, 262144)
        assert spec["payload"] == "resume/boundary_anneal_mix_step61422.pt" and spec["step"] == 61422
        assert nano_v1.plan_steps(n, 262144) == 65237 and len(nano_v1.diet_of(n)) == 14
    assert nano_v1.plan_steps("nano-castor-v1.1", 262144) == nano_v1.diet_steps(1.0, 262144) == 61422


def test_the_frames_then_carry_table():
    bed_s = {"bed/frame/copy_whole": 0.90, "bed/carry/full_value": 0.00}
    hist = {"nano-castor-v1.1-fr-cf": [dict(phase="frames_continuation", val_bpb=1.2398, bed=bed_s),
                                       dict(phase="carry_continuation", val_bpb=1.2360, bed={"bed/frame/copy_whole": 0.85})],
            "nano-castor-v1.1-cf": [dict(phase="anneal_mix", val_bpb=1.2614, bed={"bed/frame/copy_whole": 0.14}),
                                    dict(phase="carry_continuation", val_bpb=1.2367, bed={"bed/frame/copy_whole": 0.37})],
            "nano-castor-v1.1-ct": [dict(phase="control_continuation", val_bpb=1.2300, bed={"bed/frame/copy_whole": 0.18})]}
    t = nano_v1.cont2_table(hist)
    assert "| nano-castor-v1 | val bits a byte | 1.2398 | 1.2360 | 1.2367 | 1.2300 | -0.0038 | -0.0007 |" in t
    assert "| nano-castor-v1 | copy | 0.90 | 0.85 | 0.37 | 0.18 | -0.05 | +0.48 |" in t and "| nano-pollux-canon-v1 |" not in t
    # the start row falls back to the frames arm's own end when the new craft has no row yet
    t2 = nano_v1.cont2_table({"nano-castor-v1.1-fr": [dict(phase="frames_continuation", val_bpb=1.2398, bed=bed_s)]})
    assert "| nano-castor-v1 | copy | 0.90 | - | - | - | - | - |" in t2


def test_the_exact_continuation_from_a_frames_boundary_stands_at_the_carry_phase(tmp_path, corpus):
    from geolip.alephllm.model.alephlm import AlephLM
    from geolip.alephllm.train.optim import build_optimizers
    name, scale, mb, ga = "nano-castor-v1.1-fr-cf", 2e-5, 1, 1
    tps = mb * ga * nano_v1.CTX
    src = nano_v1.ALL[name]["source"]                                          # the frames craft
    steps = nano_v1.plan_steps(src, tps, scale)
    import math
    assert steps == nano_v1.diet_steps(scale, tps) + math.ceil(int(nano_v1.CONT_TOKENS * scale) / tps) and src.endswith("-fr")
    torch.manual_seed(0)
    model = AlephLM(nano_v1.craft_cfg(nano_v1.ALL[src]["arm"], nano_v1.CTX))
    opts = build_optimizers(model, 0.02, 0.95, 3e-4, decay_prefixes=nano_v1.READOUT, decay_wd=0.1)
    for p in model.parameters():
        p.grad = torch.randn_like(p) * 1e-3
    for o in opts:
        o.step()
    payload = dict(model={k: v.clone() for k, v in model.state_dict().items()}, optimizers=[o.state_dict() for o in opts], step=steps,
                   manifest=dict(phases=[dict(ph, status="done") for ph in nano_v1.diet_of(src, scale)]))
    root = tmp_path / "runs_src"
    d = root / src / "resume"
    d.mkdir(parents=True)
    torch.save(payload, str(d / f"boundary_frames_continuation_step{steps}.pt"))
    spec = nano_v1.cont_init(name, tps, scale, cont_from=str(root))
    assert spec["payload"].endswith(f"boundary_frames_continuation_step{steps}.pt") and spec["step"] == steps
    p = nano_v1.preset(name, None, mb, ga, diet_scale=scale, cont_from=str(root))
    t = nano_v1.make_trainer(p, None, str(tmp_path / "runs"), False, None, "cpu")
    sd = t.raw_model.state_dict()
    assert all(torch.equal(sd[k], payload["model"][k].float()) for k in payload["model"])
    for o, st in zip(t.optimizers, payload["optimizers"]):
        assert len(o.state_dict()["state"]) == len(st["state"]) > 0
    names = [ph["name"] for ph in t.manifest.phases]
    assert t.step == steps and names[-2:] == ["frames_continuation", "carry_continuation"]
    assert t.manifest.phases[-2]["status"] == "done" and t.manifest.phases[-1]["status"] == "active"
    assert nano_v1.unrecorded_boundary(t, []) == ("frames_continuation", True)   # the before row: the frames end, from these very weights
    assert "carried" in t.manifest.init_from["optimizers"]
