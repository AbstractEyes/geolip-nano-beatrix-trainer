# -*- coding: utf-8 -*-
"""The frames continuation (0.1.35; frames_data.py and nano_v1's continuation crafts): the framed rows are the bed's rows, one row is
one packed block, the frames arm and its control read the same text row for row, a resumed stream continues exactly without composing
the rows it skips, the mixes validate on the fineweb-edu holdout, registering only adds names, the eight crafts and their plans, the
free "before" row, and the exact start from a boundary state (weights and both optimizers). CPU only; the Hub is never touched."""
import ast
import os
import time

import numpy as np
import pytest
import torch

from geolip.nano_beatrix import bed_frames as bf
from geolip.nano_beatrix import frames_data as fd
from geolip.nano_beatrix import nano_v1
from geolip.alephllm.data.special_tokens import DOC

HERE = os.path.dirname(os.path.abspath(__file__))
BED_TRAIN = os.path.join(HERE, "..", "geolip", "nano_beatrix", "bed_train.py")
WORDS = ("the river carried silt past the old mill while the town slept and the lamps burned low over quiet streets "
         "a teacher wrote numbers on the board and the class copied every line into worn notebooks before the bell").split()


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    """A small corpus in bed_corpus.py's format (plain words, documents of 400..6,000 bytes), set for the stream children."""
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


def _bed_functions():
    """bed_train's own cap / draw_plan / frame_row, lifted out of the file (its module reads a command line and a corpus at import),
    bound to the frames phase's settings."""
    src = open(BED_TRAIN, encoding="utf-8").read()
    tree = ast.parse(src)
    keep = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ("cap", "draw_plan", "frame_row")]
    assert len(keep) == 3
    f = fd.FORM
    ns = dict(bf=bf, CTX=fd.ROW_CTX, SIZES=list(f["sizes"]), N_REC_FRAME=f["frame_records"], NEST_P=f["nest"], OVERLAP_P=f["overlap"],
              REGISTER=f["register"], TAGS=f["tags"], SIZED=f["sized"], SLICED=f["sliced"], FRAMES="dyn", ASK=f["ask"], FACTS=None,
              REC_BYTES=bf.capacity(fd.ROW_CTX, "A", f["register"])["record_bytes"], GAUGE_FLOOR={}, N_LOOKUP=2, N_BULK=0)
    exec(compile(ast.Module(body=keep, type_ignores=[]), BED_TRAIN, "exec"), ns)
    return ns


def test_the_framed_rows_are_the_beds_rows(corpus):
    bed = _bed_functions()
    packed = fd.load_corpus(corpus)
    for s in range(60):
        p = int(np.random.default_rng(s).integers(0, len(packed) - 2 * fd.ROW_BYTES - 2))
        src = packed[p:p + 2 * fd.ROW_BYTES].tobytes()
        mine, m1 = fd.frame_row(np.random.default_rng([s, 1]), src)
        theirs, m2 = bed["frame_row"](np.random.default_rng([s, 1]), src)
        assert mine == theirs and m1 == m2[0], s
        assert len(mine) == fd.ROW_BYTES


def test_a_row_is_one_block_and_the_two_arms_read_the_same_text(corpus):
    packed = fd.load_corpus(corpus)
    framed = 0
    for i in range(400):
        a, fa = fd.row_bytes(packed, 7, i, fd.P_FRAMED)
        b, fb = fd.row_bytes(packed, 7, i, 0.0)
        assert len(a) == len(b) == fd.ROW_BYTES and not fb
        rng = np.random.default_rng([7, i])
        p = int(rng.integers(0, len(packed) - 2 * fd.ROW_BYTES - 2))
        assert b == packed[p:p + fd.ROW_BYTES].tobytes()                      # the control: the corpus text at the drawn position
        if fa:
            framed += 1
            assert any(x in bf.SPECIALS for x in a[1:])                        # a framed row carries the frame bytes
        else:
            assert a == b                                                      # a plain row of the frames arm is the control's row
    assert 0.69 <= framed / 400 <= 0.81


def test_the_stream_packs_one_row_a_block_and_resumes_without_composing(corpus, monkeypatch):
    from geolip.alephllm.data.streams import build_stream
    from geolip.alephllm.data.tokenizers import build_tokenizer
    tok = build_tokenizer("byte-trigram")
    s = build_stream(fd.FRAMES_MIX, tok, 4096, 2, seed=1337)
    batches = [s.next_batch() for _ in range(3)]
    st = s.state_dict()
    gen_seed = 1337                                                            # the mix's first component keeps the stream seed
    packed = fd.load_corpus(corpus)
    for k, b in enumerate(batches):
        assert b.shape == (2, 4097)
        for r in range(2):
            i = 2 * k + r
            row, _ = fd.row_bytes(packed, gen_seed, i, fd.P_FRAMED)
            assert bytes(b[r, :4096].to(torch.uint8).tolist()) == row and int(b[r, 4096]) == int(DOC)
    nxt = s.next_batch()
    calls = []
    real = fd.frame_row
    monkeypatch.setattr(fd, "frame_row", lambda *a, **k: (calls.append(1), real(*a, **k))[1])
    again = build_stream(fd.FRAMES_MIX, tok, 4096, 2, seed=1337)
    again.load_state_dict(st)
    t0 = time.time()
    it = again._streams["nano-frames-p75"]
    it._open()                                                                  # the skip of the six consumed rows
    assert not calls and time.time() - t0 < 5.0                                 # skipped rows are never composed
    assert torch.equal(again.next_batch(), nxt)


def test_the_mixes_validate_on_the_fineweb_holdout_and_registering_only_adds():
    from geolip.alephllm.data import curriculum, streams
    from geolip.alephllm.data.tokenizers import build_tokenizer
    tok = build_tokenizer("byte-trigram")
    for mix in (fd.FRAMES_MIX, fd.CONTROL_MIX):
        v = streams.build_stream(mix, tok, 4096, 2, seed=1, role="val")
        assert v.dataset == "fineweb-edu" and v.role == "val"
    assert fd.FRAMES_MIX not in curriculum._BASE_MIXES and fd.CONTROL_MIX not in curriculum._BASE_MIXES   # the recipe fingerprint
    before = dict(streams.REGISTRY), dict(curriculum.CURRICULUM_MIXES)
    fd.register()                                                              # idempotent
    assert dict(streams.REGISTRY) == before[0] and dict(curriculum.CURRICULUM_MIXES) == before[1]


def test_the_eight_continuation_crafts_and_their_plans():
    C = nano_v1.CRAFTS_CONT
    assert list(C) == [f"{n}.1-{a}" for n in nano_v1.CRAFTS for a in ("fr", "ct")]
    assert len({c["tag"] for c in C.values()}) == 2 and all(C[n]["source"] in nano_v1.CRAFTS_V11 for n in C)
    inits = {nano_v1.gauge_init(n, 62000) for n in nano_v1.ALL}
    assert len(inits) == len(nano_v1.ALL)                                      # no two crafts share a gauge folder
    assert nano_v1.diet_steps(1.0, 262144) == 61422
    from geolip.alephllm.train.manifest import cursor_at_step
    for n in C:
        ph = nano_v1.diet_of(n)
        assert len(ph) == 14 and ph[:13] == nano_v1.diet() and ph[-1]["planned_tokens"] == nano_v1.CONT_TOKENS
        assert ph[-1]["dataset"] == (fd.FRAMES_MIX if C[n]["cont"] == "fr" else fd.CONTROL_MIX)
        placed = cursor_at_step(ph, 61422, 262144)
        assert all(p["status"] == "done" for p in placed[:13]) and placed[-1]["status"] == "active" and placed[-1]["tokens_done"] == 0
        spec = nano_v1.cont_init(n, 262144)
        assert spec["payload"] == "resume/boundary_anneal_mix_step61422.pt" and spec["prefix"] == C[n]["source"] and spec["step"] == 61422
        p = nano_v1.preset(n, None)
        assert p.init_from == spec and p.train.decay_params == nano_v1.READOUT and p.train.decay_wd == 0.1
    assert nano_v1.diet_of("nano-castor-v1.1") == nano_v1.diet() and nano_v1.version_of(list(C)) == "v1.1"


def test_the_before_and_after_table():
    bed = {"bed/frame/copy_whole": 0.10, "bed/frame/nested_whole": 0.05}
    hist = {"nano-castor-v1.1-fr": [dict(phase="anneal_mix", val_bpb=1.30, bed=bed),
                                    dict(phase="frames_continuation", val_bpb=1.31, bed={"bed/frame/copy_whole": 0.80})],
            "nano-castor-v1.1-ct": [dict(phase="anneal_mix", val_bpb=1.30, bed=bed),
                                    dict(phase="control_continuation", val_bpb=1.29, bed={"bed/frame/copy_whole": 0.12})]}
    t = nano_v1.cont_table(hist)
    assert "| nano-castor-v1 | val bits a byte | 1.3000 | 1.3100 | 1.2900 | +0.0200 |" in t
    assert "| nano-castor-v1 | copy | 0.10 | 0.80 | 0.12 | +0.68 |" in t
    assert "| nano-castor-v1 | nested | 0.05 | - | - | - |" in t and "| nano-pollux-canon-v1 | copy | - | - | - | - |" in t


def test_the_continuation_writes_the_before_row_free():
    class T:
        class manifest:
            phases = None
    from geolip.alephllm.train.manifest import cursor_at_step
    T.manifest.phases = cursor_at_step(nano_v1.diet_of("nano-castor-v1.1-fr"), 61422, 262144)
    assert nano_v1.unrecorded_boundary(T, []) == ("anneal_mix", True)          # the source's last boundary, from these very weights
    assert nano_v1.unrecorded_boundary(T, [dict(phase="anneal_mix")]) == (None, False)


def test_the_exact_continuation_carries_weights_and_both_optimizers(tmp_path, corpus):
    from geolip.alephllm.model.alephlm import AlephLM
    from geolip.alephllm.train.optim import build_optimizers
    name, scale, mb, ga = "nano-castor-v1.1-fr", 2e-5, 1, 1
    tps = mb * ga * nano_v1.CTX
    steps = nano_v1.diet_steps(scale, tps)
    src = nano_v1.ALL[name]["source"]
    torch.manual_seed(0)
    model = AlephLM(nano_v1.craft_cfg(nano_v1.ALL[src]["arm"], nano_v1.CTX))
    opts = build_optimizers(model, 0.02, 0.95, 3e-4, decay_prefixes=nano_v1.READOUT, decay_wd=0.1)
    for p in model.parameters():
        p.grad = torch.randn_like(p) * 1e-3
    for o in opts:
        o.step()                                                               # non-empty optimizer states
    payload = dict(model={k: v.clone() for k, v in model.state_dict().items()}, optimizers=[o.state_dict() for o in opts], step=steps,
                   manifest=dict(phases=[dict(ph, status="done") for ph in nano_v1.diet(scale)]))
    root = tmp_path / "runs_src"
    d = root / src / "resume"
    d.mkdir(parents=True)
    torch.save(payload, str(d / f"boundary_{nano_v1.diet(scale)[-1]['name']}_step{steps}.pt"))
    p = nano_v1.preset(name, None, mb, ga, diet_scale=scale, cont_from=str(root))
    t = nano_v1.make_trainer(p, None, str(tmp_path / "runs"), False, None, "cpu")
    sd = t.raw_model.state_dict()
    assert all(torch.equal(sd[k], payload["model"][k].float()) for k in payload["model"])
    for o, st in zip(t.optimizers, payload["optimizers"]):
        mine = o.state_dict()
        assert len(mine["state"]) == len(st["state"]) > 0
        for k in st["state"]:
            for kk, v in st["state"][k].items():
                assert torch.equal(torch.as_tensor(mine["state"][k][kk]), torch.as_tensor(v)), (k, kk)
    assert t.step == steps and t.manifest.phases[-1]["status"] == "active" and t.manifest.phases[-1]["name"] == "frames_continuation"
    assert "carried" in t.manifest.init_from["optimizers"]
