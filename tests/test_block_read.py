# -*- coding: utf-8 -*-
"""The block read (0.1.96): a stored core whose ninth heads carry the path, read with the path switched off in chosen path blocks. Its
ends are the text read exactly (no block off = the text read's bits and split, every path block off = its path-off read); a set in
between reads the forward with those blocks' ninth heads on the anchored rotation; the path blocks are restored after each set; the
default sets, the record and resume; the text read's own numbers unchanged by the shared loop. CPU only; the Hub is never touched."""
import itertools
import json
import math

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from geolip.nano_beatrix import anchored_rope as ar
from geolip.nano_beatrix import grid_rows as gr
from geolip.nano_beatrix import ninth_head as nhm
from geolip.nano_beatrix import path_rotary as prm
from geolip.nano_beatrix import rotary_scale as rs
from test_key_code import _save_core
from test_ninth_head import _held, short  # noqa: F401
from test_path_rotary import space_table  # noqa: F401
from test_value_addr import _craft


def _core(tmp_path, rot="path", path_blocks=(1, 2, 3), seed=2):
    from safetensors.torch import save_file
    m, st = _craft(n_ctx=4096, seed=seed)
    va = ar.ValueAddress(m.cfg.d_model, st)
    with torch.no_grad():
        va.table.normal_(std=0.5)
    ar.patch_value(m, va)
    kw = dict(units=prm.Units(), form="next", path_blocks=path_blocks) if rot == "path" else {}
    nh = nhm.NinthHeads(m.cfg.d_model, 6, rot, seed=seed, **kw)
    with torch.no_grad():
        for o in nh.o:
            o.weight.mul_(3.0)                                                               # a trained head's weight (not the birth)
    nhm.patch(m, nh)
    d = tmp_path / "core"
    sj = _save_core(m, d, va)
    save_file(nhm.state(nh), str(d / "extra_head.safetensors"))
    sj["ninth_head"] = nhm.scale_field(nh)
    json.dump(sj, open(d / "scale.json", "w", encoding="utf-8"), indent=1)
    return m, nh, d


def _bits(m, xs):
    with torch.no_grad():
        lg = m(xs[:, :-1]).logits
    return float(F.cross_entropy(lg.float().reshape(-1, 256), xs[:, 1:].reshape(-1), reduction="sum")) / lg.shape[0] / lg.shape[1] / math.log(2)


def _rows(held, n=2):
    docs = rs.load_heldout(held)[:n]
    return torch.tensor(np.stack([np.frombuffer(bytes([rs.DOC]) + dd[:gr.CTX], dtype=np.uint8) for dd in docs]).astype(np.int64))


def test_the_ends_are_the_text_read_and_the_sets_between_are_the_forward(tmp_path, short):   # noqa: F811
    m, nh, d = _core(tmp_path)
    held = _held(tmp_path / "held.jsonl")
    tr = rs.text_read(str(d), str(tmp_path / "text.json"), held, n_docs=2, mb=2, dev="cpu", log=lambda s: None)
    subsets = [o for k in range(4) for o in itertools.combinations((1, 2, 3), k)]
    order = subsets[1:4] + [()] + subsets[4:]                                                # the empty set read after others: restored
    rec = rs.block_read(str(d), str(tmp_path / "group.json"), held, sets=order, n_docs=2, mb=2, dev="cpu", log=lambda s: None)
    by = {tuple(r["off"]): r for r in rec["reads"]}
    assert [tuple(r["off"]) for r in rec["reads"]] == [tuple(o) for o in order]
    assert by[()]["bits"] == tr["bits"] and by[()]["by_repeat"] == tr["by_repeat"]          # the text read exactly
    assert by[(1, 2, 3)]["bits"] == tr["path_off"]["bits"] and by[(1, 2, 3)]["by_repeat"] == tr["path_off"]["by_repeat"]
    xs = _rows(held)
    for o in subsets[1:-1]:                                                                  # a set between: those blocks anchored
        nh.path_blocks = tuple(b for b in (1, 2, 3) if b not in o)
        assert abs(by[o]["bits"] - _bits(m, xs)) < 1e-6, o
        nh.path_blocks = (1, 2, 3)
    assert abs(by[()]["bits"] - _bits(m, xs)) < 1e-6
    vals = [by[o]["bits"] for o in subsets]
    assert len({round(v, 9) for v in vals}) == len(vals)                                     # every set its own forward
    assert rec["path"] == dict(form="next", head="ninth", k=prm.K, blocks=[1, 2, 3]) and rec["docs"] == 2 and rec["version"] == rs._PKG_VERSION
    again = rs.block_read(str(d), str(tmp_path / "group.json"), held, sets=[()], n_docs=2, mb=2, dev="cpu", log=lambda s: None)
    assert again == json.load(open(tmp_path / "group.json", encoding="utf-8")) and len(again["reads"]) == len(order)   # resume-first


def test_every_block_path_core_and_the_default_sets(tmp_path, short):                        # noqa: F811
    m, nh, d = _core(tmp_path, path_blocks=None, seed=3)                                     # 0.1.94's form: the path in every block
    held = _held(tmp_path / "held.jsonl")
    tr = rs.text_read(str(d), str(tmp_path / "text.json"), held, n_docs=2, mb=2, dev="cpu", log=lambda s: None)
    rec = rs.block_read(str(d), str(tmp_path / "group.json"), held, n_docs=2, mb=2, dev="cpu", log=lambda s: None)
    assert [r["off"] for r in rec["reads"]] == [[]] + [[b] for b in range(1, 7)] + [list(range(1, 7))]
    assert rec["reads"][0]["bits"] == tr["bits"] and rec["reads"][-1]["bits"] == tr["path_off"]["bits"]
    assert rec["path"]["blocks"] == [1, 2, 3, 4, 5, 6] and "blocks" not in tr["path"]
    xs = _rows(held)
    nh.path_blocks = (1, 2, 4, 5, 6)                                                         # block 3 off by hand
    assert abs(rec["reads"][3]["bits"] - _bits(m, xs)) < 1e-6
    nh.path_blocks = None


def test_bad_sets_and_cores_are_refused(tmp_path, short):                                    # noqa: F811
    _, _, d = _core(tmp_path)
    held = _held(tmp_path / "held.jsonl")
    for sets in ([(4,)], [(), ()], [(1,), (1, 1)]):
        with pytest.raises(AssertionError):
            rs.block_read(str(d), str(tmp_path / "g.json"), held, sets=sets, n_docs=2, mb=2, dev="cpu", log=lambda s: None)
    (tmp_path / "plain").mkdir()
    _, _, dp = _core(tmp_path / "plain", rot="plain")
    with pytest.raises(AssertionError):
        rs.block_read(str(dp), str(tmp_path / "gp.json"), held, n_docs=2, mb=2, dev="cpu", log=lambda s: None)
    assert rs.group_name("abc") == "abc_group"
