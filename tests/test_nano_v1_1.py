# -*- coding: utf-8 -*-
"""The nano v1.1 crafts (nano_v1.py, 0.1.33): the v1 crafts again with the output layer's readout on AdamW at decay 0.1. The v1 crafts
stay exactly as they were. CPU only; the Hub is never touched."""
import dataclasses

import torch

from geolip.nano_beatrix import automodel, lastfix_screen, nano_v1
from geolip.nano_beatrix.names import run_tag

V1 = list(nano_v1.CRAFTS)
V11 = list(nano_v1.CRAFTS_V11)


def test_the_v11_crafts_are_the_v1_crafts_under_a_new_name():
    assert V11 == [f"{n}.1" for n in V1] and set(nano_v1.ALL) == set(V1) | set(V11)
    for n in V1:
        a, b = nano_v1.CRAFTS[n], nano_v1.CRAFTS_V11[f"{n}.1"]
        assert {k: b[k] for k in a} == a                                         # the same body, seed, code and text
        assert b["train"] == dict(decay_params=("head.w_h.", "head.w_s."), decay_wd=0.1)
        assert "train" not in a and "tag" not in a                               # the v1 crafts are untouched
    assert nano_v1.train_of("nano-castor-v1") == {} and nano_v1.train_of("nano-castor-v1.1")["decay_wd"] == 0.1


def test_the_readout_is_the_screens_readout():
    assert nano_v1.READOUT == lastfix_screen.HEAD_PARAMS


def test_the_v11_birth_is_the_v1_birth(tmp_path):
    from safetensors.torch import load_file
    a = load_file(nano_v1.write_birth("nano-castor-v1", str(tmp_path / "v1")))
    b = load_file(nano_v1.write_birth("nano-castor-v1.1", str(tmp_path / "v11")))
    assert set(a) == set(b) and all(torch.equal(a[k], b[k]) for k in a)


def test_only_the_v11_preset_moves_the_readout_and_nothing_else_changes(tmp_path):
    p1 = nano_v1.preset("nano-castor-v1", nano_v1.write_birth("nano-castor-v1", str(tmp_path)))
    p11 = nano_v1.preset("nano-castor-v1.1", nano_v1.write_birth("nano-castor-v1.1", str(tmp_path)))
    t1, t11 = dataclasses.asdict(p1.train), dataclasses.asdict(p11.train)
    assert tuple(t1["decay_params"]) == () and t1["decay_wd"] == 0.0
    assert tuple(t11["decay_params"]) == nano_v1.READOUT and t11["decay_wd"] == 0.1
    assert {k: v for k, v in t1.items() if k not in ("decay_params", "decay_wd")} == \
        {k: v for k, v in t11.items() if k not in ("decay_params", "decay_wd")}
    m1, m11 = p1.model.to_dict(), p11.model.to_dict()
    assert m11.pop("name") == "nano-castor-v1.1" and m1.pop("name") == "nano-castor-v1" and m1 == m11
    assert p1.curriculum == p11.curriculum and p1.hf_repo == p11.hf_repo
    # a smoke run's overrides still win over the version's fields
    assert dataclasses.asdict(nano_v1.preset("nano-castor-v1.1", nano_v1.write_birth("nano-castor-v1.1", str(tmp_path)),
                                             decay_wd=0.0, decay_params=()).train)["decay_wd"] == 0.0


def test_the_trainer_split_puts_exactly_the_two_readout_matrices_on_adamw_at_01(tmp_path):
    from geolip.alephllm.model.alephlm import AlephLM
    from geolip.alephllm.train.optim import build_optimizers
    from geolip.nano_beatrix.bodies import craft_cfg
    model = AlephLM(craft_cfg("castor-both", nano_v1.CTX))
    names = {id(p): n for n, p in model.named_parameters()}
    tc = nano_v1.preset("nano-castor-v1.1", nano_v1.write_birth("nano-castor-v1.1", str(tmp_path))).train
    # the library trainer's own call (train/trainer.py): build_optimizers(model, muon_lr, momentum, adam_lr, decay_prefixes, decay_wd)
    muon, adamw = build_optimizers(model, tc.muon_lr, tc.muon_momentum, tc.adam_lr, decay_prefixes=tuple(tc.decay_params),
                                   decay_wd=tc.decay_wd)
    assert isinstance(adamw, torch.optim.AdamW) and len(adamw.param_groups) == 2
    dec = adamw.param_groups[1]
    assert sorted(names[id(p)] for p in dec["params"]) == ["head.w_h.weight", "head.w_s.weight"] and dec["weight_decay"] == 0.1
    assert adamw.param_groups[0]["weight_decay"] == 0.0 and dec["lr"] == tc.adam_lr == 3e-4
    on_muon = {names[id(p)] for g in muon.param_groups for p in g["params"]}
    assert "head.w_h.weight" not in on_muon and "head.w_s.weight" not in on_muon
    # under the v1 recipe the same two matrices train on Muon (the change is real)
    m1, _ = build_optimizers(model, tc.muon_lr, tc.muon_momentum, tc.adam_lr)
    on_m1 = {names[id(p)] for g in m1.param_groups for p in g["params"]}
    assert {"head.w_h.weight", "head.w_s.weight"} <= on_m1
    assert on_m1 - on_muon == {"head.w_h.weight", "head.w_s.weight"}


def test_the_two_versions_gauge_records_never_share_a_name():
    assert nano_v1.gauge_init("nano-castor-v1", 4000) == "v1c_4000"                 # v1's names as before
    assert nano_v1.gauge_init("nano-castor-v1.1", 4000) == "v11c_4000"
    a = run_tag(nano_v1.gauge_args("nano-castor-v1", nano_v1.gauge_init("nano-castor-v1", 4000), "x"))
    b = run_tag(nano_v1.gauge_args("nano-castor-v1.1", nano_v1.gauge_init("nano-castor-v1.1", 4000), "x"))
    assert a != b and "v11c_4000" in b


def test_the_cards_name_the_version_and_the_v1_card_is_unchanged():
    v1 = nano_v1.card_in_training()
    assert "Castor and Pollux, v1\n" in v1 and "v1.1" not in v1 and "colab/nano_v1_2s.ipynb" in v1
    v11 = nano_v1.card_in_training(V11)
    assert "Castor and Pollux, v1.1\n" in v11 and all(f"`{n}`" in v11 for n in V11)
    assert "decoupled weight decay 0.1" in v11 and "colab/nano_v1_1.ipynb" in v11 and "folders stay as they were" in v11
    hist = {n: [{"phase": "anneal_mix", "step": 61422, "tokens": 16.1e9, "val_bpb": 1.2345,
                 "bed": {k: 0.5 for k in nano_v1.HEADLINE}}] for n in V11}
    am = automodel.card("AbstractPhil/alephllm-nano-castor-pollux", hist, {n: 9_000_000 for n in V11})
    assert "Castor and Pollux, v1.1\n" in am and all(f"[`{n}`](./{n})" in am for n in V11)
    assert f'subfolder="{V11[0]}"' in am and "decoupled weight decay 0.1" in am and "colab/nano_v1_1.ipynb" in am
    am1 = automodel.card("AbstractPhil/alephllm-nano-castor-pollux", {n: hist[f"{n}.1"] for n in V1}, {n: 9_000_000 for n in V1})
    assert "Castor and Pollux, v1\n" in am1 and "v1.1" not in am1
