# -*- coding: utf-8 -*-
"""The AutoModel packaging (automodel.py) on a tiny craft of the v1 kind (dual hubs, Canon layers, the byte table): the vendored code,
the config, the weights, the parity with the library, the upload list's refusals, the card. CPU only, nothing leaves the machine. The
load gate needs transformers: it runs when transformers is importable or NANO_TF_PKGS names a folder that holds it."""
import importlib.util
import json
import os

import pytest
import torch

from geolip.nano_beatrix import automodel as AM
from geolip.nano_beatrix.bodies import craft_cfg

CRAFT = "tiny-v1"


def _tiny_cfg():
    from geolip.alephllm.presets import AlephLMConfig
    d = craft_cfg("altdc-both", 256).to_dict()
    d.update(name=CRAFT, d_model=64, n_heads=2, bank_ff=64, n_layers=4, hub_layers=(0, 2))
    return AlephLMConfig.from_dict(d)


def _run(tmp_path, steps=3):
    """A finished run folder in the trainer's layout: the manifest, the final bf16 checkpoint, the final resume state."""
    from safetensors.torch import save_file
    from geolip.alephllm.model.alephlm import AlephLM
    cfg = _tiny_cfg()
    torch.manual_seed(0)
    m = AlephLM(cfg)
    with torch.no_grad():                                            # trained-looking weights: no tensor left at its birth value
        for p in m.parameters():
            p.add_(0.01 * torch.randn_like(p))
    run = tmp_path / "runs" / CRAFT
    (run / "checkpoints").mkdir(parents=True)
    (run / "resume").mkdir()
    sd = {k: v.detach().clone().contiguous() for k, v in m.state_dict().items()}
    save_file({k: (v.to(torch.bfloat16) if v.is_floating_point() else v) for k, v in sd.items()},
              str(run / "checkpoints" / f"step_{steps:08d}.safetensors"), metadata={"step": str(steps), "format": "bf16"})
    torch.save({"step": steps, "model": sd}, str(run / "resume" / f"boundary_anneal_mix_step{steps}.pt"))
    json.dump({"model_config": cfg.to_dict(), "steps": steps, "tokens_seen": 1234,
               "phases": [{"name": "anneal_mix", "status": "done"}]}, open(run / "manifest.json", "w", encoding="utf-8"))
    return tmp_path / "runs", run


def _staged(tmp_path):
    runs, run = _run(tmp_path)
    root, gates, work = tmp_path / "pkg", tmp_path / "pkg" / "gates", tmp_path / "pkg" / "work" / CRAFT
    AM.stage_code(str(root), str(gates))
    AM.stage_config(CRAFT, str(run / "manifest.json"), str(root), str(gates))
    AM.stage_weights(CRAFT, str(run), str(root), str(gates))
    par = AM.stage_parity(CRAFT, str(root), str(work), str(gates))
    return root, gates, work, run, par


def test_the_code_is_vendored_with_every_patch_and_no_missing_sibling(tmp_path):
    rec = AM.stage_code(str(tmp_path), str(tmp_path / "gates"))
    assert rec["pass"] and rec["patches"] == {"alephlm.py": 4, "presets.py": 4} and not rec["missing_siblings"]
    for f in AM.VENDORED:
        assert "from .." not in open(tmp_path / f, encoding="utf-8").read()
    assert set(rec["import_graph"][AM.WRAPPER]) == {"alephlm", "presets"}


def test_a_tiny_craft_passes_config_weights_and_parity(tmp_path):
    root, gates, work, run, par = _staged(tmp_path)
    cfg = json.load(open(root / CRAFT / "config.json", encoding="utf-8"))
    assert cfg["auto_map"]["AutoModelForCausalLM"] == "modeling_nano_beatrix.NanoBeatrixForCausalLM"
    assert cfg["torch_dtype"] == "float32" and cfg["use_cache"] is False and cfg["hub_ckpt"] == 0 and cfg["bank_byte_table"] == "add"
    assert all((root / CRAFT / f).read_bytes() == (root / f).read_bytes() for f in AM.CODE)       # the copy equals the top
    assert par["pass"] and all(v["vendored_vs_library_max_abs"] == 0.0 for v in par["files"].values())
    from safetensors import safe_open
    with safe_open(str(root / CRAFT / "model.safetensors"), "pt") as fh:
        assert fh.metadata()["step"] == "3" and all(k.startswith("model.") for k in fh.keys())


def test_the_weights_gate_refuses_fp32_weights_that_are_not_the_checkpoint(tmp_path):
    runs, run = _run(tmp_path)
    p = run / "resume" / "boundary_anneal_mix_step3.pt"
    payload = torch.load(str(p), weights_only=False)
    k = next(k for k, v in payload["model"].items() if v.is_floating_point() and v.numel() > 8)
    payload["model"][k] = payload["model"][k] + 0.5
    torch.save(payload, str(p))
    root = tmp_path / "pkg"
    AM.stage_code(str(root), str(root / "gates"))
    AM.stage_config(CRAFT, str(run / "manifest.json"), str(root), str(root / "gates"))
    with pytest.raises(SystemExit, match="weights"):
        AM.stage_weights(CRAFT, str(run), str(root), str(root / "gates"))


def test_an_unfinished_diet_is_refused(tmp_path):
    runs, run = _run(tmp_path)
    man = json.load(open(run / "manifest.json", encoding="utf-8"))
    man["phases"].append({"name": "more", "status": "planned"})
    json.dump(man, open(run / "manifest.json", "w", encoding="utf-8"))
    root = tmp_path / "pkg"
    AM.stage_code(str(root), str(root / "gates"))
    AM.stage_config(CRAFT, str(run / "manifest.json"), str(root), str(root / "gates"))
    with pytest.raises(AssertionError, match="not finished"):
        AM.stage_weights(CRAFT, str(run), str(root), str(root / "gates"))


def test_the_upload_list_needs_every_gate_on_these_very_bytes(tmp_path):
    root, gates, work, run, par = _staged(tmp_path)
    with pytest.raises(FileNotFoundError):                            # no load gate yet
        AM.check_gated(str(root), [CRAFT])
    AM.gate(str(gates / CRAFT), "load", True, sha256=AM.hashes(str(root / CRAFT), AM.PACKAGE))
    files = AM.check_gated(str(root), [CRAFT])
    assert {p for p, _ in files} == set(AM.CODE) | {f"{CRAFT}/{f}" for f in AM.PACKAGE}
    with open(root / CRAFT / "model.safetensors", "r+b") as fh:       # one byte changes after the gates
        fh.seek(-1, 2)
        b = fh.read(1)
        fh.seek(-1, 2)
        fh.write(bytes([b[0] ^ 1]))
    with pytest.raises(AssertionError, match="other bytes"):
        AM.check_gated(str(root), [CRAFT])


def test_the_final_files_are_the_anneal_boundary_then_latest():
    assert AM.final_files({"steps": 61422}) == (61422, "checkpoints/step_00061422.safetensors",
                                                  "resume/boundary_anneal_mix_step61422.pt", "resume/latest.pt")


def test_the_card_names_each_model_and_how_to_load_it():
    from geolip.nano_beatrix.nano_v1 import CRAFTS, HEADLINE
    hist = {n: [{"phase": "anneal_mix", "step": 61422, "tokens": 16.1e9, "val_bpb": 1.2345, "bed": {k: 0.5 for k in HEADLINE}}]
            for n in CRAFTS}
    text = AM.card("AbstractPhil/alephllm-nano-castor-pollux", hist, {n: 9_000_000 for n in CRAFTS})
    assert text.startswith("---\nlicense: cc-by-nc-sa-4.0") and "library_name: transformers" in text
    assert all(f"[`{n}`](./{n})" in text for n in CRAFTS) and "1.2345" in text and "9,000,000" in text
    assert 'subfolder="nano-pollux-dual-canon-v1"' in text and "trust_remote_code=True" in text


TF = os.environ.get("NANO_TF_PKGS")


@pytest.mark.skipif(not (importlib.util.find_spec("transformers") or (TF and os.path.isdir(TF))),
                    reason="the load gate needs transformers (installed, or NANO_TF_PKGS)")
def test_the_load_gate_through_transformers(tmp_path):
    root, gates, work, run, par = _staged(tmp_path)
    files = work / "files"
    files.mkdir()
    for f in AM.PACKAGE:
        (files / f).write_bytes((root / CRAFT / f).read_bytes())
    AM.load_child(CRAFT, str(root), str(work), str(gates), TF)       # the library unimportable in the child; both routes, both files
    rec = json.load(open(gates / CRAFT / "load.json", encoding="utf-8"))
    assert rec["pass"] and len(rec["routes"]) == 4
    assert all(r["max_abs_vs_parity_logits"] == 0.0 and r["every_tensor_equals_the_file"] for r in rec["routes"].values())


@pytest.mark.skipif(not (importlib.util.find_spec("transformers") or (TF and os.path.isdir(TF))),
                    reason="the load gate needs transformers (installed, or NANO_TF_PKGS)")
def test_the_load_gate_fails_a_wrapper_that_leaves_the_byte_table_empty(tmp_path, monkeypatch):
    text = open(os.path.join(AM.TEMPLATE, AM.WRAPPER), encoding="utf-8").read()
    a, b = text.index("    @torch.no_grad()\n    def _init_weights"), text.index("    def forward")
    tpl = tmp_path / "tpl"
    tpl.mkdir()
    (tpl / AM.WRAPPER).write_text(text[:a] + "    def _init_weights(self, module):\n        pass\n\n" + text[b:], encoding="utf-8")
    monkeypatch.setattr(AM, "TEMPLATE", str(tpl))                    # the 2s / 3 wrappers' form: nothing refilled
    root, gates, work, run, par = _staged(tmp_path)
    files = work / "files"
    files.mkdir()
    for f in AM.PACKAGE:
        (files / f).write_bytes((root / CRAFT / f).read_bytes())
    with pytest.raises(AssertionError, match="load gate"):
        AM.load_child(CRAFT, str(root), str(work), str(gates), TF)
