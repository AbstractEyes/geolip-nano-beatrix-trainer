# -*- coding: utf-8 -*-
"""automodel.py - the nano v1 crafts as transformers AutoModel packages in their repo (AbstractPhil/alephllm-nano-castor-pollux).

The layout: the model code once at the repo's top, because transformers looks for remote code at the top of a repo even when the
config and weights come from a subfolder, and a byte-identical copy in each craft's folder, so a craft folder downloaded alone loads
too. Each craft's folder gains config.json, model.safetensors (bf16, the default) and model.fp32.safetensors (the fp32 variant) beside
its training record. Nothing is uploaded unless every gate passed on exactly the bytes being uploaded.

  code      the library's model files (the installed geolip.alephllm, this trainer's pin) vendored flat with the asserted import
            patches, and the wrapper (hf_template/modeling_nano_beatrix.py); every relative import has a sibling file
  config    each craft's config.json from its run's manifest (every AlephLMConfig field, the auto_map, use_cache false, fp32); the
            code copied into the craft's folder byte for byte
  weights   model.safetensors (the final bf16 checkpoint re-keyed under "model.") and model.fp32.safetensors (the fp32 masters of the
            final resume state); the fp32 tensors rounded to bf16 equal the bf16 file bit for bit
  parity    the vendored code against the library: the configs equal; each weight file loads strictly into both; the logits are equal
            exactly on fixed byte rows (saved for the load gate)
  load      AutoModelForCausalLM.from_pretrained in a child process where the library cannot be imported, with fresh caches, offline,
            by both routes (the top folder with subfolder=<craft>, and the craft's folder alone): every tensor equals the file's, the
            logits equal the parity stage's, a short greedy continuation
  push      the code, the card and every craft's package files in one commit; hubload: the load gate again from the repo itself

  index     crafts.json at the repo's top (0.1.43): every packaged folder with its body, its arm in plain words, what it continued from,
            its start and end step, its parameters and its end reads; the card's tables are rendered from the same facts
  config    (0.1.43) also carries `craft` (what the model is: name, body, arm, source, phase, step, bytes) and `activated` (the mechanisms
            its body switches on, in plain words), so a loaded model says what it is: model.config.craft, model.config.activated

Usage (the notebook's last cell): release(out, repo, crafts, token) fetches each finished craft's final files from the repo, packages,
gates, writes the index and the card and pushes; crafts=None packages every craft folder the repo holds (the bodies and their
continuation arms, 0.1.43).
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time

TEMPLATE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hf_template")
VENDORED = ["address.py", "alephlm.py", "attention.py", "bank.py", "canon.py", "embedding.py", "governor.py", "head.py", "lastfix.py",
            "presets.py"]
WRAPPER = "modeling_nano_beatrix.py"
CODE = VENDORED + [WRAPPER]
PACKAGE = CODE + ["config.json", "model.safetensors", "model.fp32.safetensors"]
WEIGHTS = (("bf16", "model.safetensors", None), ("fp32", "model.fp32.safetensors", "fp32"))
INDEX = "crafts.json"                                   # 0.1.43: the repo's index of its packaged crafts, at the top beside the card
PROMPT = "The history of astronomy begins"
_DEFER = ("from importlib import import_module as _im  # VENDORED: the loader resolves relative imports to flat sibling files; "
          "this one is resolved by name, when the library is installed")
# (file, the library's text, its vendored form, how many times the library's text must occur)
PATCHES = [
    ("alephlm.py", "from ..presets import AlephLMConfig", "from .presets import AlephLMConfig", 1),
    ("alephlm.py", "            from .fusion import Fusion\n",
     "            " + _DEFER + "\n            Fusion = _im(\"geolip.alephllm.model.fusion\").Fusion\n", 1),
    ("alephlm.py", "        from .fusion import cat_caches, zero_cache_rows\n",
     "        " + _DEFER + "\n        cat_caches, zero_cache_rows = (getattr(_im(\"geolip.alephllm.model.fusion\"), n) for n in "
     "(\"cat_caches\", \"zero_cache_rows\"))\n", 1),
    ("alephlm.py", "        from .fusion import snapshot, restore_rows\n",
     "        " + _DEFER + "\n        snapshot, restore_rows = (getattr(_im(\"geolip.alephllm.model.fusion\"), n) for n in "
     "(\"snapshot\", \"restore_rows\"))\n", 1),
    ("presets.py", "    from .data.curriculum import curriculum_phases, _BASE_STAGE_TOKENS\n",
     "    " + _DEFER + "\n    curriculum_phases, _BASE_STAGE_TOKENS = (getattr(_im(\"geolip.alephllm.data.curriculum\"), n) for n in "
     "(\"curriculum_phases\", \"_BASE_STAGE_TOKENS\"))\n", 1),
    ("presets.py", "    from .data.curriculum import curriculum_phases\n",
     "    " + _DEFER + "\n    curriculum_phases = _im(\"geolip.alephllm.data.curriculum\").curriculum_phases\n", 1),
    ("presets.py", "    from .data.curriculum import CURRICULUM_MIXES as _MIXES\n",
     "    " + _DEFER + "\n    _MIXES = _im(\"geolip.alephllm.data.curriculum\").CURRICULUM_MIXES\n", 1),
    ("presets.py", "    from .data.streams import REGISTRY as _REG, RF_DEEP_TEXT_BYTES as _DEEP\n",
     "    " + _DEFER + "\n    _REG, _DEEP = (getattr(_im(\"geolip.alephllm.data.streams\"), n) for n in (\"REGISTRY\", \"RF_DEEP_TEXT_BYTES\"))\n", 1),
]


def log(*a):
    print("[automodel]", *a, flush=True)


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 22), b""):
            h.update(blk)
    return h.hexdigest()


def hashes(folder: str, names) -> dict:
    return {f: sha256(os.path.join(folder, f)) for f in names}


def relative_imports(path: str) -> list:
    """The names transformers' remote-code loader demands as sibling files (its two patterns)."""
    text = open(path, encoding="utf-8").read()
    names = re.findall(r"^\s*import\s+\.(\S+)\s*$", text, flags=re.MULTILINE)
    names += re.findall(r"^\s*from\s+\.(\S+)\s+import", text, flags=re.MULTILINE)
    return sorted(set(names))


def gate(gates_dir: str, name: str, ok: bool, **facts) -> dict:
    os.makedirs(gates_dir, exist_ok=True)
    rec = {"gate": name, "pass": bool(ok), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **facts}
    json.dump(rec, open(os.path.join(gates_dir, f"{name}.json"), "w", encoding="utf-8"), indent=1, default=str)
    log(f"GATE {name}: {'PASS' if ok else 'FAIL'}")
    if not ok:
        raise SystemExit(f"GATE FAILED: {name} ({os.path.join(gates_dir, name + '.json')})")
    return rec


# ------------------------------------------------------------------ the stages
def vendor(dst: str) -> dict:
    """The library's model files (the installed geolip.alephllm) into the flat folder dst, patched. -> {file: patches applied}."""
    import geolip.alephllm as A
    root = os.path.dirname(A.__file__)
    applied = {}
    for f in VENDORED:
        src = os.path.join(root, f) if f == "presets.py" else os.path.join(root, "model", f)
        text = open(src, encoding="utf-8").read().replace("\r\n", "\n")
        for pf, old, new, n in PATCHES:
            if pf != f:
                continue
            k = text.count(old)
            assert k == n, f"{f}: the patch target occurs {k} times, not {n}: {old!r}"
            text = text.replace(old, new)
            applied[f] = applied.get(f, 0) + 1
        assert "from .." not in text, f"{f}: a parent-package import is left"
        open(os.path.join(dst, f), "w", encoding="utf-8", newline="\n").write(text)
    return applied


def stage_code(root: str, gates: str) -> dict:
    """The code at the top of the package root: the vendored model files and the wrapper; every relative import has a sibling."""
    os.makedirs(root, exist_ok=True)
    applied = vendor(root)
    shutil.copyfile(os.path.join(TEMPLATE, WRAPPER), os.path.join(root, WRAPPER))
    graph, missing = {}, []
    for f in sorted(CODE):
        graph[f] = relative_imports(os.path.join(root, f))
        missing += [f"{f}: .{n}" for n in graph[f] if not os.path.isfile(os.path.join(root, n + ".py"))]
    import geolip.alephllm as A
    return gate(os.path.join(gates, "_code"), "code", not missing, missing_siblings=missing, import_graph=graph, patches=applied,
                library=A.__version__, sha256=hashes(root, CODE))


def identity(craft: str, man: dict) -> dict:
    """0.1.43: what the packaged model is, for its config: the craft's name, its body and its arm in plain words, what it continued from,
    the step its own training started at, the phase its weights ended on, the step and the bytes seen. The names come from nano_v1's
    table when the craft is in it; a craft outside the table carries what its manifest says."""
    from .nano_v1 import ALL, TOKENS_PER_STEP, arm_text, body_of, start_step
    done = [p["name"] for p in (man.get("phases") or []) if p.get("status") == "done"]
    d = dict(name=craft, body=craft, body_text=None, arm=None, source=None, start_step=None, step=int(man["steps"]),
             tokens=int(man.get("tokens_seen") or 0), phase=(done[-1] if done else None))
    if craft in ALL:
        d.update(body=body_of(craft), body_text=ALL[body_of(craft)]["body"], arm=arm_text(craft), source=ALL[craft].get("source"),
                 start_step=start_step(craft, TOKENS_PER_STEP))
    return d


def activated(mc: dict) -> dict:
    """0.1.43: the mechanisms a body switches on, in plain words, from its model config: which blocks carry splat hubs (dual, with rotary
    codes), what the other blocks carry, the Canon layers, the expert bank and its byte table, the last block's bound, the head's mode,
    the position table."""
    n, hubs = int(mc.get("n_layers") or 0), [int(i) for i in (mc.get("hub_layers") or [])]
    soft = [i for i in range(n) if i not in hubs]
    softmax = "rotary softmax" if mc.get("attn_rotary") else "softmax with a position table"
    blocks = lambda xs: ", ".join(str(i) for i in xs)
    if hubs:
        attention = (f"splat hubs in blocks {blocks(hubs)}" + (" (dual)" if mc.get("hub_dual") else "") + (", rotary codes" if mc.get("hub_rotary") else "")
                     + (f"; {softmax} in blocks {blocks(soft)}" if soft else ""))
    else:
        attention = f"{softmax} in every block"
    return dict(attention=attention, hub_blocks=hubs, hub_dual=bool(mc.get("hub_dual")), hub_rotary=bool(mc.get("hub_rotary")),
                softmax_blocks=soft, softmax=(softmax if soft else None), canon=(mc.get("canon") or None),
                bank_experts=mc.get("bank_experts"), byte_table=(mc.get("bank_byte_table") or None), last_fix=(mc.get("bank_last_fix") or None),
                head_mode=(mc.get("head_mode") or None), position_table=bool(mc.get("pos_table")))


def stage_config(craft: str, manifest: str, root: str, gates: str) -> dict:
    """The craft's config.json from its run's manifest (0.1.43: with `craft` and `activated`, what the model is), and the code copied
    into its folder byte for byte."""
    out = os.path.join(root, craft)
    os.makedirs(out, exist_ok=True)
    man = json.load(open(manifest, encoding="utf-8"))
    mc = dict(man["model_config"])
    assert mc["name"] == craft and mc.get("fusion") is None, (mc["name"], craft)
    cfg = dict(mc, hub_ckpt=0)
    cfg.update(craft=identity(craft, man), activated=activated(mc))   # 0.1.43: a loaded model says what it is
    cfg.update(model_type="nano-beatrix", architectures=["NanoBeatrixForCausalLM"],
               auto_map={"AutoConfig": "modeling_nano_beatrix.NanoBeatrixConfig",
                         "AutoModelForCausalLM": "modeling_nano_beatrix.NanoBeatrixForCausalLM"},
               torch_dtype="float32", use_cache=False)        # the bf16 file loads exactly into fp32 parameters
    json.dump(cfg, open(os.path.join(out, "config.json"), "w", encoding="utf-8", newline="\n"), indent=1)
    for f in CODE:
        shutil.copyfile(os.path.join(root, f), os.path.join(out, f))
    same = hashes(out, CODE) == hashes(root, CODE)
    return gate(os.path.join(gates, craft), "config", same, config=cfg, code_equals_the_top=same,
                sha256=hashes(out, CODE + ["config.json"]))


def final_files(man: dict) -> tuple:
    """(step, the final bf16 checkpoint, the final resume state, its fallback) as paths inside the craft's run folder. The resume state
    is the boundary of the last phase done (anneal_mix for a body; the continuation phase for an arm, 0.1.43), else latest.pt."""
    step = int(man["steps"])
    done = [p["name"] for p in (man.get("phases") or []) if p.get("status") == "done"]
    last = done[-1] if done else "anneal_mix"
    return (step, f"checkpoints/step_{step:08d}.safetensors", f"resume/boundary_{last}_step{step}.pt", "resume/latest.pt")


def stage_weights(craft: str, run_dir: str, root: str, gates: str, require_done: bool = True) -> dict:
    import torch
    from safetensors import safe_open
    from safetensors.torch import load_file, save_file
    out = os.path.join(root, craft)
    man = json.load(open(os.path.join(run_dir, "manifest.json"), encoding="utf-8"))
    step, ck, rs, rs_alt = final_files(man)
    ck, rs = os.path.join(run_dir, ck), os.path.join(run_dir, rs if os.path.exists(os.path.join(run_dir, rs)) else rs_alt)
    if require_done:
        undone = [p["name"] for p in man["phases"] if p["status"] != "done"]
        assert not undone, f"{craft}: the diet is not finished ({undone})"
    with safe_open(ck, "pt") as fh:
        meta = fh.metadata() or {}
    assert meta.get("step") == str(step) and meta.get("format") == "bf16", meta
    sd = load_file(ck)
    assert all(v.dtype == torch.bfloat16 for v in sd.values() if v.is_floating_point())
    common = {"format": "pt", "craft": craft, "step": str(step), "tokens": str(int(man["tokens_seen"]))}
    save_file({"model." + k: v for k, v in sd.items()}, os.path.join(out, "model.safetensors"), metadata=dict(common, dtype="bfloat16"))
    payload = torch.load(rs, map_location="cpu", weights_only=False)
    assert int(payload["step"]) == step, (payload["step"], step)
    m = {k: v.detach().cpu().contiguous() for k, v in payload["model"].items()}
    assert set(m) == set(sd), sorted(set(m) ^ set(sd))[:8]
    assert all(v.dtype == torch.float32 for v in m.values() if v.is_floating_point())
    differ = [k for k, v in m.items() if v.is_floating_point()
              and not torch.equal(v.to(torch.bfloat16).reshape(-1).view(torch.int16), sd[k].reshape(-1).view(torch.int16))]
    save_file({"model." + k: v for k, v in m.items()}, os.path.join(out, "model.fp32.safetensors"), metadata=dict(common, dtype="float32"))
    return gate(os.path.join(gates, craft), "weights", not differ, step=step, tokens=int(man["tokens_seen"]), tensors=len(sd),
                elements=sum(v.numel() for v in sd.values()), differing=differ[:20], source_checkpoint=ck, source_resume=rs,
                sha256=hashes(out, [w[1] for w in WEIGHTS]))


def fixed_inputs(context: int = 4096) -> dict:
    import torch
    g = torch.Generator().manual_seed(0)
    return {"a": torch.randint(0, 256, (2, 96), generator=g), "b": torch.randint(0, 256, (1, min(700, context)), generator=g)}


def stage_parity(craft: str, root: str, work: str, gates: str) -> dict:
    import importlib
    import torch
    from safetensors.torch import load_file
    from geolip.alephllm.model.alephlm import AlephLM as LibAlephLM
    from geolip.alephllm.presets import AlephLMConfig as LibConfig
    out = os.path.join(root, craft)
    os.makedirs(work, exist_ok=True)
    pkg = "vend_" + re.sub(r"\W", "_", craft)
    pdir = os.path.join(work, pkg)
    shutil.rmtree(pdir, ignore_errors=True)
    os.makedirs(pdir)
    open(os.path.join(pdir, "__init__.py"), "w").close()
    for f in VENDORED:
        shutil.copyfile(os.path.join(out, f), os.path.join(pdir, f))
    for m in [m for m in sys.modules if m == pkg or m.startswith(pkg + ".")]:
        del sys.modules[m]                              # a second packaging in this process reads its own copy
    sys.path.insert(0, work)
    try:
        V = importlib.import_module(f"{pkg}.alephlm")
        VP = importlib.import_module(f"{pkg}.presets")
    finally:
        sys.path.remove(work)
    pc = json.load(open(os.path.join(out, "config.json"), encoding="utf-8"))
    kw = {k: pc[k] for k in LibConfig().to_dict()}
    cfg_v, cfg_l = VP.AlephLMConfig.from_dict(kw), LibConfig.from_dict(kw)
    norm = lambda d: json.loads(json.dumps(d))          # tuples and lists compare equal
    config_ok = norm(cfg_v.to_dict()) == norm(cfg_l.to_dict())
    xs = fixed_inputs(int(pc["context"]))
    torch.save(xs, os.path.join(work, "ref_inputs.pt"))
    facts, ok = {}, config_ok
    for tag, fname, _ in WEIGHTS:
        sd = {k[len("model."):]: (v.float() if v.is_floating_point() else v) for k, v in load_file(os.path.join(out, fname)).items()}
        mv, ml = V.AlephLM(cfg_v), LibAlephLM(cfg_l)
        mv.load_state_dict(sd, strict=True)
        ml.load_state_dict(sd, strict=True)
        mv.eval()
        ml.eval()
        worst = 0.0
        with torch.no_grad():
            for name, x in xs.items():
                a, b = mv(x)[0].float(), ml(x)[0].float()
                assert torch.isfinite(b).all()
                worst = max(worst, float((a - b).abs().max()))
                torch.save(b, os.path.join(work, f"ref_logits_{tag}_{name}.pt"))
        facts[tag] = {"file": fname, "vendored_vs_library_max_abs": worst, "parameters": sum(p.numel() for p in ml.parameters())}
        log(f"{craft} {tag}: vendored against the library, max abs logit difference {worst} (must be 0.0)")
        ok = ok and worst == 0.0
    return gate(os.path.join(gates, craft), "parity", ok, config_equal=config_ok, files=facts, sha256=hashes(out, PACKAGE))


def _load_once(src: str, subfolder: str | None, variant: str | None, from_hub: bool):
    from transformers import AutoModelForCausalLM
    kw = dict(trust_remote_code=True, output_loading_info=True)
    if variant:
        kw["variant"] = variant
    if subfolder:
        kw["subfolder"] = subfolder
    if from_hub:
        kw["token"] = False
    return AutoModelForCausalLM.from_pretrained(src, **kw)


def stage_load(craft: str, root: str, work: str, gates: str, from_hub: bool = False, tf_pkgs: str | None = None) -> dict:
    """The load gate. In THIS process the library must be unimportable and the caches fresh before transformers loads: it runs in a
    child (load_child). root: the package root folder, or the repo id when from_hub; the loaded tensors must equal the gated copy of
    the craft's package (<work>/files)."""
    for name in ("geolip", "amoe"):
        sys.modules[name] = None
    fresh = os.path.join(work, "fresh_hub" if from_hub else "fresh_local")
    shutil.rmtree(fresh, ignore_errors=True)
    os.environ["HF_HOME"] = os.path.join(fresh, "hf_home")
    os.environ["HF_MODULES_CACHE"] = os.path.join(fresh, "hf_modules")
    os.environ.pop("HF_TOKEN", None)
    if not from_hub:
        os.environ["HF_HUB_OFFLINE"] = "1"
    if tf_pkgs and os.path.isdir(tf_pkgs):
        sys.path.append(tf_pkgs)                       # last: the environment's own packages win
    import torch
    import transformers
    from safetensors.torch import load_file
    files = os.path.join(work, "files")
    xs = torch.load(os.path.join(work, "ref_inputs.pt"))
    routes = [("top + subfolder", root, craft)] + ([] if from_hub else [("craft folder", os.path.join(root, craft), None)])
    facts, ok = {"transformers": transformers.__version__, "torch": torch.__version__, "source": root, "routes": {}}, True
    for route, src, sub in routes:
        for tag, fname, variant in WEIGHTS:
            model, info = _load_once(src, sub, variant, from_hub)
            model.eval()
            problems = {k: (sorted(v) if isinstance(v, (set, list, tuple)) else v) for k, v in info.items() if v}
            ref = load_file(os.path.join(files, fname))
            got = model.state_dict()
            tensors_ok = set(got) == set(ref) and all(torch.equal(got[k].float(), ref[k].float()) for k in ref)
            worst = 0.0
            with torch.no_grad():
                for name, x in xs.items():
                    lg = model(input_ids=x).logits.float()
                    worst = max(worst, float((lg - torch.load(os.path.join(work, f"ref_logits_{tag}_{name}.pt"))).abs().max()))
                ids = torch.tensor([list(PROMPT.encode("utf-8"))])
                gen = model.generate(ids, max_new_tokens=32, do_sample=False)
            text = bytes(int(i) for i in gen[0]).decode("utf-8", errors="replace")
            lib = sorted(m for m in sys.modules if m.split(".")[0] in ("geolip", "amoe") and sys.modules[m] is not None)
            good = not problems and tensors_ok and worst <= 1e-5 and not lib and type(model).__name__ == "NanoBeatrixForCausalLM"
            facts["routes"][f"{route} / {tag}"] = {
                "file": fname, "loading_problems": problems, "every_tensor_equals_the_file": tensors_ok,
                "max_abs_vs_parity_logits": worst, "library_modules_loaded": lib, "greedy_32_bytes": text,
                "param_dtypes": sorted({str(p.dtype) for p in model.parameters()})}
            log(f"{craft} [{route} / {tag}] transformers {transformers.__version__}; problems {problems or 'none'}; tensors equal "
                f"{tensors_ok}; max abs against the parity logits {worst}; greedy {text!r}")
            ok = ok and good
            del model
    return gate(os.path.join(gates, craft), "hubload" if from_hub else "load", ok, **facts, sha256=hashes(files, PACKAGE))


def load_child(craft: str, root: str, work: str, gates: str, tf_pkgs: str | None = None, from_hub: bool = False) -> None:
    """The load gate in a fresh interpreter that runs this file as a script (never the package: the library stays unimported)."""
    cmd = [sys.executable, os.path.abspath(__file__), "load", craft, root, work, gates, tf_pkgs or "", "hub" if from_hub else ""]
    env = dict(os.environ)
    env.pop("HF_TOKEN", None)
    rc = subprocess.call(cmd, env=env)
    g = "hubload" if from_hub else "load"
    assert rc == 0, f"the {g} gate of {craft} failed (exit {rc}): {os.path.join(gates, craft, g + '.json')}"


def package(crafts: list, runs: str, root: str, tf_pkgs: str | None = None, require_done: bool = True) -> str:
    """Every craft's package under <root>/<craft>/ with the code at <root>'s top, through every gate. runs: the folder holding each
    craft's run folder (<runs>/<craft>/manifest.json, checkpoints/, resume/). Gates under <root>/gates/. -> root."""
    gates = os.path.join(root, "gates")
    stage_code(root, gates)
    for c in crafts:
        work = os.path.join(root, "work", c)
        os.makedirs(work, exist_ok=True)
        stage_config(c, os.path.join(runs, c, "manifest.json"), root, gates)
        stage_weights(c, os.path.join(runs, c), root, gates, require_done)
        stage_parity(c, root, work, gates)
        files = os.path.join(work, "files")
        shutil.rmtree(files, ignore_errors=True)
        os.makedirs(files)
        for f in PACKAGE:
            shutil.copyfile(os.path.join(root, c, f), os.path.join(files, f))
        load_child(c, root, work, gates, tf_pkgs)
    return root


def check_gated(root: str, crafts: list) -> list:
    """The upload list [(path in the repo, local file)], refused unless every gate passed on exactly these bytes."""
    gates = os.path.join(root, "gates")
    rec = json.load(open(os.path.join(gates, "_code", "code.json"), encoding="utf-8"))
    assert rec["pass"], "the code gate did not pass"
    stale = [f for f, h in rec["sha256"].items() if sha256(os.path.join(root, f)) != h]
    assert not stale, f"the code gate was run on other bytes: {stale}"
    files = [(f, os.path.join(root, f)) for f in CODE]
    for c in crafts:
        out = os.path.join(root, c)
        now = hashes(out, PACKAGE)
        for g in ("config", "weights", "parity", "load"):
            r = json.load(open(os.path.join(gates, c, f"{g}.json"), encoding="utf-8"))
            assert r["pass"], f"{c}: the gate {g} did not pass"
            stale = [f for f, h in (r.get("sha256") or {}).items() if now.get(f) != h]
            assert not stale, f"{c}: the gate {g} was run on other bytes: {stale}"
        files += [(f"{c}/{f}", os.path.join(out, f)) for f in PACKAGE]
    for top in ("README.md", INDEX):                  # the card and the index (0.1.43) ride along when written
        p = os.path.join(root, top)
        if os.path.exists(p):
            files.append((top, p))
    return files


# ------------------------------------------------------------------ the repo
def fetch_finals(repo: str, crafts: list, runs: str, token: str | None = None) -> None:
    """Each craft's manifest, final bf16 checkpoint and final resume state from the repo into <runs>/<craft>/ (a fresh machine)."""
    from huggingface_hub import errors as hf_errors, hf_hub_download
    missing = getattr(hf_errors, "RemoteEntryNotFoundError", hf_errors.EntryNotFoundError)
    for c in crafts:
        def fetch(path, required=True):                   # lands at <runs>/<craft>/<path>
            try:
                hf_hub_download(repo, f"{c}/{path}", token=token, local_dir=runs, force_download=path == "manifest.json")
            except missing:
                if required:
                    raise
                return False
            return True
        fetch("manifest.json")
        man = json.load(open(os.path.join(runs, c, "manifest.json"), encoding="utf-8"))
        step, ck, rs, rs_alt = final_files(man)
        fetch(ck)
        if not fetch(rs, required=False):
            fetch(rs_alt)
        log(f"{c}: the final files of step {step:,} fetched from {repo}")


ARM_WHAT = {"fr": "the diet's text with keyed records written in small frames, and questions that ask a record back",
            "ct": "the diet's text alone, the same steps: the comparison for the other arms",
            "cf": "framed records re-written along the text as carries, with the question far after: the body learns to re-write a record on its way to a distant question",
            "nt": "notes in real words taken from the text itself (a definition, an abbreviation, a quantity, a capital, a place, a year), re-written on the way and asked back"}
ARM_COLS = (("the bed's held-out bits a byte", "bed/text/heldout_bits", 4), ("copy / nested / repeated key", None, 2),
            ("1,024-byte frame", "bed/scaling/frame_1024_whole", 2), ("far question, own carries", "bed/carry/full_value", 2),
            ("carries a row", "bed/carry/carries_per_row", 1))
NOTE_COLS = (("note given, 2,048+ bytes", "bed/note_planted/far_ge2048_full_value", 2),
             ("own notes: rows whose note held the key", "bed/note_self/rows_with_key", 2),
             ("own notes: answered at 2,048+ bytes", "bed/note_self/far_ge2048_full_value", 2), ("own notes a row", "bed/note_self/carries_per_row", 1))


def end_of(hist: dict, n: str, reads: dict | None, k: str):
    """0.1.43: a craft's read at its end: the end gauge record's scalar when `reads` carries it (a later protocol's re-read of the end
    checkpoint wins), else the last boundary row's bed value; val_bpb from the row itself."""
    e = (hist.get(n) or [{}])[-1]
    if k == "val_bpb":
        return e.get("val_bpb")
    if reads and n in reads and reads[n].get(k) is not None:
        return reads[n][k]
    return (e.get("bed") or {}).get(k)


def arms_section(hist: dict, params: dict, reads: dict | None = None) -> str:
    """0.1.43: the card's continuation arms, body by body: what each arm trains on, then a table a body (the body at the diet's end and
    each of its arms at its end: the folder, the steps it trained, the bed's reads; the note arm's own columns when one is packaged).
    Empty when no packaged craft is an arm."""
    from .nano_v1 import ALL, ARM_TEXT, CONT_TOKENS, TOKENS_PER_STEP, arm_text, body_of, plan_steps, start_step

    def f(x, nd):
        return "-" if x is None else f"{x:.{nd}f}"
    arms = [n for n in ALL if n in params and ALL[n].get("source")]
    if not arms:
        return ""
    kinds = []
    for n in arms:
        if ALL[n]["cont"] not in kinds:
            kinds.append(ALL[n]["cont"])
    bodies = [b for b in ALL if not ALL[b].get("source") and any(body_of(n) == b for n in arms)]
    notes = any(end_of(hist, n, reads, "bed/note_self/rows_with_key") is not None for n in arms)
    what = "\n".join(f"| {ARM_TEXT[k]} | `-{k}` | {ARM_WHAT[k]} |" for k in kinds)
    out = [f"""## The continuation arms

Each body's diet ended at step {plan_steps(bodies[0], TOKENS_PER_STEP):,}. From there an arm takes the same body {math.ceil(CONT_TOKENS / TOKENS_PER_STEP):,} steps further
({CONT_TOKENS / 1e9:.1f} billion bytes, the same optimizer settings) on a mix that teaches it to write and read its own notes; a second-stage arm
continues an arm's end the same way (its folder suffix chains the two: `-fr-cf` is frames, then carry). Every arm folder is a model like
the four above, with its own training record, and loads the same way.

| arm | folder suffix | what it trains on |
|---|---|---|
{what}

One table a body: its folders, the steps each trained, and the nano bed's checks at each end on the same 192 held-out rows. *The bed's
held-out bits a byte* is the bed's own fineweb-edu read (not the column of the first table). *Far question, own carries*: a keyed record
early in the window and the question 2,048 bytes or more after it, the body free to re-write the record on the way; the share of rows
answered whole, and *carries a row* how many re-writes it made. A dash: the read does not apply to that folder."""]
    if notes:
        out.append("""The note arm's own reads: *note given, 2,048+ bytes* is a real-word note from the text planted after its fact with the question 2,048
bytes or more after, the share answered whole; *own notes* are the same rows with no note planted: the share of rows whose own written
note held the asked key, the share answered whole at 2,048 bytes or more, and the notes it wrote a row.""")
    hdr = "| folder | trained | " + " | ".join(c[0] for c in ARM_COLS) + (" | " + " | ".join(c[0] for c in NOTE_COLS) if notes else "") + " |"
    sep = "|" + "---|" * (2 + len(ARM_COLS) + (len(NOTE_COLS) if notes else 0))
    for body in bodies:
        rows = [hdr, sep]
        for n in [body] + [a for a in arms if body_of(a) == body]:
            if n not in params:
                continue
            end = (hist.get(n) or [{}])[-1].get("step")
            if end is None:
                trained = "-"
            elif n == body:
                trained = f"0 to {end:,} (the diet)"
            else:
                trained = f"{arm_text(n)}: {start_step(n):,} to {end:,}" + ("" if ALL[n]["source"] == body else f", from `{ALL[n]['source']}`")
            cells = []
            for label, k, nd in ARM_COLS:
                if k is None:
                    cells.append(" / ".join(f(end_of(hist, n, reads, kk), nd)
                                            for kk in ("bed/frame/copy_whole", "bed/frame/nested_whole", "bed/frame/redefined_key_whole")))
                else:
                    cells.append(f(end_of(hist, n, reads, k), nd))
            if notes:
                cells += [f(end_of(hist, n, reads, k), nd) for _, k, nd in NOTE_COLS]
            rows.append(f"| [`{n}`](./{n}) | {trained} | " + " | ".join(cells) + " |")
        out.append(f"### {ALL[body]['body']}\n\n" + "\n".join(rows))
    return "\n\n".join(out)


def index(hist: dict, params: dict, reads: dict | None = None) -> dict:
    """0.1.43: the repo's index of its packaged crafts (crafts.json at the top): folder -> what it is (body, arm, kind, source, start and
    end step, phase, bytes, parameters) and its end reads, the facts the card's tables are rendered from."""
    from .nano_v1 import ALL, arm_text, body_of, start_step
    out = {}
    for n in ALL:
        if n not in params:
            continue
        e = (hist.get(n) or [{}])[-1]
        keys = sorted(set(e.get("bed") or {}) | set((reads or {}).get(n) or {}))
        out[n] = dict(body=body_of(n), body_text=ALL[body_of(n)]["body"], arm=arm_text(n), kind=ALL[n].get("cont"), source=ALL[n].get("source"),
                      start_step=start_step(n), step=e.get("step"), tokens=e.get("tokens"), phase=e.get("phase"), val_bpb=e.get("val_bpb"),
                      parameters=params[n], reads={k: end_of(hist, n, reads, k) for k in keys})
    return out


def card(repo: str, hist: dict, params: dict, reads: dict | None = None) -> str:
    """The repo's card once the crafts are packages: what each craft is, its final reads, how to load it (the version from the crafts:
    a v1.1 card states the readout's recipe and that the v1 folders stay); 0.1.43: the continuation arms body by body (arms_section),
    `reads` the end gauge records' scalars a craft (end_of)."""
    from .nano_v1 import ALL as CRAFTS, V11_NOTE, version_of
    ver = version_of(list(params))
    note = V11_NOTE if ver == "v1.1" else ""
    recipe = (" The output layer's two readout matrices trained with AdamW (0.0003, decoupled weight decay 0.1) instead of Muon."
              if ver == "v1.1" else "")
    nb = "colab/nano_v1_1.ipynb" if ver == "v1.1" else "colab/nano_v1_2s.ipynb"

    def f(x, nd):
        return "-" if x is None else f"{x:.{nd}f}"
    rows = []
    for n, c in CRAFTS.items():
        if n not in params or c.get("source"):             # the first table: the bodies; the arms have their own section
            continue
        e = (hist.get(n) or [{}])[-1]
        b = e.get("bed") or {}
        rows.append(f"| [`{n}`](./{n}) | {c['body']} | {params[n]:,} | {f(e.get('val_bpb'), 4)} | "
                    f"{f(b.get('bed/frame/copy_whole'), 2)} / {f(b.get('bed/frame/nested_whole'), 2)} / "
                    f"{f(b.get('bed/frame/redefined_key_whole'), 2)} | {f(b.get('bed/scaling/frame_1024_whole'), 2)} | "
                    f"{f(b.get('bed/window/planted_2048_bits'), 2)} / {f(b.get('bed/window/planted_3584_bits'), 2)} |")
    first = next((n for n in CRAFTS if n in params and not CRAFTS[n].get("source")), next(iter(params)))
    arms = arms_section(hist, params, reads)
    arms_md = f"\n{arms}\n" if arms else ""
    arms_intro = ("\n\nBeside each body sit its continuation arms (below): the same body taken further on a mix that teaches it to write and read its own\n"
                  "notes. Every folder is a model with its full training record, and every folder loads the same way." if arms else "")
    arm_record = (" An arm folder keeps the same record for its own steps: checkpoints at its two mid gauges and its end (an fp8 copy of the end),\n"
                  "the end's resume state, the gauge records of its start state and its reads, `manifest.json`, `runs/`, `v1_boundaries.json`." if arms else "")
    nb_more = (", the arm notebooks `colab/nano_v1_1_frames.ipynb`, `nano_v1_1_carry.ipynb`, `nano_v1_1_frames_carry.ipynb`,\n"
               "`nano_v1_1_notes.ipynb`, and `colab/nano_v1_1_release.ipynb`, the packaging" if arms else "")
    says = (" `config.json` says what the model is (`craft`: its body, its arm, what it continued from, the phase and step its weights\n"
            "ended on; `activated`: the mechanisms its body switches on), and `crafts.json` at the top of the repo indexes every packaged\n"
            "folder with its end reads: the tables above are rendered from it.")
    return f"""---
license: cc-by-nc-sa-4.0
language: en
library_name: transformers
pipeline_tag: text-generation
tags:
- byte-level
- aleph
- splat-attention
- custom_code
---

# AlephLLM nano: Castor and Pollux, {ver}

Four small byte-level AlephLLM models (about 9 million parameters each, a 4,096-byte window), trained from birth on Beatrix 2s's
diet: 16.1 billion bytes in thirteen phases (wikitext-103, fineweb-edu, nine curriculum stages, two anneals), 61,422 steps each.
They read and write raw UTF-8 bytes: no tokenizer.{note}{arms_intro}

| model | the body | parameters | bits a byte (fineweb-edu holdout) | recall: copy / nested / repeated key | 1,024-byte frame | planted record at 2,048 / 3,584 bytes (bits) |
|---|---|---|---|---|---|---|
{chr(10).join(rows)}

The recall columns are the nano bed's checks at the end of the train: a keyed record asked back whole (copy, a nested question,
a key that was redefined), the same read on 1,024-byte frames, and the bits needed to recall a planted record from 2,048 and 3,584
bytes away (lower is better).
{arms_md}
## Use

```python
import torch
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("{repo}", subfolder="{first}",
                                             trust_remote_code=True)   # variant="fp32" for the fp32 weights
ids = torch.tensor([list("The history of astronomy".encode("utf-8"))])
out = model.generate(ids, max_new_tokens=64, do_sample=False)
print(bytes(out[0].tolist()).decode("utf-8", errors="replace"))
```

The model code sits at the top of the repo (transformers loads remote code from there) and again in each model's folder, so a
folder downloaded alone loads with `from_pretrained(<folder>, trust_remote_code=True)`. Each folder holds `model.safetensors` (bf16)
and `model.fp32.safetensors` (the fp32 weights the training ended with); the bf16 file is the fp32 file rounded. Generation re-reads
the whole window each step (no cache).{says}

## Training

The recipe is Beatrix 2s's: Muon at 0.02 on the trunk's matrices, Adam at 0.0003 on the rest, 200 warmup steps then a flat rate,
gradient clip 1.0, 262,144 bytes a step.{recipe} Each model started from exactly the weights its strongest run on the nano bed started from.
Each folder keeps the whole training record: `checkpoints/`, `resume/`, `manifest.json`, `runs/` (TensorBoard: the training curves,
the health census, the evals with bits by position, and the bed's checks at intervals under `bed/`), `bed_gauges/` and
`v1_boundaries.json` (one row a phase).{arm_record} The trainer: [geolip-nano-beatrix-trainer](https://github.com/AbstractEyes/geolip-nano-beatrix-trainer)
(`nano_v1.py`, `automodel.py`, `{nb}`{nb_more}) on [AlephLLM](https://github.com/AbstractEyes/alephllm). The license follows
the training data's share-alike and non-commercial terms, as Beatrix 2s's does.
"""


def repo_crafts(repo: str, token=None) -> list:
    """0.1.43: the craft folders the repo holds (a top-level folder with a manifest.json that nano_v1's table knows), in the table's order."""
    from huggingface_hub import HfApi
    from .nano_v1 import ALL
    paths = [f.path for f in HfApi(token=token).list_repo_tree(repo, recursive=True)]
    have = {p.split("/")[0] for p in paths if p.count("/") == 1 and p.endswith("/manifest.json")}
    return [n for n in ALL if n in have]


def end_reads(repo: str, craft: str, step: int, token=None) -> dict:
    """0.1.43: the scalars of a craft's gauge records at `step` in the repo (bed_gauges/, and any regauge_*/ folder: a later protocol's
    re-read of the same checkpoint, applied last so it wins), through the trainer's own scalar reader; {} when the step has no record."""
    from huggingface_hub import HfApi, hf_hub_download
    from .nano_v1 import gauge_scalars
    paths = [f.path for f in HfApi(token=token).list_repo_tree(repo, path_in_repo=craft, recursive=True)]
    recs = [p for p in paths if p.count("/") == 2 and p.split("/")[1].startswith(("bed_gauges", "regauge_")) and p.endswith("_gauge.json")
            and "_from_" in p and f"_{step}_" in p.split("_from_")[-1]]
    recs.sort(key=lambda p: ("/regauge_" in p, p))
    out = {}
    for p in recs:
        sc = gauge_scalars(json.load(open(hf_hub_download(repo, p, token=token), encoding="utf-8")))
        out.update({k: v for k, v in sc.items() if v is not None})
    return out


def release(out: str, repo: str, crafts: list | None, token: str, tf_pkgs: str | None = None, fetch: bool = True) -> str:
    """The end of the train: each craft's final files (fetched from the repo unless fetch=False), the packages through every gate, the
    index and the card from the repo's boundary tables and end gauge records, one commit, then the load gate again from the repo.
    crafts=None: every craft folder the repo holds (0.1.43: the bodies and their continuation arms). -> the commit id."""
    from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download
    from .nano_v1 import ALL, version_of
    crafts = list(crafts) if crafts else repo_crafts(repo, token)
    assert crafts, f"{repo}: no craft folder to package"
    runs, root = os.path.join(out, "runs"), os.path.join(out, "automodel")
    if fetch:
        fetch_finals(repo, crafts, runs, token)
    package(crafts, runs, root, tf_pkgs)
    hist, params, reads = {}, {}, {}
    for c in crafts:
        hist[c] = json.load(open(hf_hub_download(repo, f"{c}/v1_boundaries.json", token=token, force_download=True), encoding="utf-8"))
        params[c] = json.load(open(os.path.join(root, "gates", c, "parity.json"), encoding="utf-8"))["files"]["bf16"]["parameters"]
        reads[c] = end_reads(repo, c, int(hist[c][-1]["step"]), token) if hist[c] else {}
    json.dump(index(hist, params, reads), open(os.path.join(root, INDEX), "w", encoding="utf-8", newline="\n"), indent=1)
    open(os.path.join(root, "README.md"), "w", encoding="utf-8", newline="\n").write(card(repo, hist, params, reads))
    ops = [CommitOperationAdd(path_in_repo=p, path_or_fileobj=f) for p, f in check_gated(root, crafts)]
    arms = " and their continuation arms" if any(ALL[c].get("source") for c in crafts if c in ALL) else ""
    info = HfApi(token=token).create_commit(repo_id=repo, operations=ops,
                                            commit_message=f"the nano {version_of(crafts)} models{arms} as AutoModel packages (bf16 default, fp32 variant)")
    log(f"pushed {len(ops)} files to {repo}: {info.oid}")
    for c in crafts:
        load_child(c, repo, os.path.join(root, "work", c), os.path.join(root, "gates"), tf_pkgs, from_hub=True)
    return info.oid


if __name__ == "__main__":
    # the load gate's child: python automodel.py load <craft> <root> <work> <gates> [tf_pkgs] [hub]
    if sys.path and os.path.abspath(sys.path[0]) == os.path.dirname(os.path.abspath(__file__)):
        sys.path.pop(0)                                # the package's folder is not a place to import from
    a = sys.argv[1:]
    if a and a[0] == "load":
        stage_load(a[1], a[2], a[3], a[4], tf_pkgs=(a[5] or None) if len(a) > 5 else None, from_hub=len(a) > 6 and a[6] == "hub")
