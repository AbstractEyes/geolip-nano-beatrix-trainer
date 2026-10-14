# -*- coding: utf-8 -*-
"""The runner's Python entry point: the progress-line parser, the run labels, a dry grid from a dict spec, the resume file."""
import os
import sys

from geolip.nano_beatrix.bed_runner import expand, parse_step, run, run_label, run_key
from geolip.nano_beatrix import bed_runner as br

LINE = ("[23:17:22 +    25s] step 201/12000 loss 1.849 ema 2.049 (2.957 bpb) gn 0.43 lr 1.00e-03 | 294k tok/s | spent 0.4 min, "
        "left 21.9 min | value bits d0+:5.68 d128+:5.65 d256+:5.67 d512+:5.68 w:-")


def test_parse_step_reads_the_trainer_line():
    st = parse_step(LINE)
    assert st == {"step": 201, "total": 12000, "bpb": 2.957, "tok_s": 294, "spent": 0.4, "left": 21.9}


def test_parse_step_ignores_other_lines():
    assert parse_step("[23:38:11 +  1274s] training done in 21.2 min") is None
    assert parse_step("held-out 1.7712 bits a byte") is None


def test_run_label_names_arm_seed_and_register():
    assert run_label(["--arm=castor-both", "--seed=1", "--frames=dyn", "--register=math"]) == "castor-both s1 math"
    assert run_label(["--arm=smr-fixed", "--seed=2", "--ctx=4096"]) == "smr-fixed s2"
    assert run_label(["--arm=castor-both", "--seed=1", "--frames=dyn", "--register=caption", "--frame_records=8"]) == "castor-both s1 caption r8"
    assert run_label(["--arm=pollux-both", "--seed=2", "--frames=dyn", "--nest=0.6", "--overlap=0.3", "--ask=nested"]) == "pollux-both s2 caption nested n0.6 o0.3"
    assert run_label(["--arm=castor-both", "--seed=1", "--frames=dyn", "--facts=64", "--ask=weights"]) == "castor-both s1 caption weights f64"


def test_dry_run_from_a_dict_spec(tmp_path):
    spec = {"arms": "castor-both,pollux-both", "seeds": "1,2", "register": "caption,math,code", "ctx": 4096, "mb": 8, "steps": 12000,
            "frames": "dyn", "sizes": "32,128,256,512"}
    res = run(spec, str(tmp_path), dry=True)
    assert res["runs"] == 12 and res["done_before"] == 0 and res["ran"] == [] and res["failed"] is None
    assert os.path.exists(os.path.join(tmp_path, "runner.log"))
    runs = expand(spec)
    assert len({run_key(r) for r in runs}) == 12 and all("--sizes=32,128,256,512" in r for r in runs)


def test_resume_skips_the_finished_runs(tmp_path):
    spec = {"arms": "castor-both,pollux-both", "seeds": "1", "ctx": 1024, "steps": 200}
    runs = expand(spec)
    with open(os.path.join(tmp_path, "runner_done.txt"), "w", encoding="utf-8") as f:
        f.write(run_key(runs[0]) + "\n")
    res = run(spec, str(tmp_path), dry=True)
    assert res["runs"] == 2 and res["done_before"] == 1


def test_parallel_is_a_runner_key_and_the_dry_plan_ignores_it(tmp_path):
    spec = {"arms": "castor-both,pollux-both", "seeds": "1,2", "ctx": 4096, "steps": 12000, "parallel": 4}
    assert len(expand(spec)) == 4 and all("--parallel" not in a for r in expand(spec) for a in r)
    res = run(spec, str(tmp_path), dry=True, parallel=4)
    assert res["runs"] == 4 and res["failed"] is None
    assert "4 at a time" in open(os.path.join(tmp_path, "runner.log"), encoding="utf-8").read()


class _FakeApi:
    """A Hub that fails `fails` times, then takes the uploads."""
    calls = []
    fails = 0

    def __init__(self, token=None):
        self.token = token

    def upload_file(self, **kw):
        _FakeApi.calls.append(("file", kw["path_in_repo"]))
        if _FakeApi.fails > 0:
            _FakeApi.fails -= 1
            raise RuntimeError("504 Gateway Time-out")

    def upload_folder(self, **kw):
        _FakeApi.calls.append(("folder", kw["path_in_repo"]))


def _fake_hub(monkeypatch, fails):
    import types
    mod = types.ModuleType("huggingface_hub")
    mod.HfApi = _FakeApi
    _FakeApi.calls, _FakeApi.fails = [], fails
    monkeypatch.setitem(sys.modules, "huggingface_hub", mod)
    monkeypatch.setenv("HF_TOKEN", "x")
    waits = []
    monkeypatch.setattr(br.time, "sleep", lambda s: waits.append(s))
    return waits


def _record(tmp_path, tag, listed=True):
    bed = tmp_path / "bed"
    bed.mkdir(exist_ok=True)
    (bed / f"bed_{tag}.json").write_text("{}", encoding="utf-8")
    if listed:                                                        # finished under a push: the runner lists it before the ship
        with (tmp_path / "runner_pushlist.txt").open("a", encoding="utf-8") as f:
            f.write(f"bed_{tag}.json\n")
    (bed / tag).mkdir(exist_ok=True)
    (bed / tag / "model.safetensors").write_bytes(b"0")
    return str(bed / f"bed_{tag}.json")


def test_ship_retries_a_hub_fault_then_succeeds(monkeypatch, tmp_path):
    waits = _fake_hub(monkeypatch, fails=2)
    lines = []
    rec = _record(tmp_path, "a_s1")
    assert br.ship("org/repo", rec, str(tmp_path), lines.append) is True
    assert waits == [15, 30] and [c for c in _FakeApi.calls if c[0] == "folder"] == [("folder", "bed/weights/a_s1")]
    assert "a_s1" in br._shipped(str(tmp_path)) and br.pending_records(str(tmp_path)) == []
    assert any("attempt 3" in l for l in lines) and sum("retry in" in l for l in lines) == 2


def test_ship_gives_up_without_raising_and_leaves_the_record_pending(monkeypatch, tmp_path):
    waits = _fake_hub(monkeypatch, fails=99)
    lines = []
    rec = _record(tmp_path, "b_s1")
    assert br.ship("org/repo", rec, str(tmp_path), lines.append, attempts=3) is False
    assert waits == [15, 30] and any(l.startswith("UNSHIPPED b_s1") for l in lines)
    assert br.pending_records(str(tmp_path)) == [rec]
    _FakeApi.fails = 0
    assert br.ship_pending("org/repo", str(tmp_path), lines.append, attempts=1) == [] and br.pending_records(str(tmp_path)) == []


def test_pending_records_skips_shipped_smokes_and_unlisted(tmp_path):
    a, b = _record(tmp_path, "a_s1"), _record(tmp_path, "b_s1")
    _record(tmp_path, "c_s1_smoke_frdyn")
    d = _record(tmp_path, "d_s1_ctx4096", listed=False)                 # a run finished without a push (a price-line run): never pending
    (tmp_path / "runner_shipped.txt").write_text("a_s1\n", encoding="utf-8")
    assert br.pending_records(str(tmp_path)) == [b]
    assert set(br.pending_records(str(tmp_path), all_records=True)) == {b, d}


def test_resume_treats_a_record_on_disk_as_done(tmp_path):
    from geolip.nano_beatrix.names import run_tag
    runs = br.expand({"arms": "smr-fixed,altn-fixed", "seeds": "1", "ctx": "4096", "steps": "3000"})
    _record(tmp_path, run_tag(dict(a.lstrip("-").split("=", 1) for a in runs[0])))
    res = br.run({"arms": "smr-fixed,altn-fixed", "seeds": "1", "ctx": "4096", "steps": "3000"}, out=str(tmp_path), dry=True)
    assert res["runs"] == 2 and res["done_before"] == 1 and res["unshipped"] == []
    assert br.run_key(runs[0]) in (tmp_path / "runner_done.txt").read_text(encoding="utf-8")


def test_init_is_a_grid_key_filled_from_the_run():
    runs = expand({"arms": "castor-both,pollux-both", "seeds": "1,2", "ask": "nested,overlap", "steps": "4000",
                   "init": "{arm}_s{seed}_L12000_frdyn_cap_mode_sz32-128-256_r4_n60_o30_ask{ask}_p75"})
    assert len(runs) == 8
    kv = dict(a.lstrip("-").split("=", 1) for a in runs[0])
    assert kv["init"] == f"{kv['arm']}_s{kv['seed']}_L12000_frdyn_cap_mode_sz32-128-256_r4_n60_o30_ask{kv['ask']}_p75"
    assert all(dict(a.lstrip("-").split("=", 1) for a in r)["init"].startswith(dict(a.lstrip("-").split("=", 1) for a in r)["arm"]) for r in runs)
    assert run_label(runs[0]).endswith(" cont")


def test_fetch_init_pulls_the_craft_into_the_bed(monkeypatch, tmp_path):
    import types
    mod = types.ModuleType("huggingface_hub")

    def fake_download(repo, path, local_dir=None, token=None):
        p = tmp_path / "_hub" / path
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"x")
        return str(p)
    mod.hf_hub_download = fake_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", mod)
    lines = []
    assert br.fetch_init("org/repo", str(tmp_path), "castor-both_s1_L12000", lines.append)
    assert (tmp_path / "bed" / "castor-both_s1_L12000" / "model.safetensors").exists() and (tmp_path / "bed" / "castor-both_s1_L12000" / "config.json").exists()
    assert any("fetched" in l for l in lines)
    mod.hf_hub_download = lambda *a, **k: (_ for _ in ()).throw(OSError("404"))
    assert not br.fetch_init("org/repo", str(tmp_path), "nobody_s9", lines.append) and any("not fetched" in l for l in lines)


def test_run_label_and_grid_carry_the_frame_form():
    assert run_label(["--arm=castor-both", "--seed=1", "--frames=dyn", "--frame_records=8", "--sized=1", "--sliced=32"]) == "castor-both s1 caption r8 sized sl32"
    assert run_label(["--arm=castor-both", "--seed=1", "--frames=dyn", "--sized=0", "--sliced=0"]) == "castor-both s1 caption"
    runs = expand({"arms": "castor-both", "seeds": "1", "frames": "dyn", "sized": "0,1"})
    assert len(runs) == 2 and sorted(a for r in runs for a in r if a.startswith("--sized=")) == ["--sized=0", "--sized=1"]


def test_run_label_names_a_curriculum_stage():
    assert run_label(["--arm=castor-both", "--seed=3", "--frames=dyn", "--init=castor-both_s3_L12000", "--stage=B2"]) == "castor-both s3 caption stage B2"
    assert run_label(["--arm=castor-both", "--seed=3", "--frames=dyn", "--init=castor-both_s3_L12000"]) == "castor-both s3 caption cont"
