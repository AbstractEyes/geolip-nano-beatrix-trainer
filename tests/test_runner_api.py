# -*- coding: utf-8 -*-
"""The runner's Python entry point: the progress-line parser, the run labels, a dry grid from a dict spec, the resume file."""
import os

from geolip.nano_beatrix.bed_runner import expand, parse_step, run, run_label, run_key

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
