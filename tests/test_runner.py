# -*- coding: utf-8 -*-
"""The grid runner on the CPU: the expansion, the pass-through, the done list, the record-path parse."""
from geolip.nano_beatrix import bed_runner as br


def test_grid_expands_the_product_and_passes_the_rest_through():
    args = br.parse_args(["--out=X", "--arms=smr-fixed,altn-fixed", "--seeds=1,2", "--ctx=1024", "--frames=dyn", "--sizes=32,128,256", "--facts=0,64"])
    runs = br.expand(args)
    assert len(runs) == 2 * 2 * 2
    for r in runs:
        assert "--ctx=1024" in r and "--frames=dyn" in r and "--sizes=32,128,256" in r
        assert sum(a.startswith("--arm=") for a in r) == 1 and sum(a.startswith("--seed=") for a in r) == 1
    keys = {br.run_key(r) for r in runs}
    assert len(keys) == 8
    assert any("--arm=altn-fixed" in r and "--seed=2" in r and "--facts=64" in r for r in runs)


def test_no_grid_keys_is_one_run():
    runs = br.expand(br.parse_args(["--out=X", "--arms=smr-fixed", "--seeds=1", "--steps=12000"]))
    assert runs == [["--arm=smr-fixed", "--seed=1", "--steps=12000"]]


def test_wrote_path_takes_the_last_record():
    out = "step 1\n[..] wrote E:/out/bed/bed_a.json (1.0 min)\nmore\n[..] wrote E:/out/bed/bed_b_s1_ctx4096.json (2.0 min)\n"
    assert br.wrote_path(out) == "E:/out/bed/bed_b_s1_ctx4096.json"
    assert br.wrote_path("nothing") is None


def test_dry_run_lists_the_grid(tmp_path, capsys):
    rc = br.main([f"--out={tmp_path}", "--arms=smr-fixed,altn-fixed", "--seeds=1", "--dry=1"])
    assert rc == 0
    text = capsys.readouterr().out
    assert "grid: 2 runs" in text and "--arm=altn-fixed" in text


def test_done_list_skips_finished_runs(tmp_path, capsys):
    done = tmp_path / "runner_done.txt"
    done.write_text(br.run_key(["--arm=smr-fixed", "--seed=1"]) + "\n", encoding="utf-8")
    rc = br.main([f"--out={tmp_path}", "--arms=smr-fixed,altn-fixed", "--seeds=1", "--dry=1"])
    assert rc == 0
    assert "2 runs, 1 done before, 1 to run" in capsys.readouterr().out
