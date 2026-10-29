# -*- coding: utf-8 -*-
"""0.1.103: the entry's input (--key_entry=rot|code, --ptr_pos=all). key_code.entry_offsets gives each index entry's address positions
after its SEP (SEP + 1 .. SEP + L - 1, the positions that write the second .. last digit) the key offsets of the SEP, from the bytes at or
before each position: on bed_train's own batches (AG48 at 4x4 and 8x3, AG at 8x2, key-hard rows included) exactly those positions change
and nothing else (no question's value byte, no record's byte, no framed row); a written non-digit ends the rule. The key rotation's entry
form turns the told head's pairs there by the SEP's phases, value for value, and keeps the hold; the key code's entry form adds the SEP's
code there. The pointer term's positions at --ptr_pos=all are the entries' address positions (grid_rows.index_positions). Off, every
form is 0.1.102's (the fields, the phases, the code). The names and the runner carry the forms. CPU only; the Hub is never touched."""
import os

import numpy as np
import pytest
import torch

from geolip.nano_beatrix import bed_runner as br
from geolip.nano_beatrix import grid_rows as gr
from geolip.nano_beatrix import key_code as kc
from geolip.nano_beatrix import key_rotary as kr
from geolip.nano_beatrix.names import run_tag
from test_scale import _bed_batch

SRC = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "geolip", "nano_beatrix", "bed_train.py"), encoding="utf-8").read()
CASES = [("AG48", "4x4"), ("AG48", "8x3"), ("AG", "8x2")]


def _grid_batch(layout, code):
    r, n = gr.CODES[code]
    x, _, _, aux = _bed_batch(4, 2, layout, "std", GRID_CODE=code, GR_RADIX=r, GR_LEVELS=n)(3)
    rows = []                                                          # each grid meta's batch row (the entries' rows, in order)
    for b in aux["ihead_row"].tolist():
        if b not in rows:
            rows.append(b)
    assert len(rows) == len(aux["metas"]) == 4
    return x, aux, rows, r, n


def _want(x, aux, rows, n):
    """{(b, t): s} for every index entry's address position t after its SEP s (the rule's positions, from the metas)."""
    want = {}
    for b, m in zip(rows, aux["metas"]):
        for e in m["index"]:
            s = e["a_pos"] - 1
            assert int(x[b, s]) == gr.SEP
            for lag in range(1, n):
                want[(b, s + lag)] = s
    return want


@pytest.mark.parametrize("layout,code", CASES)
def test_the_rule_codes_each_entrys_address_positions_and_nothing_else(layout, code):
    x, aux, rows, r, n = _grid_batch(layout, code)
    off = kc.key_offsets(x)
    ext = kc.entry_offsets(x, off, r, n)
    want = _want(x, aux, rows, n)
    changed = {(int(b), int(t)) for b, t in (ext != off).any(-1).nonzero().tolist()}
    assert changed == set(want) and len(want) == 4 * len(aux["metas"][0]["index"]) * (n - 1)
    for (b, t), s in want.items():
        assert torch.equal(ext[b, t], off[b, s]) and bool((off[b, t] == kc.VOCAB).all())    # the SEP's offsets; nothing coded there before
    on, on_e = off[..., 0] != kc.VOCAB, ext[..., 0] != kc.VOCAB
    assert torch.equal(on_e & ~on, (ext != off).any(-1))                                  # the rule only adds coded positions
    for b, m in zip(rows, aux["metas"]):                                                  # never a question's span, never a record's bytes
        for q in m["questions"]:
            assert not bool((ext[b, q["sep"] + 1:q["end"] + 1] != off[b, q["sep"] + 1:q["end"] + 1]).any())
        for rec in m["recs"]:
            assert not bool((ext[b, rec["lo"]:rec["hi"]] != off[b, rec["lo"]:rec["hi"]]).any())


def test_a_written_non_digit_ends_the_rule():
    x, aux, rows, r, n = _grid_batch("AG48", "4x4")
    b, e = rows[0], aux["metas"][0]["index"][3]
    s = e["a_pos"] - 1
    y = x.clone()
    y[b, s + 2] = ord("q")                                                               # the second digit written as a letter
    ext = kc.entry_offsets(y, kc.key_offsets(y), r, n)
    off = kc.key_offsets(y)
    assert torch.equal(ext[b, s + 1], off[b, s])                                         # the first address position still coded
    assert bool((ext[b, s + 2:s + n] == kc.VOCAB).all())                                 # from the letter on, nothing
    y[b, s + 1] = ord("0") + r                                                           # a digit outside the code's radix
    ext = kc.entry_offsets(y, kc.key_offsets(y), r, n)
    assert bool((ext[b, s + 1:s + n] == kc.VOCAB).all())


@pytest.mark.parametrize("layout,code", CASES)
def test_the_rotations_entry_form_turns_by_the_seps_phases_and_keeps_the_hold(layout, code):
    x, aux, rows, r, n = _grid_batch(layout, code)
    off_r = kr.KeyRotary(4, 0, k=9, pairs=16, norm=2, rho=2.18, code=code)
    on_r = kr.KeyRotary(4, 0, k=9, pairs=16, norm=2, rho=2.18, code=code, entry=True)
    off_r.set(x)
    on_r.set(x)
    want = _want(x, aux, rows, n)
    mask = torch.zeros_like(off_r.on)
    for (b, t), s in want.items():
        mask[b, t] = True
        assert torch.equal(on_r.phi[b, t], on_r.phi[b, s]) and torch.equal(on_r.phi[b, s], off_r.phi[b, s])   # bitwise the SEP's
    assert torch.equal(on_r.on, off_r.on | mask) and not bool((off_r.on & mask).any())
    assert torch.equal(on_r.phi[~mask], off_r.phi[~mask]) and torch.equal(on_r.hold, off_r.hold)
    assert kr.record(on_r) == dict(kr.record(off_r), entry=True) and "entry" not in kr.record(off_r)


@pytest.mark.parametrize("layout,code", CASES)
def test_the_codes_entry_form_adds_the_seps_code(layout, code):
    x, aux, rows, r, n = _grid_batch(layout, code)
    plain = kc.KeyCode(16, "fixed")
    entry = kc.KeyCode(16, "fixed", entry=True, radix=r, levels=n)
    c0, c1 = plain.code(x), entry.code(x)
    want = _want(x, aux, rows, n)
    mask = torch.zeros(x.shape, dtype=torch.bool)
    for (b, t), s in want.items():
        mask[b, t] = True
        assert torch.equal(c1[b, t], c0[b, s]) and bool((c0[b, t] == 0).all()) and float(c0[b, s].abs().sum()) > 0
    assert torch.equal(c1[~mask], c0[~mask])
    rec = kc.record(entry)
    assert rec == dict(kc.record(plain), entry=True, radix=r, levels=n) and "entry" not in kc.record(plain)


@pytest.mark.parametrize("layout,code", CASES)
def test_the_pointer_terms_positions_at_all_are_the_entries_address_positions(layout, code):
    x, aux, rows, r, n = _grid_batch(layout, code)
    sep = aux["ihead_pos"].long()
    pos = torch.stack([sep + j for j in range(n)], 1)                                    # ptr_log_mass's expression at --ptr_pos=all
    want = torch.tensor([list(p) for m in aux["metas"] for p in gr.index_positions(m)])
    assert torch.equal(pos, want)
    assert "torch.stack([sep + j for j in range(PTR_NPOS)], 1).reshape(-1)" in SRC
    assert 'PTR_NPOS = {"sep": 1, "both": 2, "all": GR_LEVELS}[PTR_POS]' in SRC


def test_the_flags_and_their_guards_in_the_source():
    assert 'KEY_ENTRY = ARGS.get("key_entry", "")' in SRC and 'assert KEY_ENTRY in ("", "rot", "code")' in SRC
    assert 'entry=KEY_ENTRY == "code"' in SRC and 'entry=KEY_ENTRY == "rot"' in SRC
    assert 'assert PTR_POS in ("sep", "both", "all")' in SRC
    for line in SRC.split("\n"):                                                         # every entry-form branch sits under its flag
        if '_sj["key_code"].update(entry=True' in line or '_sj["key_rot"]["entry"] = True' in line:
            assert line.startswith(" " * 20), line


def test_the_names_carry_the_forms_only_when_set():
    base = {"arm": "alhena-both", "seed": "1", "steps": "24000", "ctx": "4096", "grid": "ag", "grid_p": "0.5", "rotary": "anchored",
            "head": "digit", "head_lam": "1", "grid_comps": "48", "grid_code": "4x4", "rotary_content": "4", "grid_hard": "0.5",
            "value_addr": "1", "ptr_lam": "0.1", "key_rot": "1", "key_pairs": "16", "key_norm": "2", "key_rho": "2.18", "series": "a48entry"}
    t0 = run_tag(dict(base))
    assert "_pt10b4h0_krt16r218_ahdigit1_a48entry" in t0
    assert run_tag(dict(base, key_entry="rot", ptr_pos="all")) == t0.replace("_pt10b4h0_krt16r218_", "_pt10b4h0pa_krt16r218e_")
    assert run_tag(dict(base, key_code="learned", key_entry="code")) == t0.replace("_pt10b4h0_krt16r218_", "_pt10b4h0_kcle_krt16r218_")
    assert run_tag(dict(base, key_code="learned")) == t0.replace("_pt10b4h0_krt16r218_", "_pt10b4h0_kcl_krt16r218_")


def test_the_runner_passes_the_forms():
    runs = br.expand({"arms": "alhena-both", "seeds": "1", "steps": "24000", "key_entry": "rot,code"})
    assert [[a for a in r if a.startswith("--key_entry=")] for r in runs] == [["--key_entry=rot"], ["--key_entry=code"]]
