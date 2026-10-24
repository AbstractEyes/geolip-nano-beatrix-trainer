# -*- coding: utf-8 -*-
"""The wide-band screen (0.1.73): the anchored split as a parameter. The default split is 0.1.72's bit for bit (its frequencies, its record
field, its names); ALHENA-W (4 content pairs) turns by its written formula. CPU only; the Hub is never touched."""
import math

import pytest
import torch

from geolip.alephllm.model.alephlm import AlephLM
from geolip.nano_beatrix import anchored_rope as ar
from geolip.nano_beatrix import bed_runner as br
from geolip.nano_beatrix.bodies import craft_cfg
from geolip.nano_beatrix.names import run_tag

ALHENA_F = {"arm": "alhena-both", "seed": "1", "ctx": "4096", "mb": "8", "steps": "24000", "frames": "dyn", "sizes": "32,128,256,512",
            "frame_records": "4", "nest": "0.3", "overlap": "0.0", "ask": "any", "register": "caption", "tags": "mode", "plant": "1.0",
            "grid": "ag", "grid_p": "0.5", "head": "digit", "head_lam": "1", "head_at": "both", "head_lam_i": "1", "series": "find"}


def _freqs_0172():
    """0.1.72's frequencies, written out as that version computed them."""
    base = 10000.0 * 4.0 ** (12 / 10)
    w = [base ** (-j / 6) for j in range(6)] + [0.0] * 2 + [2 * math.pi * k / 9 for k in range(1, 5)] * 2
    kind = [0] * 6 + [1] * 2 + [2] * 4 + [3] * 4
    return w, kind


def test_the_default_split_is_0172_bit_for_bit():
    w, kind = ar.frequencies()
    w0, k0 = _freqs_0172()
    assert w.tolist() == w0 and kind.tolist() == k0
    s = ar.spec()
    assert set(s) == {"form", "pairs", "rel_pairs", "content_pairs", "anchor_freqs", "levels", "ntk_scale", "base", "null", "w", "kind"}
    assert (s["pairs"], s["rel_pairs"], s["content_pairs"], s["anchor_freqs"], s["levels"]) == (16, 6, 2, 4, 2)
    assert s["w"] == w0 and s["kind"] == k0 and ar.AnchorStreams().content_pairs == 2
    assert ar.content_of(s) == 2


def test_the_wide_split_is_its_written_formula():
    w, kind = ar.frequencies(4)
    assert kind.tolist() == [0] * 4 + [1] * 4 + [2] * 4 + [3] * 4
    base = 10000.0 * 4.0 ** (8 / 6)
    assert w[:4].tolist() == [base ** (-j / 4) for j in range(4)] and w[4:8].tolist() == [0.0] * 4
    assert w[8:].tolist() == _freqs_0172()[0][8:]                                   # the anchor levels unchanged
    lam = [2 * math.pi / float(x) for x in w[:4]]
    assert [round(x, 1) for x in lam] == [6.3, 99.7, 1583.3, 25132.7]
    s = ar.spec(4)
    assert (s["rel_pairs"], s["content_pairs"]) == (4, 4) and ar.content_of(s) == 4
    with pytest.raises(AssertionError):
        ar.content_of(dict(s, content_pairs=2))                                      # a field that is not its own split is refused
    with pytest.raises(AssertionError):
        ar.frequencies(7)                                                           # the relative band keeps at least two pairs


def _expected_logits(q, k, dq, dk, c):
    w, kind = ar.frequencies(c)
    P = ar.PAIRS
    qc = torch.complex(q[..., :P].double(), q[..., P:].double())
    kc = torch.complex(k[..., :P].double(), k[..., P:].double())
    n = q.shape[-2]
    m_idx = torch.arange(n, dtype=torch.float64)
    out = torch.zeros(q.shape[0], q.shape[1], n, n, dtype=torch.float64)
    for j in range(P):
        prod = qc[..., j][..., :, None] * kc[..., j].conj()[..., None, :]
        if kind[j] == 0:
            ang = w[j] * (m_idx[:, None] - m_idx[None, :])
        elif kind[j] == 1:
            ang = torch.zeros(n, n, dtype=torch.float64)
        else:
            lev = int(kind[j]) - 2
            ang = w[j] * (dq[:, :, lev].double()[:, None, :, None] - dk[:, :, lev].double()[:, None, None, :])
        out += (prod * torch.exp(1j * ang)).real
    return out


@pytest.mark.parametrize("c", [2, 4])
def test_the_rotation_at_either_split_is_the_written_formula(c):
    torch.manual_seed(c)
    B, h, n = 2, 3, 40
    q, k = torch.randn(B, h, n, 2 * ar.PAIRS), torch.randn(B, h, n, 2 * ar.PAIRS)
    dq, dk = torch.randint(0, 9, (B, n, 2)), torch.randint(0, 9, (B, n, 2))
    st = ar.AnchorStreams(c)
    st.set(dk, dq)

    class Fake:
        _anchor = st
    qr, kr = ar._anchored_rotate(Fake(), q, k)
    got = qr.double() @ kr.double().transpose(-1, -2)
    exp = _expected_logits(q, k, dq, dk, c)
    assert torch.allclose(got, exp, atol=1e-4), float((got - exp).abs().max())


def test_the_wide_patch_keeps_the_state_dict():
    torch.manual_seed(1)
    m = AlephLM(craft_cfg("alhena-both", 256))
    before = {k: v.clone() for k, v in m.state_dict().items()}
    assert ar.patch(m, ar.AnchorStreams(4)) == 6
    after = m.state_dict()
    assert before.keys() == after.keys() and all(torch.equal(before[k], after[k]) for k in before)


def test_the_wide_runs_never_share_a_name():
    f = run_tag(ALHENA_F)
    assert f == "alhena-both_s1_L24000_ctx4096_frdyn_cap_mode_sz32-128-256-512_r4_n30_o0_askany_gag50_ahdigit1i1_find"
    assert run_tag(dict(ALHENA_F, rotary_content="2")) == f                          # the default split names nothing new
    base = {k: v for k, v in ALHENA_F.items() if k not in ("head_at", "head_lam_i")}
    w = run_tag(dict(base, rotary_content="4", series="wide"))
    wf = run_tag(dict(ALHENA_F, rotary_content="4", series="wide"))
    a3 = run_tag(dict(base, seed="3", series="scalerep"))
    assert w == "alhena-both_s1_L24000_ctx4096_frdyn_cap_mode_sz32-128-256-512_r4_n30_o0_askany_gag50_cp4_ahdigit1_wide"
    assert wf == "alhena-both_s1_L24000_ctx4096_frdyn_cap_mode_sz32-128-256-512_r4_n30_o0_askany_gag50_cp4_ahdigit1i1_wide"
    assert a3 == "alhena-both_s3_L24000_ctx4096_frdyn_cap_mode_sz32-128-256-512_r4_n30_o0_askany_gag50_ahdigit1_scalerep"
    assert len({f, w, wf, a3}) == 4


def test_the_runner_passes_the_split_and_labels_it():
    runs = br.expand(dict(arms="alhena-both", seeds="1,2", grid="ag", grid_p="0.5", head="digit", head_lam="1", rotary_content="4",
                          series="wide"))
    assert len(runs) == 2 and all("--rotary_content=4" in r for r in runs)
    assert "content pairs 4" in br.run_label(runs[0])
    plain = br.expand(dict(arms="alhena-both", seeds="1", grid="ag", grid_p="0.5", head="digit", head_lam="1", series="find"))
    assert "content pairs" not in br.run_label(plain[0])
