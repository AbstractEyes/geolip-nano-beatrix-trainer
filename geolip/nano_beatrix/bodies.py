# -*- coding: utf-8 -*-
"""bodies.py - the nano bed's bodies as one importable table (0.1.26): an arm name -> the library config of the craft
(geolip.alephllm AlephLMConfig). bed_train.py builds its crafts from here, and so do the nano v1 crafts (nano_v1.py), so a body
means one thing everywhere. An arm is "<attention>-<bank>":
  attention  sm     softmax + the learned position table             smr    softmax + rotary (the table kept)
             smrn   softmax + rotary, no table                       hub    every block a splat hub (the chunked scan, aleph address)
             hubr   hubs with the address rotary                     hubrn  hubs with the address rotary, no position table
             hubd   hubs with two address masses a block (dual)      alt    hub / softmax+rotary alternating
             altn   alternating, both rotaries                       castor = smr, pollux = altn (the nano twin pair)
             hubdb  six dual hubs on the bigger address              altd   Pollux's layout with dual hubs
             d5s1   five dual hubs + one rotary softmax block (3)    d4s2   four dual hubs + two rotary softmax blocks (1, 3)
             altdc, hubdc, polluxc  altd, hubd and Pollux with Canon layers on every block's mixer and bank inputs
  bank       free   the aleph address alone                          fixed  an installed byte-class table replaces the amplitudes
             both   the address plus the installed table (learned)
The shared body: d256, six blocks, eight heads, hub address 32 @ D16, bank seven experts x ff256, head 64 @ D16, gate -3, hub chunk 128.

0.1.57: THE SIZES (Phil 2026-10-19: "create a micro variant then, step up from nano"; plans/2026-10-19_pollux_oven_plan.md). The micro
tier is the nano body widened and deepened by one rule: every address scales with the width (the hub's K = d/8 at D = d/16, the head's
K = d/4 at D = d/16, the bank's ff = d, seven experts), the softmax blocks keep 64 dims a head (d/64 heads; the nano's eight heads of 32
stay as they were), and an arm's layout is written for the size's depth (the hubs in every other block for the alternating arms, every
block for the hub arms). craft_cfg(arm, ctx, args, size): size "nano" (the default) builds the bed's bodies bit for bit; "micro26" =
d384 x 8 blocks (25.8 M), "micro45" = d512 x 8 (44.9 M), "micro66" = d512 x 12 (66.1 M), the counts at context 4,096 with the position
table (AlephLM.param_count; the nano is 9.3 M). The size also arrives in args ("size": the bed's command line, the gauge child). The
fixed-layout arms (hubdb's bigger address, d5s1 and d4s2's six-block layouts) are the nano's and refuse another size."""
from __future__ import annotations

from geolip.alephllm.presets import AlephLMConfig

N_LAYERS = 6
BASE = dict(d_model=256, n_layers=N_LAYERS, n_heads=8, hub_layers=(), hub_K=32, hub_D=16, tau=0.1, bank_experts=7, bank_ff=256,
            head_K=64, head_D=16, gate_init=-3.0, hub_chunk=128)
# 0.1.57: the sizes: width, depth and the softmax heads; the addresses follow the width (size_fields)
SIZES = {"nano": dict(d_model=256, n_layers=6, n_heads=8),
         "micro26": dict(d_model=384, n_layers=8, n_heads=6),
         "micro45": dict(d_model=512, n_layers=8, n_heads=8),
         "micro66": dict(d_model=512, n_layers=12, n_heads=8)}
MILLIONS = {"nano": 9.3, "micro26": 25.8, "micro45": 44.9, "micro66": 66.1}     # parameters at context 4,096, in millions (tests/test_micro.py)
NANO_ONLY = ("hubdb", "d5s1", "d4s2")                                         # layouts written for six blocks or the nano's address


def size_fields(size: str) -> dict:
    """The config fields a size sets: its width, depth and heads, and every address scaled with the width (the nano's own ratios: hub
    K = d/8 at D = d/16, head K = d/4 at D = d/16, the bank's ff = d)."""
    s = SIZES[size]
    d = s["d_model"]
    return dict(s, hub_K=d // 8, hub_D=d // 16, bank_ff=d, head_K=d // 4, head_D=d // 16)


def _attention(L: int = N_LAYERS) -> dict:
    A = {"sm": {}, "smr": dict(attn_rotary=True), "smrn": dict(attn_rotary=True, pos_table=False),
         "hub": dict(hub_layers=tuple(range(L))), "hubr": dict(hub_layers=tuple(range(L)), hub_rotary=True),
         "hubrn": dict(hub_layers=tuple(range(L)), hub_rotary=True, pos_table=False),
         "hubd": dict(hub_layers=tuple(range(L)), hub_dual=True),
         "alt": dict(hub_layers=tuple(range(0, L, 2)), attn_rotary=True),
         "altn": dict(hub_layers=tuple(range(0, L, 2)), attn_rotary=True, hub_rotary=True)}
    A["castor"] = A["smr"]            # the twins' names: Castor = the rotary softmax body (the mortal twin, the growing memory)
    A["pollux"] = A["altn"]           # Pollux = the all-rotary alternating body (the immortal twin, the constant decode memory)
    # the dual-hub bodies: each hub keeps two address masses, one rotary (sharp near) and one plain (distance-free, far reach)
    A["hubdb"] = dict(A["hubd"], hub_K=64, hub_D=32, hub_const=2)        # six dual hubs on the bigger address (two books of 64 in 32)
    A["altd"] = dict(hub_layers=tuple(range(0, L, 2)), attn_rotary=True, hub_dual=True)   # Pollux's layout with dual hubs
    A["d5s1"] = dict(hub_layers=(0, 1, 2, 4, 5), attn_rotary=True, hub_dual=True)        # five dual hubs, one rotary softmax block (3)
    A["d4s2"] = dict(hub_layers=(0, 2, 4, 5), attn_rotary=True, hub_dual=True)           # four dual hubs, two rotary softmax blocks (1, 3)
    # Canon layers (a causal kernel-4 convolution with a residual on each block's mixer and bank inputs, born zero; arXiv 2512.17351)
    A["altdc"] = dict(A["altd"], canon="AC")
    A["hubdc"] = dict(A["hubd"], canon="AC")
    A["polluxc"] = dict(A["altn"], canon="AC")
    return A


ATTENTION = _attention()
_ATTENTION_OF = {"nano": ATTENTION}
BANK = {"free": "", "fixed": "fixed", "both": "add"}


def attention_of(size: str) -> dict:
    """The arm layouts written for a size's depth (the nano's table is ATTENTION itself)."""
    if size not in _ATTENTION_OF:
        _ATTENTION_OF[size] = _attention(SIZES[size]["n_layers"])
    return _ATTENTION_OF[size]


def craft_cfg(arm: str, ctx: int, args: dict | None = None, size: str | None = None) -> AlephLMConfig:
    """The config of the bed craft `arm` at context `ctx`. args: the run's command-line values (strings); the hub's address size
    (hub_K, hub_D, hub_const), rotary_base and hub_rotary_frac override the body's when present (the bed's capacity screens).
    size (0.1.57): one of SIZES; None reads args["size"] and falls back to "nano" (the bed's bodies, bit for bit)."""
    attn, bank = arm.split("-")
    assert attn in ATTENTION and bank in BANK, f"unknown arm {arm!r}: attention in {sorted(ATTENTION)}, bank in {sorted(BANK)}"
    args = args or {}
    size = size or args.get("size") or "nano"
    assert size in SIZES, f"unknown size {size!r}: one of {sorted(SIZES)}"
    assert size == "nano" or attn not in NANO_ONLY, f"the arm {arm!r} is written for the nano body alone, not the {size} size"
    extra = {}
    if "rotary_base" in args:
        extra["rotary_base"] = float(args["rotary_base"])
    if "hub_rotary_frac" in args:
        extra["hub_rotary_frac"] = float(args["hub_rotary_frac"])
    for k in ("hub_K", "hub_D", "hub_const"):          # the hub's address size (the capacity question)
        if k in args:
            extra[k] = int(args[k])
    if args.get("head_mode"):                          # 0.1.34: the head's form (the library's head_mode: dial / dialraw / dialplus / pergain)
        extra["head_mode"] = str(args["head_mode"])
    return AlephLMConfig(**{**dict(BASE, name=f"bed-{arm}", context=ctx), **size_fields(size), **attention_of(size)[attn], **extra},
                         bank_byte_table=BANK[bank])
