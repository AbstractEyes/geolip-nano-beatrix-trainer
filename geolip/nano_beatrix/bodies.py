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
The shared body: d256, six blocks, eight heads, hub address 32 @ D16, bank seven experts x ff256, head 64 @ D16, gate -3, hub chunk 128."""
from __future__ import annotations

from geolip.alephllm.presets import AlephLMConfig

N_LAYERS = 6
BASE = dict(d_model=256, n_layers=N_LAYERS, n_heads=8, hub_layers=(), hub_K=32, hub_D=16, tau=0.1, bank_experts=7, bank_ff=256,
            head_K=64, head_D=16, gate_init=-3.0, hub_chunk=128)


def _attention() -> dict:
    L = N_LAYERS
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
BANK = {"free": "", "fixed": "fixed", "both": "add"}


def craft_cfg(arm: str, ctx: int, args: dict | None = None) -> AlephLMConfig:
    """The config of the bed craft `arm` at context `ctx`. args: the run's command-line values (strings); the hub's address size
    (hub_K, hub_D, hub_const), rotary_base and hub_rotary_frac override the body's when present (the bed's capacity screens)."""
    attn, bank = arm.split("-")
    assert attn in ATTENTION and bank in BANK, f"unknown arm {arm!r}: attention in {sorted(ATTENTION)}, bank in {sorted(BANK)}"
    args = args or {}
    extra = {}
    if "rotary_base" in args:
        extra["rotary_base"] = float(args["rotary_base"])
    if "hub_rotary_frac" in args:
        extra["hub_rotary_frac"] = float(args["hub_rotary_frac"])
    for k in ("hub_K", "hub_D", "hub_const"):          # the hub's address size (the capacity question)
        if k in args:
            extra[k] = int(args[k])
    return AlephLMConfig(**{**dict(BASE, name=f"bed-{arm}", context=ctx), **ATTENTION[attn], **extra}, bank_byte_table=BANK[bank])
