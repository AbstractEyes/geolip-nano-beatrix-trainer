# -*- coding: utf-8 -*-
"""names.py - the pieces of a run's tag that must never collide."""


def init_suffix(init, arm, seed):
    """The starting craft's tag as it enters a continued run's tag: this run's own arm and seed are stripped from its front when
    they match, the rest stays. Two continuations of one recipe from different crafts then never share a name (a 12,000-step
    continuation from a 1,024-context craft and one from a 4,096-context craft once did, and the second overwrote the first)."""
    p = f"{arm}_s{seed}_"
    return init[len(p):] if init.startswith(p) else init


def run_tag(args: dict) -> str:
    """The run's tag from its argument dict (the command-line values as strings): the single source of the trainer's record and weights
    names, and of the runner's resume (a run whose record exists on disk is done, whatever the done list says). The defaults here are
    the trainer's; a new tag piece is added here and nowhere else."""
    arm, seed = args["arm"], int(args.get("seed", "1"))
    smoke = args.get("smoke", "0") == "1"
    steps = int(args.get("steps", "20" if smoke else "3000"))
    ctx = int(args.get("ctx", "1024"))
    p_plant = float(args.get("plant", "0.25"))
    frames = args.get("frames", "")
    register, tags = args.get("register", "caption"), args.get("tags", "mode")
    n_lookup, n_bulk, n_facts = int(args.get("n_lookup", "2")), int(args.get("n_bulk", "0")), int(args.get("facts", "0"))
    ask = args.get("ask", "lookup" if frames == "A" else "any")
    sizes = [int(x) for x in args.get("sizes", "32,128,256").split(",")]
    n_rec_frame = int(args.get("frame_records", "4"))
    nest_p, overlap_p = float(args.get("nest", "0.3")), float(args.get("overlap", "0.0"))
    fr = ("" if not frames else f"_fr{frames}_{register[:3]}_{tags}"
          + (f"_nl{n_lookup}_nb{n_bulk}" if frames == "A" else f"_sz{'-'.join(map(str, sizes))}_r{n_rec_frame}_n{int(nest_p * 100)}_o{int(overlap_p * 100)}")
          + (f"_M{n_facts}" if n_facts else "") + f"_ask{ask}"
          + (f"_p{int(round(p_plant * 100))}" if p_plant != 1.0 else ""))   # the diet, when not every row is framed
    return (f"{arm}_s{seed}" + ("_smoke" if smoke else "") + (f"_L{steps}" if steps > 2500 and not smoke else "")   # the short grid keeps its bare names
            + (f"_dense{args.get('records')}" if int(args.get("records", "1")) > 1 else "")
            + (f"_b{args['rotary_base']}" if "rotary_base" in args else "") + (f"_f{args['hub_rotary_frac']}" if "hub_rotary_frac" in args else "")
            + (f"_K{args['hub_K']}D{args['hub_D']}c{args.get('hub_const', '1')}" if "hub_K" in args else "")
            + (f"_ctx{ctx}" if ctx != 1024 else "") + (f"_c{args['cues']}" if args.get("cues", "5") != "5" else "") + fr
            + (f"_from_{init_suffix(args['init'], arm, seed)}" if "init" in args else ""))   # a continued run names its starting craft
