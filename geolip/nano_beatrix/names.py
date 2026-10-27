# -*- coding: utf-8 -*-
"""names.py - the pieces of a run's tag that must never collide."""


def init_suffix(init, arm, seed):
    """The starting craft's tag as it enters a continued run's tag: this run's own arm and seed are stripped from its front when
    they match, the rest stays. Two continuations of one recipe from different crafts then never share a name (a 12,000-step
    continuation from a 1,024-context craft and one from a 4,096-context craft once did, and the second overwrote the first)."""
    p = f"{arm}_s{seed}_"
    return init[len(p):] if init.startswith(p) else init


def partner_args(args: dict) -> dict:
    """A pair run's partner as a run of its own (0.1.24): the arguments its tag is made from. Its arm and starting craft are the pair's
    partner arm and pair_init; the run's arm is its partner; its role is "s" (it learns from the run's arm, xc one) or "m" (mutual)."""
    a = {k: v for k, v in args.items() if k not in ("init", "pair_init", "pair_role")}
    a.update(arm=args["pair"], pair=args["arm"], pair_role="s" if args.get("xc", "one") == "one" else "m")
    if args.get("pair_init"):
        a["init"] = args["pair_init"]
    if args.get("init"):
        a["pair_init"] = args["init"]
    return a


def run_tag(args: dict) -> str:
    """The run's tag from its argument dict (the command-line values as strings): the single source of the trainer's record and weights
    names, and of the runner's resume (a run whose record exists on disk is done, whatever the done list says). The defaults here are
    the trainer's; a new tag piece is added here and nowhere else. A curriculum block (--stage=<name>, 0.1.22) carries its stage name in
    place of its starting craft's tag: a chain of continuations would otherwise nest every earlier tag into the next (a second block's
    tag ran to 259 characters); the record keeps the starting craft (init). The stage name must be unique within its chain."""
    arm, seed = args["arm"], int(args.get("seed", "1"))
    smoke = args.get("smoke", "0") == "1"
    gauge = args.get("gauge_only", "0") == "1"                # 0.1.26: a gauge read of a stored craft (no step): the tag ends _gauge
    steps = 0 if gauge else int(args.get("steps", "20" if smoke else "3000"))
    ctx = int(args.get("ctx", "1024"))
    p_plant = float(args.get("plant", "0.25"))
    frames = args.get("frames", "")
    register, tags = args.get("register", "caption"), args.get("tags", "mode")
    n_lookup, n_bulk, n_facts = int(args.get("n_lookup", "2")), int(args.get("n_bulk", "0")), int(args.get("facts", "0"))
    ask = args.get("ask", "lookup" if frames == "A" else "any")
    sizes = [int(x) for x in args.get("sizes", "32,128,256").split(",")]
    n_rec_frame = int(args.get("frame_records", "4"))
    nest_p, overlap_p = float(args.get("nest", "0.3")), float(args.get("overlap", "0.0"))
    sized, sliced = args.get("sized", "0") == "1", int(args.get("sliced", "0"))     # 0.1.21: size-typed frames, sliced frames
    fr = ("" if not frames else f"_fr{frames}_{register[:3]}_{tags}" + ("_sized" if sized else "") + (f"_sl{sliced}" if sliced else "")
          + (f"_nl{n_lookup}_nb{n_bulk}" if frames == "A" else f"_sz{'-'.join(map(str, sizes))}_r{n_rec_frame}_n{int(nest_p * 100)}_o{int(overlap_p * 100)}")
          + (f"_M{n_facts}" if n_facts else "") + f"_ask{ask}"
          + (f"_p{int(round(p_plant * 100))}" + ("d" if args.get("plain_doc", "0") == "1" else "")   # the diet, when not every row is
             if p_plant != 1.0 else ""))         # framed; 0.1.90: "d" when the plain rows open at a DOC (--plain_doc=1)
    # 0.1.24: the learning rate and the warmup, when not the trainer's defaults (a rate screen's runs would otherwise share one name)
    lr, warm = float(args.get("lr", "1e-3")), int(args.get("warmup", "5" if smoke else "100"))
    hp = (f"_lr{lr:g}" if lr != 1e-3 else "") + (f"_wu{warm}" if warm != (5 if smoke else 100) else "")
    # 0.1.31: the optimizer, when not pure Adam: AdamW's decoupled decay and its set when not the stream's matrices ("_wd0.1", "_wd1.2head")
    wd, wd_set = float(args.get("wd", "0")), args.get("wd_set", "mat")
    hp += (f"_wd{wd:g}" + ("" if wd_set == "mat" else wd_set)) if wd else ""
    # 0.1.34: the head's form when not the dual head ("_hddial", "_hdpergain", ...; the library's head_mode)
    hp += f"_hd{args['head_mode']}" if args.get("head_mode") else ""
    # 0.1.24: a pair run (--pair): the partner's arm, this craft's role (t teaches its partner, s learns from it, m both ways), where the
    # cross term reads (v the asked values' bytes, a every byte), its weight and its first step when not 1 and 0
    pair = args.get("pair", "")
    pr = ""
    if pair:
        role = args.get("pair_role") or ("t" if args.get("xc", "one") == "one" else "m")
        where = {"value": "v", "all": "a"}[args.get("xc_pos", "value")]
        xw, xf = float(args.get("xc_w", "1.0")), int(args.get("xc_from", "0"))
        pr = f"_pair-{pair}-{role}{where}" + (f"w{xw:g}" if xw != 1.0 else "") + (f"f{xf}" if xf else "")
    # 0.1.70: the scale test's cores: the grid rows and their share of a batch ("_gag50"), the rotation when not the house's ("_ranc"),
    # the address head and its weight ("_ahdigit1"), and a series name that keeps a re-run of a stored recipe from overwriting it;
    # 0.1.72: the head's term at the index entries too (--head_at=both) and its weight ("_ahdigit1i1")
    # 0.1.73: the anchored split's content pairs when not ALHENA's 2 ("_cp4": ALHENA-W)
    # 0.1.74: the grid rows' address as one byte ("_ab1"; the two digit bytes name nothing)
    # 0.1.75: the key-hard share of the grid rows ("_kh50" for .5; none names nothing)
    # 0.1.76: the value-path address ("_va1"; off names nothing)
    # 0.1.78: the pointer term, its weight, block and head ("_pt10b4h0" for lambda .1 on block 4's head 0; off names nothing)
    # 0.1.81: the key code at the end-marker ("_kcf" fixed, "_kcl" learned; off names nothing)
    # 0.1.82: the key in the told head's rotation ("_krt"; off names nothing)
    # 0.1.83: the larger grid row ("_gc48": 48 compartments; 24 names nothing) and the long keys ("_kl"; the standard keys name nothing)
    # 0.1.87: the binding's forms after "_krt": sixteen bound pairs ("16"; eight name nothing) and the key's norm ("n"; off names nothing)
    # 0.1.88: the hold at the records' SEPs only ("r") and the hold's length in hundredths when it is not 4.16 ("218" for 2.18)
    scale =((f"_g{args['grid']}{int(round(float(args.get('grid_p', '0.5')) * 100))}" if args.get("grid") else "")
             + (f"_r{args['rotary'][:3]}" if args.get("rotary") else "")
             + (f"_cp{int(args['rotary_content'])}" if int(args.get("rotary_content", "2")) != 2 else "")
             + (f"_gc{int(args['grid_comps'])}" if int(args.get("grid_comps", "24")) != 24 else "")
             + ("_kl" if args.get("grid_keys", "std") == "long" else "")
             + (f"_ab{int(args['grid_abytes'])}" if int(args.get("grid_abytes", "2")) != 2 else "")
             + (f"_kh{int(round(float(args['grid_hard']) * 100))}" if float(args.get("grid_hard", "0")) > 0 else "")
             + ("_va1" if args.get("value_addr", "0") == "1" else "")
             + (f"_pt{int(round(float(args['ptr_lam']) * 100))}b{int(args.get('ptr_block', '4'))}h{int(args.get('ptr_head', '0'))}"
                + ("p2" if args.get("ptr_pos", "sep") == "both" else "") + ("sg" if args.get("ptr_stop", "0") == "1" else "")
                if float(args.get("ptr_lam", "0")) > 0 else "")
             + ({"fixed": "_kcf", "learned": "_kcl"}[args["key_code"]] if args.get("key_code") else "")
             + ("_krt" + ("16" if args.get("key_pairs", "8") == "16" else "") + ("n" if args.get("key_norm", "0") == "1" else "")
                + ("r" if args.get("key_norm", "0") == "2" else "")
                + (f"{int(round(float(args['key_rho']) * 100))}" if float(args.get("key_rho", "4.16")) != 4.16 else "")
                if args.get("key_rot", "0") == "1" else "")
             + (f"_ah{args['head']}{float(args.get('head_lam', '1.0')):g}" if args.get("head") else "")
             + (f"i{float(args.get('head_lam_i', '1.0')):g}" if args.get("head") and args.get("head_at", "question") == "both" else "")
             + (f"_{args['series']}" if args.get("series") else ""))
    return (f"{arm}_s{seed}" + ("_smoke" if smoke else "") + (f"_L{steps}" if steps > 2500 and not smoke else "")   # the short grid keeps its bare names
            + (f"_dense{args.get('records')}" if int(args.get("records", "1")) > 1 else "")
            + (f"_b{args['rotary_base']}" if "rotary_base" in args else "") + (f"_f{args['hub_rotary_frac']}" if "hub_rotary_frac" in args else "")
            + (f"_K{args['hub_K']}D{args['hub_D']}c{args.get('hub_const', '1')}" if "hub_K" in args else "")
            + (f"_{args['size']}" if args.get("size", "nano") != "nano" else "") + hp      # 0.1.57: the body's size when not the nano's
            + (f"_ctx{ctx}" if ctx != 1024 else "") + (f"_c{args['cues']}" if args.get("cues", "5") != "5" else "") + fr + pr + scale
            + (f"_st{args['stage']}" if args.get("stage") else                            # 0.1.22: a curriculum block names its stage
               (f"_from_{init_suffix(args['init'], arm, seed)}" if "init" in args else ""))   # a continued run names its starting craft
            + ("_carry" if args.get("carry_gauge") == "1" else "")               # 0.1.37: a gauge read with the carry gauge (reading with notes)
            + (("_rows" + str(args.get("carry_rows"))) if args.get("carry_gauge") == "1" and str(args.get("carry_rows") or "192") != "192" else "")   # 0.1.39: the row count when not 192
            + ("_plain" if args.get("carry_gauge") == "1" and args.get("carry_plain") == "1" else "")                                   # 0.1.39: the plain-row read
            + ("_note" if args.get("note_gauge") == "1" else "")                                                                      # 0.1.41: the note rows read
            + ("_gauge" if gauge else ""))
