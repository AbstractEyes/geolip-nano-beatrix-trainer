# -*- coding: utf-8 -*-
"""bed_table.py - the nano bed's results as tables. Reads every bed_<arm>_s<seed>[_L<steps>][_dense<K>][_<variant>]
[_ctx<N>][_from].json in DIR (bed_train.py outputs; smokes excluded) and prints markdown: the main table per context and fetch-distance set
(one row per run: steps, records a row, continued from, held-out bits a byte, the dispatch-off delta, digit and upper bits, the planted fetch
first byte / all five at every distance, the online value bits in the last window by band, the last block's census home share and entropy,
the cost: tokens/s and minutes), the seed means (--pairs), the per-class bits table, and the cost table of every prototype when DIR holds
bed_timing.json (train ms/step, tokens/s, peak GB; inference ms/row).
Usage: bed_table.py DIR [--pairs]"""
import glob
import io
import json
import os
import sys
import contextlib
from collections import defaultdict


def main(argv: list) -> None:
    """Print the index for the run records in argv[0] (a folder); '--pairs' adds the seed-pair tables."""
    D = argv[0]
    rows = []
    for p in sorted(glob.glob(os.path.join(D, "bed_*_s[0-9]*.json"))):
        if "smoke" in os.path.basename(p) or "timing" in os.path.basename(p):
            continue
        r = json.load(open(p, encoding="utf-8"))
        if r.get("smoke"):
            continue
        rows.append(r)
    rows.sort(key=lambda r: (r.get("ctx", 1024), r.get("records", 1), r["steps"], r["arm"], r["seed"]))
    framed = [r for r in rows if r.get("frames")]          # framed rows: their own table at the end (the planted fetch does not apply)
    rows = [r for r in rows if not r.get("frames")]


    def dist_key(r):
        return (r.get("ctx", 1024), tuple(sorted(r["fetch"], key=int)))


    if not rows:
        print("no runs")
    else:
        classes = list(rows[0]["heldout_bpb_by_class"])

        def home(r, blk="last"):
            c = r["dispatch_census"].get(blk, {})
            vals = [v["home_share"] for v in c.values()]
            return sum(vals) / len(vals) if vals else float("nan")

        def ent(r, blk="last"):
            c = r["dispatch_census"].get(blk, {})
            vals = [v["entropy_bits"] for v in c.values()]
            return sum(vals) / len(vals) if vals else float("nan")

        def formed_by(r, band="0", thr=3.0):
            """The first logged step at which the band's value bits fall under thr (the unigram rate is ~4.4; 3.0 = the copy is clearly forming)."""
            for c in r["curve"]:
                v = (c.get("value_bits") or {}).get(band)
                if v is not None and v < thr:
                    return c["step"]
            return None

        def last_value_bits(r):
            for c in reversed(r["curve"]):
                vb = c.get("value_bits") or {}
                if any(v is not None for v in vb.values()):
                    return vb
            return {}

        groups = []
        for r in rows:
            if dist_key(r) not in groups:
                groups.append(dist_key(r))
        for ctx, dists in groups:
            grp = [r for r in rows if dist_key(r) == (ctx, dists)]
            print(f"Context {ctx:,} bytes; the fetch at {' / '.join(dists)} ({len(grp)} runs):")
            print("| arm | seed | steps | records a row | continued from | held-out bits a byte | dispatch off (delta) | digit | upper | " + " | ".join(f"fetch first {d}" for d in dists) + " | " + " | ".join(f"all five {d}" for d in dists) + " | value bits (train, last window) <128 / 128-256 / 256-512 / 512+ | copy formed by (step; <128 band / 512+ band under 3.0 bits) | home share (last block) | dispatch entropy bits (last) | k tok/s | min |")
            print("|---|---|---|---|---|---|---|---|---|" + "---|" * (2 * len(dists)) + "---|---|---|---|---|---|")
            for r in grp:
                f, c = r["fetch"], r["heldout_bpb_by_class"]
                vb = last_value_bits(r)
                vbs = " / ".join(f"{vb[k]:.2f}" if vb.get(k) is not None else "-" for k in ("0", "128", "256", "512"))
                print(f"| {r['arm']} | {r['seed']} | {r['steps']:,} | {r.get('records', 1)} | {r.get('init') or '-'} | {r['heldout_bpb']:.4f} | {r['dispatch_off_bpb']:.4f} (+{r['dispatch_off_bpb'] - r['heldout_bpb']:.3f}) | {c['digit']:.3f} | {c['upper']:.3f} | "
                      + " | ".join(f"{f[d]['first_byte']:.2f}" for d in dists) + " | " + " | ".join(f"{f[d]['all_five']:.2f}" for d in dists)
                      + f" | {vbs} | {formed_by(r, '0') or '-'} / {formed_by(r, '512') or '-'} | {home(r):.2f} | {ent(r):.2f} | {r['tok_per_s'] / 1e3:.0f} | {r['train_minutes']:.1f} |")
            print()

        if "--pairs" in argv:
            def variant(r):
                c = r.get("config", {})
                tag = []
                if c.get("hub_K", 32) != 32 or c.get("hub_D", 16) != 16 or c.get("hub_const", 1) != 1:
                    tag.append(f"K{c.get('hub_K')}D{c.get('hub_D')}c{c.get('hub_const', 1)}")
                if c.get("rotary_base", 10000.0) != 10000.0:
                    tag.append(f"base{c['rotary_base']:g}")
                if c.get("hub_rotary_frac", 1.0) != 1.0:
                    tag.append(f"frac{c['hub_rotary_frac']}")
                if r.get("init"):
                    tag.append("continued")
                return r["arm"] + (" " + " ".join(tag) if tag else "")

            for ctx, dists in groups:
                by = defaultdict(list)
                for r in rows:
                    if dist_key(r) == (ctx, dists):
                        by[(variant(r), r["steps"], r.get("records", 1))].append(r)
                multi = {k: v for k, v in by.items() if len(v) > 1}
                if not multi:
                    continue
                print(f"Seed means at context {ctx:,} (n = seeds):")
                print("| arm | steps | records | n | held-out bits | " + " | ".join(f"fetch first {d}" for d in dists) + " | value bits 512+ |")
                print("|---|---|---|---|---|" + "---|" * len(dists) + "---|")
                for (arm, st, rec), rs in sorted(multi.items()):
                    m = lambda key: sum(key(r) for r in rs) / len(rs)
                    vb = [last_value_bits(r).get("512") for r in rs]
                    vb = [v for v in vb if v is not None]
                    print(f"| {arm} | {st:,} | {rec} | {len(rs)} | {m(lambda r: r['heldout_bpb']):.4f} | " + " | ".join(f"{m(lambda r, d=d: r['fetch'][d]['first_byte']):.2f}" for d in dists)
                          + f" | {(sum(vb) / len(vb)) if vb else float('nan'):.2f} |")
                print()

        posrows = [r for r in rows if r.get("heldout_bpb_by_position")]
        if posrows:
            for ctx in sorted({r.get("ctx", 1024) for r in posrows}):
                grp = [r for r in posrows if r.get("ctx", 1024) == ctx]
                bands = [b for b in grp[0]["heldout_bpb_by_position"]["row"] if grp[0]["heldout_bpb_by_position"]["row"][b] is not None]
                print(f"Held-out bits a byte BY POSITION in the window (context {ctx:,}; the bands in bytes from the row start; the counts in the JSON):")
                print("| arm | seed | steps | records | continued from | " + " | ".join(bands) + " |")
                print("|---|---|---|---|---|" + "---|" * len(bands))
                for r in grp:
                    bp = r["heldout_bpb_by_position"]["row"]
                    print(f"| {r['arm']} | {r['seed']} | {r['steps']:,} | {r.get('records', 1)} | {r.get('init') or '-'} | " + " | ".join(f"{bp[b]:.3f}" if bp.get(b) is not None else "-" for b in bands) + " |")
                print()
        print("Bits a byte by byte class (held-out):")
        print("| arm | seed | steps | records | ctx | " + " | ".join(classes) + " |")
        print("|---|---|---|---|---|" + "---|" * len(classes))
        for r in rows:
            c = r["heldout_bpb_by_class"]
            print(f"| {r['arm']} | {r['seed']} | {r['steps']:,} | {r.get('records', 1)} | {r.get('ctx', 1024):,} | " + " | ".join(f"{c[k]:.3f}" if c[k] is not None else "-" for k in classes) + " |")

    for tp in sorted(glob.glob(os.path.join(D, "bed_timing*.json"))):
        if "_smoke" in os.path.basename(tp):
            continue
        t = json.load(open(tp, encoding="utf-8"))
        print(f"\nThe time cost of every prototype ({t['device']}; the training step = forward + backward + Adam at {t['mb']} x {t['ctx']} bytes, bf16; inference = one no-grad forward of 16 rows; identical conditions, back to back):")
        print("| prototype | params | train ms/step | train k tok/s | train peak GB | inference ms/row | inference k tok/s |")
        print("|---|---|---|---|---|---|---|")
        for a, v in t["arms"].items():
            print(f"| {a} | {v['params'] / 1e6:.2f}M | {v['train_ms_per_step']:.1f} | {v['train_tok_per_s'] / 1e3:.0f} | {v['train_peak_gb']:.2f} | {v['infer_ms_per_row']:.2f} | {v['infer_tok_per_s'] / 1e3:.0f} |")
        b = t["arms"].get("sm-free")
        for a, v in t["fixes"].items():
            rel = f" ({v['train_ms_per_step'] / b['train_ms_per_step'] - 1:+.1%})" if b else ""
            reli = f" ({v['infer_ms_per_row'] / b['infer_ms_per_row'] - 1:+.1%})" if b else ""
            print(f"| sm-free + last-block {a} | {v['params'] / 1e6:.2f}M | {v['train_ms_per_step']:.1f}{rel} | {v['train_tok_per_s'] / 1e3:.0f} | {v['train_peak_gb']:.2f} | {v['infer_ms_per_row']:.2f}{reli} | {v['infer_tok_per_s'] / 1e3:.0f} |")
        for a, v in t["teachers"].items():
            print(f"| {a} (inference at {v['ctx']:,}, 2 rows) | {v['params'] / 1e6:.1f}M | | | | {v['infer_ms_per_row']:.1f} | {v['infer_tok_per_s'] / 1e3:.1f} |")


    if framed:
        print("\nFRAMED ROWS (every training row framed; text = the plain held-out bits a byte; outside = the bits between the frames inside framed held-out rows; "
              "the data frames' body bits by frame size; the whole value fetched from a data frame by its size at depth 0, with the first byte and the trials in brackets; "
              "the nested whole = every data frame at depth 1 or deeper, trial-weighted):")
        sizes = sorted({int(k.split("/")[1]) for r in framed for k in r["heldout_bits_by_frame"]["by_kind_size_segment"] if k.startswith("data/")}
                       | {int(k.split("/")[1]) for r in framed for k in r["fetch_by_frame"] if k.startswith("data/")})
        print("| arm | seed | ctx | steps | register | tags | sizes | nest | overlap | ask | facts | text | outside | "
              + " | ".join(f"data {s} body" for s in sizes) + " | " + " | ".join(f"whole value {s}" for s in sizes) + " | nested whole | tok/s | min |")
        print("|---|---|---|---|---|---|---|---|---|---|---|---|---|" + "---|" * (2 * len(sizes)) + "---|---|---|")
        for r in framed:
            fr, bf, ff = r["frames"], r["heldout_bits_by_frame"], r["fetch_by_frame"]
            seg = bf["by_kind_size_segment"]
            body = " | ".join(f"{seg['data/%d/body' % s]['bits']:.2f}" if "data/%d/body" % s in seg else "-" for s in sizes)
            fetch = " | ".join(f"{ff['data/%d/depth0' % s]['full_value']:.2f} ({ff['data/%d/depth0' % s]['first_byte']:.2f}, n {ff['data/%d/depth0' % s]['n']})"
                               if "data/%d/depth0" % s in ff else "-" for s in sizes)
            nested = [v for k, v in ff.items() if k.startswith("data/") and not k.endswith("depth0")]
            n_nest = sum(v["n"] for v in nested)
            nest_s = f"{sum(v['full_value'] * v['n'] for v in nested) / n_nest:.2f} (n {n_nest})" if n_nest else "-"
            print(f"| {r['arm']} | {r['seed']} | {r.get('ctx', 1024):,} | {r['steps']:,} | {fr.get('register')} | {fr.get('tags')} | {'/'.join(map(str, fr.get('sizes', [])))} | "
                  f"{fr.get('nest')} | {fr.get('overlap')} | {fr.get('ask')} | {fr.get('facts')} | {r['heldout_bpb']:.4f} | {bf['outside']['bits']:.3f} | {body} | {fetch} | {nest_s} | "
                  f"{r['tok_per_s'] / 1e3:.0f} | {r['train_minutes']:.1f} |")


def render(d: str, pairs: bool = True) -> str:
    """The index as a markdown string (a notebook cell shows it with IPython.display.Markdown)."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        main([d] + (["--pairs"] if pairs else []))
    return buf.getvalue()


if __name__ == "__main__":
    main(sys.argv[1:])
