# SPDX-License-Identifier: Apache-2.0
"""Break down a Polaris opstats CSV.

    python analyze_opstats.py __OUTPUT/rn50_e2e/STATS/p100a-*opstats.csv

Uses the csv module rather than awk because attrs / input_tensors /
tensor_attributes contain embedded commas, which shifts every column after 15.
"""
import csv
import glob
import sys
from collections import defaultdict


def num(row, key, cast=float, default=0):
    try:
        return cast(row[key])
    except (KeyError, TypeError, ValueError):
        return default


def main(pattern):
    paths = sorted(glob.glob(pattern))
    if not paths:
        print(f"no files matched {pattern}")
        return
    if len(paths) > 1:
        print(f"WARNING: {len(paths)} files matched; using {paths[0]} only\n")

    with open(paths[0], newline="") as fh:
        rows = list(csv.DictReader(fh))

    print(f"{paths[0]}\n{len(rows)} ops\n")

    tot = sum(num(r, "cycles") for r in rows)

    # ---- cycles by optype -------------------------------------------------
    agg = defaultdict(lambda: [0, 0.0, 0.0, 0.0, 0.0])  # n, cycles, matrix, rd, wr
    for r in rows:
        a = agg[r["optype"]]
        a[0] += 1
        a[1] += num(r, "cycles")
        a[2] += num(r, "matrix_cycles")
        a[3] += num(r, "mem_rd_cycles")
        a[4] += num(r, "mem_wr_cycles")

    print(f"{'optype':22} {'n':>4} {'cycles':>12} {'%':>6} {'matrix':>12} {'mem_rd':>12} {'mem_wr':>12}")
    for name, (n, c, m, rd, wr) in sorted(agg.items(), key=lambda kv: -kv[1][1]):
        print(f"{name:22} {n:>4} {c:>12,.0f} {c/tot*100 if tot else 0:>5.1f}% "
              f"{m:>12,.0f} {rd:>12,.0f} {wr:>12,.0f}")
    print(f"{'TOTAL':22} {len(rows):>4} {tot:>12,.0f}")

    # ---- hotspots ---------------------------------------------------------
    print(f"\n--- top 12 ops by cycles ---")
    print(f"{'op':>5} {'optype':20} {'cycles':>11} {'%':>6} {'bottleneck':14} {'in->out shape'}")
    for r in sorted(rows, key=lambda r: -num(r, "cycles"))[:12]:
        c = num(r, "cycles")
        print(f"{r.get('opnum',''):>5} {r['optype']:20} {c:>11,.0f} {c/tot*100 if tot else 0:>5.1f}% "
              f"{r.get('rsrc_bnck',''):14} {str(r.get('input_tensors',''))[:38]}")

    # ---- resource bottleneck mix -----------------------------------------
    bn = defaultdict(lambda: [0, 0.0])
    for r in rows:
        b = bn[r.get("rsrc_bnck", "?")]
        b[0] += 1
        b[1] += num(r, "cycles")
    print(f"\n--- bottleneck mix ---")
    for k, (n, c) in sorted(bn.items(), key=lambda kv: -kv[1][1]):
        print(f"  {k:14} {n:>4} ops  {c:>12,.0f} cycles  {c/tot*100 if tot else 0:5.1f}%")

    # ---- weights and LUT --------------------------------------------------
    params = sum(num(r, "inParamCount", int) for r in rows)
    with_params = sum(1 for r in rows if num(r, "inParamCount", int) > 0)
    print(f"\n--- weights ---")
    print(f"  total inParamCount = {params:,}  ({with_params} of {len(rows)} ops report any)")
    if params == 0:
        print("  -> no weight traffic modelled at all")

    lut = defaultdict(int)
    for r in rows:
        lut[str(r.get("uses_perf_lookup", "?"))] += 1
    print(f"\n--- perf lookup (LUT) ---")
    for k, v in sorted(lut.items()):
        print(f"  uses_perf_lookup={k}: {v} ops")
    srcs = defaultdict(int)
    for r in rows:
        srcs[str(r.get("lut_hit_source", "") or "<none>")] += 1
    for k, v in sorted(srcs.items(), key=lambda kv: -kv[1]):
        print(f"  lut_hit_source={k}: {v} ops")

    # ---- fusion -----------------------------------------------------------
    fused = sum(1 for r in rows if str(r.get("fused", "")).lower() in ("true", "1"))
    removed = sum(1 for r in rows if str(r.get("removed", "")).lower() in ("true", "1"))
    print(f"\n--- fusion ---\n  fused={fused}  removed={removed}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else
         "__OUTPUT/rn50_e2e/STATS/p100a-*opstats.csv")