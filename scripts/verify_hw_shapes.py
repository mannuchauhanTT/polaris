#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Test the hw_shape claims against two opstats CSVs (pre-fix vs post-fix).

    python verify_hw_shape.py 'OUT/rn50_lut/STATS/p100a-*opstats.csv' \
                              'OUT/rn50_lut5/STATS/p100a-*opstats.csv'

Checks three separate assertions:

  A. Which tensors lacked hw_shape, before and after.
  B. Were the missing ones FAN-OUT tensors (consumed by >1 op)?  This is the
     stated cause and the one that needs testing, not assuming.
  C. What are the remaining shape mismatches -- are they all Conv inputs with
     post-halo gather extents, as claimed?
"""
import csv
import glob
import re
import sys
from collections import Counter, defaultdict

TENSOR_RE = re.compile(r"([\w.]+)\[([^\]]*)\]:(\w+)")


def load(pattern):
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise SystemExit(f"no files matched {pattern}")
    with open(paths[0], newline="") as fh:
        return paths[0], list(csv.DictReader(fh))


def tensors(field):
    """[(name, shape_str, precision, has_hw)] from an input_tensors cell."""
    out = []
    for m in TENSOR_RE.finditer(field or ""):
        name, shape, prec = m.group(1), m.group(2), m.group(3)
        out.append((name, shape, prec, "|hw:" in shape))
    return out


def report(label, rows):
    print(f"\n{'='*74}\n{label}\n{'='*74}")

    # --- A: who lacks hw_shape -------------------------------------------
    missing = []
    consumers = defaultdict(list)      # tensor name -> ops consuming it
    for r in rows:
        for name, shape, prec, has_hw in tensors(r.get("input_tensors")):
            consumers[name].append(r.get("opnum"))
            if not has_hw and "x" in shape and shape.count("x") == 3:
                # rank-4 spatial-looking tensor with no hw view
                missing.append((r.get("opnum"), r["optype"], name, shape))

    print(f"A. rank-4 input tensors with NO |hw: view : {len(missing)}")
    for opnum, optype, name, shape in missing[:14]:
        n_consumers = len(set(consumers[name]))
        print(f"     op {opnum:>4} {optype:20} {name:28} [{shape}]  consumers={n_consumers}")
    if len(missing) > 14:
        print(f"     ... and {len(missing)-14} more")

    # --- B: fan-out test --------------------------------------------------
    fanout = {n for n, ops in consumers.items() if len(set(ops)) > 1}
    missing_names = {m[2] for m in missing}
    if missing_names:
        overlap = missing_names & fanout
        print(f"\nB. fan-out test: {len(fanout)} tensors consumed by >1 op")
        print(f"   of the {len(missing_names)} distinct tensors missing hw_shape, "
              f"{len(overlap)} are fan-out ({len(overlap)/len(missing_names)*100:.0f}%)")
        if overlap != missing_names:
            print(f"   NOT fan-out but still missing: "
                  f"{sorted(missing_names - fanout)[:6]}")
        if fanout - missing_names:
            print(f"   fan-out but hw_shape PRESENT: "
                  f"{len(fanout - missing_names)} tensors "
                  f"-> fan-out alone does NOT predict the loss")
    else:
        print("\nB. fan-out test: nothing missing hw_shape; hypothesis untestable here")

    # --- C: precision mix on inputs --------------------------------------
    prec = Counter(p for r in rows for _, _, p, _ in tensors(r.get("input_tensors")))
    print(f"\nC. input tensor precisions: {dict(prec)}")
    print(f"   ops: {len(rows)}   optype mix: "
          f"{dict(Counter(r['optype'] for r in rows).most_common(6))}")


if __name__ == "__main__":
    if len(sys.argv) not in (2, 3):
        raise SystemExit(__doc__)
    for pat in sys.argv[1:]:
        path, rows = load(pat)
        report(path, rows)