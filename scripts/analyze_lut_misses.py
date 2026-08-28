#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Explain every LUT miss by comparing the emitted key against the LUT.

    python analyze_lut_misses.py bh_p100a_resnet50_lut.yaml opstats.csv

Unlike diff_lut_keys.py (which compared a subset of fields against the CLOSEST
entry and could report "0 fields differ" on a genuine miss), this:

  * reads the exact `lut_key` tuple ttsim emitted for each op (present even on
    miss, per device.py)
  * maps the tuple positions to the LUT's named key fields by op arity
  * for each miss, finds the LUT entry that agrees on shape+op_code and reports
    exactly which remaining fields differ
  * tallies mismatches by (optype, field) so the dominant blocker per op is clear

The tuple layout (from lookup_operator_perf.py):
  9-tuple  (1 input): op_code, w0,z0,y0,x0, layout0, dtype0, mem0, fidelity
  16-tuple (2 input): + w1,z1,y1,x1, layout1, dtype1, mem1
  23-tuple (3 input): + w2,z2,y2,x2, layout2, dtype2, mem2
"""
import ast
import csv
import glob
import sys
from collections import defaultdict

import yaml

FIELDS_9 = ["op_code", "input_0_w_pad_logical", "input_0_z_pad_logical",
            "input_0_y_pad_logical", "input_0_x_pad_logical",
            "input_0_layout", "input_0_datatype", "input_0_memory", "math_fidelity"]
FIELDS_1 = ["input_{n}_w_pad_logical", "input_{n}_z_pad_logical",
            "input_{n}_y_pad_logical", "input_{n}_x_pad_logical",
            "input_{n}_layout", "input_{n}_datatype", "input_{n}_memory"]


def key_fields(n_inputs):
    fields = list(FIELDS_9)
    for n in range(1, n_inputs):
        fields += [f.format(n=n) for f in FIELDS_1]
    return fields


def norm(v):
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, (int, float)):
        return str(int(v)) if float(v).is_integer() else str(v)
    s = str(v).strip()
    return s.replace("DataType.", "").replace("Layout.", "").upper() if s else s


def load_lut(path):
    lut = yaml.safe_load(open(path))
    by_op = defaultdict(list)
    for e in lut["entries"]:
        by_op[norm(e["key"]["op_code"])].append(e["key"])
    return by_op


def main(lut_path, csv_glob):
    by_op = load_lut(lut_path)
    paths = sorted(glob.glob(csv_glob))
    if not paths:
        raise SystemExit(f"no csv matched {csv_glob}")
    rows = list(csv.DictReader(open(paths[0], newline="")))

    hits = sum(1 for r in rows if str(r.get("uses_perf_lookup", "")).lower() in ("true", "1"))
    print(f"LUT     : {lut_path}  ({sum(len(v) for v in by_op.values())} entries)")
    print(f"opstats : {paths[0]}  ({len(rows)} ops)")
    print(f"hits    : {hits}/{len(rows)}\n")

    field_tally = defaultdict(int)         # (optype, field) -> count
    optype_miss = defaultdict(int)
    no_entry = defaultdict(int)

    for r in rows:
        if str(r.get("uses_perf_lookup", "")).lower() in ("true", "1"):
            continue
        optype = r["optype"]
        raw = r.get("lut_key")
        if not raw:
            field_tally[(optype, "<no key built>")] += 1
            optype_miss[optype] += 1
            continue
        try:
            key = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            field_tally[(optype, "<unparseable key>")] += 1
            continue

        optype_miss[optype] += 1
        n_inputs = 1 + (len(key) - 9) // 7 if len(key) >= 9 else 1
        fields = key_fields(n_inputs)
        keyed = {f: norm(v) for f, v in zip(fields, key)}
        opc = keyed.get("op_code", norm(optype))

        cands = by_op.get(opc, [])
        if not cands:
            no_entry[opc] += 1
            field_tally[(optype, f"op_code '{opc}' absent from LUT")] += 1
            continue

        # among entries agreeing on the shape fields, find the closest and
        # report the non-shape fields that differ; if none agree on shape,
        # the miss IS shape.
        shape_fields = [f for f in fields if f.endswith("_pad_logical")]
        shape_matches = [c for c in cands
                         if all(norm(c.get(f)) == keyed.get(f) for f in shape_fields)]
        if not shape_matches:
            # shape is the blocker; report which shape field(s) no entry matches
            best = min(cands, key=lambda c: sum(
                1 for f in shape_fields if norm(c.get(f)) != keyed.get(f)))
            for f in shape_fields:
                if norm(best.get(f)) != keyed.get(f):
                    field_tally[(optype, f)] += 1
            continue

        best = min(shape_matches, key=lambda c: sum(
            1 for f in fields if norm(c.get(f)) != keyed.get(f)))
        diffs = [f for f in fields if norm(best.get(f)) != keyed.get(f)]
        if not diffs:
            field_tally[(optype, "<all fields match but still missed!>")] += 1
        for f in diffs:
            field_tally[(optype, f)] += 1

    print("=== misses by optype ===")
    for ot, n in sorted(optype_miss.items(), key=lambda kv: -kv[1]):
        print(f"  {ot:24} {n}")

    print("\n=== mismatched fields (optype, field) -> count ===")
    for (ot, f), n in sorted(field_tally.items(), key=lambda kv: -kv[1]):
        print(f"  {n:>4}  {ot:20} {f}")

    print("\n=== field totals across all optypes ===")
    tot = defaultdict(int)
    for (ot, f), n in field_tally.items():
        tot[f] += n
    for f, n in sorted(tot.items(), key=lambda kv: -kv[1]):
        print(f"  {n:>4}  {f}")

    if no_entry:
        print("\n=== op_codes with NO LUT entry (naming mismatch) ===")
        for opc, n in sorted(no_entry.items(), key=lambda kv: -kv[1]):
            print(f"  {n:>4}  {opc}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    main(sys.argv[1], sys.argv[2])