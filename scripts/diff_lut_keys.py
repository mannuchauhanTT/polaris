#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Why did each LUT lookup miss?

    python diff_lut_keys.py bh_p100a_resnet50_lut.yaml 'OUT/STATS/p100a-*opstats.csv'

Polaris emits ``lut_key`` for every op even on a miss (device.py:930), so the
keys it built are recoverable from the opstats CSV.  This compares them against
the entries in a local LUT and, for each miss, reports which of the entry's key
field VALUES are absent from the emitted tuple -- i.e. the fields that differ.

Field order inside the tuple is not assumed: comparison is by value membership,
which is enough to identify the culprit field (shape number, datatype, layout,
memory, math_fidelity) without knowing the 9/16/23-tuple layout.
"""
import ast
import csv
import glob
import sys
from collections import defaultdict

import yaml


def norm(v):
    """Canonical form for value comparison across yaml / tuple representations."""
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, (int, float)):
        return str(int(v)) if float(v).is_integer() else str(v)
    return str(v).strip().lower().replace("datatype.", "").replace("layout.", "")


def main(lut_path, csv_pattern):
    lut = yaml.safe_load(open(lut_path))
    entries = lut["entries"]

    by_op = defaultdict(list)
    for e in entries:
        by_op[norm(e["key"]["op_code"])].append(e)
    print(f"LUT {lut_path}: {len(entries)} entries, "
          f"op_codes {sorted(by_op)}\n")

    paths = sorted(glob.glob(csv_pattern))
    if not paths:
        raise SystemExit(f"no files matched {csv_pattern}")
    rows = list(csv.DictReader(open(paths[0], newline="")))
    print(f"opstats {paths[0]}: {len(rows)} ops\n")

    hits = [r for r in rows if str(r.get("uses_perf_lookup", "")).lower() in ("true", "1")]
    print(f"hits {len(hits)} / {len(rows)}")
    if not any(r.get("lut_key") for r in rows):
        print("\nNo lut_key emitted -> no LUT was configured for this run.")
        print("Point operator_lookup_file in the arch spec at the local LUT first.")
        return

    missing_field_tally = defaultdict(int)

    for r in rows:
        if str(r.get("uses_perf_lookup", "")).lower() in ("true", "1"):
            continue
        raw = r.get("lut_key")
        if not raw:
            print(f"  op {r.get('opnum'):>4} {r['optype']:20} NO KEY BUILT "
                  f"(unsupported arity / missing shape)")
            continue
        try:
            key = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            print(f"  op {r.get('opnum'):>4} {r['optype']:20} unparseable key: {raw[:70]}")
            continue

        kvals = {norm(x) for x in (key if isinstance(key, tuple) else (key,))}
        opc = norm(key[0]) if isinstance(key, tuple) and key else norm(r["optype"])
        cands = by_op.get(opc, [])
        if not cands:
            print(f"  op {r.get('opnum'):>4} {r['optype']:20} op_code {opc!r} "
                  f"NOT IN LUT at all")
            missing_field_tally[f"op_code:{opc}"] += 1
            continue

        # the closest entry is the one with fewest unmatched field values
        best, best_missing = None, None
        for e in cands:
            miss = {f: v for f, v in e["key"].items() if norm(v) not in kvals}
            if best_missing is None or len(miss) < len(best_missing):
                best, best_missing = e, miss
        print(f"  op {r.get('opnum'):>4} {r['optype']:20} closest entry differs on "
              f"{len(best_missing)} field(s): "
              f"{ {f: best_missing[f] for f in list(best_missing)[:4]} }")
        for f in best_missing:
            missing_field_tally[f] += 1

    if missing_field_tally:
        print("\n--- which key fields most often fail to match ---")
        for f, n in sorted(missing_field_tally.items(), key=lambda kv: -kv[1]):
            print(f"  {f:34} {n:>4} ops")
        print("\nA shape field dominating => hw_shape absent on that tensor.")
        print("datatype/layout/memory dominating => tensor state differs, not shape.")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    main(sys.argv[1], sys.argv[2])