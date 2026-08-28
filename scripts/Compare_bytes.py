# SPDX-License-Identifier: Apache-2.0
"""Compare bytes-per-element between two Polaris runs.

    python compare_bytes.py 'RUN_A/*opstats.csv' 'RUN_B/*opstats.csv'

Cycle totals alone cannot tell you whether a dtype change reduced modelled
traffic: element counts, bytes, tiling and core counts all move together.  This
prints inElems/inBytes per optype for both runs so the implied bytes-per-element
is visible, which is the thing a dtype change should actually alter.
"""
import csv
import glob
import sys
from collections import defaultdict


def fnum(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def load(pattern):
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise SystemExit(f"no files matched {pattern}")
    with open(paths[0], newline="") as fh:
        return paths[0], list(csv.DictReader(fh))


def agg(rows):
    d = defaultdict(lambda: defaultdict(float))
    for r in rows:
        a = d[r["optype"]]
        a["n"] += 1
        for f in ("inElems", "outElems", "inBytes", "outBytes", "cycles",
                  "mem_rd_cycles", "mem_wr_cycles", "memory_traffic"):
            a[f] += fnum(r.get(f))
    return d


def main(pa, pb):
    na, a = load(pa)
    nb, b = load(pb)
    da, db = agg(a), agg(b)
    print(f"A = {na}\nB = {nb}\n")

    hdr = (f"{'optype':22} {'n':>3} | {'A B/elem':>9} {'B B/elem':>9} | "
           f"{'A inBytes':>13} {'B inBytes':>13} {'bytes x':>8} | {'cycles x':>9}")
    print(hdr)
    print("-" * len(hdr))
    for k in sorted(set(da) | set(db), key=lambda k: -db.get(k, {}).get("cycles", 0)):
        x, y = da.get(k, {}), db.get(k, {})
        bpe_a = x.get("inBytes", 0) / x["inElems"] if x.get("inElems") else 0
        bpe_b = y.get("inBytes", 0) / y["inElems"] if y.get("inElems") else 0
        bx = y.get("inBytes", 0) / x["inBytes"] if x.get("inBytes") else 0
        cx = y.get("cycles", 0) / x["cycles"] if x.get("cycles") else 0
        print(f"{k:22} {int(y.get('n', x.get('n', 0))):>3} | {bpe_a:>9.2f} {bpe_b:>9.2f} | "
              f"{x.get('inBytes', 0):>13,.0f} {y.get('inBytes', 0):>13,.0f} {bx:>8.2f} | {cx:>9.2f}")

    for label, d in (("A", da), ("B", db)):
        tb = sum(v["inBytes"] for v in d.values())
        te = sum(v["inElems"] for v in d.values())
        tc = sum(v["cycles"] for v in d.values())
        print(f"\n{label}: inElems {te:,.0f}  inBytes {tb:,.0f}  "
              f"({tb/te if te else 0:.2f} B/elem)  cycles {tc:,.0f}")

    print("\nIf B/elem did not drop from ~4 to ~2, the dtype never reached the tensors.")
    print("If B/elem dropped but bytes went UP, element counts grew -- suspect tiling/"
          "padding or a changed shard/core count per op.")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    main(sys.argv[1], sys.argv[2])