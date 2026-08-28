#!/usr/bin/env python3
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Generate _HALO_EXT_Y rows from a tt-metal profiler CSV.

    python gen_halo_ext_y.py merged_ops_<model>-<arch>.csv

ttsim/ops/desc/ttsim_layout.py keys the halo-extended Y as

    _HALO_EXT_Y[(nhw, C, kH, kW, pH, pW, is_transpose)] = max_out_nsticks_per_core * num_cores_nhw

Every one of those fields is recorded in the HaloDeviceOperation ATTRIBUTES
column (SlidingWindowConfig + max_out_nsticks_per_core), so the rows can be
derived directly from a hardware run rather than hand-transcribed.

The shipped table has 17 entries keyed at nhw = 65536 / 16384 (a 64x64 / 32x32
input geometry).  ResNet-50 at 224 needs nhw = 211600 / 200704 / 50176 / 12544 /
3136, none of which are present -- so halo_sinf falls back to propagating the
pre-halo hw_shape and every downstream Conv keys on the un-gathered size.
"""
import ast
import re
import sys

import pandas as pd

HALO_OP = "HaloDeviceOperation"


def parse_attrs(raw: str) -> dict:
    """The ATTRIBUTES cell is a dict-ish string using ';' as separator."""
    out = {}
    for m in re.finditer(r"'(\w+)':\s*'([^']*)'", str(raw)):
        out[m.group(1)] = m.group(2)
    return out


def parse_sliding_window(cfg: str) -> dict:
    """Pull the fields the LUT key needs out of SlidingWindowConfig(...)."""
    def grab(pat, default=None):
        m = re.search(pat, cfg)
        return m.group(1) if m else default

    batch = grab(r"batch_size=(\d+)")
    in_hw = grab(r"input_hw=\((\d+);\s*(\d+)\)")
    m = re.search(r"input_hw=\((\d+);\s*(\d+)\)", cfg)
    in_h, in_w = (int(m.group(1)), int(m.group(2))) if m else (None, None)
    m = re.search(r"window_hw=\((\d+);\s*(\d+)\)", cfg)
    kh, kw = (int(m.group(1)), int(m.group(2))) if m else (None, None)
    # padding=((top; bottom); (left; right))
    m = re.search(r"padding=\(\((\d+);\s*(\d+)\);\s*\((\d+);\s*(\d+)\)\)", cfg)
    ph, pw = (int(m.group(1)), int(m.group(3))) if m else (None, None)
    ncores = grab(r"num_cores_nhw=(\d+)")
    is_tp = grab(r"is_transpose=(\w+)", "false")
    return dict(batch=int(batch) if batch else None, in_h=in_h, in_w=in_w,
                kh=kh, kw=kw, ph=ph, pw=pw,
                num_cores_nhw=int(ncores) if ncores else None,
                is_transpose=(str(is_tp).lower() == "true"))


def main(path):
    df = pd.read_csv(path)
    halo = df[df["OP CODE"] == HALO_OP]
    if halo.empty:
        raise SystemExit(f"no {HALO_OP} rows in {path}")

    CIN = "INPUT_0_X_PAD[LOGICAL]"
    OY = "OUTPUT_0_Y_PAD[LOGICAL]"

    def as_int(v):
        return int(str(v).split("[")[0])

    rows, mismatches = {}, []
    for _, r in halo.iterrows():
        a = parse_attrs(r["ATTRIBUTES"])
        cfg = a.get("config", "")
        sw = parse_sliding_window(cfg)
        nsticks = a.get("max_out_nsticks_per_core")
        if None in (sw["batch"], sw["in_h"], sw["kh"], sw["ph"], sw["num_cores_nhw"]) or nsticks is None:
            continue
        nhw = sw["batch"] * sw["in_h"] * sw["in_w"]
        C = as_int(r[CIN])
        ext_y = int(nsticks) * sw["num_cores_nhw"]
        observed = as_int(r[OY])
        if ext_y != observed:
            mismatches.append((nhw, C, ext_y, observed))
        key = (nhw, C, sw["kh"], sw["kw"], sw["ph"], sw["pw"], sw["is_transpose"])
        rows.setdefault(key, (ext_y, sw["num_cores_nhw"], int(nsticks)))

    print(f"# derived from {path}")
    print(f"# {len(halo)} Halo ops -> {len(rows)} distinct keys")
    print("# key: (nhw, C, kH, kW, pH, pW, is_transpose)")
    print("#   value = max_out_nsticks_per_core * num_cores_nhw")
    print("_HALO_EXT_Y_RESNET50_224 = {")
    for key, (ext_y, ncores, nsticks) in sorted(rows.items(), key=lambda kv: -kv[0][0]):
        nhw, C, kh, kw, ph, pw, tp = key
        print(f"    ({nhw:>7}, {C:>5}, {kh}, {kw}, {ph}, {pw}, {tp}): {ext_y:>7},"
              f"   # {nsticks} sticks x {ncores} cores")
    print("}")

    if mismatches:
        print(f"\n# WARNING: {len(mismatches)} rows where nsticks*cores != OUTPUT_0_Y:")
        for nhw, C, calc, obs in mismatches:
            print(f"#   nhw={nhw} C={C}: computed {calc} vs recorded {obs}")
    else:
        print("\n# all rows: max_out_nsticks_per_core * num_cores_nhw == OUTPUT_0_Y exactly")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    main(sys.argv[1])