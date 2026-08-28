#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Can we fix the three LUT-miss causes from the workload side?

    python probe_lut_fixes.py

Reads tensor state through the SAME helpers the key builder uses
(tools.profiling.shape_canonical), so the answers reflect what actually lands
in the key rather than what the tensor prints.

  1. input_0_memory  (73 misses) -- is memory settable / does to_memory_config tag it?
  2. hw_shape        (42 misses) -- is it writable, and does it survive fan-out?
  3. math_fidelity   (36 misses) -- does it reach op.attrs?
"""
import ttsim.front.ttnn as ttnn
import ttsim.front.ttnn.minitorch_shim as torch
from ttsim.front.ttnn.device import open_device, set_default_device
from ttsim.front.ttnn.tensor import ttnn_random

from tools.profiling.shape_canonical import (
    tensor_datatype, tensor_layout_str, tensor_memory_str,
)

DEV = open_device()
set_default_device(DEV)


def _dt(t):
    """tensor_datatype(tensor, op_precision); op_precision disambiguates
    BF16-stored-as-numpy.float16 from IEEE FP16, so pass None and fall back."""
    for args in ((t, None), (t, "bfloat8_b"), (t,)):
        try:
            return tensor_datatype(*args)
        except TypeError:
            continue
        except Exception as exc:
            return f"<{type(exc).__name__}>"
    return "<unavailable>"


def _safe(fn, t):
    try:
        return fn(t)
    except Exception as exc:
        return f"<{type(exc).__name__}>"


def state(label, t):
    print(f"  {label:38} mem={_safe(tensor_memory_str, t)!s:26} "
          f"layout={_safe(tensor_layout_str, t)!s:10} dt={_dt(t)!s:12} "
          f"hw_shape={getattr(t, 'hw_shape', None)}")


print("=" * 78)
print("1. MEMORY  (LUT wants DEV_1_L1_HEIGHT_SHARDED; 73 ops mismatch)")
print("=" * 78)
t = ttnn.from_torch(ttnn_random((1, 1, 50176, 64), -1, 1, dtype=torch.bfloat16),
                    dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)
state("fresh from_torch", t)

try:
    mc = ttnn.create_sharded_memory_config_(
        [56, 64], ttnn.CoreGrid(x=8, y=8),
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.ShardOrientation.ROW_MAJOR, True)
    print(f"  create_sharded_memory_config_ -> {mc!r}")
    t2 = ttnn.to_memory_config(t, mc)
    state("after to_memory_config(HEIGHT)", t2)
except Exception as exc:
    print(f"  to_memory_config path FAILED: {type(exc).__name__}: {exc}")

for attr in ("memory_config", "memory", "_memory_config", "mem_config", "buffer_type"):
    print(f"  tensor.{attr:16} = {getattr(t, attr, '<absent>')!r}")

# can we tag it directly?
for attr in ("memory_config", "_memory_config", "memory"):
    try:
        setattr(t, attr, "DEV_1_L1_HEIGHT_SHARDED")
        print(f"  set tensor.{attr} -> tensor_memory_str now {tensor_memory_str(t)!r}")
    except Exception as exc:
        print(f"  set tensor.{attr} failed: {type(exc).__name__}")

print()
print("=" * 78)
print("2. HW_SHAPE  (42 ops key on NCHW because hw_shape is absent)")
print("=" * 78)
a = ttnn.from_torch(ttnn_random((16, 1024, 14, 14), -1, 1, dtype=torch.bfloat16),
                    dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)
state("NCHW tensor as created", a)
try:
    a.hw_shape = [1, 1, 16 * 14 * 14, 1024]
    state("after setting hw_shape", a)
    print("  -> WRITABLE: workload can backfill hw_shape on fan-out tensors")
except Exception as exc:
    print(f"  hw_shape NOT writable: {type(exc).__name__}: {exc}")

print()
print("=" * 78)
print("3. MATH_FIDELITY  (36 ops mismatch on LoFi)")
print("=" * 78)
w = ttnn.from_torch(ttnn_random((64, 64, 1, 1), -1, 1, dtype=torch.bfloat16),
                    dtype=ttnn.bfloat8_b, layout=ttnn.ROW_MAJOR_LAYOUT)
b = ttnn.from_torch(ttnn_random((64,), -1, 1, dtype=torch.bfloat16),
                    dtype=ttnn.bfloat8_b, layout=ttnn.ROW_MAJOR_LAYOUT)
try:
    ck = ttnn.init_device_compute_kernel_config(
        DEV.arch(), math_fidelity=ttnn.MathFidelity.LoFi)
    print(f"  compute_kernel_config -> {ck!r}")
except Exception as exc:
    ck = None
    print(f"  init_device_compute_kernel_config FAILED: {exc}")

out = ttnn.conv2d(input_tensor=a, weight_tensor=w, bias_tensor=b,
                  in_channels=1024, out_channels=64, batch_size=16,
                  input_height=14, input_width=14, kernel_size=(1, 1),
                  stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1,
                  device=DEV, compute_config=ck)
out = out[0] if isinstance(out, (tuple, list)) else out
print(f"  conv2d accepted compute_config, out={getattr(out,'shape',out)}")

g = DEV.get_graph() if hasattr(DEV, "get_graph") else None
if g is not None:
    try:
        ops = list(g.get_ordered_nodes()) if hasattr(g, "get_ordered_nodes") else []
        last = g.get_op(ops[-1]) if ops else None
        if last is not None:
            attrs = getattr(last, "attrs", None)
            print(f"  last op {ops[-1]} attrs = {attrs}")
            print("  -> does attrs carry math_fidelity? "
                  f"{'YES' if attrs and any('fidelity' in str(k).lower() for k in attrs) else 'NO'}")
    except Exception as exc:
        print(f"  graph inspection failed: {type(exc).__name__}: {exc}")
else:
    print("  device exposes no get_graph(); inspect op.attrs via the opstats 'attrs' column")