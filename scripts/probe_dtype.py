# SPDX-License-Identifier: Apache-2.0
"""Which dtype spelling actually sticks on a ttsim tensor?

    python probe_dtype.py

Both ttnn_random(dtype=torch.bfloat16) and from_torch(dtype=ttnn.bfloat8_b)
report float32, so tensors are 4 bytes/element regardless of what we ask for --
which inflates every modelled byte of memory traffic.  This tries every
plausible spelling and reports which ones the Tensor actually honours.
"""
import ttsim.front.ttnn as ttnn
import ttsim.front.ttnn.minitorch_shim as torch
from ttsim.front.ttnn.device import open_device, set_default_device
from ttsim.front.ttnn.tensor import ttnn_random

set_default_device(open_device())
SHAPE = (1, 1, 32, 32)


def report(label, t):
    dt = getattr(t, "dtype", "<no dtype attr>")
    bpe = getattr(t, "bytes_per_element", None)
    nb = getattr(t, "nbytes", None)
    print(f"  {label:46} dtype={str(dt):12} bpe={bpe} nbytes={nb}")


print("what dtype objects exist?")
print(f"  ttnn.bfloat8_b   = {getattr(ttnn, 'bfloat8_b', None)!r}  type={type(getattr(ttnn,'bfloat8_b',None)).__name__}")
print(f"  ttnn.bfloat16    = {getattr(ttnn, 'bfloat16', None)!r}")
print(f"  torch.bfloat16   = {getattr(torch, 'bfloat16', None)!r}  type={type(getattr(torch,'bfloat16',None)).__name__}")
print(f"  torch.float32    = {getattr(torch, 'float32', None)!r}")

print("\nttnn_random with various dtype spellings:")
for label, dt in [("dtype=torch.bfloat16", getattr(torch, "bfloat16", None)),
                  ("dtype=ttnn.bfloat16", getattr(ttnn, "bfloat16", None)),
                  ("dtype=ttnn.bfloat8_b", getattr(ttnn, "bfloat8_b", None)),
                  ("dtype='bfloat8_b' (string)", "bfloat8_b"),
                  ("dtype='bfloat16' (string)", "bfloat16")]:
    if dt is None:
        print(f"  {label:46} <unavailable>")
        continue
    try:
        report(label, ttnn_random(SHAPE, -1, 1, dtype=dt))
    except Exception as exc:
        print(f"  {label:46} FAILED {type(exc).__name__}: {exc}")

print("\nfrom_torch on a float32 source, various target dtypes:")
src = ttnn_random(SHAPE, -1, 1, dtype=getattr(torch, "float32", None))
for label, dt in [("dtype=ttnn.bfloat8_b", getattr(ttnn, "bfloat8_b", None)),
                  ("dtype=ttnn.bfloat16", getattr(ttnn, "bfloat16", None))]:
    if dt is None:
        continue
    try:
        report(label, ttnn.from_torch(src, dtype=dt, layout=ttnn.TILE_LAYOUT))
    except Exception as exc:
        print(f"  {label:46} FAILED {type(exc).__name__}: {exc}")

print("\nis there a cast op?")
for name in ("typecast", "to_dtype", "cast", "as_dtype"):
    fn = getattr(ttnn, name, None)
    print(f"  ttnn.{name:10} {'present' if fn else 'absent'}")
    if fn:
        try:
            report(f"ttnn.{name}(t, bfloat8_b)", fn(src, ttnn.bfloat8_b))
        except Exception as exc:
            print(f"    call failed: {type(exc).__name__}: {exc}")

print("\ncan dtype be set directly on the tensor?")
t = ttnn_random(SHAPE, -1, 1, dtype=getattr(torch, "float32", None))
try:
    t.dtype = ttnn.bfloat8_b
    report("after t.dtype = ttnn.bfloat8_b", t)
except Exception as exc:
    print(f"  assignment failed: {type(exc).__name__}: {exc}")