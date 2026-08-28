# What does the ttsim device report for its compute grid?
from ttsim.front.ttnn.device import open_device
d = open_device()
fn = getattr(d, "compute_with_storage_grid_size", None)
print("device type      :", type(d).__name__)
print("has method       :", callable(fn))
if callable(fn):
    g = fn()
    print("returns          :", repr(g), type(g).__name__)
    print("len              :", len(g) if hasattr(g, "__len__") else "n/a")
    for attr in ("x", "y"):
        print(f"  .{attr}            :", getattr(g, attr, "<absent>"))
    if hasattr(g, "__getitem__"):
        try:
            print("  [0], [1]       :", g[0], g[1])
        except Exception as e:
            print("  indexing failed:", e)
print("\nother device attrs:", [a for a in dir(d) if not a.startswith('_')])