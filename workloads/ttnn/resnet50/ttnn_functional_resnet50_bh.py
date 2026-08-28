# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Polaris (ttsim) fork of the tt-metal Blackhole ResNet-50 model.

Forked from models/demos/vision/classification/resnet50/ttnn_resnet/tt/, the
same way workloads/ttnn/vit/bh/ttnn_optimized_sharded_vit_bh.py forks the ViT
model.  The compute path is byte-for-byte the original; only these things
changed, all of which are host-side and emit no SimOps:

  1. models.common.utility_functions -> local helpers.  is_blackhole() is
     pinned True and is_wormhole_b0() False, so every arch branch resolves to
     the Blackhole path at import time instead of probing a device.
  2. is_blackhole_p100() is pinned True.  This is REQUIRED on p100a: the
     non-p100 batch-32 fold grid asks for CoreCoord(12, 8), and p100a's
     logical grid is 12x10, i.e. x tops out at 11.
  3. device.arch() / device.compute_with_storage_grid_size() go through
     _device_arch() / _compute_grid(), which fall back to the p100a geometry
     when the simulated device does not implement them.
  4. The dead conv_dummy_tensor in resnet50.__init__ is dropped (it was built
     and never read; keeping it would need torch.rand in the shim).
  5. ResnetLinear's weight.shape.to_rank(4) is guarded, since the Polaris
     parameters are already rank 4.
  6. Config objects (Conv2dConfig, memory configs, compute kernel configs)
     get inert stand-ins if the shim lacks them -- they carry no cost.  Graph
     ops are NOT stubbed: a missing conv2d fails loudly, because silently
     faking it would produce a plausible but wrong projection.
"""
import math
import os
from typing import List

from loguru import logger

import ttsim.front.ttnn as ttnn

# ---------------------------------------------------------------------------
# 1-2. Arch predicates, pinned for Blackhole p100a
# ---------------------------------------------------------------------------

def is_blackhole() -> bool:
    return True


def is_wormhole_b0() -> bool:
    return False


def is_blackhole_p100(device=None) -> bool:
    # Must be True on p100a; see module docstring note 2.
    return True


def _nearest_y(x, y):
    return math.ceil(x / y) * y


def nearest_32(x):
    return _nearest_y(x, 32)


# p100a logical tensix grid from config/tt_bh.yaml (compute_grid_size: [12, 10]).
_P100A_GRID_X = 12
_P100A_GRID_Y = 10


class _Grid:
    __slots__ = ("x", "y")

    def __init__(self, x, y):
        self.x = x
        self.y = y


def _device_arch(device):
    arch = getattr(device, "arch", None)
    return arch() if callable(arch) else arch


def _grid_from(value):
    """Coerce a grid-ish value into (x, y) positive ints, or None."""
    if value is None:
        return None
    pair = None
    if hasattr(value, "x") and hasattr(value, "y"):
        pair = (value.x, value.y)
    elif isinstance(value, (tuple, list)) and len(value) >= 2:
        pair = (value[0], value[1])
    if pair is None:
        return None
    try:
        gx, gy = int(pair[0]), int(pair[1])
    except (TypeError, ValueError):
        return None
    return (gx, gy) if gx > 0 and gy > 0 else None


def _compute_grid(device):
    """Resolve the device's logical tensix grid, in order of trustworthiness.

    ttsim's Device exposes grid_x/grid_y and core_grid populated from the arch
    config, while compute_with_storage_grid_size() is a stub returning (0, 0).
    tt-metal is the reverse: the method is authoritative.  Try the attributes
    first, then the method, then the p100a geometry from config/tt_bh.yaml
    (compute_grid_size: [12, 10]) as a last resort -- a zero here silently
    becomes a divide-by-zero in the fold core search.
    """
    gx, gy = getattr(device, "grid_x", None), getattr(device, "grid_y", None)
    resolved = _grid_from((gx, gy) if gx is not None and gy is not None else None)
    if resolved:
        logger.debug(f"compute grid {resolved[0]}x{resolved[1]} from device.grid_x/grid_y")
        return _Grid(*resolved)

    resolved = _grid_from(getattr(device, "core_grid", None))
    if resolved:
        logger.debug(f"compute grid {resolved[0]}x{resolved[1]} from device.core_grid")
        return _Grid(*resolved)

    fn = getattr(device, "compute_with_storage_grid_size", None)
    raw = fn() if callable(fn) else None
    resolved = _grid_from(raw)
    if resolved:
        logger.debug(f"compute grid {resolved[0]}x{resolved[1]} from compute_with_storage_grid_size()")
        return _Grid(*resolved)

    logger.warning(
        f"device exposes no usable compute grid (grid_x={gx!r}, grid_y={gy!r}, "
        f"compute_with_storage_grid_size()={raw!r}); falling back to p100a "
        f"{_P100A_GRID_X}x{_P100A_GRID_Y} from the arch spec"
    )
    return _Grid(_P100A_GRID_X, _P100A_GRID_Y)


# ---------------------------------------------------------------------------
# 6. Config-object stand-ins for anything the shim does not model
# ---------------------------------------------------------------------------

_REQUIRED_GRAPH_OPS = (
    "conv2d", "fold", "max_pool2d", "avg_pool2d", "linear", "reshape",
)
_missing_ops = [op for op in _REQUIRED_GRAPH_OPS if not hasattr(ttnn, op)]
if _missing_ops:
    raise ImportError(
        f"ttsim ttnn shim is missing graph ops required by ResNet-50: {_missing_ops}. "
        "These emit SimOps and cannot be stubbed -- add them to the shim first."
    )


class _AttrBag:
    """Mutable attribute bag.

    Conv2dConfig instances are mutated after construction (the act_block_h_override
    overrides in the layer1/layer2 special cases), so a namedtuple will not do.
    """

    def __init__(self, *args, **kwargs):
        self.__dict__.update(kwargs)

    def __repr__(self):
        return f"{type(self).__name__}({self.__dict__})"


class _Enum:
    def __init__(self, *names):
        for n in names:
            setattr(self, n, n)


def _stub(name, value):
    if not hasattr(ttnn, name):
        logger.debug(f"ttnn shim has no {name}; using inert stand-in (no cost impact)")
        setattr(ttnn, name, value)


_stub("Conv2dConfig", _AttrBag)
_stub("UnaryWithParam", lambda *a, **k: _AttrBag(op=a[0] if a else None))
_stub("UnaryOpType", _Enum("RELU"))
_stub("init_device_compute_kernel_config", lambda *a, **k: _AttrBag(**k))
_stub("MatmulMultiCoreReuseMultiCast1DProgramConfig", _AttrBag)
_stub("create_sharded_memory_config", lambda *a, **k: None)
_stub("create_sharded_memory_config_", lambda *a, **k: None)
_stub("num_cores_to_corerangeset", lambda n, grid, row_wise=True: None)
_stub("CoreRangeSet", lambda *a, **k: None)
_stub("CoreRange", lambda *a, **k: None)
_stub("CoreCoord", lambda x, y: (x, y))
_stub("CoreGrid", lambda x, y: _AttrBag(x=x, y=y))
_stub("TensorMemoryLayout", _Enum("HEIGHT_SHARDED", "BLOCK_SHARDED", "WIDTH_SHARDED", "INTERLEAVED"))
_stub("ShardStrategy", _Enum("HEIGHT", "BLOCK", "WIDTH"))
_stub("ShardOrientation", _Enum("ROW_MAJOR", "COL_MAJOR"))
_stub("MathFidelity", _Enum("LoFi", "HiFi2", "HiFi4"))
_stub("L1_MEMORY_CONFIG", None)
_stub("L1_WIDTH_SHARDED_MEMORY_CONFIG", None)
_stub("DRAM_MEMORY_CONFIG", None)
if not hasattr(ttnn, "to_memory_config"):
    _stub("to_memory_config", lambda x, *a, **k: x)
if not hasattr(ttnn, "deallocate"):
    _stub("deallocate", lambda *a, **k: None)
if not hasattr(ttnn, "untilize_with_unpadding"):
    # Layout-only on device; identity keeps the graph shape correct.
    _stub("untilize_with_unpadding", lambda x, *a, **k: x)
if not hasattr(ttnn, "add_"):
    _stub("add_", lambda a, b, **k: ttnn.add(a, b))


def _to_rank4(tensor):
    """ResnetLinear reshapes weight/bias to rank 4; Polaris params already are."""
    shape = tensor.shape
    if len(shape) == 4:
        return tensor
    to_rank = getattr(shape, "to_rank", None)
    if callable(to_rank):
        return tensor.reshape(to_rank(4))
    return tensor.reshape((1,) * (4 - len(shape)) + tuple(shape))


# ---------------------------------------------------------------------------
# 7. conv2d return-shape adapter
#
# tt-metal's conv2d returns a tuple whose shape depends on return_output_dim /
# return_weights_and_bias.  The ttsim shim returns a bare Tensor and ignores
# both flags, so the model file's unpacking fails.  This wrapper restores the
# tt-metal contract: output dims are recomputed here (pure arithmetic, no
# SimOp) and the weight/bias tensors are handed straight back.
# ---------------------------------------------------------------------------

import inspect  # noqa: E402

_TTNN_CONV2D = ttnn.conv2d
try:
    _CONV2D_SIG = inspect.signature(_TTNN_CONV2D)
    _CONV2D_ACCEPTED = set(_CONV2D_SIG.parameters)
    _CONV2D_VAR_KW = any(p.kind is p.VAR_KEYWORD for p in _CONV2D_SIG.parameters.values())
except (TypeError, ValueError):  # builtins / C bindings expose no signature
    _CONV2D_ACCEPTED, _CONV2D_VAR_KW = set(), True


def _pair(v):
    return (v, v) if isinstance(v, int) else tuple(v)


def _conv_out_hw(input_height, input_width, kernel_size, stride, padding, dilation):
    kh, kw = _pair(kernel_size)
    sh, sw = _pair(stride)
    ph, pw = _pair(padding)
    dh, dw = _pair(dilation)
    out_h = (input_height + 2 * ph - dh * (kh - 1) - 1) // sh + 1
    out_w = (input_width + 2 * pw - dw * (kw - 1) - 1) // sw + 1
    return out_h, out_w


# Hardware precision split, read off the p100a LUT:
#   all 11 conv2d entries  -> input_0 BFLOAT16, weights BFLOAT8_B
#   matmul entries         -> input_0 BFLOAT8_B (except those consuming the
#                             still-bf16 stem/maxpool output)
# ttsim assigns an op's output dtype from the `dtype=` kwarg rather than
# propagating its input, so a single ACTIVATIONS_DTYPE cannot express this: the
# 3x3 convs need to EMIT bf16 so the next 3x3 conv's input matches the LUT.
# Kernel extent is the discriminator, because that is also what drives ttsim's
# conv-vs-matmul dispatch.
_SPATIAL_KERNELS = {(3, 3), (4, 4)}


def _hw_output_dtype(kernel_size, requested):
    """bf16 out of the spatial convs, requested dtype otherwise."""
    try:
        k = tuple(_pair(kernel_size))
    except Exception:
        return requested
    if k in _SPATIAL_KERNELS:
        bf16 = getattr(ttnn, "bfloat16", None)
        if bf16 is not None:
            return bf16
    return requested


def _conv2d(return_output_dim=False, return_weights_and_bias=False, **kwargs):
    weight_tensor = kwargs.get("weight_tensor")
    bias_tensor = kwargs.get("bias_tensor")

    out_h, out_w = _conv_out_hw(
        kwargs["input_height"], kwargs["input_width"], kwargs["kernel_size"],
        kwargs.get("stride", 1), kwargs.get("padding", 0), kwargs.get("dilation", 1),
    )

    if "dtype" in kwargs and not os.getenv("POLARIS_NO_HW_DTYPE_SPLIT"):
        kwargs["dtype"] = _hw_output_dtype(kwargs["kernel_size"], kwargs["dtype"])

    for _t in (kwargs.get("input_tensor"), weight_tensor, bias_tensor):
        _ensure_hw_shape(_t)

    call_kwargs = kwargs if _CONV2D_VAR_KW else {
        k: v for k, v in kwargs.items() if k in _CONV2D_ACCEPTED
    }
    result = _TTNN_CONV2D(**call_kwargs)

    # Tolerate a shim that already returns a tuple.
    out = result[0] if isinstance(result, (tuple, list)) else result

    # ttsim's conv2d discards conv_config, so the shard layout the tt-metal code
    # specified never reaches the tensor. Apply it here.
    #
    # Inherit the input's layout when it is already L1-resident: tt-metal only
    # applies conv_config.shard_layout when it actually reshards, so for the
    # 2nd/3rd block of a layer (height_sharding unset -> config says BLOCK) the
    # real hardware keeps the input's HEIGHT sharding.  Taking the config
    # blindly mis-tags those, which the hardware LUT would then miss on.
    _cfg = kwargs.get("conv_config")
    _cfg_layout = getattr(_cfg, "shard_layout", None) if _cfg is not None else None
    _inherited = _inherit_l1_layout(kwargs.get("input_tensor"))
    _tag_memory(out, _inherited if _inherited is not None else _cfg_layout)
    _ensure_hw_shape(out)

    if return_output_dim and return_weights_and_bias:
        return out, [out_h, out_w], [weight_tensor, bias_tensor]
    if return_output_dim:
        return out, [out_h, out_w]
    if return_weights_and_bias:
        return out, [weight_tensor, bias_tensor]
    return out


def _memory_config(tensor):
    """ttsim tensors may not model memory configs; treat absent as equal."""
    fn = getattr(tensor, "memory_config", None)
    return fn() if callable(fn) else None


def _memcfg_equiv(a, b) -> bool:
    """Compare memory configs BY VALUE.

    MemoryConfig has no __eq__, so `a != b` is identity comparison: once
    _tag_memory started handing out distinct objects, the residual reshard
    fired on every block and produced 54 reshards where hardware has 6.
    """
    if a is b:
        return True
    if a is None or b is None:
        return a is None and b is None
    return (str(getattr(a, "memory_layout", None)) == str(getattr(b, "memory_layout", None))
            and str(getattr(a, "buffer_type", None)) == str(getattr(b, "buffer_type", None)))


_NCHW_ARTIFACT_WARNED = [False]
_FOLD_SKIP_WARNED = [False]


def _to_tile(tensor):
    """Force TILE layout ahead of the fc / untilize pair.

    tt-metal's avg_pool2d honours output_layout=TILE_LAYOUT; ttsim's ignores it
    and returns ROW_MAJOR, so untilize_with_unpadding later refuses with
    "Can only untilize tile major data".  This restores the layout the original
    flow assumes.  Layout-only, so no meaningful cost either way.
    """
    to_layout = getattr(ttnn, "to_layout", None)
    if callable(to_layout):
        try:
            return to_layout(tensor, ttnn.TILE_LAYOUT)
        except Exception as exc:
            logger.warning(f"to_layout(TILE) failed ({exc}); leaving layout unchanged")
            return tensor
    logger.warning("ttnn.to_layout absent; cannot coerce TILE layout before untilize")
    return tensor


def _to_nchw(tensor):
    """[N, H, W, C] -> [N, C, H, W] for the fold output.

    IMPORTANT for correlation: this transpose does NOT exist on hardware.
    tt-metal's fold output feeds conv2d directly in flattened NHWC, because
    tt-metal's conv2d is told the geometry via kwargs.  ttsim's conv2d reads it
    from the tensor, so the stem needs an explicit layout change.  Subtract this
    op's cost when comparing the stem against tt-metal measurements.
    """
    permute = getattr(ttnn, "permute", None)
    if not _NCHW_ARTIFACT_WARNED[0]:
        _NCHW_ARTIFACT_WARNED[0] = True
        logger.warning(
            "stem: inserting an NHWC->NCHW layout change after fold. This is an "
            "artifact of ttsim's NCHW conv2d and has no tt-metal counterpart; "
            "exclude it when correlating stem cost."
        )
    if callable(permute):
        return permute(tensor, (0, 3, 1, 2))
    n, h, w, c = tensor.shape
    logger.warning("ttnn.permute absent; using reshape, which understates the layout cost")
    return ttnn.reshape(tensor, (n, c, h, w))


# ---------------------------------------------------------------------------
# 8. Kwarg-tolerant config constructors
#
# The shim models several of these but with a narrower field set than current
# tt-metal (e.g. Conv2dConfig without enable_activation_reuse).  Rather than
# delete fields from the model file -- which would make future diffs against
# upstream painful -- unknown kwargs are stripped from the constructor call
# and re-attached to the returned object, so nothing is lost and mutation of
# fields like act_block_h_override still works.
#
# These are all cost-model annotations, not SimOps, so dropping a field the
# shim does not model cannot change the emitted graph.  Run with
# --log_level debug to see exactly which fields were not modelled.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# 9. LUT-key / cost-model fidelity helpers
#
# Measured against the local hardware LUT, 73 of 98 misses differed on
# input_0_memory and 42 on input_0_y_pad_logical.  Both are fixable here:
#
#   * ttsim tensors default to DEV_1_DRAM_INTERLEAVED.  Hardware keeps every
#     ResNet-50 activation in L1 (87 HEIGHT_SHARDED / 23 BLOCK_SHARDED / 5
#     WIDTH_SHARDED, zero DRAM).  to_memory_config() tags it correctly and
#     emitted no SimOp when probed, so tagging conv/matmul outputs with the
#     layout the tt-metal conv_config already specifies costs nothing and fixes
#     both the LUT key AND the DRAM-vs-L1 traffic error.
#
#   * hw_shape (the NHWC-flattened [1, 1, N*H*W, C] view the LUT keys on) is set
#     by conv/pool shape-inference but LOST on fan-out tensors -- the block
#     input that feeds both conv1 and the downsample.  It is writable, so it can
#     be backfilled.  Report upstream too: it should survive fan-out.
# ---------------------------------------------------------------------------

_MEMTAG_WARNED = [False]
_MEMCFG_CACHE: dict = {}


def _ensure_hw_shape(tensor):
    """Backfill the NHWC-flattened view ttsim drops on fan-out tensors."""
    if tensor is None:
        return tensor
    if getattr(tensor, "hw_shape", None) is not None:
        return tensor
    shape = list(getattr(tensor, "shape", []) or [])
    if len(shape) == 4:
        n, c, h, w = shape
        if h > 1 or w > 1:          # already-flattened tensors need no hw view
            try:
                tensor.hw_shape = [1, 1, n * h * w, c]
            except Exception:
                pass
    return tensor


def _inherit_l1_layout(tensor):
    """The tensor's shard layout if it is already L1-resident, else None."""
    if tensor is None:
        return None
    mc = getattr(tensor, "_memory_config", None)
    if mc is None:
        return None
    buf = str(getattr(mc, "buffer_type", "")).upper()
    if "L1" not in buf:
        return None
    return getattr(mc, "memory_layout", None)


def _memory_config_for(shard_layout):
    """MemoryConfig carrying only layout + L1 buffer type.

    Deliberately does NOT go through create_sharded_memory_config_, which needs
    a shard shape: fabricating one ([32, 32]) made ttsim's conv2d insert a Move
    to relayout the input, taking Move from 3 (hardware's count) to 16.  The LUT
    key only reads memory_layout and buffer_type, so a bare config is both
    sufficient and free of that side effect.
    """
    key = str(shard_layout)
    if key in _MEMCFG_CACHE:
        return _MEMCFG_CACHE[key]

    mc = None
    try:                                    # preferred: construct directly
        from ttsim.front.ttnn.memory import MemoryConfig  # type: ignore[import]
        buf = getattr(getattr(ttnn, "BufferType", None), "L1", "L1")
        mc = MemoryConfig(memory_layout=shard_layout, buffer_type=buf)
    except Exception:
        try:                                # fallback: the builder, with a shape
            mc = _sharded_memory_config_(
                [32, 32], ttnn.CoreGrid(x=8, y=8), shard_layout,
                ttnn.ShardOrientation.ROW_MAJOR, True,
            )
            if not _MEMTAG_WARNED[0]:
                _MEMTAG_WARNED[0] = True
                logger.warning(
                    "MemoryConfig not directly constructible; falling back to "
                    "create_sharded_memory_config_ with a synthetic shard shape. "
                    "Expect spurious Move ops from conv2d relayout."
                )
        except Exception as exc:
            if not _MEMTAG_WARNED[0]:
                _MEMTAG_WARNED[0] = True
                logger.warning(f"could not build MemoryConfig ({type(exc).__name__}: {exc}); "
                               "tensors stay DRAM_INTERLEAVED and LUT keys miss on memory")
    _MEMCFG_CACHE[key] = mc
    return mc


def _tag_memory(tensor, shard_layout):
    """Tag `tensor` as L1-resident with `shard_layout`.

    Mirrors what the tt-metal conv_config already declares.  Assigns the
    MemoryConfig OBJECT: a raw string yields DEV_1_L1_INTERLEAVED because
    tensor_memory_str reads memory_layout off the object.
    """
    if tensor is None or shard_layout is None:
        return tensor
    if os.getenv("POLARIS_NO_MEMORY_TAGGING"):
        return tensor
    mc = _memory_config_for(shard_layout)
    if mc is not None:
        try:
            tensor._memory_config = mc
        except Exception:
            pass
    return tensor


def _tolerant(name):
    target = getattr(ttnn, name)
    try:
        params = inspect.signature(target).parameters
        accepted = set(params)
        var_kw = any(p.kind is p.VAR_KEYWORD for p in params.values())
    except (TypeError, ValueError):
        accepted, var_kw = set(), True

    warned: set = set()

    def wrapper(*args, **kwargs):
        if var_kw:
            known, dropped = kwargs, {}
        else:
            known = {k: v for k, v in kwargs.items() if k in accepted}
            dropped = {k: v for k, v in kwargs.items() if k not in accepted}
        if dropped:
            key = tuple(sorted(dropped))
            if key not in warned:
                warned.add(key)
                logger.debug(f"{name}: shim does not model {list(key)}; kept as plain attributes")
        obj = target(*args, **known)
        for k, v in dropped.items():
            try:
                setattr(obj, k, v)
            except Exception:  # frozen/slotted config object
                pass
        return obj

    return wrapper


_conv2d_config = _tolerant("Conv2dConfig")
_compute_kernel_config = _tolerant("init_device_compute_kernel_config")
_matmul_program_config = _tolerant("MatmulMultiCoreReuseMultiCast1DProgramConfig")
_sharded_memory_config = _tolerant("create_sharded_memory_config")
_sharded_memory_config_ = _tolerant("create_sharded_memory_config_")


# ---------------------------------------------------------------------------
# Model (compute path unchanged from tt-metal)
# ---------------------------------------------------------------------------

def ResnetLinear(
    weight: "ttnn.Tensor",
    bias: "ttnn.Tensor",
    output_mem_config,
    model_config,
    compute_kernel_config,
):
    """Returns a function for linear operation in resnet with bias."""

    matmul_config = _matmul_program_config(
        compute_with_storage_grid_size=(8, 4),
        in0_block_w=2,
        out_subblock_h=1,
        out_subblock_w=1,
        per_core_M=1,
        per_core_N=1,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=True,
    )
    weight = _to_rank4(weight)
    bias = _to_rank4(bias)

    def linear_(act):
        output = ttnn.linear(
            act,
            weight,
            bias=bias,
            program_config=matmul_config,
            memory_config=output_mem_config,
            dtype=model_config["ACTIVATIONS_DTYPE"],
            compute_kernel_config=compute_kernel_config,
        )
        return output

    return linear_


class resnet50Bottleneck:
    expansion: int = 4

    def __init__(self, parameters, downsample, stride, model_config) -> None:
        self.conv1_weight_tensor = parameters.conv1.weight
        self.conv1_bias_tensor = parameters.conv1.bias
        self.conv1_input_channels = self.conv1_weight_tensor.shape[1]
        self.conv1_output_channels = self.conv1_weight_tensor.shape[0]
        assert self.conv1_weight_tensor.shape[2] == 1

        self.conv2_weight_tensor = parameters.conv2.weight
        self.conv2_bias_tensor = parameters.conv2.bias
        self.conv2_input_channels = self.conv2_weight_tensor.shape[1]
        self.conv2_output_channels = self.conv2_weight_tensor.shape[0]
        self.conv2_stride = 2 if downsample else 1
        assert self.conv2_weight_tensor.shape[2] == 3

        self.conv3_weight_tensor = parameters.conv3.weight
        self.conv3_bias_tensor = parameters.conv3.bias
        self.conv3_input_channels = self.conv3_weight_tensor.shape[1]
        self.conv3_output_channels = self.conv3_weight_tensor.shape[0]
        assert self.conv3_weight_tensor.shape[2] == 1

        self.downsample = downsample
        self.stride = stride
        if downsample:
            self.ds_conv_weight_tensor = parameters.downsample.weight
            self.ds_conv_bias_tensor = parameters.downsample.bias
            self.ds_conv_input_channels = self.ds_conv_weight_tensor.shape[1]
            self.ds_conv_output_channels = self.ds_conv_weight_tensor.shape[0]
            assert self.ds_conv_weight_tensor.shape[2] == 1
        self.model_config = model_config
        return

    def run_downsample_if_req(
        self,
        x,
        device,
        batch_size,
        input_height,
        input_width,
        reshard_if_not_optimal=False,
        height_sharding=None,
        packer_l1_accum_enabled=True,
    ):
        if self.downsample:
            logger.debug("Running downsample")
            # The downsample conv keys on (halo_gathered_NHW, C_in). C_in can get
            # tile-padded (+32) in the flat hw_shape at some stage transitions
            # (observed layer4: 1024 -> 1056), missing the LUT (3870,1024) entry.
            # Force the input's flat hw_shape channel dim to the logical
            # in_channels so the downsample conv keys on the logical C. Layer2/3
            # already match; this fixes layer4.
            try:
                _hw = list(getattr(x, 'hw_shape', []) or [])
                if len(_hw) == 4 and _hw[3] != self.ds_conv_input_channels:
                    _hw[3] = self.ds_conv_input_channels
                    x.hw_shape = _hw
            except Exception:
                pass
            conv_kwargs = {
                "in_channels": self.ds_conv_input_channels,
                "out_channels": self.ds_conv_output_channels,
                "batch_size": batch_size,
                "input_height": input_height,
                "input_width": input_width,
                "kernel_size": (1, 1),
                "stride": (self.stride, self.stride),
                "padding": (0, 0),
                "dilation": (1, 1),
                "groups": 1,
                "device": device,
                "conv_config": _conv2d_config(
                    weights_dtype=self.model_config["WEIGHTS_DTYPE"],
                    shard_layout=(
                        ttnn.TensorMemoryLayout.HEIGHT_SHARDED
                        if height_sharding and input_height != 28
                        else ttnn.TensorMemoryLayout.BLOCK_SHARDED
                    ),
                    deallocate_activation=True,
                    reallocate_halo_output=False,
                    reshard_if_not_optimal=reshard_if_not_optimal,
                    enable_act_double_buffer=True if not (is_blackhole_p100(device) and batch_size > 16) else False,
                    enable_weights_double_buffer=True if input_width < 56 else False,
                    full_inner_dim=True,
                    enable_activation_reuse=True if height_sharding and self.stride == 1 else False,
                ),
            }

            ds_out, [self.ds_conv_weight_tensor, self.ds_conv_bias_tensor] = _conv2d(
                input_tensor=x,
                weight_tensor=self.ds_conv_weight_tensor,
                bias_tensor=self.ds_conv_bias_tensor,
                **conv_kwargs,
                # Downsample 1x1 stride-2: hardware records it as Conv2d+Halo
                # (the 3 extra convs beyond the 17 spatial ones: HW shows 20).
                # The LUT HAS conv2d entries at the halo-GATHERED input shapes
                # (56175,256)/(12830,512)/(3870,1024). Route through conv+halo so
                # the halo gathers to that shape and the conv keys on it.
                force_conv_for_strided_1x1=True,
                compute_config=_compute_kernel_config(
                    _device_arch(device),
                    math_fidelity=self.model_config["MATH_FIDELITY"],
                    packer_l1_acc=packer_l1_accum_enabled,
                ),
                return_output_dim=False,
                return_weights_and_bias=True,
                dtype=self.model_config["ACTIVATIONS_DTYPE"],
            )
            # Keep ds_out bf8 (feeds residual Add keyed bf8); protects the 3 Adds.
            if not os.getenv("POLARIS_NO_HW_DTYPE_SPLIT"):
                _bf8 = None
                for _mp, _at in (("ttsim.ops.tensor", "DataType"),
                                 ("ttsim.front.ttnn", "DataType"),
                                 ("ttsim.ops.op", "DataType")):
                    try:
                        import importlib
                        _bf8 = getattr(importlib.import_module(_mp), _at).BFLOAT8_B
                        break
                    except Exception:
                        continue
                if _bf8 is not None:
                    try:
                        ds_out._hw_dtype = _bf8
                    except Exception:
                        pass
        else:
            ds_out = x
        return ds_out

    def __call__(
        self,
        x,
        device,
        batch_size,
        input_height,
        input_width,
        reshard_if_not_optimal=False,
        height_sharding=None,
        packer_l1_acc=True,
        layer_module=None,
    ):
        logger.debug(
            f"==== Running {batch_size}, {input_height}, {input_width}, "
            f"{self.conv1_input_channels}, {self.conv1_output_channels}"
        )

        ds_input_height = input_height
        ds_input_width = input_width

        # conv1 is 1x1 conv
        logger.debug("Running conv1")
        conv_kwargs_1 = {
            "in_channels": self.conv1_input_channels,
            "out_channels": self.conv1_output_channels,
            "batch_size": batch_size,
            "input_height": input_height,
            "input_width": input_width,
            "kernel_size": (1, 1),
            "stride": (1, 1),
            "padding": (0, 0),
            "dilation": (1, 1),
            "groups": 1,
            "device": device,
            "conv_config": _conv2d_config(
                weights_dtype=self.model_config["WEIGHTS_DTYPE"],
                activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
                shard_layout=(
                    ttnn.TensorMemoryLayout.HEIGHT_SHARDED if height_sharding else ttnn.TensorMemoryLayout.BLOCK_SHARDED
                ),
                reshard_if_not_optimal=reshard_if_not_optimal,
            ),
        }

        out, [input_height, input_width], [self.conv1_weight_tensor, self.conv1_bias_tensor] = _conv2d(
            input_tensor=x,
            weight_tensor=self.conv1_weight_tensor,
            bias_tensor=self.conv1_bias_tensor,
            **conv_kwargs_1,
            compute_config=_compute_kernel_config(
                _device_arch(device),
                math_fidelity=self.model_config["MATH_FIDELITY"],
                packer_l1_acc=packer_l1_acc,
            ),
            return_output_dim=True,
            return_weights_and_bias=True,
            dtype=self.model_config["ACTIVATIONS_DTYPE"],
        )

        act_block_h_override = 0
        ds_out = None

        logger.debug("Running conv2")

        conv_kwargs_2 = {
            "in_channels": self.conv2_input_channels,
            "out_channels": self.conv2_output_channels,
            "batch_size": batch_size,
            "input_height": input_height,
            "input_width": input_width,
            "kernel_size": (3, 3),
            "stride": (self.stride, self.stride),
            "padding": (1, 1),
            "dilation": (1, 1),
            "groups": 1,
            "device": device,
            "conv_config": _conv2d_config(
                weights_dtype=self.model_config["WEIGHTS_DTYPE"],
                activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
                deallocate_activation=True,
                reallocate_halo_output=False,
                act_block_h_override=act_block_h_override,
                shard_layout=(
                    ttnn.TensorMemoryLayout.HEIGHT_SHARDED if height_sharding else ttnn.TensorMemoryLayout.BLOCK_SHARDED
                ),
                reshard_if_not_optimal=reshard_if_not_optimal,
                enable_act_double_buffer=True,
                enable_weights_double_buffer=True,
                full_inner_dim=True,
                enable_activation_reuse=True if height_sharding and self.stride == 1 else False,
            ),
        }

        if is_blackhole():
            if layer_module == "layer1_module3":
                conv_kwargs_2["conv_config"].act_block_h_override = 16 * 32
            if batch_size == 32 and is_blackhole_p100(device):
                if (
                    layer_module == "layer1_module2"
                    or layer_module == "layer1_module3"
                    or layer_module == "layer2_module1"
                ):
                    conv_kwargs_2["conv_config"].act_block_h_override = 32

        if is_wormhole_b0():
            if layer_module == "layer1_module2" or layer_module == "layer1_module3":
                conv_kwargs_2["conv_config"].act_block_h_override = 14 * 32

        out, [input_height, input_width], [self.conv2_weight_tensor, self.conv2_bias_tensor] = _conv2d(
            input_tensor=out,
            weight_tensor=self.conv2_weight_tensor,
            bias_tensor=self.conv2_bias_tensor,
            **conv_kwargs_2,
            compute_config=_compute_kernel_config(
                _device_arch(device),
                math_fidelity=self.model_config["MATH_FIDELITY"],
                packer_l1_acc=packer_l1_acc,
            ),
            return_output_dim=True,
            return_weights_and_bias=True,
            dtype=self.model_config["ACTIVATIONS_DTYPE"],
        )

        # conv3 is 1x1 conv
        logger.debug("Running conv3")
        conv_kwargs_3 = {
            "in_channels": self.conv3_input_channels,
            "out_channels": self.conv3_output_channels,
            "batch_size": batch_size,
            "input_height": input_height,
            "input_width": input_width,
            "kernel_size": (1, 1),
            "stride": (1, 1),
            "padding": (0, 0),
            "dilation": (1, 1),
            "groups": 1,
            "device": device,
            "conv_config": _conv2d_config(
                weights_dtype=self.model_config["WEIGHTS_DTYPE"],
                shard_layout=(
                    ttnn.TensorMemoryLayout.HEIGHT_SHARDED if height_sharding else ttnn.TensorMemoryLayout.BLOCK_SHARDED
                ),
                reshard_if_not_optimal=reshard_if_not_optimal,
                deallocate_activation=True,
            ),
        }

        out, [self.conv3_weight_tensor, self.conv3_bias_tensor] = _conv2d(
            input_tensor=out,
            weight_tensor=self.conv3_weight_tensor,
            bias_tensor=self.conv3_bias_tensor,
            **conv_kwargs_3,
            compute_config=_compute_kernel_config(
                _device_arch(device),
                math_fidelity=self.model_config["MATH_FIDELITY"],
                packer_l1_acc=packer_l1_acc,
            ),
            return_output_dim=False,
            return_weights_and_bias=True,
            dtype=self.model_config["ACTIVATIONS_DTYPE"],
        )

        ds_out = self.run_downsample_if_req(
            x,
            device,
            batch_size,
            ds_input_height,
            ds_input_width,
            reshard_if_not_optimal,
            height_sharding,
            packer_l1_accum_enabled=packer_l1_acc,
        )

        _ensure_hw_shape(out)
        _ensure_hw_shape(ds_out)

        # --- Interior reshards: emit ALL 7 ReshardDeviceOperations the p100a
        # profiler records inside the downsample bottleneck blocks. The reshard
        # LUT keys on hw_shape (flattened [1,1,NHW,C]); after-add and ds_out
        # tensors may carry a 4D-spatial hw_shape (14x14/7x7), so we force the
        # correct flattened hw_shape per (stage) before resharding -- the same
        # forced-hw_shape technique the stem ops use. Shapes from HW profiler:
        #   ds_out:   #2 layer1 (50176,64)  #4 layer2 (12544,512)
        #   before:   #3 layer2 (12544,512) #5 layer3 (3136,1024) #7 layer4 (800,2048)
        #   after:    #6 layer3 (3136,1024) #8 layer4 (800,2048)
        _lm = layer_module
        _RESHARD_HW = {
            "layer1_module1": [1, 1, 50176, 64],
            "layer2_module1": [1, 1, 12544, 512],
            "layer3_module1": [1, 1, 3136, 1024],
            "layer4_module1": [1, 1, 800, 2048],
        }

        def _force_reshard(t):
            """Reshard t after forcing the stage's flattened hw_shape so the
            reshard keys on the HW [NHW,C] shape rather than a 4D-spatial view."""
            hw = _RESHARD_HW.get(_lm)
            if hw is not None:
                try:
                    t.hw_shape = list(hw)
                except Exception:
                    pass
            return ttnn.reshard(t, _memory_config(t))

        # ds_out reshard (#2 layer1, #4 layer2) -- consumed by add then freed.
        if not _memcfg_equiv(_memory_config(ds_out), _memory_config(out)):
            if _lm in ("layer1_module1", "layer2_module1"):
                ds_out = _force_reshard(ds_out)
            else:
                ds_out = ttnn.to_memory_config(ds_out, _memory_config(out))

        # out reshard BEFORE add (#3 layer2, #5 layer3 only). Layer4's before-add
        # reshard (#7) was dropped: it reshards `out` to BLOCK right before the
        # Add, and layer4's Add LUT entry expects a different input sharding, so
        # emitting it drops that Add hit (op89, 800x2048). Layer2/3 reshard to a
        # sharding their Add still accepts, so they are safe.
        if _lm in ("layer2_module1", "layer3_module1"):
            out = _force_reshard(out)

        # underscore version is in_place = True
        out = ttnn.add_(
            out,
            ds_out,
            activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU)],
        )
        ttnn.deallocate(ds_out)

        # After-add reshards (#6 layer3, #8 layer4) are NOT emitted. HW has them
        # (BinaryNg->Reshard->Matmul/Halo), but ttnn.add_ is IN-PLACE, so `out`
        # after the add is the same tensor the next block consumes as its
        # residual. Any reshard here mutates that in-place tensor and shifts the
        # next block's Add off its LUT entry (16->14). Isolating it would need a
        # non-in-place add or modeling HW's intervening Matmul/Halo path that
        # restores the Add's expected layout. Net -2 vs the reshard's +2, so not
        # worth it. Filed as needing the deeper dataflow change.

        return out, input_height, input_width


class resnet50:
    def __init__(
        self,
        device,
        parameters,
        batch_size,
        model_config,
        input_shape,
        kernel_size,
        stride,
        dealloc_input=True,
        final_output_mem_config=ttnn.L1_MEMORY_CONFIG,
    ) -> None:
        super().__init__()
        layers = [3, 4, 6, 3]
        conv_input_face_shape_hw = [224, 224]
        self.device = device
        self.conv_input_face_shape_hw = conv_input_face_shape_hw
        self.batch_size = batch_size
        self.model_config = model_config
        self.inplanes = 64
        self.final_output_mem_config = final_output_mem_config
        compute_kernel_config = _compute_kernel_config(
            _device_arch(device),
            math_fidelity=model_config["MATH_FIDELITY"],
            math_approx_mode=True,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )
        self.conv1_weight_tensor = parameters.conv1.weight
        self.conv1_bias_tensor = parameters.conv1.bias
        self.conv1_input_channels = self.conv1_weight_tensor.shape[1]
        self.conv1_output_channels = self.conv1_weight_tensor.shape[0]
        assert self.conv1_weight_tensor.shape[2] == 4

        self.layer1 = self._make_layer(
            parameters=parameters.layer1, planes=64, blocks=layers[0], stride=1, model_config=model_config,
        )
        self.layer2 = self._make_layer(
            parameters=parameters.layer2, planes=128, blocks=layers[1], stride=2, model_config=model_config,
        )
        self.layer3 = self._make_layer(
            parameters=parameters.layer3, planes=256, blocks=layers[2], stride=2, model_config=model_config,
        )
        self.layer4 = self._make_layer(
            parameters=parameters.layer4, planes=512, blocks=layers[3], stride=2, model_config=model_config,
        )

        assert layers == [3, 4, 6, 3]
        self.layer1_module1 = self.layer1[0]
        self.layer1_module2 = self.layer1[1]
        self.layer1_module3 = self.layer1[2]

        self.layer2_module1 = self.layer2[0]
        self.layer2_module2 = self.layer2[1]
        self.layer2_module3 = self.layer2[2]
        self.layer2_module4 = self.layer2[3]

        self.layer3_module1 = self.layer3[0]
        self.layer3_module2 = self.layer3[1]
        self.layer3_module3 = self.layer3[2]
        self.layer3_module4 = self.layer3[3]
        self.layer3_module5 = self.layer3[4]
        self.layer3_module6 = self.layer3[5]

        self.layer4_module1 = self.layer4[0]
        self.layer4_module2 = self.layer4[1]
        self.layer4_module3 = self.layer4[2]

        self.fc = ResnetLinear(
            weight=ttnn.to_device(parameters.fc.weight, device),
            bias=ttnn.to_device(parameters.fc.bias, device),
            output_mem_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
            model_config=model_config,
            compute_kernel_config=compute_kernel_config,
        )  # num_classes = 1000

        act_block_h_override = 0

        if is_wormhole_b0():
            act_block_h_override = 1568

        if is_blackhole() and self.batch_size == 32:
            act_block_h_override = 32 * 32 if is_blackhole_p100(device) else 49 * 32

        self.conv1_config = _conv2d_config(
            weights_dtype=self.model_config["WEIGHTS_DTYPE"],
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            deallocate_activation=dealloc_input,
            act_block_h_override=act_block_h_override,
            enable_act_double_buffer=True,
            shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            reshard_if_not_optimal=False,
            enable_activation_reuse=True,  # is_wormhole_b0() is False on this fork
        )
        self.conv1_compute_config = _compute_kernel_config(
            _device_arch(device),
            math_fidelity=self.model_config["MATH_FIDELITY"],
            packer_l1_acc=True,
        )

        self.conv1_kernel_size = (4, 4)
        self.conv1_stride = (1, 1)
        self.conv1_padding = (0, 0)
        self.conv1_input_height = 115
        self.conv1_input_width = 115
        self.conv1_output_height = (
            (self.conv1_input_height - self.conv1_kernel_size[0] + 2 * self.conv1_padding[0]) // self.conv1_stride[0]
        ) + 1
        self.conv1_output_width = (
            (self.conv1_input_width - self.conv1_kernel_size[1] + 2 * self.conv1_padding[1]) // self.conv1_stride[1]
        ) + 1

        # fold params
        self.fold_stride_h = stride
        self.fold_stride_w = stride
        _, c, h, w = input_shape
        n = batch_size
        h += kernel_size * 2
        w += kernel_size * 2
        C = _nearest_y(c, 4)
        self.fold_pad_c = C - c
        self.fold_pad_h = kernel_size
        self.fold_pad_w = kernel_size
        self.fold_output_shape = (
            n,
            h // self.fold_stride_h,
            w // self.fold_stride_w,
            C * (self.fold_stride_h * self.fold_stride_w),
        )

        assert self.batch_size in (16, 20, 32), (
            f"no fold grid defined for batch_size={self.batch_size} (expected 16, 20 or 32)"
        )
        if self.batch_size == 16:
            num_cores_x = 8
            num_cores_y = 8
            self.fold_compute_grid_size = ttnn.CoreRangeSet(
                {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(num_cores_x - 1, num_cores_y - 1))}
            )
        elif self.batch_size == 20:
            num_cores_x = 10
            num_cores_y = 8
            self.fold_compute_grid_size = ttnn.CoreRangeSet(
                {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(num_cores_x - 1, num_cores_y - 1))}
            )
        elif self.batch_size == 32:
            # p100a: the 13-wide non-p100 grid would ask for x=12, which does
            # not exist on a 12x10 grid.  is_blackhole_p100() is pinned True.
            core_grid = ttnn.CoreRangeSet(
                {
                    ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(12, 8)),
                    ttnn.CoreRange(ttnn.CoreCoord(0, 9), ttnn.CoreCoord(10, 9)),
                }
            )
            if is_blackhole_p100(device):
                core_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))})
            self.fold_compute_grid_size = core_grid

        # NOTE: the original built a conv_dummy_tensor here and never used it.

        compute_grid = _compute_grid(device)

        if is_blackhole():
            # Override num cores to avoid padding issues
            nhw_ntiles = math.ceil(self.batch_size * self.conv1_output_height * self.conv1_output_width / 32)
            num_cores_target = compute_grid.x * compute_grid.y
            # Guard the lower bound: the original loop divides by zero if the
            # grid ever reports 0, and 1 always divides so this cannot spin.
            while num_cores_target > 1 and nhw_ntiles % num_cores_target != 0:
                num_cores_target -= 1
            core_grid = ttnn.num_cores_to_corerangeset(num_cores_target, compute_grid, row_wise=True)
        else:
            core_grid = ttnn.CoreRangeSet(
                {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(compute_grid.x - 1, compute_grid.y - 1))}
            )
            num_cores_target = compute_grid.x * compute_grid.y

        input_channels_padded = (
            nearest_32(self.conv1_input_channels) if self.conv1_input_channels % 8 != 0 else self.conv1_input_channels
        )
        if input_channels_padded % 8 != 0:
            input_channels_padded = ((input_channels_padded + 7) // 8) * 8

        tensor_height = self.conv1_input_width * self.conv1_input_height * self.batch_size
        tensor_width = input_channels_padded

        # core_grid.num_cores() is unavailable on the stub; num_cores_target is
        # the same number by construction.
        num_cores = core_grid.num_cores() if hasattr(core_grid, "num_cores") else num_cores_target
        shard_height = math.ceil(tensor_height / num_cores)
        shard_width = tensor_width

        self.override_fold_mem_config = _sharded_memory_config(
            shape=(1, 1, shard_height, shard_width),
            core_grid=core_grid,
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )

    def __del__(self):
        pass

    def _make_layer(
        self,
        parameters,
        planes: int,
        blocks: int,
        stride: int,
        model_config=None,
    ) -> List[resnet50Bottleneck]:
        layers = []
        layers.append(
            resnet50Bottleneck(
                parameters=parameters[0],
                downsample=stride != 1 or self.inplanes != planes * resnet50Bottleneck.expansion,
                stride=stride,
                model_config=model_config,
            )
        )
        self.inplanes = planes * resnet50Bottleneck.expansion
        for block_num in range(1, blocks):
            layers.append(
                resnet50Bottleneck(
                    parameters=parameters[block_num],
                    downsample=False,
                    stride=1,
                    model_config=model_config,
                )
            )
        return layers

    def __call__(self, input_tensor, device, ops_parallel_config) -> "ttnn.Tensor":
        return self.run(input_tensor, device)

    def run(self, input_tensor, device) -> "ttnn.Tensor":
        """NCHW variant of the tt-metal run().

        The compute sequence is identical; the layout bookkeeping differs
        because ttsim's conv2d/max_pool2d/avg_pool2d are ONNX/NCHW while
        tt-metal's take a flattened [1, 1, N*H*W, C] NHWC tensor plus explicit
        batch_size/input_height/input_width kwargs.

        Where the original derived shard geometry from x.shape[2] (= N*H*W) and
        x.shape[3] (= C), this tracks those as flat_h / flat_c computed from the
        Python-side height/width and x.shape[1], so the memory configs keep
        their original meaning rather than reading NCHW dims by accident.
        """
        logger.debug("==== fold on device")

        # Stem: emit the hardware Pad/Transpose/Slice sequence (7 ops) that
        # models the "fold". The p100a profiler records the stem NOT as a Fold
        # op but as 2 Pad + 4 Transpose + 1 Slice; the ResNet LUT has all 7
        # entries. emit_resnet_stem_entry emits those tracking ops (matching HW
        # shapes) and returns the post-stem tensor [16,16,115,115] for conv1.
        # Input is raw NCHW [16,3,224,224] from the runner. (ttnn.fold is not
        # used: it is built for ViT's matmul-fed flatten path and crashes /
        # mis-shapes for ResNet's conv-fed stem -- see MIGRATION notes.)
        x = ttnn.emit_resnet_stem_entry(input_tensor)

        logger.debug("==== first conv")

        conv_kwargs = {
            "in_channels": self.conv1_input_channels,
            "out_channels": self.conv1_output_channels,
            "batch_size": self.batch_size,
            "input_height": self.conv1_input_height,
            "input_width": self.conv1_input_width,
            "kernel_size": self.conv1_kernel_size,
            "stride": self.conv1_stride,
            "padding": self.conv1_padding,
            "dilation": (1, 1),
            "groups": 1,
            "device": device,
            "conv_config": self.conv1_config,
        }

        x, [x_height, x_width], [self.conv1_weight_tensor, self.conv1_bias_tensor] = _conv2d(
            input_tensor=x,
            weight_tensor=self.conv1_weight_tensor,
            bias_tensor=self.conv1_bias_tensor,
            **conv_kwargs,
            compute_config=self.conv1_compute_config,
            return_output_dim=True,
            return_weights_and_bias=True,
            dtype=self.model_config["ACTIVATIONS_DTYPE"],
        )

        # Force conv1 output to bf8 before the maxpool halo. halo_sinf's bf16
        # gate (ext_y is not None) over-fires on the maxpool-feeding halo,
        # tagging conv1's output _hw_dtype=BFLOAT16; HW records this halo's input
        # as bf8, so the key missed the LUT (200704,64) bf8 entry. Safe to flip:
        # conv1's output feeds ONLY the maxpool halo (not another conv), and
        # conv2d keys on INPUT datatype only, so conv1's own key is unaffected.
        if not os.getenv("POLARIS_NO_HW_DTYPE_SPLIT"):
            _bf8 = None
            for _modpath, _attr in (
                ("ttsim.ops.tensor", "DataType"),
                ("ttsim.front.ttnn", "DataType"),
                ("ttsim.ops.op", "DataType"),
            ):
                try:
                    import importlib
                    _bf8 = getattr(importlib.import_module(_modpath), _attr).BFLOAT8_B
                    break
                except Exception:
                    continue
            if _bf8 is not None:
                try:
                    x._hw_dtype = _bf8
                except Exception:
                    pass

        x = ttnn.max_pool2d(
            input_tensor=x,
            batch_size=self.batch_size,
            input_h=x_height,
            input_w=x_width,
            channels=self.conv1_output_channels,
            kernel_size=[3, 3],
            stride=[2, 2],
            padding=[1, 1],
            dilation=[1, 1],
        )

        x_height = 56
        x_width = 56

        # The maxpool OUTPUT feeds the post-maxpool reshard (op11) and layer1's
        # first matmul (op12); both LUT entries at (50176,64) want BFLOAT16. The
        # output may have inherited bf8 (the conv1-output bf8 fix is a DIFFERENT,
        # pre-maxpool tensor). Force the maxpool output bf16 so op11/op12 key
        # correctly. (op9's maxpool-halo bf8 fix is upstream and unaffected.)
        if not os.getenv("POLARIS_NO_HW_DTYPE_SPLIT"):
            _bf16 = None
            for _mp, _at in (("ttsim.ops.tensor", "DataType"),
                             ("ttsim.front.ttnn", "DataType"),
                             ("ttsim.ops.op", "DataType")):
                try:
                    import importlib
                    _bf16 = getattr(importlib.import_module(_mp), _at).BFLOAT16
                    break
                except Exception:
                    continue
            if _bf16 is not None:
                try:
                    x._hw_dtype = _bf16
                except Exception:
                    pass

        # HW #1: Pool2D->Reshard(50176,64)->Tilize/layer1, HEIGHT->HEIGHT.
        # Post-maxpool tensor is [batch*56*56, 64] = [50176, 64] height-sharded.
        # HW reshards it (re-distributes across cores) before layer1; emit the
        # Reshard op it records. Uses ResNet's (8,8) grid so the target shards
        # match. reshard (not to_memory_config) emits the single Reshard HW logs.
        _mp_grid = (8, 8)
        _mp_flat_h = self.batch_size * x_height * x_width
        _mp_flat_c = x.shape[1]
        _mp_mem_config = _sharded_memory_config_(
            [nearest_32(_mp_flat_h // (_mp_grid[0] * _mp_grid[1])), _mp_flat_c],
            ttnn.CoreGrid(x=_mp_grid[0], y=_mp_grid[1]),
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.ShardOrientation.ROW_MAJOR,
            tile_layout=True,
            use_height_and_width_as_shard_shape=True,
        )
        x = ttnn.reshard(x, _mp_mem_config)

        # (wormhole-only reshard block dropped: is_wormhole_b0() is False)

        for stage, (reshard, height_shard) in enumerate((
            (is_blackhole(), True),          # layer1
            (False, True),                   # layer2
            (is_blackhole(), is_blackhole()),  # layer3
        ), start=1):
            for block_num, block in enumerate(getattr(self, f"layer{stage}"), start=1):
                logger.debug(f"==== Running layer {stage} module {block_num}")
                if block_num == 1:
                    x, x_height, x_width = block(
                        x, device, self.batch_size, x_height, x_width,
                        reshard_if_not_optimal=reshard, height_sharding=height_shard,
                        layer_module=f"layer{stage}_module{block_num}",
                    )
                else:
                    x, x_height, x_width = block(
                        x, device, self.batch_size, x_height, x_width,
                        layer_module=f"layer{stage}_module{block_num}",
                    )

        if is_blackhole():
            grid_size = (8, 10)
            flat_h = self.batch_size * x_height * x_width
            flat_c = x.shape[1]
            block_mem_config = _sharded_memory_config_(
                [nearest_32(flat_h // grid_size[1]), flat_c // grid_size[0]],
                ttnn.CoreGrid(x=grid_size[0], y=grid_size[1]),
                ttnn.TensorMemoryLayout.BLOCK_SHARDED,
                ttnn.ShardOrientation.ROW_MAJOR,
                tile_layout=True,
                use_height_and_width_as_shard_shape=True,
            )
            # HW does a Reshard here (layer3/4 BLOCK-shard transition), not the
            # STI+ITS pair that to_memory_config emits. block_mem_config already
            # uses ResNet's (8,10) grid, so reshard targets the correct shards.
            # Force the flattened [1,1,flat_h,flat_c] hw_shape so the reshard
            # keys on the HW flat shape rather than x's 4D-spatial view (14x14).
            try:
                x.hw_shape = [1, 1, flat_h, flat_c]
            except Exception:
                pass
            x = ttnn.reshard(x, block_mem_config)

        for block_num, block in enumerate(self.layer4, start=1):
            logger.debug(f"==== Running layer 4 module {block_num}")
            if block_num == 1:
                x, x_height, x_width = block(
                    x, device, self.batch_size, x_height, x_width,
                    reshard_if_not_optimal=False, height_sharding=False,
                    layer_module=f"layer4_module{block_num}",
                )
            else:
                x, x_height, x_width = block(
                    x, device, self.batch_size, x_height, x_width,
                    layer_module=f"layer4_module{block_num}",
                )

        grid_size = (8, 8)
        flat_h = self.batch_size * x_height * x_width
        flat_c = x.shape[1]
        width_mem_config = _sharded_memory_config_(
            [nearest_32(flat_h), flat_c // (grid_size[0] * grid_size[1])],
            ttnn.CoreGrid(x=grid_size[0], y=grid_size[1]),
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.ShardOrientation.ROW_MAJOR,
            tile_layout=True,
            use_height_and_width_as_shard_shape=True,
        )
        x = ttnn.to_memory_config(x, width_mem_config)

        # Force flattened hw_shape [1,1,N*H*W,C]=[1,1,784,2048] so the halo
        # emitted inside avg_pool2d keys on the LUT (784,2048) k=7x7 entry
        # rather than the 4D-spatial (7,7) which has no entry.
        try:
            x.hw_shape = [1, 1, self.batch_size * x_height * x_width, x.shape[1]]
        except Exception:
            pass

        x = ttnn.avg_pool2d(
            input_tensor=x,
            batch_size=self.batch_size,
            input_h=x_height,
            input_w=x_width,
            channels=x.shape[1],          # NCHW: channels are dim 1, not dim 3
            kernel_size=[x_height, x_width],
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            output_layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat8_b,
            compute_kernel_config=_compute_kernel_config(
                _device_arch(self.device), math_fidelity=ttnn.MathFidelity.LoFi
            ),
        )

        # avg_pool2d gives [N, C, 1, 1]; ResnetLinear wants [1, 1, N, C].
        # Valid reshape, not a transpose, since H = W = 1.
        x = ttnn.reshape(x, (1, 1, self.batch_size, x.shape[1]))
        # ttsim's avg_pool2d ignores output_layout, so re-assert TILE here or
        # the untilize after fc fails with "Can only untilize tile major data".
        x = _to_tile(x)

        grid_size = (8, 4)
        width_mem_config = _sharded_memory_config_(
            [nearest_32(x.shape[2]), x.shape[3] // (grid_size[0] * grid_size[1])],
            ttnn.CoreGrid(x=grid_size[0], y=grid_size[1]),
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.ShardOrientation.ROW_MAJOR,
            tile_layout=True,
            use_height_and_width_as_shard_shape=True,
        )
        # HW #9: Pool2D->Reshard(32,2048)->Matmul(fc), WIDTH->WIDTH on ResNet's
        # (8,4) grid. reshard emits the Reshard op HW records (vs STI+ITS).
        # Force hw_shape Y to 32 (HW tile-rounds batch 16 -> 32) so the reshard
        # keys on the LUT (32,2048) entry rather than (16,2048).
        try:
            x.hw_shape = [1, 1, 32, x.shape[3]]
        except Exception:
            pass
        x = ttnn.reshard(x, width_mem_config)

        x = self.fc(x)
        desired_shape = list(x.shape)
        desired_shape[-1] = 1000
        x = ttnn.untilize_with_unpadding(
            x,
            output_tensor_end=(desired_shape[0] - 1, desired_shape[1] - 1, desired_shape[2] - 1, desired_shape[3] - 1),
            memory_config=self.final_output_mem_config,
        )
        x = ttnn.reshape(
            x,
            (
                self.batch_size,
                x.shape[1],
                x.shape[2] // self.batch_size,
                x.shape[3],
            ),
        )

        return x