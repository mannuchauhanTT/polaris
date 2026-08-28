# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Dual-mode ResNet-50 (Blackhole p100) tests: tt-metal hardware + Polaris (ttsim).

Structured after the optimized-sharded ViT (BH) port.  Two differences worth
knowing:

1. ``ttnn_functional_resnet50`` exposes classes, not per-stage free functions,
   so the granularity here is bottleneck -> layer -> fc -> whole model rather
   than one test per op.
2. The ViT port commented its ``@pytest.mark.parametrize`` decorators out and
   relied on defaults.  That would break the requested test id, so instead
   ``_parametrize`` is a no-op under Polaris and the real decorator on
   hardware, letting one file serve both.

Polaris copy lives at workloads/ttnn/resnet50/bh/run_ttnn_functional_resnet50_bh.py,
matching the run_ttnn_optimized_sharded_vit_bh.py naming.  pytest still collects
it when given the path explicitly -- the test_*.py glob only applies to directory
collection -- so the tt-metal side can be the same file or a copy named
test_resnet50_performant.py.

Hardware:
    pytest .../resnet50/blackhole/tests/test_resnet50_performant.py::test_run_resnet50_inference

Polaris:
    python .../test_resnet50_performant.py bottleneck
"""
import argparse
import sys

sys.path.append(".")

import math
import os

IS_POLARIS = os.getenv('IRD_ARCH_NAME', '') == ''

from loguru import logger  # noqa E402

if not IS_POLARIS:
    import pytest  # noqa E402
    import torch  # type: ignore[no-redef] # noqa E402
    import ttnn  # type: ignore[no-redef, import] # noqa E402
    from ttnn.model_preprocessing import preprocess_model_parameters  # type: ignore[import] # noqa F401, E402
else:
    import ttsim.front.ttnn as ttnn
    import ttsim.front.ttnn.minitorch_shim as torch  # type: ignore[no-redef]

    import workloads.ttnn.resnet50.ttnn_functional_resnet50_bh as ttnn_resnet50
    from ttsim.front.ttnn.device import set_default_device
    from ttsim.front.ttnn.tensor import ttnn_random
    torch_random = ttnn_random  # type: ignore[no-redef]

if not IS_POLARIS:
    from models.common.utility_functions import torch_random   # type: ignore[import, no-redef] # noqa F401, E402
    from models.demos.vision.classification.resnet50.ttnn_resnet.tt import (   # type: ignore[import] # noqa F401, E402
        ttnn_functional_resnet50 as ttnn_resnet50,  # type: ignore[no-redef]
    )
    # The Polaris fork needs its own is_blackhole()/is_blackhole_p100(); for a
    # p100 target both are `lambda *_: True`, since every arch branch in the
    # model file then resolves to the Blackhole path.
    from models.demos.vision.classification.resnet50.ttnn_resnet.tt.ttnn_functional_resnet50_model_utils import (   # type: ignore[import] # noqa E402
        is_blackhole_p100,
    )
    from tests.ttnn.utils_for_testing import assert_with_pcc   # type: ignore[import] # noqa F401, E402

if IS_POLARIS:
    from workloads.ttnn.resnet50.resnet50_polaris_params_bh import (  # noqa: E402
        config_dict,
        polaris_parameters_bottleneck,
        polaris_parameters_fc,
        polaris_parameters_layer,
        polaris_parameters_resnet50,
    )


# ---------------------------------------------------------------------------
# Shared constants / helpers
# ---------------------------------------------------------------------------

EXPANSION = 4

# ttnn.MathFidelity may not be modelled by the ttsim front-end; keep the lookup
# soft so import time never fails under Polaris.
_MATH_FIDELITY_LOFI = getattr(getattr(ttnn, "MathFidelity", None), "LoFi", None)

# Per-layer geometry.  in_channels/planes/blocks/stride mirror _make_layer;
# input_height/width are what resnet50.run() feeds each layer for 224x224 input.
#
# height_sharding / reshard are the Blackhole resolutions of the flags in
# resnet50.run() -- note layer3 is HEIGHT sharded on BH (`height_shard =
# is_blackhole()`) where Wormhole block-shards it, and layer4 is block sharded
# on both.
_LAYER_SPECS = {
    "layer1": dict(in_channels=64,   planes=64,  blocks=3, stride=1, input_height=56, input_width=56, height_sharding=True,  reshard=True),
    "layer2": dict(in_channels=256,  planes=128, blocks=4, stride=2, input_height=56, input_width=56, height_sharding=True,  reshard=False),
    "layer3": dict(in_channels=512,  planes=256, blocks=6, stride=2, input_height=28, input_width=28, height_sharding=True,  reshard=True),
    "layer4": dict(in_channels=1024, planes=512, blocks=3, stride=2, input_height=14, input_width=14, height_sharding=False, reshard=False),
}

# resnet50.__init__ only builds a fold grid for these; anything else leaves
# self.fold_compute_grid_size unset.  On p100 the batch-32 fold grid is
# overridden to a single 8x8 range and act double buffering is disabled.
_SUPPORTED_BATCH_SIZES = (16, 20, 32)


def _nearest_y(x: int, y: int) -> int:
    return math.ceil(x / y) * y


def nearest_32(x: int) -> int:
    return _nearest_y(x, 32)


def _conv_out_dim(in_dim: int, stride: int) -> int:
    """3x3, pad 1, dilation 1."""
    return (in_dim + 2 - 3) // stride + 1


_DTYPE_ALIASES = {
    "bfloat8_b": "bfloat8_b", "bfloat8": "bfloat8_b", "bfp8": "bfloat8_b",
    "bfp8_b": "bfloat8_b", "bf8": "bfloat8_b", "bf8_b": "bfloat8_b",
    "bfloat16": "bfloat16", "bf16": "bfloat16",
    "float32": "float32", "fp32": "float32",
}


# DO NOT substitute bfloat16 for bfloat8_b by default.
#
# A standalone probe suggests ttsim ignores the dtype kwarg: from_torch(dtype=
# ttnn.bfloat8_b) returns a tensor reporting float32.  That reading is WRONG for
# the graph.  Measured per-optype bytes/element with bfloat8_b requested:
#
#   Halo 1.07, Move 1.00, ShardedToInterleaved 1.00,
#   InterleavedToSharded 1.00, TilizeWithValPadding 1.00
#
# i.e. bfloat8_b IS honoured at 1 B/elem for the data-movement ops.  Forcing
# bfloat16 doubled all of them (Halo 51.7 MB -> 96.7 MB) and pushed the total
# from 4.49M to 4.75M cycles, further from hardware, not closer.
#
# Substitution is therefore opt-in: set POLARIS_DTYPE_SUBSTITUTION=1 to enable.
# Left in place because MatMul (2.19 B/elem) and Conv (2.50 B/elem) show only
# PARTIAL propagation, so some tensors are still heavier than the hardware's
# bfloat8_b -- worth revisiting once that is understood.
_DTYPE_SUBSTITUTIONS = {"bfloat8_b": "bfloat16"}
_DTYPE_SUBST_WARNED: set = set()


def _resolve_dtype(value):
    """Accept a ttnn dtype object or a name from all_workloads.yaml.

    YAML instance dicts can only carry strings, so 'BFLOAT8_B' has to be mapped
    to a ttnn dtype here.  Unknown names raise rather than silently falling
    back, because a wrong dtype changes modelled bandwidth without failing.
    """
    if value is None:
        return value
    if not isinstance(value, str):
        return value
    key = value.strip().lower()
    attr = _DTYPE_ALIASES.get(key)
    if attr is None:
        raise ValueError(
            f"unknown dtype {value!r}; expected one of {sorted(set(_DTYPE_ALIASES))}"
        )

    if IS_POLARIS and os.getenv("POLARIS_DTYPE_SUBSTITUTION"):
        sub = _DTYPE_SUBSTITUTIONS.get(attr)
        if sub is not None:
            if attr not in _DTYPE_SUBST_WARNED:
                _DTYPE_SUBST_WARNED.add(attr)
                logger.warning(
                    f"POLARIS_DTYPE_SUBSTITUTION is set: replacing {attr} with {sub}. "
                    f"This DOUBLES modelled bytes on the data-movement ops, which "
                    f"measurement shows ttsim already models at 1 B/elem for {attr}. "
                    f"Only useful for isolating dtype effects."
                )
            attr = sub

    resolved = getattr(ttnn, attr, None)
    if resolved is None:
        raise ValueError(f"ttnn has no dtype {attr!r} (shim may not model it)")
    return resolved


def _resolve_fidelity(value):
    """Accept a ttnn.MathFidelity or a name like 'LoFi' / 'HiFi2'."""
    if value is None or not isinstance(value, str):
        return value
    mf = getattr(ttnn, "MathFidelity", None)
    if mf is None:
        logger.warning(f"shim models no MathFidelity; ignoring math_fidelity={value!r}")
        return None
    for cand in (value, value.strip(), value.strip().capitalize()):
        if hasattr(mf, cand):
            return getattr(mf, cand)
    for name in dir(mf):
        if name.lower() == value.strip().lower():
            return getattr(mf, name)
    raise ValueError(f"unknown math_fidelity {value!r}")


def _make_model_config(act_dtype=None, weight_dtype=None, math_fidelity=None) -> dict:
    """Defaults match the hardware run: BFLOAT8_B / BFLOAT8_B / LoFi."""
    # Defaults are expressed as names so they pass through the substitution
    # logic rather than bypassing it as raw dtype objects.
    act = _resolve_dtype(act_dtype if act_dtype is not None else "BFLOAT8_B")
    wt = _resolve_dtype(weight_dtype if weight_dtype is not None else "BFLOAT8_B")
    mf = _resolve_fidelity(math_fidelity)
    return {
        "MATH_FIDELITY": mf if mf is not None else _MATH_FIDELITY_LOFI,
        "WEIGHTS_DTYPE": wt,
        "ACTIVATIONS_DTYPE": act,
    }


def _cfg_kwargs(cfg: dict, fn) -> dict:
    """Pick out the keys of a Polaris instance dict that `fn` actually accepts.

    `bs` and `model_name` are framework-level metadata (they label rows in the
    stats CSV), not workload arguments -- the ViT run_* wrappers ignore cfg
    entirely for that reason.  They are skipped silently here; anything else
    unrecognised is warned about, since it is probably a typo.

    Note `bs` is deliberately NOT mapped onto batch_size: resnet50.__init__
    only builds a fold grid for batch 16/20/32, so pass batch_size explicitly
    in the instance dict if you want to override it.
    """
    import inspect

    if not cfg:
        return {}
    reserved = {"bs", "model_name"}
    accepted = {
        name for name, param in inspect.signature(fn).parameters.items()
        if param.kind is param.POSITIONAL_OR_KEYWORD and name != "device"
    }
    unknown = set(cfg) - accepted - reserved
    if unknown:
        logger.warning(f"{fn.__name__}: ignoring unsupported cfg keys {sorted(unknown)}")
    return {k: v for k, v in cfg.items() if k in accepted}


def _compute_kernel_config(*args, **kwargs):
    """Route through the fork's kwarg-tolerant wrapper under Polaris.

    ttsim's init_device_compute_kernel_config / create_sharded_memory_config_
    take a narrower kwarg set than tt-metal's.  The model fork already wraps
    them; the tests need the same treatment for the calls they make directly.
    """
    if IS_POLARIS:
        return ttnn_resnet50._compute_kernel_config(*args, **kwargs)
    return ttnn.init_device_compute_kernel_config(*args, **kwargs)


def _sharded_memory_config_(*args, **kwargs):
    if IS_POLARIS:
        return ttnn_resnet50._sharded_memory_config_(*args, **kwargs)
    return ttnn.create_sharded_memory_config_(*args, **kwargs)


def _arch(device):
    fn = getattr(device, "arch", None)
    return fn() if callable(fn) else fn


def _assert_shape(output, expected: list, name: str) -> None:
    """Shape check that tolerates tile padding on any dim.

    conv2d/matmul outputs are frequently padded up to a 32 multiple on the
    flattened NHW dim, so an exact match is too strict for a shape-only check.
    """
    actual = list(output.shape)
    ok = len(actual) == len(expected) and all(
        a == e or a == nearest_32(e) for a, e in zip(actual, expected)
    )
    assert ok, f"{name}: expected output shape {expected} (tile padding allowed), but got {actual}"
    logger.info(f"{name}: obtained expected output shape {actual}")


if not IS_POLARIS:
    def _parametrize(*args, **kwargs):
        return pytest.mark.parametrize(*args, **kwargs)

    _DEVICE_PARAMS = [{"l1_small_size": 24576}]
    _PERF_PARAMS = (
        (16, ttnn.bfloat8_b, ttnn.bfloat8_b, ttnn.MathFidelity.LoFi),
        (32, ttnn.bfloat8_b, ttnn.bfloat8_b, ttnn.MathFidelity.LoFi),
    )
    _PRETRAINED_PARAMS = [True, False]
else:
    def _parametrize(*args, **kwargs):
        def _identity(fn):
            return fn
        return _identity

    _DEVICE_PARAMS: list = []
    _PERF_PARAMS: tuple = ()
    _PRETRAINED_PARAMS: list = []


# ---------------------------------------------------------------------------
# Hardware-only helpers
# ---------------------------------------------------------------------------

def _load_torch_model(use_pretrained_weight=True, model_location_generator=None):
    """torchvision ResNet-50 in eval mode, via the tt-metal helper when present."""
    try:
        from models.demos.vision.classification.resnet50.common.common import (  # type: ignore[import]
            load_torch_model,
        )
        return load_torch_model(model_location_generator).eval()
    except ImportError:
        import torchvision  # type: ignore[import]

        weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V1 if use_pretrained_weight else None
        return torchvision.models.resnet50(weights=weights).eval()


def _custom_preprocessor():
    """ResNet needs the custom preprocessor: it folds BatchNorm into each conv
    and produces the conv bias tensors the model file indexes unconditionally.
    """
    try:
        from models.demos.vision.classification.resnet50.ttnn_resnet.tt.custom_preprocessing import (  # type: ignore[import]
            custom_preprocessor,
        )
        return custom_preprocessor
    except ImportError:
        from models.demos.ttnn_resnet.tt.custom_preprocessing import custom_preprocessor  # type: ignore[import]
        return custom_preprocessor


def _to_tt_activation(device, torch_input_nchw, act_dtype):
    """NCHW float torch tensor -> [1, 1, N*H*W, C] tilized ttnn activation.

    Left interleaved in L1: ttnn.conv2d shards its own input, so the test does
    not have to guess a shard spec.
    """
    torch_input_nhwc = torch.permute(torch_input_nchw, (0, 2, 3, 1))
    n, h, w, c = torch_input_nhwc.shape
    torch_input_flat = torch_input_nhwc.reshape(1, 1, n * h * w, c)
    return ttnn.from_torch(
        torch_input_flat,
        dtype=act_dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )


def _from_tt_activation(tt_output, batch_size, height, width, channels):
    """[1, 1, N*H*W, C_padded] ttnn output -> NCHW torch tensor."""
    output = ttnn.to_torch(tt_output)
    output = output.reshape(batch_size, height, width, -1)
    output = output[:, :, :, :channels]
    return torch.permute(output, (0, 3, 1, 2))


def _polaris_activation(batch_size, channels, height, width, act_dtype):
    """Activations in NCHW, which is what ttsim's conv2d expects.

    This deliberately DIVERGES from the hardware path.  tt-metal's conv2d takes
    a flattened [1, 1, N*H*W, C] NHWC tensor plus explicit batch_size /
    input_height / input_width kwargs and ignores the tensor's own layout;
    ttsim's conv2d is ONNX convention and convolves the last two dims of an
    [N, C, H, W] tensor, ignoring those kwargs.  Feeding it the flattened shape
    silently convolves over the wrong axes with C_in = 1 -- it does not raise,
    because ttsim does not validate in_channels against the tensor.

    Safe to do here because resnet50Bottleneck never inspects tensor shapes: it
    threads tensors between convs and tracks input_height/input_width as plain
    Python ints.  Created pre-shaped, so no host-side Permute/Reshape SimOp.
    """
    return ttnn.from_torch(
        torch_random((batch_size, channels, height, width), -0.1, 0.1, dtype=torch.bfloat16),
        dtype=act_dtype,
        layout=ttnn.TILE_LAYOUT,
    )


# ---------------------------------------------------------------------------
# Test functions
# ---------------------------------------------------------------------------

def test_resnet50_bottleneck(
    device,
    batch_size=16,
    layer_name="layer1",
    block_index=0,
    act_dtype=None,
    weight_dtype=None,
    math_fidelity=None,
    *,
    model_location_generator=None,
):
    """One resnet50Bottleneck.  Defaults to layer1_module1, the downsample block."""
    if IS_POLARIS:
        set_default_device(device)
    torch.manual_seed(0)

    spec = _LAYER_SPECS[layer_name]
    planes = spec["planes"]
    out_channels = planes * EXPANSION
    first_block = block_index == 0
    in_channels = spec["in_channels"] if first_block else out_channels
    stride = spec["stride"] if first_block else 1
    downsample = first_block and (stride != 1 or in_channels != out_channels)
    input_height, input_width = spec["input_height"], spec["input_width"]
    output_height = _conv_out_dim(input_height, stride)
    output_width = _conv_out_dim(input_width, stride)
    layer_module = f"{layer_name}_module{block_index + 1}"

    model_config = _make_model_config(act_dtype, weight_dtype, math_fidelity)

    if not IS_POLARIS:
        torch_model = _load_torch_model(model_location_generator=model_location_generator)
        torch_block = getattr(torch_model, layer_name)[block_index]

        torch_input = torch_random(
            (batch_size, in_channels, input_height, input_width), -1, 1, dtype=torch.float32
        )
        torch_output = torch_block(torch_input)

        parameters = preprocess_model_parameters(
            initialize_model=lambda: torch_block,
            custom_preprocessor=_custom_preprocessor(),
            device=device,
        )

        tt_block = ttnn_resnet50.resnet50Bottleneck(
            parameters=parameters,
            downsample=downsample,
            stride=stride,
            model_config=model_config,
        )

        tt_input = _to_tt_activation(device, torch_input, model_config["ACTIVATIONS_DTYPE"])

        # reshard_if_not_optimal=True because the input is interleaved here
        # rather than arriving pre-sharded from an upstream block; run() passes
        # spec["reshard"] instead when the block sits mid-graph.
        tt_output, out_h, out_w = tt_block(
            tt_input,
            device,
            batch_size,
            input_height,
            input_width,
            reshard_if_not_optimal=True,
            height_sharding=spec["height_sharding"],
            layer_module=layer_module,
        )
        output = _from_tt_activation(tt_output, batch_size, out_h, out_w, out_channels)
        assert_with_pcc(torch_output, output.to(torch_output.dtype), 0.98)
    else:
        parameters = polaris_parameters_bottleneck(
            in_channels, planes, stride, downsample=downsample,
            weights_dtype=model_config["WEIGHTS_DTYPE"],
        )
        tt_block = ttnn_resnet50.resnet50Bottleneck(
            parameters=parameters,
            downsample=downsample,
            stride=stride,
            model_config=model_config,
        )
        tt_input = _polaris_activation(
            batch_size, in_channels, input_height, input_width, model_config["ACTIVATIONS_DTYPE"]
        )

        tt_output, out_h, out_w = tt_block(
            tt_input,
            device,
            batch_size,
            input_height,
            input_width,
            reshard_if_not_optimal=True,
            height_sharding=spec["height_sharding"],
            layer_module=layer_module,
        )
        assert (out_h, out_w) == (output_height, output_width), (
            f"Expected spatial dims {(output_height, output_width)}, but got {(out_h, out_w)}"
        )
        _assert_shape(
            tt_output,
            [batch_size, out_channels, output_height, output_width],
            layer_module,
        )


def run_resnet50_bottleneck(wlname: str, device, cfg: dict):
    return test_resnet50_bottleneck(device, **_cfg_kwargs(cfg, test_resnet50_bottleneck))


def test_resnet50_layer(
    device,
    layer_name="layer1",
    batch_size=16,
    act_dtype=None,
    weight_dtype=None,
    math_fidelity=None,
    *,
    model_location_generator=None,
):
    """All bottlenecks of one of layer1..layer4, run back to back."""
    if IS_POLARIS:
        set_default_device(device)
    torch.manual_seed(0)

    spec = _LAYER_SPECS[layer_name]
    in_channels = spec["in_channels"]
    planes = spec["planes"]
    blocks = spec["blocks"]
    stride = spec["stride"]
    out_channels = planes * EXPANSION
    input_height, input_width = spec["input_height"], spec["input_width"]
    output_height = _conv_out_dim(input_height, stride)
    output_width = _conv_out_dim(input_width, stride)
    downsample = stride != 1 or in_channels != out_channels

    model_config = _make_model_config(act_dtype, weight_dtype, math_fidelity)

    def _build(block_parameters):
        tt_blocks = [
            ttnn_resnet50.resnet50Bottleneck(
                parameters=block_parameters[0],
                downsample=downsample,
                stride=stride,
                model_config=model_config,
            )
        ]
        for block_num in range(1, blocks):
            tt_blocks.append(
                ttnn_resnet50.resnet50Bottleneck(
                    parameters=block_parameters[block_num],
                    downsample=False,
                    stride=1,
                    model_config=model_config,
                )
            )
        return tt_blocks

    def _run(tt_blocks, tt_input):
        x, x_height, x_width = tt_input, input_height, input_width
        for block_num, tt_block in enumerate(tt_blocks):
            x, x_height, x_width = tt_block(
                x,
                device,
                batch_size,
                x_height,
                x_width,
                # only the first block sees an interleaved input
                reshard_if_not_optimal=(block_num == 0),
                height_sharding=spec["height_sharding"] if block_num == 0 else None,
                layer_module=f"{layer_name}_module{block_num + 1}",
            )
        return x, x_height, x_width

    if not IS_POLARIS:
        torch_model = _load_torch_model(model_location_generator=model_location_generator)
        torch_layer = getattr(torch_model, layer_name)

        torch_input = torch_random(
            (batch_size, in_channels, input_height, input_width), -1, 1, dtype=torch.float32
        )
        torch_output = torch_layer(torch_input)

        parameters = preprocess_model_parameters(
            initialize_model=lambda: torch_layer,
            custom_preprocessor=_custom_preprocessor(),
            device=device,
        )

        tt_output, out_h, out_w = _run(
            _build(parameters), _to_tt_activation(device, torch_input, model_config["ACTIVATIONS_DTYPE"])
        )
        output = _from_tt_activation(tt_output, batch_size, out_h, out_w, out_channels)
        assert_with_pcc(torch_output, output.to(torch_output.dtype), 0.97)
    else:
        parameters = polaris_parameters_layer(
            in_channels, planes, blocks, stride, weights_dtype=model_config["WEIGHTS_DTYPE"]
        )
        tt_input = _polaris_activation(
            batch_size, in_channels, input_height, input_width, model_config["ACTIVATIONS_DTYPE"]
        )
        tt_output, out_h, out_w = _run(_build(parameters), tt_input)
        assert (out_h, out_w) == (output_height, output_width), (
            f"Expected spatial dims {(output_height, output_width)}, but got {(out_h, out_w)}"
        )
        _assert_shape(
            tt_output,
            [batch_size, out_channels, output_height, output_width],
            layer_name,
        )


def run_resnet50_layer1(wlname: str, device, cfg: dict):
    kwargs = _cfg_kwargs(cfg, test_resnet50_layer)
    kwargs["layer_name"] = "layer1"
    return test_resnet50_layer(device, **kwargs)


def run_resnet50_layer2(wlname: str, device, cfg: dict):
    kwargs = _cfg_kwargs(cfg, test_resnet50_layer)
    kwargs["layer_name"] = "layer2"
    return test_resnet50_layer(device, **kwargs)


def run_resnet50_layer3(wlname: str, device, cfg: dict):
    kwargs = _cfg_kwargs(cfg, test_resnet50_layer)
    kwargs["layer_name"] = "layer3"
    return test_resnet50_layer(device, **kwargs)


def run_resnet50_layer4(wlname: str, device, cfg: dict):
    kwargs = _cfg_kwargs(cfg, test_resnet50_layer)
    kwargs["layer_name"] = "layer4"
    return test_resnet50_layer(device, **kwargs)


def test_resnet50_fc(
    device,
    batch_size=16,
    in_features=2048,
    num_classes=1000,
    act_dtype=None,
    weight_dtype=None,
    math_fidelity=None,
    *,
    model_location_generator=None,
):
    """The ResnetLinear head.

    Note the hardcoded matmul program config: (8, 4) grid, fuse_batch=True,
    per_core_M=1 -- so M must fit one tile, i.e. batch_size <= 32.
    """
    if IS_POLARIS:
        set_default_device(device)
    torch.manual_seed(0)

    model_config = _make_model_config(act_dtype, weight_dtype, math_fidelity)
    padded_classes = nearest_32(num_classes)

    compute_kernel_config = _compute_kernel_config(
        _arch(device),
        math_fidelity=model_config["MATH_FIDELITY"],
        math_approx_mode=True,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )

    grid_size = (8, 4)
    width_mem_config = _sharded_memory_config_(
        [nearest_32(batch_size), in_features // (grid_size[0] * grid_size[1])],
        ttnn.CoreGrid(x=grid_size[0], y=grid_size[1]),
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.ShardOrientation.ROW_MAJOR,
        tile_layout=True,
        use_height_and_width_as_shard_shape=True,
    )

    if not IS_POLARIS:
        torch_model = _load_torch_model(model_location_generator=model_location_generator)
        torch_fc = torch_model.fc

        torch_input = torch_random((batch_size, in_features), -1, 1, dtype=torch.float32)
        torch_output = torch_fc(torch_input)

        parameters = preprocess_model_parameters(
            initialize_model=lambda: torch_fc,
            custom_preprocessor=_custom_preprocessor(),
            device=device,
        )

        tt_input = ttnn.from_torch(
            torch_input.reshape(1, 1, batch_size, in_features),
            dtype=model_config["ACTIVATIONS_DTYPE"],
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )
        tt_input = ttnn.to_memory_config(tt_input, width_mem_config)

        tt_fc = ttnn_resnet50.ResnetLinear(
            weight=ttnn.to_device(parameters.weight, device),
            bias=ttnn.to_device(parameters.bias, device),
            output_mem_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
            model_config=model_config,
            compute_kernel_config=compute_kernel_config,
        )
        output = ttnn.to_torch(tt_fc(tt_input))
        output = output.reshape(batch_size, -1)[:, :num_classes]
        assert_with_pcc(torch_output, output.to(torch_output.dtype), 0.99)
    else:
        parameters = polaris_parameters_fc(in_features, num_classes)
        tt_input = ttnn.from_torch(
            torch_random((1, 1, batch_size, in_features), -0.1, 0.1, dtype=torch.bfloat16),
            dtype=model_config["ACTIVATIONS_DTYPE"],
            layout=ttnn.TILE_LAYOUT,
        )
        tt_input = ttnn.to_memory_config(tt_input, width_mem_config)

        tt_fc = ttnn_resnet50.ResnetLinear(
            weight=parameters.weight,
            bias=parameters.bias,
            output_mem_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
            model_config=model_config,
            compute_kernel_config=compute_kernel_config,
        )
        _assert_shape(tt_fc(tt_input), [1, 1, batch_size, padded_classes], "fc")


def run_resnet50_fc(wlname: str, device, cfg: dict):
    return test_resnet50_fc(device, **_cfg_kwargs(cfg, test_resnet50_fc))


def _run_resnet50(
    device,
    batch_size=16,
    image_size=224,
    image_channels=3,
    num_classes=1000,
    act_dtype=None,
    weight_dtype=None,
    math_fidelity=None,
    use_pretrained_weight=True,
    model_location_generator=None,
):
    """Whole model: fold -> conv1 -> maxpool -> layer1..4 -> avgpool -> fc."""
    if IS_POLARIS:
        set_default_device(device)
    torch.manual_seed(0)

    assert batch_size in _SUPPORTED_BATCH_SIZES, (
        f"resnet50.__init__ only builds a fold grid for {_SUPPORTED_BATCH_SIZES}, got {batch_size}"
    )
    if not IS_POLARIS and batch_size > 16 and is_blackhole_p100(device):
        # Not fatal, but the p100 path disables downsample act double buffering
        # and clamps several act_block_h overrides -- worth seeing in the log.
        logger.warning(f"p100 with batch_size={batch_size}: act double buffering is disabled")

    model_config = _make_model_config(act_dtype, weight_dtype, math_fidelity)
    input_shape = (batch_size, image_channels, image_size, image_size)
    # fold halo pad / stride: 224 + 2*3 = 230, 230 // 2 = 115 = conv1_input_height
    fold_pad = 3
    fold_stride = 2

    def _build(parameters):
        return ttnn_resnet50.resnet50(
            device=device,
            parameters=parameters,
            batch_size=batch_size,
            model_config=model_config,
            input_shape=input_shape,
            kernel_size=fold_pad,
            stride=fold_stride,
        )

    if not IS_POLARIS:
        torch_model = _load_torch_model(use_pretrained_weight, model_location_generator)
        torch_input = torch_random(input_shape, -1, 1, dtype=torch.float32)
        torch_output = torch_model(torch_input)

        parameters = preprocess_model_parameters(
            initialize_model=lambda: torch_model,
            custom_preprocessor=_custom_preprocessor(),
            device=device,
        )
        tt_model = _build(parameters)

        # resnet50.run() folds on device, so it wants a row-major NHWC input.
        torch_input_nhwc = torch.permute(torch_input, (0, 2, 3, 1))
        tt_input = ttnn.from_torch(
            torch_input_nhwc,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        output = ttnn.to_torch(tt_model(tt_input, device, ops_parallel_config={}))
        output = output.reshape(batch_size, -1)[:, :num_classes]
        assert_with_pcc(torch_output, output.to(torch_output.dtype), 0.985)
    else:
        parameters = polaris_parameters_resnet50(
            num_classes=num_classes, weights_dtype=model_config["WEIGHTS_DTYPE"]
        )
        tt_model = _build(parameters)

        # Created already POST-fold, in NCHW: [N, nearest_y(C,4)*s*s, H', W'].
        # ttsim's fold is broken (op.py reshape arity), and its conv2d is NCHW,
        # so pre-shaping past both host-side steps is the only path today --
        # the same technique the ViT port uses for pixel_values.
        # 224 + 2*3 = 230, 230 // 2 = 115;  nearest_y(3, 4) * 2 * 2 = 16.
        # POST-fold NCHW input [N, nearest_y(C,4)*s*s, H', W'] = [16,16,115,115].
        # Fed directly to the model; the fork's run() detects the post-fold shape
        # and skips ttnn.fold (which is designed for ViT's matmul-fed flatten
        # path and produces a flattened output the NCHW stem conv can't consume).
        # RAW NCHW input [N, C, H, W] = [16, 3, 224, 224], fed to the stem
        # emitter (ttnn.emit_resnet_stem_entry). op0 Pad keys its input on the
        # last two dims (H,W)=(224,224), matching the LUT pad entry. NCHW order
        # is required: NHWC [16,224,224,3] would key op0 on (224,3) and miss.
        tt_input = ttnn.from_torch(
            torch_random(
                (batch_size, image_channels, image_size, image_size), -1, 1,
                dtype=torch.bfloat16,
            ),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        # Hardware runs the stem-fold output as L1 HEIGHT_SHARDED (the whole
        # network is 100% L1; nothing is DRAM-resident). ttsim defaults a
        # freshly-created tensor to DRAM_INTERLEAVED, so the stem Halo keys its
        # input_0_memory as DRAM and misses the LUT (which records
        # DEV_1_L1_HEIGHT_SHARDED). Tag it to match. Safe: this is the graph
        # input, consumed only by the stem Halo -- no fan-out, no shared-tensor
        # aliasing. Opt out via POLARIS_NO_MEMORY_TAGGING for A/B.
        if not os.getenv("POLARIS_NO_MEMORY_TAGGING"):
            try:
                from ttsim.front.ttnn.memory import MemoryConfig  # type: ignore
                _bt = getattr(getattr(ttnn, "BufferType", None), "L1", "L1")
                _sl = getattr(getattr(ttnn, "TensorMemoryLayout", None),
                              "HEIGHT_SHARDED", "HEIGHT_SHARDED")
                tt_input._memory_config = MemoryConfig(
                    memory_layout=_sl, buffer_type=_bt
                )
            except Exception:
                pass
        output = tt_model(tt_input, device, ops_parallel_config={})
        _assert_shape(output, [batch_size, 1, 1, num_classes], "resnet50")


def test_resnet50(
    device,
    batch_size=16,
    image_size=224,
    image_channels=3,
    num_classes=1000,
    act_dtype=None,
    weight_dtype=None,
    math_fidelity=None,
    *,
    model_location_generator=None,
):
    return _run_resnet50(
        device,
        batch_size=batch_size,
        image_size=image_size,
        image_channels=image_channels,
        num_classes=num_classes,
        act_dtype=act_dtype,
        weight_dtype=weight_dtype,
        math_fidelity=math_fidelity,
        model_location_generator=model_location_generator,
    )


def run_resnet50(wlname: str, device, cfg: dict):
    return test_resnet50(device, **_cfg_kwargs(cfg, test_resnet50))


# ---------------------------------------------------------------------------
# pytest entry point
#
#   CMD=models/demos/vision/classification/resnet50/blackhole/tests/\
#   test_resnet50_performant.py::test_run_resnet50_inference\
#   [True-16-DataType.BFLOAT8_B-DataType.BFLOAT8_B-MathFidelity.LoFi-device_params0]
#
# Decorator order matters for that id: pytest builds ids bottom-up, so
# use_pretrained_weight -> (batch, act, weight, fidelity) -> device_params.
# ---------------------------------------------------------------------------

@_parametrize("device_params", _DEVICE_PARAMS, indirect=True)
@_parametrize("batch_size, act_dtype, weight_dtype, math_fidelity", _PERF_PARAMS)
@_parametrize("use_pretrained_weight", _PRETRAINED_PARAMS)
def test_run_resnet50_inference(
    device,
    use_pretrained_weight=True,
    batch_size=16,
    act_dtype=None,
    weight_dtype=None,
    math_fidelity=None,
    *,
    model_location_generator=None,
):
    return _run_resnet50(
        device,
        batch_size=batch_size,
        act_dtype=act_dtype,
        weight_dtype=weight_dtype,
        math_fidelity=math_fidelity,
        use_pretrained_weight=use_pretrained_weight,
        model_location_generator=model_location_generator,
    )


# ---------------------------------------------------------------------------
# Registry and CLI
# ---------------------------------------------------------------------------

_STANDALONE_RUN_SPECS: list[tuple[str, object, str]] = [
    ("bottleneck", run_resnet50_bottleneck, "resnet50-bh-bottleneck"),
    ("layer1", run_resnet50_layer1, "resnet50-bh-layer1"),
    ("layer2", run_resnet50_layer2, "resnet50-bh-layer2"),
    ("layer3", run_resnet50_layer3, "resnet50-bh-layer3"),
    ("layer4", run_resnet50_layer4, "resnet50-bh-layer4"),
    ("fc", run_resnet50_fc, "resnet50-bh-fc"),
    ("resnet50", run_resnet50, "resnet50-bh"),
]

_STANDALONE_VALID_SHORT_NAMES = frozenset(s[0] for s in _STANDALONE_RUN_SPECS)


def run_one(callback, wlname: str, cfg: dict):
    if IS_POLARIS:
        from ttsim.front.ttnn.device import close_device, open_device
        device = open_device()
    else:
        from ttnn import close_device, open_device
        device = open_device(device_id=0)
    callback(wlname, device, cfg)
    close_device(device)


def standalone(test_name: str | None = None) -> None:
    """Run all standalone ResNet-50 (BH) tests, or a single test by short name."""
    if test_name is None:
        for _short, fn, wlname in _STANDALONE_RUN_SPECS:
            run_one(fn, wlname, {})
        return
    if test_name not in _STANDALONE_VALID_SHORT_NAMES:
        valid = ", ".join(sorted(_STANDALONE_VALID_SHORT_NAMES))
        logger.error(
            f"Unknown test {test_name}. Valid names: {valid}"
        )
        sys.exit(1)
    for short, fn, wlname in _STANDALONE_RUN_SPECS:
        if short == test_name:
            run_one(fn, wlname, {})
            return


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stdout, level="INFO")
    parser = argparse.ArgumentParser(
        description="Run ResNet-50 (Blackhole p100) standalone tests."
    )
    parser.add_argument(
        "test",
        nargs="?",
        metavar="TEST",
        default="bottleneck",
        help=(
            "Run only this test by short name, "
            "e.g. bottleneck, layer1, fc, resnet50. If omitted, runs 'bottleneck'."
        ),
    )
    _args = parser.parse_args()
    standalone(_args.test)