# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Polaris (ttsim) parameter providers for the Blackhole (p100) ResNet-50 port.

Mirrors ``workloads/ttnn/vit/bh/vit_polaris_params_bh.py``: on hardware the
parameter tree comes from ``preprocess_model_parameters`` + the ResNet custom
preprocessor (which folds BatchNorm into the convs and produces a bias for
every conv).  Under Polaris there is no torch checkpoint, so we synthesize a
structurally identical tree of correctly shaped random ttnn tensors.

Shapes the model file (``ttnn_functional_resnet50``) actually reads:

    parameters.conv1.weight            [64, 16, 4, 4]   (folded stem conv)
    parameters.conv1.bias              [64]             (rank-1; see _conv_bias)
    parameters.layerN[i].conv1.weight  [planes, in_ch, 1, 1]
    parameters.layerN[i].conv2.weight  [planes, planes, 3, 3]
    parameters.layerN[i].conv3.weight  [4*planes, planes, 1, 1]
    parameters.layerN[i].downsample.*  [4*planes, in_ch, 1, 1]   (first block only)
    parameters.fc.weight               [1, 1, 2048, 1024]  (transposed + tile padded)
    parameters.fc.bias                 [1, 1, 1, 1024]

The asserts in ``resnet50Bottleneck.__init__`` / ``resnet50.__init__`` read
``weight.shape[0..2]``, so the ranks and kernel extents above must be exact.
"""
import math

import ttsim.front.ttnn as ttnn
import ttsim.front.ttnn.minitorch_shim as torch  # type: ignore[no-redef]
from ttsim.front.ttnn.tensor import ttnn_random as torch_random

# ---------------------------------------------------------------------------
# Model topology / config
# ---------------------------------------------------------------------------

config_dict = {
    "image_size": 224,
    "image_channels": 3,
    "num_classes": 1000,
    # stem: ttnn.fold(stride=2) with a 3-pixel halo pad, so the folded conv1
    # sees 115x115 x (nearest_y(3, 4) * 2 * 2) = 115x115x16 with a 4x4 kernel.
    "fold_stride": 2,
    "fold_pad": 3,
    "conv1_kernel_size": 4,
    "conv1_output_channels": 64,
    # [3, 4, 6, 3] is asserted in resnet50.__init__ -- do not change.
    "layers": [3, 4, 6, 3],
    "planes": [64, 128, 256, 512],
    "strides": [1, 2, 2, 2],
    "expansion": 4,
    "fc_in_features": 2048,
}


def _nearest_y(x: int, y: int) -> int:
    return math.ceil(x / y) * y


def nearest_32(x: int) -> int:
    return _nearest_y(x, 32)


class ParameterDict(dict):
    """dict with attribute access, so ``parameters.layer1[0].conv2.weight`` works.

    ``preprocess_model_parameters`` returns ttnn's own ParameterDict; the model
    file only ever uses attribute access on it plus integer indexing on the
    per-layer lists, which a plain ``list`` covers.
    """

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as exc:  # pragma: no cover
            raise AttributeError(key) from exc

    def __setattr__(self, key, value):
        self[key] = value


# ---------------------------------------------------------------------------
# Tensor factories
# ---------------------------------------------------------------------------


def _conv_weight(out_channels: int, in_channels: int, kernel_h: int, kernel_w: int,
                 dtype=ttnn.bfloat16):
    """Conv weights stay in torch OIHW layout on host; ttnn.conv2d prepares them."""
    return ttnn.from_torch(
        torch_random((out_channels, in_channels, kernel_h, kernel_w), -0.1, 0.1, dtype=torch.bfloat16),
        dtype=dtype,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )


def _conv_bias(out_channels: int, dtype=ttnn.bfloat16):
    """Conv bias as rank-1 (C_out,).

    On hardware the custom preprocessor emits [1, 1, 1, C_out] and tt-metal's
    conv2d accepts it, but ttsim's conv shape inference (ttsim/ops/desc/nn.py,
    conv_sinf) requires (C_out,) and raises otherwise.  Nothing in the model
    file inspects the bias rank -- only weight.shape[0..2] is read -- so rank-1
    is safe here.
    """
    return ttnn.from_torch(
        torch_random((out_channels,), -0.1, 0.1, dtype=torch.bfloat16),
        dtype=dtype,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )


def polaris_parameters_conv(out_channels: int, in_channels: int, kernel_size: int,
                            weights_dtype=ttnn.bfloat16) -> ParameterDict:
    return ParameterDict(
        weight=_conv_weight(out_channels, in_channels, kernel_size, kernel_size, dtype=weights_dtype),
        bias=_conv_bias(out_channels, dtype=weights_dtype),
    )


# ---------------------------------------------------------------------------
# Bottleneck / layer / whole-model parameter trees
# ---------------------------------------------------------------------------


def polaris_parameters_bottleneck(in_channels: int, planes: int, stride: int = 1,
                                  downsample: bool = False,
                                  weights_dtype=ttnn.bfloat16) -> ParameterDict:
    """Parameters for one ``resnet50Bottleneck``.

    ``stride`` is only carried by the block object itself (conv2 uses it); the
    weight shapes do not depend on it.
    """
    out_channels = planes * config_dict["expansion"]
    parameters = ParameterDict(
        conv1=polaris_parameters_conv(planes, in_channels, 1, weights_dtype),
        conv2=polaris_parameters_conv(planes, planes, 3, weights_dtype),
        conv3=polaris_parameters_conv(out_channels, planes, 1, weights_dtype),
    )
    if downsample:
        parameters.downsample = polaris_parameters_conv(out_channels, in_channels, 1, weights_dtype)
    return parameters


def polaris_parameters_layer(in_channels: int, planes: int, blocks: int, stride: int,
                             weights_dtype=ttnn.bfloat16) -> list:
    """Parameters for one of layer1..layer4, as an indexable list of blocks.

    Downsample lives on the first block only, and exactly when
    ``stride != 1 or in_channels != planes * expansion`` -- same predicate as
    ``resnet50._make_layer``, so layer1 (64 -> 256) does have a downsample.
    """
    expansion = config_dict["expansion"]
    out_channels = planes * expansion
    layer = [
        polaris_parameters_bottleneck(
            in_channels, planes, stride,
            downsample=(stride != 1 or in_channels != out_channels),
            weights_dtype=weights_dtype,
        )
    ]
    for _ in range(1, blocks):
        layer.append(
            polaris_parameters_bottleneck(out_channels, planes, 1, downsample=False,
                                          weights_dtype=weights_dtype)
        )
    return layer


def polaris_parameters_fc(in_features: int = 2048, num_classes: int = 1000,
                          weights_dtype=ttnn.bfloat8_b) -> ParameterDict:
    """Parameters for ``ResnetLinear``.

    The weight is already transposed to [K, N] and the N dim is padded to a
    tile multiple (1000 -> 1024) because the hardcoded matmul program config
    uses a (8, 4) grid with per_core_N=1: N = 32 cores * 32 = 1024, and
    in0_block_w=2 => K = 2 * 32 * 32 = 2048.
    """
    padded_classes = nearest_32(num_classes)
    return ParameterDict(
        weight=ttnn.from_torch(
            torch_random((1, 1, in_features, padded_classes), -0.05, 0.05, dtype=torch.bfloat16),
            dtype=weights_dtype,
            layout=ttnn.TILE_LAYOUT,
        ),
        bias=ttnn.from_torch(
            torch_random((1, 1, 1, padded_classes), -0.05, 0.05, dtype=torch.bfloat16),
            dtype=weights_dtype,
            layout=ttnn.TILE_LAYOUT,
        ),
    )


def polaris_parameters_resnet50(num_classes: int = 1000,
                                weights_dtype=ttnn.bfloat16) -> ParameterDict:
    """Full parameter tree consumed by ``resnet50.__init__``."""
    image_channels = config_dict["image_channels"]
    fold_stride = config_dict["fold_stride"]
    # resnet50.__init__ asserts conv1.weight.shape[2] == 4 and derives
    # conv1_input_channels from shape[1].
    conv1_in_channels = _nearest_y(image_channels, 4) * (fold_stride * fold_stride)

    parameters = ParameterDict(
        conv1=polaris_parameters_conv(
            config_dict["conv1_output_channels"],
            conv1_in_channels,
            config_dict["conv1_kernel_size"],
            weights_dtype,
        ),
        fc=polaris_parameters_fc(config_dict["fc_in_features"], num_classes),
    )

    in_channels = config_dict["conv1_output_channels"]
    for idx, (planes, blocks, stride) in enumerate(
        zip(config_dict["planes"], config_dict["layers"], config_dict["strides"]), start=1
    ):
        parameters[f"layer{idx}"] = polaris_parameters_layer(
            in_channels, planes, blocks, stride, weights_dtype
        )
        in_channels = planes * config_dict["expansion"]

    return parameters
