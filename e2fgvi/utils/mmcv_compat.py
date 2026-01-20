import logging
from typing import Optional

import torch
from torch import nn


_LOGGER = logging.getLogger(__name__)


try:  # pragma: no cover - optional dependency
    from torchvision.ops import deform_conv2d as _deform_conv2d
except Exception:  # pragma: no cover - optional dependency
    _deform_conv2d = None


def constant_init(module: nn.Module, val: float = 0.0, bias: float = 0.0) -> None:
    if hasattr(module, "weight") and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(module.bias, bias)


class ConvModule(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        norm_cfg: Optional[dict] = None,
        act_cfg: Optional[dict] = None,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.norm = None
        if norm_cfg is not None:
            raise RuntimeError("norm_cfg is not supported in this lightweight ConvModule.")
        self.activation = None
        if act_cfg is not None:
            act_type = act_cfg.get("type", "ReLU")
            if act_type == "ReLU":
                self.activation = nn.ReLU(inplace=True)
            else:
                raise RuntimeError(f"Unsupported activation: {act_type}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


def load_checkpoint(model: nn.Module, filename: str, strict: bool = True) -> None:
    if filename.startswith("http"):
        state_dict = torch.hub.load_state_dict_from_url(filename, map_location="cpu")
    else:
        state_dict = torch.load(filename, map_location="cpu")
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    model.load_state_dict(state_dict, strict=strict)


class ModulatedDeformConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        deform_groups: int = 1,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if _deform_conv2d is None:
            raise RuntimeError("torchvision.ops.deform_conv2d is required for E2FGVI.")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        self.dilation = dilation if isinstance(dilation, tuple) else (dilation, dilation)
        self.groups = groups
        self.deform_groups = deform_groups
        weight = torch.empty(out_channels, in_channels // groups, *self.kernel_size)
        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None
        nn.init.kaiming_uniform_(self.weight, a=1)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor, offset: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return modulated_deform_conv2d(
            x,
            offset,
            mask,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
            self.deform_groups,
        )


def modulated_deform_conv2d(
    x: torch.Tensor,
    offset: torch.Tensor,
    mask: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    stride: tuple[int, int],
    padding: tuple[int, int],
    dilation: tuple[int, int],
    groups: int,
    deform_groups: int,
) -> torch.Tensor:
    if _deform_conv2d is None:
        raise RuntimeError("torchvision.ops.deform_conv2d is required for E2FGVI.")
    if deform_groups <= 1 and groups == 1:
        return _deform_conv2d(
            x,
            offset,
            weight,
            bias,
            stride=stride,
            padding=padding,
            dilation=dilation,
            mask=mask,
        )

    if groups != 1:
        raise RuntimeError("Grouped convolution is not supported in this compat layer.")

    in_channels = x.shape[1]
    kh, kw = weight.shape[2], weight.shape[3]
    offset_channels = 2 * kh * kw
    mask_channels = kh * kw
    if in_channels % deform_groups != 0:
        raise RuntimeError("Input channels must be divisible by deform_groups.")

    channels_per_group = in_channels // deform_groups
    outputs = None
    for g in range(deform_groups):
        x_g = x[:, g * channels_per_group : (g + 1) * channels_per_group]
        weight_g = weight[:, g * channels_per_group : (g + 1) * channels_per_group]
        offset_g = offset[:, g * offset_channels : (g + 1) * offset_channels]
        mask_g = mask[:, g * mask_channels : (g + 1) * mask_channels]
        out_g = _deform_conv2d(
            x_g,
            offset_g,
            weight_g,
            bias=None,
            stride=stride,
            padding=padding,
            dilation=dilation,
            mask=mask_g,
        )
        outputs = out_g if outputs is None else outputs + out_g

    if bias is not None:
        outputs = outputs + bias.view(1, -1, 1, 1)

    return outputs
