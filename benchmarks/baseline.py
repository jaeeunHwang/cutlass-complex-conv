import torch
import torch.nn as nn


def apply_complex(conv_real, conv_imag, x):
    return (
        (conv_real(x.real) - conv_imag(x.imag)).to(torch.complex64)
        + 1j * (conv_real(x.imag) + conv_imag(x.real)).to(torch.complex64)
    )


class ReferenceComplexConv2d(nn.Module):
    """Reference PyTorch ComplexConv2d used by the benchmark suite."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
    ):
        super().__init__()

        self.conv_real = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.conv_imag = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )

    def forward(self, x):
        if not x.is_complex():
            raise ValueError("ReferenceComplexConv2d expects a complex tensor")
        return apply_complex(self.conv_real, self.conv_imag, x)
