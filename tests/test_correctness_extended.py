import pytest
import torch
import torch.nn as nn

from complex_conv2d import ComplexConv2d


def _apply_complex(conv_real, conv_imag, x):
    return (
        (conv_real(x.real) - conv_imag(x.imag)).to(torch.complex64)
        + 1j * (conv_real(x.imag) + conv_imag(x.real)).to(torch.complex64)
    )


class ReferenceComplexConv2d(nn.Module):
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
        return _apply_complex(self.conv_real, self.conv_imag, x)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize(
    "shape,out_channels,kernel_size,stride,padding",
    [
        # Representative 3x3 layers used in the benchmark suite.
        ((1, 32, 32, 32), 32, 3, 1, 1),
        ((1, 32, 56, 56), 64, 3, 1, 1),
        ((1, 64, 28, 28), 128, 3, 1, 1),
        ((1, 128, 14, 14), 256, 3, 1, 1),
        ((1, 256, 7, 7), 512, 3, 1, 1),

        # Additional coverage for supported Conv2d parameters.
        ((1, 32, 32, 32), 64, 1, 1, 0),
        ((1, 32, 32, 32), 64, 3, 2, 1),
        ((1, 32, 32, 32), 64, 3, 1, 0),
        ((2, 32, 32, 32), 64, 3, 1, 1),
        ((1, 32, 32, 48), 64, 3, 1, 1),
    ],
)
def test_complex_conv2d_matches_pytorch_reference(
    shape,
    out_channels,
    kernel_size,
    stride,
    padding,
):
    torch.manual_seed(0)

    n, in_channels, h, w = shape

    reference = ReferenceComplexConv2d(
        in_channels,
        out_channels,
        kernel_size,
        stride=stride,
        padding=padding,
    ).cuda().eval()

    cutlass = ComplexConv2d(
        in_channels,
        out_channels,
        kernel_size,
        stride=stride,
        padding=padding,
        bias=False,
        device="cuda",
    ).eval()

    with torch.no_grad():
        cutlass.weight_real.copy_(reference.conv_real.weight)
        cutlass.weight_imag.copy_(reference.conv_imag.weight)

    x = torch.randn(
        n,
        in_channels,
        h,
        w,
        device="cuda",
        dtype=torch.complex64,
    )

    with torch.no_grad():
        y_ref = reference(x)
        y_cutlass = cutlass(x)

    torch.testing.assert_close(
        y_cutlass,
        y_ref,
        rtol=1e-3,
        atol=1e-3,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_planar_fast_path_matches_public_forward():
    torch.manual_seed(1)

    layer = ComplexConv2d(
        32,
        64,
        3,
        padding=1,
        bias=False,
        device="cuda",
    ).eval()

    x = torch.randn(
        1,
        32,
        32,
        32,
        device="cuda",
        dtype=torch.complex64,
    )

    with torch.no_grad():
        y_public = layer(x)

        x_real = x.real.permute(0, 2, 3, 1).contiguous()
        x_imag = x.imag.permute(0, 2, 3, 1).contiguous()

        y_real, y_imag = layer.forward_planar(x_real, x_imag)

        y_planar = torch.complex(
            y_real.permute(0, 3, 1, 2).contiguous(),
            y_imag.permute(0, 3, 1, 2).contiguous(),
        )

    torch.testing.assert_close(
        y_planar,
        y_public,
        rtol=1e-6,
        atol=1e-6,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_rejects_wrong_input_channel_count():
    layer = ComplexConv2d(
        32,
        64,
        3,
        padding=1,
        bias=False,
        device="cuda",
    ).eval()

    x = torch.randn(
        1,
        16,
        32,
        32,
        device="cuda",
        dtype=torch.complex64,
    )

    with pytest.raises(ValueError, match="input channels"):
        layer(x)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_rejects_non_complex64_input():
    layer = ComplexConv2d(
        32,
        64,
        3,
        padding=1,
        bias=False,
        device="cuda",
    ).eval()

    x = torch.randn(
        1,
        32,
        32,
        32,
        device="cuda",
        dtype=torch.float32,
    )

    with pytest.raises(ValueError, match="complex64"):
        layer(x)


def test_rejects_unsupported_options():
    with pytest.raises(NotImplementedError, match="bias"):
        ComplexConv2d(32, 64, 3, bias=True)

    with pytest.raises(NotImplementedError, match="groups"):
        ComplexConv2d(32, 64, 3, groups=2)

    with pytest.raises(NotImplementedError, match="dilation"):
        ComplexConv2d(32, 64, 3, dilation=2)

    with pytest.raises(NotImplementedError, match="padding_mode"):
        ComplexConv2d(32, 64, 3, padding_mode="reflect")
