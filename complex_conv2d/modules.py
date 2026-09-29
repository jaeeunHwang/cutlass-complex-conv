import torch
from torch import nn

from ._extension import cutlass_complex_conv


def _pair(value, name):
    if isinstance(value, int):
        return (value, value)

    if (
        isinstance(value, (tuple, list))
        and len(value) == 2
        and all(isinstance(v, int) for v in value)
    ):
        return (int(value[0]), int(value[1]))

    raise TypeError(f"{name} must be an int or a pair of ints")


class ComplexConv2d(nn.Module):
    """Inference-only complex-valued 2D convolution backed by CUTLASS.

    The public ``forward`` method accepts a ``torch.complex64`` tensor in NCHW
    format. ``forward_planar`` exposes the faster internal path and accepts
    separate real/imaginary float32 tensors in contiguous NHWC format.

    Current limitations:
      - CUDA only
      - torch.complex64 only
      - forward inference only (no autograd/backward)
      - bias=False
      - dilation=1
      - groups=1
      - padding_mode="zeros"
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=False,
        padding_mode="zeros",
        device=None,
        dtype=None,
    ):
        super().__init__()

        if in_channels <= 0 or out_channels <= 0:
            raise ValueError("in_channels and out_channels must be positive")

        kernel_size = _pair(kernel_size, "kernel_size")
        stride = _pair(stride, "stride")
        padding = _pair(padding, "padding")
        dilation = _pair(dilation, "dilation")

        if kernel_size[0] <= 0 or kernel_size[1] <= 0:
            raise ValueError("kernel_size must be positive")
        if stride[0] <= 0 or stride[1] <= 0:
            raise ValueError("stride must be positive")
        if padding[0] < 0 or padding[1] < 0:
            raise ValueError("padding must be non-negative")

        if dilation != (1, 1):
            raise NotImplementedError("dilation != 1 is not supported yet")
        if groups != 1:
            raise NotImplementedError("groups != 1 is not supported yet")
        if bias:
            raise NotImplementedError("bias=True is not supported yet")
        if padding_mode != "zeros":
            raise NotImplementedError(
                'padding_mode other than "zeros" is not supported yet'
            )
        if dtype not in (None, torch.complex64):
            raise NotImplementedError("only torch.complex64 is supported")

        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.padding_mode = padding_mode

        # PyTorch-facing weights use KCRS layout. They are prepacked to KRSC
        # lazily before the first CUTLASS invocation.
        weight_shape = (
            self.out_channels,
            self.in_channels,
            self.kernel_size[0],
            self.kernel_size[1],
        )

        self.weight_real = nn.Parameter(
            torch.empty(weight_shape, device=device, dtype=torch.float32),
            requires_grad=False,
        )
        self.weight_imag = nn.Parameter(
            torch.empty(weight_shape, device=device, dtype=torch.float32),
            requires_grad=False,
        )

        nn.init.kaiming_uniform_(self.weight_real)
        nn.init.kaiming_uniform_(self.weight_imag)

        self._clear_prepacked_weights()

    def _clear_prepacked_weights(self):
        self._wr_krsc = None
        self._wi_krsc = None
        self._weight_version = None

    def _apply(self, fn, recurse=True):
        # nn.Module.to()/cuda()/cpu() may replace or move Parameter storage.
        # Cached packed weights must never outlive the storage they came from.
        self._clear_prepacked_weights()
        result = super()._apply(fn, recurse=recurse)
        self._clear_prepacked_weights()
        return result

    @torch.no_grad()
    def _prepack_weights(self):
        if not self.weight_real.is_cuda or not self.weight_imag.is_cuda:
            raise RuntimeError("ComplexConv2d weights must be on CUDA")

        self._wr_krsc = self.weight_real.permute(0, 2, 3, 1).contiguous()
        self._wi_krsc = self.weight_imag.permute(0, 2, 3, 1).contiguous()

        self._weight_version = (
            self.weight_real._version,
            self.weight_imag._version,
        )

    def _ensure_prepacked(self):
        version = (
            self.weight_real._version,
            self.weight_imag._version,
        )

        if (
            self._wr_krsc is None
            or self._wi_krsc is None
            or self._weight_version != version
        ):
            self._prepack_weights()

    def forward_planar(self, x_real, x_imag):
        """Run the fast planar path.

        Args:
            x_real: contiguous CUDA float32 tensor in NHWC layout.
            x_imag: contiguous CUDA float32 tensor in NHWC layout.

        Returns:
            ``(y_real, y_imag)`` as contiguous NHWC float32 tensors.
        """
        self._ensure_prepacked()

        return cutlass_complex_conv.forward(
            x_real,
            x_imag,
            self._wr_krsc,
            self._wi_krsc,
            self.stride[0],
            self.stride[1],
            self.padding[0],
            self.padding[1],
        )

    def forward(self, x):
        """Run complex Conv2d on a torch.complex64 NCHW tensor."""
        if not x.is_cuda:
            raise ValueError("ComplexConv2d input must be on CUDA")
        if x.dtype != torch.complex64:
            raise ValueError(
                f"ComplexConv2d expects torch.complex64 input, got {x.dtype}"
            )
        if x.dim() != 4:
            raise ValueError(
                f"ComplexConv2d expects a 4D NCHW tensor, got {x.dim()}D"
            )
        if x.size(1) != self.in_channels:
            raise ValueError(
                f"expected {self.in_channels} input channels, got {x.size(1)}"
            )
        if x.device != self.weight_real.device:
            raise ValueError(
                "input and ComplexConv2d weights must be on the same CUDA device"
            )

        # PyTorch complex NCHW -> planar float32 NHWC.
        x_real = x.real.permute(0, 2, 3, 1).contiguous()
        x_imag = x.imag.permute(0, 2, 3, 1).contiguous()

        y_real, y_imag = self.forward_planar(x_real, x_imag)

        # Planar NHWC -> PyTorch complex NCHW.
        y_real = y_real.permute(0, 3, 1, 2).contiguous()
        y_imag = y_imag.permute(0, 3, 1, 2).contiguous()

        return torch.complex(y_real, y_imag)
