"""Import the CUDA extension built during package installation."""

try:
    from . import _C as cutlass_complex_conv
except ImportError as exc:
    raise ImportError(
        "The cutlass-complex-conv CUDA extension is not built. "
        "Install the project from the repository root with "
        "`pip install . --no-build-isolation`."
    ) from exc

__all__ = ["cutlass_complex_conv"]
