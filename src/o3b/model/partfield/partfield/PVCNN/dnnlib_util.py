# Minimal stub of the upstream PartField ``dnnlib_util.py``.
#
# The original file (https://github.com/nv-tlabs/PartField) pulls in boto3,
# loguru and psutil for training-time utilities. The feature-extraction code
# vendored here only imports ``ScopedTorchProfiler`` and ``printarr`` (both
# referenced solely from commented-out debug lines), so we provide no-op
# replacements to avoid those dependencies.

from contextlib import ContextDecorator


class ScopedTorchProfiler(ContextDecorator):
    """No-op replacement for the upstream torch-profiler context decorator."""

    def __init__(self, unique_name: str):
        self.unique_name = unique_name

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False


def printarr(*arrs, **kwargs):
    """No-op replacement for the upstream array-debugging helper."""
