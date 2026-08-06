import os

__version__ = "0.1.0"

# Omni6DPose stores depth and instance masks as OpenEXR.  Since OpenCV 4.5.5 the
# EXR codec ships in the wheels but is refused at runtime unless this is set
# (see opencv/opencv#21326) — cv2.imread then *raises*, which the modality
# loaders turn into a silent None and the shard build turns into a missing item.
#
# It must be set before the first EXR decode in the process: cv2 latches the
# decision inside initOpenEXR, so a read that fails once keeps failing for the
# lifetime of the process even after the variable is set afterwards.  Package
# import is the earliest point every o3b entry point passes through.
# setdefault, so an explicit OPENCV_IO_ENABLE_OPENEXR=0 in the environment still
# wins — nothing in o3b flips it back off.
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

# Torch multiprocessing settings the platform's job preamble exports (slurm
# needs file_system sharing, see o3b.dataloading).  Guarded so the torch import
# only happens when the environment actually asks for something — package
# import is otherwise torch-free, and this must run before any DataLoader
# worker is spawned, which package import is the earliest point to guarantee.
if os.environ.get("MP_SHARING_STRATEGY") or os.environ.get("MP_START_METHOD"):
    from o3b.dataloading import apply_mp_env

    apply_mp_env()
