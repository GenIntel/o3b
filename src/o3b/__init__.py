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
