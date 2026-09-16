"""Ported verbatim from od3d ``od3d/od3d_datasets/objectron/enum.py``.

The OD3D_CATEGORIES maps are dropped: o3b aligns categories onto
UCO3D_CATEGORIES instead (see o3b.dataset.uco3d).
"""
from o3b.data.ext_enum import StrEnum
from o3b.dataset.uco3d.enum import UCO3D_CATEGORIES

class OBJECTRON_SUBSETS(StrEnum):
    TEST = "test"
    TRAIN = "train"

class OBJECTRON_CATEGORIES(StrEnum):
    BIKE = "bike"
    BOOK = "book"
    BOTTLE = "bottle"
    CAMERA = "camera"
    CEREAL_BOX = "cereal_box"
    CHAIR = "chair"
    CUP = "cup"
    LAPTOP = "laptop"
    SHOE = "shoe"


# Objectron's 9 categories onto UCO3D's vocabulary.  Authored here rather than
# ported: od3d never mapped Objectron onto UCO3D at all, so there was nothing to
# copy.  The target is chosen by the *symmetry* the UCO3D orientation tree gives
# it, not by the closest-sounding word — that code is what the pose metric
# consumes.
#
# cereal_box -> BOX, not CARTON: carton / packet / crate are all
# [1, 1, -1] (continuous rotation about the vertical), while a flat rectangular
# cereal box is [1, 1, 2] — a half turn about the vertical maps it onto itself
# and nothing finer does.  BOX carries exactly that code.
MAP_CATEGORIES_OBJECTRON_TO_UCO3D = {
    OBJECTRON_CATEGORIES.BIKE: UCO3D_CATEGORIES.BICYCLE,
    OBJECTRON_CATEGORIES.BOOK: UCO3D_CATEGORIES.BOOK,
    OBJECTRON_CATEGORIES.BOTTLE: UCO3D_CATEGORIES.BOTTLE,
    OBJECTRON_CATEGORIES.CAMERA: UCO3D_CATEGORIES.CAMERA,
    OBJECTRON_CATEGORIES.CEREAL_BOX: UCO3D_CATEGORIES.BOX,
    OBJECTRON_CATEGORIES.CHAIR: UCO3D_CATEGORIES.CHAIR,
    OBJECTRON_CATEGORIES.CUP: UCO3D_CATEGORIES.CUP,
    OBJECTRON_CATEGORIES.LAPTOP: UCO3D_CATEGORIES.LAPTOP_COMPUTER,
    OBJECTRON_CATEGORIES.SHOE: UCO3D_CATEGORIES.SHOE,
}
