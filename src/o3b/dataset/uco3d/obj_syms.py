"""Per-category symmetry codes for UCO3D, derived from its orientation tree.

UCO3D annotates *how* a category's canonical axes are defined (the "orient
keys" in :mod:`o3b.dataset.uco3d.map_orient_tree`) rather than annotating
symmetry directly.  An axis that no rule pins down is exactly an axis the
canonical pose cannot distinguish — i.e. a symmetry.  This module turns the
former into the latter, in the (3,) code o3b uses everywhere else:

    -1  continuous rotation about this axis
     1  no symmetry
     2  half turn (180 deg)

Ported from ``od3d/od3d_datasets/uco3d/map_obj_syms.py``; the mapping rules are
unchanged.
"""
from __future__ import annotations

import torch

from o3b.dataset.uco3d.map_orient_tree import MAP_UCO3D_CATEGORY_TO_ORIENT_KEYS
from o3b.dataset.uco3d.orient_tree import MAP_CONDITION_TO_RULE


def map_orient_keys_to_obj_syms(orient_key) -> torch.Tensor:
    """(3,) symmetry code for one category's per-axis orientation keys."""
    obj_syms = torch.zeros(3).long()
    defined_axis_count = len([ok for ok in orient_key if ok is not None])

    if defined_axis_count == 0:
        # nothing pins any axis down: fully symmetric
        obj_syms[:] = -1
        return obj_syms

    if defined_axis_count == 1:
        for o, _orient_key in enumerate(orient_key):
            if _orient_key is None:
                continue
            # the one defined axis is a rotation axis; the two free axes are
            # continuous ("axis" rules) or half-turn symmetric
            obj_syms[o] = -1
            free_code = 2 if "axis" in MAP_CONDITION_TO_RULE[_orient_key] else 1
            if orient_key[(o + 1) % 3] is None:
                obj_syms[(o + 1) % 3] = free_code
            if orient_key[(o + 2) % 3] is None:
                obj_syms[(o + 2) % 3] = free_code
    else:  # 2 or 3 axes defined (3 is overdefined — two axes fix the third)
        obj_syms[:] = 1
        for o, _orient_key in enumerate(orient_key):
            if _orient_key is None:
                continue
            if "axis" in MAP_CONDITION_TO_RULE[_orient_key]:
                # an "axis" rule fixes a line, not a direction, so the rotation
                # symmetry sits on the *other* defined axes
                if orient_key[(o + 1) % 3] is not None:
                    obj_syms[(o + 1) % 3] = 2
                if orient_key[(o + 2) % 3] is not None:
                    obj_syms[(o + 2) % 3] = 2

    return obj_syms


def map_uco3d_cat_to_obj_syms(uco3d_cats) -> torch.Tensor:
    """(N, 3) symmetry codes for a list of UCO3D category names."""
    return torch.stack(
        [map_orient_keys_to_obj_syms(MAP_UCO3D_CATEGORY_TO_ORIENT_KEYS[cat]) for cat in uco3d_cats],
        dim=0,
    )


def obj_syms_for_category(category: str):
    """(3,) float symmetry code for one UCO3D category, or None when unknown."""
    orient_key = MAP_UCO3D_CATEGORY_TO_ORIENT_KEYS.get(category)
    if orient_key is None:
        return None
    return map_orient_keys_to_obj_syms(orient_key).float()
