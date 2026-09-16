"""Ported verbatim from od3d ``od3d/od3d_datasets/handal/enum.py``.

The OD3D_CATEGORIES maps are dropped: o3b aligns categories onto
UCO3D_CATEGORIES instead (see o3b.dataset.uco3d).
"""
from o3b.data.ext_enum import StrEnum

from o3b.data.axis import AXIS

from o3b.dataset.uco3d.enum import UCO3D_CATEGORIES

class HANDAL_SUBSETS(StrEnum):
    TRAIN = "train"
    TEST = "test"

class HANDAL_CATEGORIES(StrEnum):
    CLAW_HAMMER = "claw_hammer" # toolbox downwards
    FIXED_JOINT_PLIERS = "fixed_joint_pliers"
    SLIP_JOINT_PLIERS = "slip_joint_pliers"
    LOCKING_PLIERS = "locking_pliers"
    POWER_DRILL = "power_drill"
    RATCHET = "ratchet"
    SCREWDRIVER = "screwdriver"
    ADJUSTABLE_WRENCH = "adjustable_wrench"
    COMBINATION_WRENCH = "combination_wrench" 
    LADLE = "ladle" # kitchen downwards
    MEASURING_CUP = "measuring_cup"
    MUG = "mug"
    POT = "pot_and_pan" # pots and pans
    SPATULA = "spatula"
    STRAINER = "strainer"
    UTENSIL = "utensil"
    WHISK = "whisk"

MAP_CATEGORIES_HANDAL_TO_AXIS = {
    HANDAL_CATEGORIES.CLAW_HAMMER: None, #  [AXIS.FRONT, AXIS.TOP, AXIS.RIGHT],
    HANDAL_CATEGORIES.FIXED_JOINT_PLIERS: None, # [AXIS.FRONT, AXIS.TOP, AXIS.RIGHT],
    HANDAL_CATEGORIES.SLIP_JOINT_PLIERS: None, # [AXIS.FRONT, AXIS.TOP, AXIS.RIGHT],
    HANDAL_CATEGORIES.LOCKING_PLIERS: None, # [AXIS.FRONT, AXIS.TOP, AXIS.RIGHT],
    HANDAL_CATEGORIES.POWER_DRILL: None, # [AXIS.FRONT, AXIS.TOP, AXIS.RIGHT],
    HANDAL_CATEGORIES.RATCHET: None, #[AXIS.FRONT, AXIS.TOP, AXIS.RIGHT],
    HANDAL_CATEGORIES.SCREWDRIVER: None, #  [AXIS.FRONT, AXIS.TOP, AXIS.RIGHT], # # note that this is rotation symmetric around top-bottom (-1)
    HANDAL_CATEGORIES.ADJUSTABLE_WRENCH: [AXIS.TOP, AXIS.BACK, AXIS.RIGHT], # note that this is almost rotation symmetric round top-bottom (2)
    HANDAL_CATEGORIES.COMBINATION_WRENCH: [AXIS.TOP, AXIS.BACK, AXIS.RIGHT], # note that this is rotation symmetric around front-back (2) and left-right (2) and top-bottom (2)
    HANDAL_CATEGORIES.LADLE: [AXIS.TOP, AXIS.FRONT, AXIS.LEFT], # kitchen downwards
    HANDAL_CATEGORIES.MEASURING_CUP: [AXIS.FRONT, AXIS.TOP, AXIS.RIGHT], # kitchen downwards
    HANDAL_CATEGORIES.MUG: [AXIS.BACK, AXIS.TOP, AXIS.LEFT], # kitchen downwards
    HANDAL_CATEGORIES.POT: None, # [AXIS.BOTTOM, AXIS.LEFT, AXIS.BACK], # kitchen downwards
    HANDAL_CATEGORIES.SPATULA: None, # [AXIS.BOTTOM, AXIS.LEFT, AXIS.BACK], # kitchen downwards
    HANDAL_CATEGORIES.STRAINER: [AXIS.TOP, AXIS.FRONT, AXIS.LEFT], # kitchen downwards
    HANDAL_CATEGORIES.UTENSIL: None, # kitchen downwards
    HANDAL_CATEGORIES.WHISK: None, # kitchen downwards
}

MAP_CATEGORIES_HANDAL_TO_UCO3D = {
    HANDAL_CATEGORIES.CLAW_HAMMER: UCO3D_CATEGORIES.HAMMER,
    HANDAL_CATEGORIES.FIXED_JOINT_PLIERS: UCO3D_CATEGORIES.PLIERS,
    HANDAL_CATEGORIES.SLIP_JOINT_PLIERS: UCO3D_CATEGORIES.PLIERS,
    HANDAL_CATEGORIES.LOCKING_PLIERS: UCO3D_CATEGORIES.PLIERS,
    HANDAL_CATEGORIES.POWER_DRILL: UCO3D_CATEGORIES.DRILL,
    HANDAL_CATEGORIES.RATCHET: UCO3D_CATEGORIES.SCREWDRIVER,
    HANDAL_CATEGORIES.SCREWDRIVER: UCO3D_CATEGORIES.SCREWDRIVER,
    HANDAL_CATEGORIES.ADJUSTABLE_WRENCH: UCO3D_CATEGORIES.WRENCH,
    HANDAL_CATEGORIES.COMBINATION_WRENCH: UCO3D_CATEGORIES.WRENCH,
    HANDAL_CATEGORIES.LADLE: UCO3D_CATEGORIES.LADLE,
    HANDAL_CATEGORIES.MEASURING_CUP: UCO3D_CATEGORIES.MEASURING_CUP,
    HANDAL_CATEGORIES.MUG: UCO3D_CATEGORIES.MUG,
    HANDAL_CATEGORIES.POT: UCO3D_CATEGORIES.POT,
    HANDAL_CATEGORIES.SPATULA: UCO3D_CATEGORIES.SPATULA,
    HANDAL_CATEGORIES.STRAINER: UCO3D_CATEGORIES.STRAINER,
    HANDAL_CATEGORIES.UTENSIL: UCO3D_CATEGORIES.COOKING_UTENSIL,
    HANDAL_CATEGORIES.WHISK: UCO3D_CATEGORIES.STIRRER,
}

MAP_CATEGORIES_OBJ_ORIENT_HANDAL_TO_UCO3D = {
  "claw_hammer": [
    [
      0,
      0,
      -1
    ],
    [
      0,
      1,
      0
    ],
    [
      1,
      0,
      0
    ]
  ],
  "fixed_joint_pliers": [
    [
      0,
      0,
      -1
    ],
    [
      0,
      1,
      0
    ],
    [
      1,
      0,
      0
    ]
  ],
  "slip_joint_pliers": [
    [
      0,
      0,
      -1
    ],
    [
      0,
      1,
      0
    ],
    [
      1,
      0,
      0
    ]
  ],
  "locking_pliers": [
    [
      0,
      0,
      -1
    ],
    [
      0,
      1,
      0
    ],
    [
      1,
      0,
      0
    ]
  ],
  "power_drill": [
    [
      0,
      0,
      -1
    ],
    [
      -1,
      0,
      0
    ],
    [
      0,
      1,
      0
    ]
  ],
  "ratchet": [
    [
      0,
      0,
      -1
    ],
    [
      0,
      1,
      0
    ],
    [
      1,
      0,
      0
    ]
  ],
  "screwdriver": [
    [
      0,
      1,
      0
    ],
    [
      0,
      0,
      1
    ],
    [
      1,
      0,
      0
    ]
  ],
  "adjustable_wrench": [
    [
      0,
      0,
      -1
    ],
    [
      0,
      1,
      0
    ],
    [
      1,
      0,
      0
    ]
  ],
  "combination_wrench": [
    [
      0,
      0,
      -1
    ],
    [
      0,
      1,
      0
    ],
    [
      1,
      0,
      0
    ]
  ],
  "ladle": [
    [
      0,
      0,
      1
    ],
    [
      0,
      -1,
      0
    ],
    [
      1,
      0,
      0
    ]
  ],
  "measuring_cup": [
    [
      0,
      0,
      -1
    ],
    [
      -1,
      0,
      0
    ],
    [
      0,
      1,
      0
    ]
  ],
  "mug": [
    [
      0,
      0,
      1
    ],
    [
      1,
      0,
      0
    ],
    [
      0,
      1,
      0
    ]
  ],
  "pot_and_pan": [
    [
      0,
      0,
      -1
    ],
    [
      -1,
      0,
      0
    ],
    [
      0,
      1,
      0
    ]
  ],
  "spatula": [
    [
      0,
      0,
      1
    ],
    [
      0,
      -1,
      0
    ],
    [
      1,
      0,
      0
    ]
  ],
  "strainer": [
    [
      0,
      0,
      1
    ],
    [
      0,
      -1,
      0
    ],
    [
      1,
      0,
      0
    ]
  ],
  "utensil": [
    [
      0,
      0,
      1
    ],
    [
      0,
      -1,
      0
    ],
    [
      1,
      0,
      0
    ]
  ],
  "whisk": [
    [
      0,
      1,
      0
    ],
    [
      0,
      0,
      1
    ],
    [
      1,
      0,
      0
    ]
  ]
}

MAP_CATEGORIES_HANDAL_TO_OD3D = {
    
}

MAP_CATEGORIES_HANDAL_TO_RPATH = {
    "claw_hammer": "handal_dataset_hammers", # note not claw_hammers
    "fixed_joint_pliers": "handal_dataset_fixed_joint_pliers",
    "slip_joint_pliers": "handal_dataset_slip_joint_pliers",
    "locking_pliers": "handal_dataset_locking_pliers",
    "power_drill": "handal_dataset_power_drills",
    "ratchet": "handal_dataset_ratchets",
    "screwdriver": "handal_dataset_screwdrivers",
    "adjustable_wrench": "handal_dataset_adjustable_wrenches",
    "combination_wrench": "handal_dataset_combinational_wrenches",
    "ladle": "handal_dataset_ladles",
    "measuring_cup": "handal_dataset_measuring_cups",
    "mug": "handal_dataset_mugs",
    "pot_and_pan": "handal_dataset_pots_pans", # note not pots_and_pans
    "spatula": "handal_dataset_spatulas",
    "strainer": "handal_dataset_strainers",
    "utensil": "handal_dataset_utensils",
    "whisk": "handal_dataset_whisks",
}

MAP_CATEGORIES_HANDAL_TO_GDRIVE_ID = {
    "claw_hammer": "1TA3q6JijumlEtC1eiU7nSUUyOm5UrmQ4",
    "fixed_joint_pliers": "1i3YvN3ONcmU_c3I9T1qB2N1fbsurpNw4",
    "slip_joint_pliers": "130xOW5g7o_3LGcO1mPglfeUZEUowT0E0",
    "locking_pliers": "1CQhHvSxaB3ElxVanljrJcDhJWCaZz2_n",
    "power_drill": "1VNmI1qP8oDpXaupYKdJHugHel49x2ccm",
    "ratchet": "1UciT0GkUkqGHyNMWp1_FJlsfLatwCkyw",
    "screwdriver": "1tdL81CbNH-pSEqCVZ858dGZ3nWouJbDy",
    "adjustable_wrench": "12aLw8W8Y-TwhpjoPvWXfz5j26HkDLC_J",
    "combination_wrench": "1VMyETGvnciDo2SisjbpNPULnVx-PjPXH",
    "ladle": "1gRv_luYxbTiXM55e3l_5DefpYEtPvVbd",
    "measuring_cup": "1GBZQnigGBCSKiYRmIG7bXTTIRZNMipIe",
    "mug": "1gSMIcN0-H-Ophs2B5wjF8GI30aRpQ1Bb",
    "pot_and_pan": "1CJa0mHTbsemSSMsFcjYTyHdI7ntF59bt",
    "spatula": "1Aod0dMaIgnd19N-bs5pJpQVO5Ya8J07X",
    "strainer": "1bYP3qevtmjiG3clRiP93mwVBTxyiDQFq",
    "utensil": "1ciU8IUx_fxneWQNvTTNAO3v5xoH67kKj",
    "whisk": "19-YUR5xfvP5DbalrIMALTKlEf7FS_kij",
}
