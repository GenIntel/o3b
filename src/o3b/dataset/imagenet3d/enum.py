"""Ported verbatim from od3d ``od3d/od3d_datasets/imagenet3d/enum.py``.

The OD3D_CATEGORIES maps are dropped: o3b aligns categories onto
UCO3D_CATEGORIES instead (see o3b.dataset.uco3d).
"""
from o3b.data.ext_enum import ExtEnum

from o3b.dataset.uco3d.enum import UCO3D_CATEGORIES

class IMAGENET3D_SUBSETS(str, ExtEnum):
    VAL = "val"
    TRAIN = "train"

class IMAGENET3D_CATEGORIES(str, ExtEnum):
    AEROPLANE = "aeroplane"
    AIR_HAMMER = "air_hammer"
    AIRSHIP = "airship"
    AMBULANCE = "ambulance"
    ASHTRAY = "ashtray"
    AX = "ax"
    BACKPACK = "backpack"
    BARBELL = "barbell"
    BASEBALL_BAT = "baseball_bat"
    BASKET = "basket"
    BATTERY = "battery"
    BEAKER = "beaker"
    BED = "bed"
    BENCH = "bench"
    BICYCLE = "bicycle"
    BICYCLE_BUILT_FOR_TWO = "bicycle_built_for_two"
    BICYCLE_PUMP = "bicycle_pump"
    BILLIARD_TABLE = "billiard_table"
    BLACKBOARD = "blackboard"
    BOAT = "boat"
    BOOKSHELF = "bookshelf"
    BOTTLE = "bottle"
    BOW = "bow"
    BOWL = "bowl"
    BOWLING_PIN = "bowling_pin"
    BOXING_GLOVE = "boxing_glove"
    BUCKET = "bucket"
    BUMPER_CAR = "bumper_car"
    BUS = "bus"
    CABINET = "cabinet"
    CALCULATOR = "calculator"
    CAMCORDER = "camcorder"
    CAMERA = "camera"
    CAN = "can"
    CAP = "cap"
    CAR = "car"
    CART = "cart"
    CELLPHONE = "cellphone"
    CHAIN_SAW = "chain_saw"
    CHAIR = "chair"
    CLOCK = "clock"
    COFFEE_MAKER = "coffee_maker"
    COMB = "comb"
    COMPUTER = "computer"
    CONE = "cone"
    COOKER = "cooker"
    COOLING_TOWER = "cooling_tower"
    CRUTCH = "crutch"
    CUP = "cup"
    DESK_LAMP = "desk_lamp"
    DININGTABLE = "diningtable"
    DISHWASHER = "dishwasher"
    DOOR = "door"
    DUMBBELL = "dumbbell"
    DUNE_BUGGY = "dune_buggy"
    ERASER = "eraser"
    ERLENMEYER_FLASK = "erlenmeyer_flask"
    EYEGLASSES = "eyeglasses"
    FAN = "fan"
    FAUCET = "faucet"
    FERRIS_WHEEL = "ferris_wheel"
    FILING_CABINET = "filing_cabinet"
    FIRE_EXTINGUISHER = "fire_extinguisher"
    FIRE_TRUCK = "fire_truck"
    FISH_TANK = "fish_tank"
    FLASHLIGHT = "flashlight"
    FORK = "fork"
    FORKLIFT = "forklift"
    FRENCH_HORN = "french_horn"
    FRIGATE = "frigate"
    FRISBEE = "frisbee"
    FUNNEL = "funnel"
    GARBAGE_TRUCK = "garbage_truck"
    GLIDER = "glider"
    GO_KART = "go_kart"
    GRINDER = "grinder"
    GUITAR = "guitar"
    HACKSAW = "hacksaw"
    HAIR_DRYER = "hair_dryer"
    HAMMER = "hammer"
    HAND_BARROW = "hand_barrow"
    HAND_MOWER = "hand_mower"
    HARP = "harp"
    HARVESTER = "harvester"
    HEADPHONE = "headphone"
    HELMET = "helmet"
    HIGHCHAIR = "highchair"
    HOCKEY_STICK = "hockey_stick"
    HOT_AIR_BALLOON = "hot_air_balloon"
    HOVERCRAFT = "hovercraft"
    HOURGLASS = "hourglass"
    INTERPHONE = "interphone"
    IRON = "iron"
    JAR = "jar"
    JINRIKISHA = "jinrikisha"
    KETTLE = "kettle"
    KEY = "key"
    KEYBOARD = "keyboard"
    KNIFE = "knife"
    LAPTOP = "laptop"
    LIFE_BUOY = "life_buoy"
    LIGHT_BULB = "light_bulb"
    LIGHTER = "lighter"
    LIPSTICK = "lipstick"
    MAGNIFIER = "magnifier"
    MAILBOX = "mailbox"
    MEGAPHONE = "megaphone"
    MICROPHONE = "microphone"
    MICROMETER = "micrometer"
    MILLWHEEL = "millwheel"
    MICROWAVE = "microwave"
    MOTORBIKE = "motorbike"
    MOUSE = "mouse"
    PADLOCK = "padlock"
    PAINTBRUSH = "paintbrush"
    PAN = "pan"
    PAY_PHONE = "pay_phone"
    PEN = "pen"
    PENCIL = "pencil"
    PIANO = "piano"
    PILLOW = "pillow"
    PLATE = "plate"
    POT = "pot"
    POWER_DRILL = "power_drill"
    PRINTER = "printer"
    PROJECTOR = "projector"
    PUNCHING_BAG = "punching_bag"
    RACKET = "racket"
    RADIATOR = "radiator"
    RECREATIONAL_VEHICLE = "recreational_vehicle"
    REFRIGERATOR = "refrigerator"
    REMOTE_CONTROL = "remote_control"
    RIDING_MOWER = "riding_mower"
    RIFLE = "rifle"
    ROAD_POLE = "road_pole"
    SALT_OR_PEPPER_SHAKER = "salt_or_pepper_shaker"
    SATELLITE_DISH = "satellite_dish"
    SAX = "sax"
    SCHOOL_BUS = "school_bus"
    SCISSORS = "scissors"
    SCREWDRIVER = "screwdriver"
    SEGWAY = "segway"
    SEWING_MACHINE = "sewing_machine"
    SHAVER = "shaver"
    SHOE = "shoe"
    SHOPPING_CART = "shopping_cart"
    SHOVEL = "shovel"
    SIGN = "sign"
    SINK = "sink"
    SKATE = "skate"
    SKATEBOARD = "skateboard"
    SLIPPER = "slipper"
    SNOWMOBILE = "snowmobile"
    SOFA = "sofa"
    SOMBRERO = "sombrero"
    SPEAKER = "speaker"
    SPOON = "spoon"
    STAPLER = "stapler"
    STOVE = "stove"
    SUITCASE = "suitcase"
    SUNDIAL = "sundial"
    SYRINGE = "syringe"
    TABLE_TENNIS_TABLE = "table_tennis_table"
    TANK = "tank"
    TEAPOT = "teapot"
    TELEPHONE = "telephone"
    TOASTER = "toaster"
    TOILET = "toilet"
    TOOTHBRUSH = "toothbrush"
    TRACTOR = "tractor"
    TRAIN = "train"
    TRASH_BIN = "trash_bin"
    TREADMILL = "treadmill"
    TRICYCLE = "tricycle"
    TROLLEYBUS = "trolleybus"
    TROPHY = "trophy"
    TROWEL = "trowel"
    TUB = "tub"
    TVMONITOR = "tvmonitor"
    UNICYCLE = "unicycle"
    VENDING_MACHINE = "vending_machine"
    VIOLIN = "violin"
    WASHER = "washer"
    WASHING_MACHINE = "washing_machine"
    WATCH = "watch"
    WHEELCHAIR = "wheelchair"
    WHISK = "whisk"
    WIND_TURBINE = "wind_turbine"
    YURT = "yurt"

MAP_CATEGORIES_IMAGENET3D_TO_UCO3D = {
    IMAGENET3D_CATEGORIES.AEROPLANE: UCO3D_CATEGORIES.AIRPLANE,
    IMAGENET3D_CATEGORIES.AIR_HAMMER: UCO3D_CATEGORIES.HAMMER,
    IMAGENET3D_CATEGORIES.AIRSHIP: UCO3D_CATEGORIES.CARGO_SHIP,
    IMAGENET3D_CATEGORIES.AMBULANCE: UCO3D_CATEGORIES.AMBULANCE,
    IMAGENET3D_CATEGORIES.ASHTRAY: UCO3D_CATEGORIES.ASHTRAY,
    IMAGENET3D_CATEGORIES.AX: UCO3D_CATEGORIES.AX,
    IMAGENET3D_CATEGORIES.BACKPACK: UCO3D_CATEGORIES.BACKPACK,
    IMAGENET3D_CATEGORIES.BARBELL: UCO3D_CATEGORIES.BARBELL,
    IMAGENET3D_CATEGORIES.BASEBALL_BAT: UCO3D_CATEGORIES.BASEBALL_BAT,
    IMAGENET3D_CATEGORIES.BASKET: UCO3D_CATEGORIES.BASKET,
    IMAGENET3D_CATEGORIES.BATTERY: UCO3D_CATEGORIES.BATTERY,
    IMAGENET3D_CATEGORIES.BEAKER: UCO3D_CATEGORIES.SHAKER,
    IMAGENET3D_CATEGORIES.BED: UCO3D_CATEGORIES.BED,
    IMAGENET3D_CATEGORIES.BENCH: UCO3D_CATEGORIES.BENCH,
    IMAGENET3D_CATEGORIES.BICYCLE: UCO3D_CATEGORIES.BICYCLE,
    IMAGENET3D_CATEGORIES.BICYCLE_BUILT_FOR_TWO: UCO3D_CATEGORIES.BICYCLE,
    IMAGENET3D_CATEGORIES.BICYCLE_PUMP: UCO3D_CATEGORIES.BICYCLE,
    IMAGENET3D_CATEGORIES.BILLIARD_TABLE: UCO3D_CATEGORIES.POOL_TABLE,
    IMAGENET3D_CATEGORIES.BLACKBOARD: UCO3D_CATEGORIES.BILLBOARD,
    IMAGENET3D_CATEGORIES.BOAT: UCO3D_CATEGORIES.BOAT,
    IMAGENET3D_CATEGORIES.BOOKSHELF: UCO3D_CATEGORIES.BOOK,
    IMAGENET3D_CATEGORIES.BOTTLE: UCO3D_CATEGORIES.BOTTLE,
    IMAGENET3D_CATEGORIES.BOW: UCO3D_CATEGORIES.BOW_WEAPON,
    IMAGENET3D_CATEGORIES.BOWL: UCO3D_CATEGORIES.BOWL,
    IMAGENET3D_CATEGORIES.BOWLING_PIN: UCO3D_CATEGORIES.ROLLING_PIN,
    IMAGENET3D_CATEGORIES.BOXING_GLOVE: UCO3D_CATEGORIES.BOXING_GLOVE,
    IMAGENET3D_CATEGORIES.BUCKET: UCO3D_CATEGORIES.BUCKET,
    IMAGENET3D_CATEGORIES.BUMPER_CAR: UCO3D_CATEGORIES.CAR_AUTOMOBILE,
    IMAGENET3D_CATEGORIES.BUS: UCO3D_CATEGORIES.BUS_VEHICLE,
    IMAGENET3D_CATEGORIES.CABINET: UCO3D_CATEGORIES.CABINET,
    IMAGENET3D_CATEGORIES.CALCULATOR: UCO3D_CATEGORIES.CALCULATOR,
    IMAGENET3D_CATEGORIES.CAMCORDER: UCO3D_CATEGORIES.CAMCORDER,
    IMAGENET3D_CATEGORIES.CAMERA: UCO3D_CATEGORIES.CAMERA,
    IMAGENET3D_CATEGORIES.CAN: UCO3D_CATEGORIES.CAN,
    IMAGENET3D_CATEGORIES.CAP: UCO3D_CATEGORIES.CAP_HEADWEAR,
    IMAGENET3D_CATEGORIES.CAR: UCO3D_CATEGORIES.CAR_AUTOMOBILE,
    IMAGENET3D_CATEGORIES.CART: UCO3D_CATEGORIES.CART,
    IMAGENET3D_CATEGORIES.CELLPHONE: UCO3D_CATEGORIES.CELLULAR_TELEPHONE,
    IMAGENET3D_CATEGORIES.CHAIN_SAW: UCO3D_CATEGORIES.HANDSAW,
    IMAGENET3D_CATEGORIES.CHAIR: UCO3D_CATEGORIES.CHAIR,
    IMAGENET3D_CATEGORIES.CLOCK: UCO3D_CATEGORIES.CLOCK,
    IMAGENET3D_CATEGORIES.COFFEE_MAKER: UCO3D_CATEGORIES.COFFEE_MAKER,
    IMAGENET3D_CATEGORIES.COMB: UCO3D_CATEGORIES.HAIRBRUSH,
    IMAGENET3D_CATEGORIES.COMPUTER: UCO3D_CATEGORIES.LAPTOP_COMPUTER,
    IMAGENET3D_CATEGORIES.CONE: UCO3D_CATEGORIES.CONE,
    IMAGENET3D_CATEGORIES.COOKER: UCO3D_CATEGORIES.COOKER,
    IMAGENET3D_CATEGORIES.COOLING_TOWER: UCO3D_CATEGORIES.CLOCK_TOWER,
    IMAGENET3D_CATEGORIES.CRUTCH: UCO3D_CATEGORIES.CRUTCH,
    IMAGENET3D_CATEGORIES.CUP: UCO3D_CATEGORIES.CUP,
    IMAGENET3D_CATEGORIES.DESK_LAMP: UCO3D_CATEGORIES.TABLE_LAMP,
    IMAGENET3D_CATEGORIES.DININGTABLE: UCO3D_CATEGORIES.DINING_TABLE,
    IMAGENET3D_CATEGORIES.DISHWASHER: UCO3D_CATEGORIES.DISHWASHER,
    IMAGENET3D_CATEGORIES.DOOR: UCO3D_CATEGORIES.DOOR,
    IMAGENET3D_CATEGORIES.DUMBBELL: UCO3D_CATEGORIES.DUMBBELL,
    IMAGENET3D_CATEGORIES.DUNE_BUGGY: UCO3D_CATEGORIES.HORSE_BUGGY,
    IMAGENET3D_CATEGORIES.ERASER: UCO3D_CATEGORIES.ERASER,
    IMAGENET3D_CATEGORIES.ERLENMEYER_FLASK: UCO3D_CATEGORIES.WATER_JUG,
    IMAGENET3D_CATEGORIES.EYEGLASSES: UCO3D_CATEGORIES.SUNGLASSES,
    IMAGENET3D_CATEGORIES.FAN: UCO3D_CATEGORIES.FAN,
    IMAGENET3D_CATEGORIES.FAUCET: UCO3D_CATEGORIES.FAUCET,
    IMAGENET3D_CATEGORIES.FERRIS_WHEEL: UCO3D_CATEGORIES.PINWHEEL,
    IMAGENET3D_CATEGORIES.FILING_CABINET: UCO3D_CATEGORIES.FILE_CABINET,
    IMAGENET3D_CATEGORIES.FIRE_EXTINGUISHER: UCO3D_CATEGORIES.FIRE_EXTINGUISHER,
    IMAGENET3D_CATEGORIES.FIRE_TRUCK: UCO3D_CATEGORIES.TRUCK,
    IMAGENET3D_CATEGORIES.FISH_TANK: UCO3D_CATEGORIES.AQUARIUM,
    IMAGENET3D_CATEGORIES.FLASHLIGHT: UCO3D_CATEGORIES.FLASHLIGHT,
    IMAGENET3D_CATEGORIES.FORK: UCO3D_CATEGORIES.FORK,
    IMAGENET3D_CATEGORIES.FORKLIFT: UCO3D_CATEGORIES.FORKLIFT,
    IMAGENET3D_CATEGORIES.FRENCH_HORN: UCO3D_CATEGORIES.BASS_HORN,
    IMAGENET3D_CATEGORIES.FRIGATE: UCO3D_CATEGORIES.CARGO_SHIP,
    IMAGENET3D_CATEGORIES.FRISBEE: UCO3D_CATEGORIES.FRISBEE,
    IMAGENET3D_CATEGORIES.FUNNEL: UCO3D_CATEGORIES.FUNNEL,
    IMAGENET3D_CATEGORIES.GARBAGE_TRUCK: UCO3D_CATEGORIES.GARBAGE_TRUCK,
    IMAGENET3D_CATEGORIES.GLIDER: UCO3D_CATEGORIES.AIRPLANE,
    IMAGENET3D_CATEGORIES.GO_KART: UCO3D_CATEGORIES.GOLFCART,
    IMAGENET3D_CATEGORIES.GRINDER: UCO3D_CATEGORIES.PEPPER_MILL,
    IMAGENET3D_CATEGORIES.GUITAR: UCO3D_CATEGORIES.GUITAR,
    IMAGENET3D_CATEGORIES.HACKSAW: UCO3D_CATEGORIES.HANDSAW,
    IMAGENET3D_CATEGORIES.HAIR_DRYER: UCO3D_CATEGORIES.HAIR_DRYER,
    IMAGENET3D_CATEGORIES.HAMMER: UCO3D_CATEGORIES.HAMMER,
    IMAGENET3D_CATEGORIES.HAND_BARROW: UCO3D_CATEGORIES.HANDCART,
    IMAGENET3D_CATEGORIES.HAND_MOWER: UCO3D_CATEGORIES.LAWN_MOWER,
    IMAGENET3D_CATEGORIES.HARP: UCO3D_CATEGORIES.VIOLIN,
    IMAGENET3D_CATEGORIES.HARVESTER: UCO3D_CATEGORIES.TRACTOR_FARM_EQUIPMENT,
    IMAGENET3D_CATEGORIES.HEADPHONE: UCO3D_CATEGORIES.EARPHONE,
    IMAGENET3D_CATEGORIES.HELMET: UCO3D_CATEGORIES.HELMET,
    IMAGENET3D_CATEGORIES.HIGHCHAIR: UCO3D_CATEGORIES.HIGHCHAIR,
    IMAGENET3D_CATEGORIES.HOCKEY_STICK: UCO3D_CATEGORIES.WALKING_STICK,
    IMAGENET3D_CATEGORIES.HOT_AIR_BALLOON: UCO3D_CATEGORIES.BALLOON,
    IMAGENET3D_CATEGORIES.HOVERCRAFT: UCO3D_CATEGORIES.RAFT,
    IMAGENET3D_CATEGORIES.HOURGLASS: UCO3D_CATEGORIES.HOURGLASS,
    IMAGENET3D_CATEGORIES.INTERPHONE: UCO3D_CATEGORIES.TELEPHONE,
    IMAGENET3D_CATEGORIES.IRON: UCO3D_CATEGORIES.IRON_FOR_CLOTHING,
    IMAGENET3D_CATEGORIES.JAR: UCO3D_CATEGORIES.JAR,
    IMAGENET3D_CATEGORIES.JINRIKISHA: UCO3D_CATEGORIES.HANDCART,
    IMAGENET3D_CATEGORIES.KETTLE: UCO3D_CATEGORIES.KETTLE,
    IMAGENET3D_CATEGORIES.KEY: UCO3D_CATEGORIES.KEY,
    IMAGENET3D_CATEGORIES.KEYBOARD: UCO3D_CATEGORIES.COMPUTER_KEYBOARD,
    IMAGENET3D_CATEGORIES.KNIFE: UCO3D_CATEGORIES.KNIFE,
    IMAGENET3D_CATEGORIES.LAPTOP: UCO3D_CATEGORIES.LAPTOP_COMPUTER,
    IMAGENET3D_CATEGORIES.LIFE_BUOY: UCO3D_CATEGORIES.LIFE_BUOY,
    IMAGENET3D_CATEGORIES.LIGHT_BULB: UCO3D_CATEGORIES.LIGHTBULB,
    IMAGENET3D_CATEGORIES.LIGHTER: UCO3D_CATEGORIES.IGNITER,
    IMAGENET3D_CATEGORIES.LIPSTICK: UCO3D_CATEGORIES.LIP_BALM,
    IMAGENET3D_CATEGORIES.MAGNIFIER: UCO3D_CATEGORIES.MIRROR,
    IMAGENET3D_CATEGORIES.MAILBOX: UCO3D_CATEGORIES.MAILBOX_AT_HOME,
    IMAGENET3D_CATEGORIES.MEGAPHONE: UCO3D_CATEGORIES.BULLHORN,
    IMAGENET3D_CATEGORIES.MICROPHONE: UCO3D_CATEGORIES.MICROPHONE,
    IMAGENET3D_CATEGORIES.MICROMETER: UCO3D_CATEGORIES.TACHOMETER,
    IMAGENET3D_CATEGORIES.MILLWHEEL: UCO3D_CATEGORIES.PINWHEEL,
    IMAGENET3D_CATEGORIES.MICROWAVE: UCO3D_CATEGORIES.MICROWAVE_OVEN,
    IMAGENET3D_CATEGORIES.MOTORBIKE: UCO3D_CATEGORIES.MOTORCYCLE,
    IMAGENET3D_CATEGORIES.MOUSE: UCO3D_CATEGORIES.MOUSE_COMPUTER_EQUIPMENT,
    IMAGENET3D_CATEGORIES.PADLOCK: UCO3D_CATEGORIES.PADLOCK,
    IMAGENET3D_CATEGORIES.PAINTBRUSH: UCO3D_CATEGORIES.PAINTBRUSH,
    IMAGENET3D_CATEGORIES.PAN: UCO3D_CATEGORIES.PAN_FOR_COOKING,
    IMAGENET3D_CATEGORIES.PAY_PHONE: UCO3D_CATEGORIES.TELEPHONE_BOOTH,
    IMAGENET3D_CATEGORIES.PEN: UCO3D_CATEGORIES.PEN,
    IMAGENET3D_CATEGORIES.PENCIL: UCO3D_CATEGORIES.PENCIL,
    IMAGENET3D_CATEGORIES.PIANO: UCO3D_CATEGORIES.PIANO,
    IMAGENET3D_CATEGORIES.PILLOW: UCO3D_CATEGORIES.PILLOW,
    IMAGENET3D_CATEGORIES.PLATE: UCO3D_CATEGORIES.PLATE,
    IMAGENET3D_CATEGORIES.POT: UCO3D_CATEGORIES.POT,
    IMAGENET3D_CATEGORIES.POWER_DRILL: UCO3D_CATEGORIES.DRILL,
    IMAGENET3D_CATEGORIES.PRINTER: UCO3D_CATEGORIES.PRINTER,
    IMAGENET3D_CATEGORIES.PROJECTOR: UCO3D_CATEGORIES.PROJECTOR,
    IMAGENET3D_CATEGORIES.PUNCHING_BAG: UCO3D_CATEGORIES.SLEEPING_BAG,
    IMAGENET3D_CATEGORIES.RACKET: UCO3D_CATEGORIES.RACKET,
    IMAGENET3D_CATEGORIES.RADIATOR: UCO3D_CATEGORIES.HEATER,
    IMAGENET3D_CATEGORIES.RECREATIONAL_VEHICLE: UCO3D_CATEGORIES.CAMPER_VEHICLE,
    IMAGENET3D_CATEGORIES.REFRIGERATOR: UCO3D_CATEGORIES.REFRIGERATOR,
    IMAGENET3D_CATEGORIES.REMOTE_CONTROL: UCO3D_CATEGORIES.REMOTE_CONTROL,
    IMAGENET3D_CATEGORIES.RIDING_MOWER: UCO3D_CATEGORIES.LAWN_MOWER,
    IMAGENET3D_CATEGORIES.RIFLE: UCO3D_CATEGORIES.RIFLE,
    IMAGENET3D_CATEGORIES.ROAD_POLE: UCO3D_CATEGORIES.POLE,
    IMAGENET3D_CATEGORIES.SALT_OR_PEPPER_SHAKER: UCO3D_CATEGORIES.SALTSHAKER,
    IMAGENET3D_CATEGORIES.SATELLITE_DISH: UCO3D_CATEGORIES.DISH_ANTENNA,
    IMAGENET3D_CATEGORIES.SAX: UCO3D_CATEGORIES.SAXOPHONE,
    IMAGENET3D_CATEGORIES.SCHOOL_BUS: UCO3D_CATEGORIES.BUS_VEHICLE,
    IMAGENET3D_CATEGORIES.SCISSORS: UCO3D_CATEGORIES.SHEARS,
    IMAGENET3D_CATEGORIES.SCREWDRIVER: UCO3D_CATEGORIES.SCREWDRIVER,
    IMAGENET3D_CATEGORIES.SEGWAY: UCO3D_CATEGORIES.MOTOR_SCOOTER,
    IMAGENET3D_CATEGORIES.SEWING_MACHINE: UCO3D_CATEGORIES.SEWING_MACHINE,
    IMAGENET3D_CATEGORIES.SHAVER: UCO3D_CATEGORIES.SHAVER_ELECTRIC,
    IMAGENET3D_CATEGORIES.SHOE: UCO3D_CATEGORIES.SHOE,
    IMAGENET3D_CATEGORIES.SHOPPING_CART: UCO3D_CATEGORIES.SHOPPING_CART,
    IMAGENET3D_CATEGORIES.SHOVEL: UCO3D_CATEGORIES.SHOVEL,
    IMAGENET3D_CATEGORIES.SIGN: UCO3D_CATEGORIES.STOP_SIGN,
    IMAGENET3D_CATEGORIES.SINK: UCO3D_CATEGORIES.SINK,
    IMAGENET3D_CATEGORIES.SKATE: UCO3D_CATEGORIES.ICE_SKATE,
    IMAGENET3D_CATEGORIES.SKATEBOARD: UCO3D_CATEGORIES.SKATEBOARD,
    IMAGENET3D_CATEGORIES.SLIPPER: UCO3D_CATEGORIES.SLIPPER_FOOTWEAR,
    IMAGENET3D_CATEGORIES.SNOWMOBILE: UCO3D_CATEGORIES.MOTOR_SCOOTER,
    IMAGENET3D_CATEGORIES.SOFA: UCO3D_CATEGORIES.SOFA,
    IMAGENET3D_CATEGORIES.SOMBRERO: UCO3D_CATEGORIES.SUNHAT,
    IMAGENET3D_CATEGORIES.SPEAKER: UCO3D_CATEGORIES.SPEAKER_STERO_EQUIPMENT,
    IMAGENET3D_CATEGORIES.SPOON: UCO3D_CATEGORIES.SPOON,
    IMAGENET3D_CATEGORIES.STAPLER: UCO3D_CATEGORIES.STAPLER_STAPLING_MACHINE,
    IMAGENET3D_CATEGORIES.STOVE: UCO3D_CATEGORIES.STOVE,
    IMAGENET3D_CATEGORIES.SUITCASE: UCO3D_CATEGORIES.SUITCASE,
    IMAGENET3D_CATEGORIES.SUNDIAL: UCO3D_CATEGORIES.CLOCK_TOWER,
    IMAGENET3D_CATEGORIES.SYRINGE: UCO3D_CATEGORIES.SYRINGE,
    IMAGENET3D_CATEGORIES.TABLE_TENNIS_TABLE: UCO3D_CATEGORIES.TABLE_TENNIS_TABLE,
    IMAGENET3D_CATEGORIES.TANK: UCO3D_CATEGORIES.ARMY_TANK,
    IMAGENET3D_CATEGORIES.TEAPOT: UCO3D_CATEGORIES.TEAPOT,
    IMAGENET3D_CATEGORIES.TELEPHONE: UCO3D_CATEGORIES.TELEPHONE,
    IMAGENET3D_CATEGORIES.TOASTER: UCO3D_CATEGORIES.TOASTER,
    IMAGENET3D_CATEGORIES.TOILET: UCO3D_CATEGORIES.TOILET,
    IMAGENET3D_CATEGORIES.TOOTHBRUSH: UCO3D_CATEGORIES.TOOTHBRUSH,
    IMAGENET3D_CATEGORIES.TRACTOR: UCO3D_CATEGORIES.TRACTOR_FARM_EQUIPMENT,
    IMAGENET3D_CATEGORIES.TRAIN: UCO3D_CATEGORIES.BULLET_TRAIN,
    IMAGENET3D_CATEGORIES.TRASH_BIN: UCO3D_CATEGORIES.TRASH_CAN,
    IMAGENET3D_CATEGORIES.TREADMILL: UCO3D_CATEGORIES.WINDMILL,
    IMAGENET3D_CATEGORIES.TRICYCLE: UCO3D_CATEGORIES.TRICYCLE,
    IMAGENET3D_CATEGORIES.TROLLEYBUS: UCO3D_CATEGORIES.BUS_VEHICLE,
    IMAGENET3D_CATEGORIES.TROPHY: UCO3D_CATEGORIES.TROPHY_CUP,
    IMAGENET3D_CATEGORIES.TROWEL: UCO3D_CATEGORIES.SHOVEL,
    IMAGENET3D_CATEGORIES.TUB: UCO3D_CATEGORIES.BATHTUB,
    IMAGENET3D_CATEGORIES.TVMONITOR: UCO3D_CATEGORIES.TELEVISION_SET,
    IMAGENET3D_CATEGORIES.UNICYCLE: UCO3D_CATEGORIES.BICYCLE,
    IMAGENET3D_CATEGORIES.VENDING_MACHINE: UCO3D_CATEGORIES.VENDING_MACHINE,
    IMAGENET3D_CATEGORIES.VIOLIN: UCO3D_CATEGORIES.VIOLIN,
    IMAGENET3D_CATEGORIES.WASHER: UCO3D_CATEGORIES.AUTOMATIC_WASHER,
    IMAGENET3D_CATEGORIES.WASHING_MACHINE: UCO3D_CATEGORIES.AUTOMATIC_WASHER,
    IMAGENET3D_CATEGORIES.WATCH: UCO3D_CATEGORIES.WATCH,
    IMAGENET3D_CATEGORIES.WHEELCHAIR: UCO3D_CATEGORIES.WHEELCHAIR,
    IMAGENET3D_CATEGORIES.WHISK: UCO3D_CATEGORIES.SHAKER,
    IMAGENET3D_CATEGORIES.WIND_TURBINE: UCO3D_CATEGORIES.WINDMILL,
    IMAGENET3D_CATEGORIES.YURT: UCO3D_CATEGORIES.YURT,
}

MAP_CATEGORIES_OBJ_ORIENT_IMAGENET3D_TO_UCO3D = {
  "unknown": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
  "aeroplane": [
    [
      1,
      0,
      0
    ],
    [
      0,
      1,
      0
    ],
    [
      0,
      0,
      1
    ]
  ],
  "air_hammer": [
    [
      -1,
      0,
      0
    ],
    [
      0,
      0,
      1
    ],
    [
      0,
      1,
      0
    ]
  ],
  "airship": [
    [
      1,
      0,
      0
    ],
    [
      0,
      1,
      0
    ],
    [
      0,
      0,
      1
    ]
  ],
  "ambulance": [
    [
      1,
      0,
      0
    ],
    [
      0,
      1,
      0
    ],
    [
      0,
      0,
      1
    ]
  ],
  "ashtray": [
    [
      1,
      0,
      0
    ],
    [
      0,
      1,
      0
    ],
    [
      0,
      0,
      1
    ]
  ],
  "ax": [
    [
      1,
      0,
      0
    ],
    [
      0,
      1,
      0
    ],
    [
      0,
      0,
      1
    ]
  ],
  "backpack": [
    [
      1,
      0,
      0
    ],
    [
      0,
      1,
      0
    ],
    [
      0,
      0,
      1
    ]
  ],
  "barbell": [
    [
      1,
      0,
      0
    ],
    [
      0,
      1,
      0
    ],
    [
      0,
      0,
      1
    ]
  ],
  "baseball_bat": [
    [
      1,
      0,
      0
    ],
    [
      0,
      1,
      0
    ],
    [
      0,
      0,
      1
    ]
  ],
  "basket": [
    [
      1,
      0,
      0
    ],
    [
      0,
      1,
      0
    ],
    [
      0,
      0,
      1
    ]
  ],
  "battery": [
    [
      1,
      0,
      0
    ],
    [
      0,
      1,
      0
    ],
    [
      0,
      0,
      1
    ]
  ],
  "beaker": [
    [
      1,
      0,
      0
    ],
    [
      0,
      1,
      0
    ],
    [
      0,
      0,
      1
    ]
  ],
  "bed": [
    [
      1,
      0,
      0
    ],
    [
      0,
      1,
      0
    ],
    [
      0,
      0,
      1
    ]
  ],
  "bench": [
    [
      1,
      0,
      0
    ],
    [
      0,
      1,
      0
    ],
    [
      0,
      0,
      1
    ]
  ],
  "bicycle": [
    [
      1,
      0,
      0
    ],
    [
      0,
      1,
      0
    ],
    [
      0,
      0,
      1
    ]
  ],
  "bicycle_built_for_two": [
    [
      1,
      0,
      0
    ],
    [
      0,
      1,
      0
    ],
    [
      0,
      0,
      1
    ]
  ],
  "bicycle_pump": [
    [
      1,
      0,
      0
    ],
    [
      0,
      1,
      0
    ],
    [
      0,
      0,
      1
    ]
  ],
  "billiard_table": [
    [
      1,
      0,
      0
    ],
    [
      0,
      1,
      0
    ],
    [
      0,
      0,
      1
    ]
  ],
  "boat": [
    [
      1,
      0,
      0
    ],
    [
      0,
      1,
      0
    ],
    [
      0,
      0,
      1
    ]
  ],
  "bookshelf": [
    [
      1,
      0,
      0
    ],
    [
      0,
      1,
      0
    ],
    [
      0,
      0,
      1
    ]
  ],
  "bottle": [
    [
      1,
      0,
      0
    ],
    [
      0,
      1,
      0
    ],
    [
      0,
      0,
      1
    ]
  ],
  "bow": [
    [
      1,
      0,
      0
    ],
    [
      0,
      1,
      0
    ],
    [
      0,
      0,
      1
    ]
  ],
  "bowl": [
    [
      1,
      0,
      0
    ],
    [
      0,
      1,
      0
    ],
    [
      0,
      0,
      1
    ]
  ],
  "bowling_pin": [
    [
      1,
      0,
      0
    ],
    [
      0,
      1,
      0
    ],
    [
      0,
      0,
      1
    ]
  ],
  "boxing_glove": [
    [
      0,
      0,
      -1
    ],
    [
      1,
      0,
      0
    ],
    [
      0,
      -1,
      0
    ]
  ],
  "bucket": [
    [
      1,
      0,
      0
    ],
    [
      0,
      1,
      0
    ],
    [
      0,
      0,
      1
    ]
  ],
  "bumper_car": [
    [
      1,
      0,
      0
    ],
    [
      0,
      1,
      0
    ],
    [
      0,
      0,
      1
    ]
  ],
  "bus": [
    [
      1,
      0,
      0
    ],
    [
      0,
      1,
      0
    ],
    [
      0,
      0,
      1
    ]
  ],
  "cabinet": [
    [
      1,
      0,
      0
    ],
    [
      0,
      1,
      0
    ],
    [
      0,
      0,
      1
    ]
  ],
  "calculator": [
    [
      1,
      0,
      0
    ],
    [
      0,
      1,
      0
    ],
    [
      0,
      0,
      1
    ]
  ],
  "camcorder": [
    [
      1,
      0,
      0
    ],
    [
      0,
      1,
      0
    ],
    [
      0,
      0,
      1
    ]
  ],
  "camera": [
    [
      1,
      0,
      0
    ],
    [
      0,
      1,
      0
    ],
    [
      0,
      0,
      1
    ]
  ],
  "can": [
    [
      1,
      0,
      0
    ],
    [
      0,
      1,
      0
    ],
    [
      0,
      0,
      1
    ]
  ],
  "cap": [
    [
      1,
      0,
      0
    ],
    [
      0,
      1,
      0
    ],
    [
      0,
      0,
      1
    ]
  ],
  "car": [
    [
      1,
      0,
      0
    ],
    [
      0,
      1,
      0
    ],
    [
      0,
      0,
      1
    ]
  ],
  "cart": [
    [
      1,
      0,
      0
    ],
    [
      0,
      1,
      0
    ],
    [
      0,
      0,
      1
    ]
  ],
  "cellphone": [
    [
      1,
      0,
      0
    ],
    [
      0,
      1,
      0
    ],
    [
      0,
      0,
      1
    ]
  ],
  "chain_saw": [
    [
      1,
      0,
      0
    ],
    [
      0,
      1,
      0
    ],
    [
      0,
      0,
      1
    ]
  ],
  "chair": [
    [
      1,
      0,
      0
    ],
    [
      0,
      1,
      0
    ],
    [
      0,
      0,
      1
    ]
  ],
  "clock": [
    [
      1,
      0,
      0
    ],
    [
      0,
      1,
      0
    ],
    [
      0,
      0,
      1
    ]
  ],
  "coffee_maker": [
    [
      1,
      0,
      0
    ],
    [
      0,
      1,
      0
    ],
    [
      0,
      0,
      1
    ]
  ],
  "comb": [
    [
      1,
      0,
      0
    ],
    [
      0,
      1,
      0
    ],
    [
      0,
      0,
      1
    ]
  ],
  "computer": [
    [
      1,
      0,
      0
    ],
    [
      0,
      1,
      0
    ],
    [
      0,
      0,
      1
    ]
  ],
  "cone": [
    [
      1,
      0,
      0
    ],
    [
      0,
      1,
      0
    ],
    [
      0,
      0,
      1
    ]
  ],
  "cooker": [
    [
      1,
      0,
      0
    ],
    [
      0,
      1,
      0
    ],
    [
      0,
      0,
      1
    ]
  ],
  "cooling_tower": [
    [
      1,
      0,
      0
    ],
    [
      0,
      1,
      0
    ],
    [
      0,
      0,
      1
    ]
  ],
  "crutch": [
    [
      1,
      0,
      0
    ],
    [
      0,
      1,
      0
    ],
    [
      0,
      0,
      1
    ]
  ],
  "cup": [
    [
      1,
      0,
      0
    ],
    [
      0,
      1,
      0
    ],
    [
      0,
      0,
      1
    ]
  ],
  "desk_lamp": [
    [
      1,
      0,
      0
    ],
    [
      0,
      1,
      0
    ],
    [
      0,
      0,
      1
    ]
  ],
  "diningtable": [
    [
      1,
      0,
      0
    ],
    [
      0,
      1,
      0
    ],
    [
      0,
      0,
      1
    ]
  ],
  "dishwasher": [
    [
      1,
      0,
      0
    ],
    [
      0,
      1,
      0
    ],
    [
      0,
      0,
      1
    ]
  ],
  "door": [
    [
      1,
      0,
      0
    ],
    [
      0,
      1,
      0
    ],
    [
      0,
      0,
      1
    ]
  ],
  "dumbbell": [
    [
      0,
      0,
      1
    ],
    [
      0,
      1,
      0
    ],
    [
      -1,
      0,
      0
    ]
  ],
  "dune_buggy": [
    [
      1,
      0,
      0
    ],
    [
      0,
      1,
      0
    ],
    [
      0,
      0,
      1
    ]
  ],
  "eraser": [
    [
      1,
      0,
      0
    ],
    [
      0,
      1,
      0
    ],
    [
      0,
      0,
      1
    ]
  ],
  "erlenmeyer_flask": [
    [
      1,
      0,
      0
    ],
    [
      0,
      1,
      0
    ],
    [
      0,
      0,
      1
    ]
  ],
  "eyeglasses": [
    [
      1,
      0,
      0
    ],
    [
      0,
      1,
      0
    ],
    [
      0,
      0,
      1
    ]
  ],
  "fan": [
    [
      1,
      0,
      0
    ],
    [
      0,
      1,
      0
    ],
    [
      0,
      0,
      1
    ]
  ],
  "faucet": [
    [
      1,
      0,
      0
    ],
    [
      0,
      1,
      0
    ],
    [
      0,
      0,
      1
    ]
  ],
  "ferris_wheel": [
    [
      1,
      0,
      0
    ],
    [
      0,
      1,
      0
    ],
    [
      0,
      0,
      1
    ]
  ],
  "filing_cabinet": [
    [
      1,
      0,
      0
    ],
    [
      0,
      1,
      0
    ],
    [
      0,
      0,
      1
    ]
  ],
  "fire_extinguisher": [
    [
      1,
      0,
      0
    ],
    [
      0,
      1,
      0
    ],
    [
      0,
      0,
      1
    ]
  ],
  "fire_truck": [
    [
      1,
      0,
      0
    ],
    [
      0,
      1,
      0
    ],
    [
      0,
      0,
      1
    ]
  ],
  "fish_tank": [
    [
      1,
      0,
      0
    ],
    [
      0,
      1,
      0
    ],
    [
      0,
      0,
      1
    ]
  ],
  "flashlight": [
    [
      -1,
      0,
      0
    ],
    [
      0,
      0,
      -1
    ],
    [
      0,
      -1,
      0
    ]
  ],
  "fork": [
    [
      -1,
      0,
      0
    ],
    [
      0,
      0,
      -1
    ],
    [
      0,
      -1,
      0
    ]
  ],
  "forklift": [
    [
      1,
      0,
      0
    ],
    [
      0,
      1,
      0
    ],
    [
      0,
      0,
      1
    ]
  ],
  "french_horn": [
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
  "garbage_truck": [
    [
      1,
      0,
      0
    ],
    [
      0,
      1,
      0
    ],
    [
      0,
      0,
      1
    ]
  ],
  "glider": [
    [
      1,
      0,
      0
    ],
    [
      0,
      1,
      0
    ],
    [
      0,
      0,
      1
    ]
  ],
  "go_kart": [
    [
      1,
      0,
      0
    ],
    [
      0,
      1,
      0
    ],
    [
      0,
      0,
      1
    ]
  ],
  "grinder": [
    [
      1,
      0,
      0
    ],
    [
      0,
      1,
      0
    ],
    [
      0,
      0,
      1
    ]
  ],
  "guitar": [
    [
      1,
      0,
      0
    ],
    [
      0,
      1,
      0
    ],
    [
      0,
      0,
      1
    ]
  ],
  "hacksaw": [
    [
      1,
      0,
      0
    ],
    [
      0,
      0,
      1
    ],
    [
      0,
      -1,
      0
    ]
  ],
  "hair_dryer": [
    [
      1,
      0,
      0
    ],
    [
      0,
      1,
      0
    ],
    [
      0,
      0,
      1
    ]
  ],
  "hammer": [
    [
      1,
      0,
      0
    ],
    [
      0,
      1,
      0
    ],
    [
      0,
      0,
      1
    ]
  ],
  "hand_barrow": [
    [
      1,
      0,
      0
    ],
    [
      0,
      1,
      0
    ],
    [
      0,
      0,
      1
    ]
  ],
  "hand_mower": [
    [
      1,
      0,
      0
    ],
    [
      0,
      1,
      0
    ],
    [
      0,
      0,
      1
    ]
  ],
  "harp": [
    [
      0,
      -1,
      0
    ],
    [
      1,
      0,
      0
    ],
    [
      0,
      0,
      1
    ]
  ],
  "harvester": [
    [
      1,
      0,
      0
    ],
    [
      0,
      1,
      0
    ],
    [
      0,
      0,
      1
    ]
  ],
  "headphone": [
    [
      1,
      0,
      0
    ],
    [
      0,
      1,
      0
    ],
    [
      0,
      0,
      1
    ]
  ],
  "helmet": [
    [
      1,
      0,
      0
    ],
    [
      0,
      1,
      0
    ],
    [
      0,
      0,
      1
    ]
  ],
  "highchair": [
    [
      1,
      0,
      0
    ],
    [
      0,
      1,
      0
    ],
    [
      0,
      0,
      1
    ]
  ],
  "hockey_stick": [
    [
      0,
      -1,
      0
    ],
    [
      1,
      0,
      0
    ],
    [
      0,
      0,
      1
    ]
  ],
  "hot_air_balloon": [
    [
      1,
      0,
      0
    ],
    [
      0,
      1,
      0
    ],
    [
      0,
      0,
      1
    ]
  ],
  "hourglass": [
    [
      1,
      0,
      0
    ],
    [
      0,
      1,
      0
    ],
    [
      0,
      0,
      1
    ]
  ],
  "hovercraft": [
    [
      1,
      0,
      0
    ],
    [
      0,
      1,
      0
    ],
    [
      0,
      0,
      1
    ]
  ],
  "interphone": [
    [
      1,
      0,
      0
    ],
    [
      0,
      1,
      0
    ],
    [
      0,
      0,
      1
    ]
  ],
  "iron": [
    [
      1,
      0,
      0
    ],
    [
      0,
      0,
      1
    ],
    [
      0,
      -1,
      0
    ]
  ],
  "jar": [
    [
      1,
      0,
      0
    ],
    [
      0,
      1,
      0
    ],
    [
      0,
      0,
      1
    ]
  ],
  "jinrikisha": [
    [
      1,
      0,
      0
    ],
    [
      0,
      1,
      0
    ],
    [
      0,
      0,
      1
    ]
  ],
  "kettle": [
    [
      1,
      0,
      0
    ],
    [
      0,
      1,
      0
    ],
    [
      0,
      0,
      1
    ]
  ],
  "key": [
    [
      -1,
      0,
      0
    ],
    [
      0,
      0,
      -1
    ],
    [
      0,
      -1,
      0
    ]
  ],
  "keyboard": [
    [
      1,
      0,
      0
    ],
    [
      0,
      0,
      -1
    ],
    [
      0,
      1,
      0
    ]
  ],
  "knife": [
    [
      1,
      0,
      0
    ],
    [
      0,
      0,
      1
    ],
    [
      0,
      -1,
      0
    ]
  ],
  "laptop": [
    [
      1,
      0,
      0
    ],
    [
      0,
      1,
      0
    ],
    [
      0,
      0,
      1
    ]
  ],
  "life_buoy": [
    [
      -1,
      0,
      0
    ],
    [
      0,
      0,
      1
    ],
    [
      0,
      1,
      0
    ]
  ],
  "light_bulb": [
    [
      -1,
      0,
      0
    ],
    [
      0,
      1,
      0
    ],
    [
      0,
      0,
      -1
    ]
  ],
  "lighter": [
    [
      -1,
      0,
      0
    ],
    [
      0,
      -1,
      0
    ],
    [
      0,
      0,
      1
    ]
  ],
  "lipstick": [
    [
      1,
      0,
      0
    ],
    [
      0,
      1,
      0
    ],
    [
      0,
      0,
      1
    ]
  ],
  "magnifier": [
    [
      1,
      0,
      0
    ],
    [
      0,
      1,
      0
    ],
    [
      0,
      0,
      1
    ]
  ],
  "mailbox": [
    [
      1,
      0,
      0
    ],
    [
      0,
      1,
      0
    ],
    [
      0,
      0,
      1
    ]
  ],
  "megaphone": [
    [
      1,
      0,
      0
    ],
    [
      0,
      1,
      0
    ],
    [
      0,
      0,
      1
    ]
  ],
  "micrometer": [
    [
      0,
      1,
      0
    ],
    [
      0,
      0,
      -1
    ],
    [
      -1,
      0,
      0
    ]
  ],
  "microphone": [
    [
      1,
      0,
      0
    ],
    [
      0,
      1,
      0
    ],
    [
      0,
      0,
      1
    ]
  ],
  "microwave": [
    [
      1,
      0,
      0
    ],
    [
      0,
      1,
      0
    ],
    [
      0,
      0,
      1
    ]
  ],
  "millwheel": [
    [
      -1,
      0,
      0
    ],
    [
      0,
      -1,
      0
    ],
    [
      0,
      0,
      1
    ]
  ],
  "motorbike": [
    [
      1,
      0,
      0
    ],
    [
      0,
      1,
      0
    ],
    [
      0,
      0,
      1
    ]
  ],
  "mouse": [
    [
      1,
      0,
      0
    ],
    [
      0,
      1,
      0
    ],
    [
      0,
      0,
      1
    ]
  ],
  "padlock": [
    [
      1,
      0,
      0
    ],
    [
      0,
      1,
      0
    ],
    [
      0,
      0,
      1
    ]
  ],
  "paintbrush": [
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
  "pan": [
    [
      1,
      0,
      0
    ],
    [
      0,
      1,
      0
    ],
    [
      0,
      0,
      1
    ]
  ],
  "pay_phone": [
    [
      1,
      0,
      0
    ],
    [
      0,
      1,
      0
    ],
    [
      0,
      0,
      1
    ]
  ],
  "pen": [
    [
      1,
      0,
      0
    ],
    [
      0,
      0,
      1
    ],
    [
      0,
      -1,
      0
    ]
  ],
  "pencil": [
    [
      1,
      0,
      0
    ],
    [
      0,
      0,
      1
    ],
    [
      0,
      -1,
      0
    ]
  ],
  "piano": [
    [
      1,
      0,
      0
    ],
    [
      0,
      1,
      0
    ],
    [
      0,
      0,
      1
    ]
  ],
  "pillow": [
    [
      -1,
      0,
      0
    ],
    [
      0,
      0,
      1
    ],
    [
      0,
      1,
      0
    ]
  ],
  "plate": [
    [
      1,
      0,
      0
    ],
    [
      0,
      1,
      0
    ],
    [
      0,
      0,
      1
    ]
  ],
  "pot": [
    [
      0,
      -1,
      0
    ],
    [
      1,
      0,
      0
    ],
    [
      0,
      0,
      1
    ]
  ],
  "power_drill": [
    [
      1,
      0,
      0
    ],
    [
      0,
      1,
      0
    ],
    [
      0,
      0,
      1
    ]
  ],
  "printer": [
    [
      1,
      0,
      0
    ],
    [
      0,
      1,
      0
    ],
    [
      0,
      0,
      1
    ]
  ],
  "projector": [
    [
      1,
      0,
      0
    ],
    [
      0,
      1,
      0
    ],
    [
      0,
      0,
      1
    ]
  ],
  "punching_bag": [
    [
      1,
      0,
      0
    ],
    [
      0,
      1,
      0
    ],
    [
      0,
      0,
      1
    ]
  ],
  "racket": [
    [
      1,
      0,
      0
    ],
    [
      0,
      1,
      0
    ],
    [
      0,
      0,
      1
    ]
  ],
  "radiator": [
    [
      1,
      0,
      0
    ],
    [
      0,
      1,
      0
    ],
    [
      0,
      0,
      1
    ]
  ],
  "recreational_vehicle": [
    [
      1,
      0,
      0
    ],
    [
      0,
      1,
      0
    ],
    [
      0,
      0,
      1
    ]
  ],
  "refrigerator": [
    [
      1,
      0,
      0
    ],
    [
      0,
      1,
      0
    ],
    [
      0,
      0,
      1
    ]
  ],
  "remote_control": [
    [
      1,
      0,
      0
    ],
    [
      0,
      1,
      0
    ],
    [
      0,
      0,
      1
    ]
  ],
  "riding_mower": [
    [
      1,
      0,
      0
    ],
    [
      0,
      1,
      0
    ],
    [
      0,
      0,
      1
    ]
  ],
  "rifle": [
    [
      1,
      0,
      0
    ],
    [
      0,
      1,
      0
    ],
    [
      0,
      0,
      1
    ]
  ],
  "road_pole": [
    [
      1,
      0,
      0
    ],
    [
      0,
      1,
      0
    ],
    [
      0,
      0,
      1
    ]
  ],
  "salt_or_pepper_shaker": [
    [
      1,
      0,
      0
    ],
    [
      0,
      1,
      0
    ],
    [
      0,
      0,
      1
    ]
  ],
  "satellite_dish": [
    [
      1,
      0,
      0
    ],
    [
      0,
      1,
      0
    ],
    [
      0,
      0,
      1
    ]
  ],
  "sax": [
    [
      1,
      0,
      0
    ],
    [
      0,
      1,
      0
    ],
    [
      0,
      0,
      1
    ]
  ],
  "school_bus": [
    [
      1,
      0,
      0
    ],
    [
      0,
      1,
      0
    ],
    [
      0,
      0,
      1
    ]
  ],
  "scissors": [
    [
      0,
      0,
      -1
    ],
    [
      1,
      0,
      0
    ],
    [
      0,
      -1,
      0
    ]
  ],
  "screwdriver": [
    [
      0,
      0,
      -1
    ],
    [
      1,
      0,
      0
    ],
    [
      0,
      -1,
      0
    ]
  ],
  "segway": [
    [
      1,
      0,
      0
    ],
    [
      0,
      1,
      0
    ],
    [
      0,
      0,
      1
    ]
  ],
  "sewing_machine": [
    [
      -1,
      0,
      0
    ],
    [
      0,
      -1,
      0
    ],
    [
      0,
      0,
      1
    ]
  ],
  "shaver": [
    [
      1,
      0,
      0
    ],
    [
      0,
      1,
      0
    ],
    [
      0,
      0,
      1
    ]
  ],
  "shoe": [
    [
      1,
      0,
      0
    ],
    [
      0,
      1,
      0
    ],
    [
      0,
      0,
      1
    ]
  ],
  "shopping_cart": [
    [
      1,
      0,
      0
    ],
    [
      0,
      1,
      0
    ],
    [
      0,
      0,
      1
    ]
  ],
  "shovel": [
    [
      1,
      0,
      0
    ],
    [
      0,
      0,
      -1
    ],
    [
      0,
      1,
      0
    ]
  ],
  "sign": [
    [
      1,
      0,
      0
    ],
    [
      0,
      1,
      0
    ],
    [
      0,
      0,
      1
    ]
  ],
  "sink": [
    [
      1,
      0,
      0
    ],
    [
      0,
      1,
      0
    ],
    [
      0,
      0,
      1
    ]
  ],
  "skate": [
    [
      1,
      0,
      0
    ],
    [
      0,
      1,
      0
    ],
    [
      0,
      0,
      1
    ]
  ],
  "skateboard": [
    [
      1,
      0,
      0
    ],
    [
      0,
      1,
      0
    ],
    [
      0,
      0,
      1
    ]
  ],
  "slipper": [
    [
      1,
      0,
      0
    ],
    [
      0,
      1,
      0
    ],
    [
      0,
      0,
      1
    ]
  ],
  "snowmobile": [
    [
      1,
      0,
      0
    ],
    [
      0,
      1,
      0
    ],
    [
      0,
      0,
      1
    ]
  ],
  "sofa": [
    [
      1,
      0,
      0
    ],
    [
      0,
      1,
      0
    ],
    [
      0,
      0,
      1
    ]
  ],
  "sombrero": [
    [
      1,
      0,
      0
    ],
    [
      0,
      1,
      0
    ],
    [
      0,
      0,
      1
    ]
  ],
  "speaker": [
    [
      1,
      0,
      0
    ],
    [
      0,
      1,
      0
    ],
    [
      0,
      0,
      1
    ]
  ],
  "spoon": [
    [
      -1,
      0,
      0
    ],
    [
      0,
      0,
      -1
    ],
    [
      0,
      -1,
      0
    ]
  ],
  "stapler": [
    [
      1,
      0,
      0
    ],
    [
      0,
      0,
      1
    ],
    [
      0,
      -1,
      0
    ]
  ],
  "stove": [
    [
      1,
      0,
      0
    ],
    [
      0,
      1,
      0
    ],
    [
      0,
      0,
      1
    ]
  ],
  "suitcase": [
    [
      1,
      0,
      0
    ],
    [
      0,
      1,
      0
    ],
    [
      0,
      0,
      1
    ]
  ],
  "sundial": [
    [
      1,
      0,
      0
    ],
    [
      0,
      1,
      0
    ],
    [
      0,
      0,
      1
    ]
  ],
  "syringe": [
    [
      1,
      0,
      0
    ],
    [
      0,
      0,
      1
    ],
    [
      0,
      -1,
      0
    ]
  ],
  "table_tennis_table": [
    [
      0,
      -1,
      0
    ],
    [
      1,
      0,
      0
    ],
    [
      0,
      0,
      1
    ]
  ],
  "tank": [
    [
      1,
      0,
      0
    ],
    [
      0,
      1,
      0
    ],
    [
      0,
      0,
      1
    ]
  ],
  "teapot": [
    [
      1,
      0,
      0
    ],
    [
      0,
      1,
      0
    ],
    [
      0,
      0,
      1
    ]
  ],
  "telephone": [
    [
      1,
      0,
      0
    ],
    [
      0,
      0,
      -1
    ],
    [
      0,
      1,
      0
    ]
  ],
  "toaster": [
    [
      1,
      0,
      0
    ],
    [
      0,
      1,
      0
    ],
    [
      0,
      0,
      1
    ]
  ],
  "toilet": [
    [
      1,
      0,
      0
    ],
    [
      0,
      1,
      0
    ],
    [
      0,
      0,
      1
    ]
  ],
  "toothbrush": [
    [
      1,
      0,
      0
    ],
    [
      0,
      1,
      0
    ],
    [
      0,
      0,
      1
    ]
  ],
  "tractor": [
    [
      1,
      0,
      0
    ],
    [
      0,
      1,
      0
    ],
    [
      0,
      0,
      1
    ]
  ],
  "train": [
    [
      1,
      0,
      0
    ],
    [
      0,
      1,
      0
    ],
    [
      0,
      0,
      1
    ]
  ],
  "trash_bin": [
    [
      1,
      0,
      0
    ],
    [
      0,
      1,
      0
    ],
    [
      0,
      0,
      1
    ]
  ],
  "treadmill": [
    [
      1,
      0,
      0
    ],
    [
      0,
      1,
      0
    ],
    [
      0,
      0,
      1
    ]
  ],
  "tricycle": [
    [
      1,
      0,
      0
    ],
    [
      0,
      1,
      0
    ],
    [
      0,
      0,
      1
    ]
  ],
  "trolleybus": [
    [
      1,
      0,
      0
    ],
    [
      0,
      1,
      0
    ],
    [
      0,
      0,
      1
    ]
  ],
  "trophy": [
    [
      1,
      0,
      0
    ],
    [
      0,
      1,
      0
    ],
    [
      0,
      0,
      1
    ]
  ],
  "trowel": [
    [
      1,
      0,
      0
    ],
    [
      0,
      0,
      -1
    ],
    [
      0,
      1,
      0
    ]
  ],
  "tub": [
    [
      0,
      -1,
      0
    ],
    [
      1,
      0,
      0
    ],
    [
      0,
      0,
      1
    ]
  ],
  "tvmonitor": [
    [
      1,
      0,
      0
    ],
    [
      0,
      1,
      0
    ],
    [
      0,
      0,
      1
    ]
  ],
  "unicycle": [
    [
      -1,
      0,
      0
    ],
    [
      0,
      -1,
      0
    ],
    [
      0,
      0,
      1
    ]
  ],
  "vending_machine": [
    [
      1,
      0,
      0
    ],
    [
      0,
      1,
      0
    ],
    [
      0,
      0,
      1
    ]
  ],
  "violin": [
    [
      1,
      0,
      0
    ],
    [
      0,
      1,
      0
    ],
    [
      0,
      0,
      1
    ]
  ],
  "washer": [
    [
      1,
      0,
      0
    ],
    [
      0,
      1,
      0
    ],
    [
      0,
      0,
      1
    ]
  ],
  "washing_machine": [
    [
      1,
      0,
      0
    ],
    [
      0,
      1,
      0
    ],
    [
      0,
      0,
      1
    ]
  ],
  "watch": [
    [
      1,
      0,
      0
    ],
    [
      0,
      1,
      0
    ],
    [
      0,
      0,
      1
    ]
  ],
  "wheelchair": [
    [
      1,
      0,
      0
    ],
    [
      0,
      1,
      0
    ],
    [
      0,
      0,
      1
    ]
  ],
  "whisk": [
    [
      -1,
      0,
      0
    ],
    [
      0,
      0,
      -1
    ],
    [
      0,
      -1,
      0
    ]
  ],
  "wind_turbine": [
    [
      1,
      0,
      0
    ],
    [
      0,
      1,
      0
    ],
    [
      0,
      0,
      1
    ]
  ],
  "yurt": [
    [
      1,
      0,
      0
    ],
    [
      0,
      1,
      0
    ],
    [
      0,
      0,
      1
    ]
  ],
  "frigate": [
    [
      1,
      0,
      0
    ],
    [
      0,
      1,
      0
    ],
    [
      0,
      0,
      1
    ]
  ],
  "frisbee": [
    [
      1,
      0,
      0
    ],
    [
      0,
      1,
      0
    ],
    [
      0,
      0,
      1
    ]
  ],
  "funnel": [
    [
      1,
      0,
      0
    ],
    [
      0,
      1,
      0
    ],
    [
      0,
      0,
      1
    ]
  ],
  "blackboard": [
    [
      1,
      0,
      0
    ],
    [
      0,
      1,
      0
    ],
    [
      0,
      0,
      1
    ]
  ]
}
