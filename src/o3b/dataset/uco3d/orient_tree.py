from o3b.data.ext_enum import StrEnum

# goal we want to map each category to a "rule" category which has multiple rules
# -> each rule either describes an axis or an vector
# -> combining multiple rules will lead to symmetries, leading to correct poses (over-determined) 
# -> also not having enough rules for object will lead to symmetry annotation (under-determined)

# core problem, sometimes grab does not aligns with action for front and top (umbrealla, strainer, pan)

class ORIENT_AXIS_RULES(StrEnum):
    AXIS_RIGHT_LEFT = "axis_right_left" # from right to left
    VEC_RIGHT_LEFT = "vec_right_left" # from right to left
    VEC_LEFT_RIGHT = "vec_left_right" # from left to right
    AXIS_FRONT_BACK = "axis_front_back" # from front to back
    VEC_FRONT_BACK = "vec_front_back" # from front to back
    VEC_BACK_FRONT = "vec_back_front" # from back to front
    AXIS_BOTTOM_TOP = "axis_bottom_top" # from bottom to top
    VEC_BOTTOM_TOP = "vec_bottom_top" # from bottom to top
    VEC_TOP_BOTTOM = "vec_top_bottom" # from top to bottom
    
    #FACE_LEFT = "face_left"
    #FACE_BACK = "face_back"
    #FACE_TOP = "face_top"

# notes: perhaps we should change bottom-top, to back-front for handles 
#       (this would align with standing things, also umbrella is defined without handle, also top-bottom for squeezing makes more sense for filtering)
# notes: perhaps swap contain/support to contain open upwards , support upwards#
# stand upward and support/contain upward are basically same for bed, chair, sofa, armchair...
#   -> if it can stand, use stand (little bit problem, everything can stand, so it is not very clear when this is used? doesnt matter, for e.g. pillow support is more clear than stand)
# notes: decorative display item accessoire not well defined
#   -> wristband armband, earring fingerring (with stone)
#       -> some one shiny direction (somtimes axis)
#       -> some wear direciton plus another shiny direction
#   -> jewelry -> one shiny direction (sometimes axis)

#   -> first top bottom eyelet (ring/wristband) then (front for shiny direction)
#   -> earring (top-bottom hanging) then (front-back front)
#   -> problem button if front-back then does not align with pin (bottom top) (and if switched pin then does not align with needle)
# notes: it would be better if stand would overrule human upward direction e.g. for scale and keyboard?
# # note problem as with bed toward and away form user is not clear what direction, perhaps below and above user lying

# note: ORIENT_CONDITIONS_EXTRINSIC_USER_OBJECT_OBJECT.AWAY_FROM_OUTFLOW, and shoot_backwards are basically same? and should be both intrinic?
# note: what about animal which moves on four feet but can also stand on two feet, does orientation change?
# note: replace toward mount vehicle/wall with toward_mount_backward and toward_mount_upward for ceiling
# note: replace human with user (so that it can also be an animal)
# note: does cutting fruits change direction?
# note: does opening book change direction?

class ORIENT_CONDITIONS_INTRINSIC(StrEnum):
    # NO USER INTERACTION (INTRINSIC)
    MOVE_BACKWARD = "move_backward" # natural moving direction is backwards
    MOVE_UPWARD = "move_upward" # natural moving direction is front (upward can be upward direction when moving or moving upwards)
    MOVE_BACKWARD_AND_FORWARD = "move_backward_and_forward" # natural moving direction is front
    CONTAIN_OR_SUPPORT_UPWARD = "contain_or_support_upward" #  e.g. table, tray, bowl, cup, bag # note there is also temporary containment
    CONTAIN_OR_SUPPORT_UPWARD_AND_DOWNWARD = "contain_or_support_upward_and_downward"
    COVER_UPWARD = "cover_upward" # top, umbrella     
    COVER_UPWARD_AND_DOWNWARD = "cover_upward_and_downward" # blanket
    ENTRY_BACKWARD = "entry_backward" # back
    STAND_UPWARD = "stand_upward" # bottom is the contact point to ground plane
    STAND_UPWARD_DOWNWARD = "stand_upward_downward" # bottom/top is the contact point to ground plane (egg)
    FACE_BACKWARD = "face_backward" # some face that is looking forward (if there are eyes)
    SPROUT_UPWARD = "sprout_upward" # natural growing direction is top (only roots grow in bottom direction) # note: for mushroom it is the had.
    SPROUT_UPWARD_AND_DOWNWARD = "sprout_upward_and_downward" 
    STEM_UPWARD = "stem_upward" # natural growing direction is top (only roots grow in bottom direction)
    WINDING_AXIS = "winding_axis"
    FLOW_BACKWARD = "flow_backward" # back wind/bullets/arrows/electric signal/light
    FLOW_BACKWARD_AND_FORWARD = "flow_backward_and_forward" # back
    UNWINDING_EXTENSION_BACKWARD = "unwinding_extension_backward"
    EYELET_AXIS = "eyelet_axis"
    TOWARD_MOUNT_BACKWARD = "toward_mount_backward" # back is facing wall/vehicle
    TOWARD_AND_AWAY_FROM_MOUNT_BACKWARD = "toward_and_away_from_mount_backward" # back-front axis for e.g. deadbolt
    TOWARD_MOUNT_UPWARD = "toward_mount_upward" # top is facing ceiling / hanger

class ORIENT_CONDITIONS_EXTRINSIC_USER_OBJECT(StrEnum):
    # USER-OBJECT INTERACTION (EXTRINSIC)
    AWAY_FROM_USER = "away_from_user" # back 
    AWAY_FROM_USER_FINGERS = "away_from_user_fingers" # back (keyboard, violin, guitar)
    TOWARD_AND_AWAY_FROM_USER = "toward_and_away_from_user" # top as human 
    USER_UPWARD = "user_upward" # top as human 
    USER_BACKWARD = "user_backward" # back as human 
    USER_UPWARD_AND_DOWNWARD = "user_upward_and_downward" # top as human 
    USER_LYING_AXIS_HEAD_FEET = "user_lying_axis_head_feet" # back-front
    USER_FORWARD_AND_BACKWARD = "user_forward_and_backward" # forward and back as human 
    ALONG_GRAB_TWO_HANDS_LEFT_RIGHT =  "along_grab_two_hands_left_right" # left-right
    ALONG_GRAB_TWO_HANDS_AWAY_FROM_FUNCTION = "along_grab_two_hands_away_from_function" # top
    ALONG_GRAB_ONE_HAND = "along_grab_one_hand" # top-bottom
    ALONG_GRAB_ONE_HAND_TOWARD_FUNCTION = "along_grab_one_hand_toward_function" # top
    ALONG_STAND_GRAB_ONE_HAND_AWAY_FROM_FUNCTION = "along_stand_grab_one_hand_grab_away_from_function" # back
    ALONG_GRAB_TWO_FINGERS_TOWARD_FUNCTION = "along_grab_two_fingers_toward_function" # top
    HAND_FINGERS = "hand_fingers" # top (glove)
    HAND_BACK = "hand_back" # back (glove)

class ORIENT_CONDITIONS_EXTRINSIC_USER_OBJECT_OBJECT(StrEnum):
    # USER-OBJECT-OBJECT INTERACTION (while human interaction)
    AWAY_FROM_OBJECT = "away_from_object" # top as human 
    TOWARD_AND_AWAY_FROM_OBJECT = "toward_and_away_from_object" # top as human 
    OBJECT_UPWARD = "object_upward" # top as object 
    OBJECT_SQUEEZE = "object_squeeze" # (e.g. lemon squeezer with handle)
    OBJECT_SQUEEZE_BACK = "object_squeeze_back"


MAP_CONDITION_TO_RULE = {
    # NO USER INTERACTION (INTRINSIC)
    ORIENT_CONDITIONS_INTRINSIC.MOVE_BACKWARD: ORIENT_AXIS_RULES.VEC_FRONT_BACK, # natural moving direction is backwards
    ORIENT_CONDITIONS_INTRINSIC.MOVE_UPWARD: ORIENT_AXIS_RULES.VEC_BOTTOM_TOP, # natural moving direction is front (upward can be upward direction when moving or moving upwards)
    ORIENT_CONDITIONS_INTRINSIC.MOVE_BACKWARD_AND_FORWARD: ORIENT_AXIS_RULES.AXIS_FRONT_BACK, # natural moving direction is front
    ORIENT_CONDITIONS_INTRINSIC.CONTAIN_OR_SUPPORT_UPWARD: ORIENT_AXIS_RULES.VEC_BOTTOM_TOP, #  e.g. table, tray, bowl, cup, bag # note there is also temporary containment
    ORIENT_CONDITIONS_INTRINSIC.CONTAIN_OR_SUPPORT_UPWARD_AND_DOWNWARD: ORIENT_AXIS_RULES.AXIS_BOTTOM_TOP,
    ORIENT_CONDITIONS_INTRINSIC.COVER_UPWARD: ORIENT_AXIS_RULES.VEC_BOTTOM_TOP, # top, umbrella     
    ORIENT_CONDITIONS_INTRINSIC.COVER_UPWARD_AND_DOWNWARD: ORIENT_AXIS_RULES.AXIS_BOTTOM_TOP, # blanket
    ORIENT_CONDITIONS_INTRINSIC.ENTRY_BACKWARD: ORIENT_AXIS_RULES.VEC_FRONT_BACK, # back
    ORIENT_CONDITIONS_INTRINSIC.STAND_UPWARD: ORIENT_AXIS_RULES.VEC_BOTTOM_TOP, # bottom is the contact point to ground plane
    ORIENT_CONDITIONS_INTRINSIC.STAND_UPWARD_DOWNWARD: ORIENT_AXIS_RULES.AXIS_BOTTOM_TOP, # bottom/top is the contact point to ground plane (egg)
    ORIENT_CONDITIONS_INTRINSIC.FACE_BACKWARD: ORIENT_AXIS_RULES.VEC_FRONT_BACK, # some face that is looking forward (if there are eyes)
    ORIENT_CONDITIONS_INTRINSIC.SPROUT_UPWARD: ORIENT_AXIS_RULES.VEC_BOTTOM_TOP, # natural growing direction is top (only roots grow in bottom direction) # note: for mushroom it is the had.
    ORIENT_CONDITIONS_INTRINSIC.SPROUT_UPWARD_AND_DOWNWARD: ORIENT_AXIS_RULES.AXIS_BOTTOM_TOP,  
    ORIENT_CONDITIONS_INTRINSIC.STEM_UPWARD: ORIENT_AXIS_RULES.VEC_BOTTOM_TOP, # natural growing direction is top (only roots grow in bottom direction)
    ORIENT_CONDITIONS_INTRINSIC.WINDING_AXIS: ORIENT_AXIS_RULES.AXIS_BOTTOM_TOP, 
    ORIENT_CONDITIONS_INTRINSIC.FLOW_BACKWARD: ORIENT_AXIS_RULES.VEC_FRONT_BACK, # back wind/bullets/arrows/electric signal/light
    ORIENT_CONDITIONS_INTRINSIC.FLOW_BACKWARD_AND_FORWARD: ORIENT_AXIS_RULES.AXIS_FRONT_BACK, # back-front
    ORIENT_CONDITIONS_INTRINSIC.UNWINDING_EXTENSION_BACKWARD: ORIENT_AXIS_RULES.VEC_FRONT_BACK, 
    ORIENT_CONDITIONS_INTRINSIC.EYELET_AXIS: ORIENT_AXIS_RULES.AXIS_BOTTOM_TOP,
    ORIENT_CONDITIONS_INTRINSIC.TOWARD_MOUNT_BACKWARD: ORIENT_AXIS_RULES.VEC_FRONT_BACK, # back is facing wall/vehicle
    ORIENT_CONDITIONS_INTRINSIC.TOWARD_AND_AWAY_FROM_MOUNT_BACKWARD: ORIENT_AXIS_RULES.AXIS_FRONT_BACK, # back-front axis for e.g. deadbolt
    ORIENT_CONDITIONS_INTRINSIC.TOWARD_MOUNT_UPWARD: ORIENT_AXIS_RULES.VEC_BOTTOM_TOP, # top is facing ceiling / hanger

    # USER-OBJECT INTERACTION (EXTRINSIC)
    ORIENT_CONDITIONS_EXTRINSIC_USER_OBJECT.AWAY_FROM_USER: ORIENT_AXIS_RULES.VEC_FRONT_BACK, # back 
    ORIENT_CONDITIONS_EXTRINSIC_USER_OBJECT.AWAY_FROM_USER_FINGERS: ORIENT_AXIS_RULES.VEC_FRONT_BACK, # back (keyboard, violin, guitar)
    ORIENT_CONDITIONS_EXTRINSIC_USER_OBJECT.TOWARD_AND_AWAY_FROM_USER: ORIENT_AXIS_RULES.AXIS_FRONT_BACK, # top as human 
    ORIENT_CONDITIONS_EXTRINSIC_USER_OBJECT.USER_UPWARD: ORIENT_AXIS_RULES.VEC_BOTTOM_TOP, # top as human 
    ORIENT_CONDITIONS_EXTRINSIC_USER_OBJECT.USER_BACKWARD: ORIENT_AXIS_RULES.VEC_FRONT_BACK, # back as human 
    ORIENT_CONDITIONS_EXTRINSIC_USER_OBJECT.USER_UPWARD_AND_DOWNWARD: ORIENT_AXIS_RULES.AXIS_BOTTOM_TOP, # top as human 
    ORIENT_CONDITIONS_EXTRINSIC_USER_OBJECT.USER_LYING_AXIS_HEAD_FEET: ORIENT_AXIS_RULES.AXIS_FRONT_BACK, # back-front
    ORIENT_CONDITIONS_EXTRINSIC_USER_OBJECT.USER_FORWARD_AND_BACKWARD: ORIENT_AXIS_RULES.AXIS_FRONT_BACK, # forward and back as human 
    ORIENT_CONDITIONS_EXTRINSIC_USER_OBJECT.ALONG_GRAB_TWO_HANDS_LEFT_RIGHT: ORIENT_AXIS_RULES.AXIS_RIGHT_LEFT, # left-right
    ORIENT_CONDITIONS_EXTRINSIC_USER_OBJECT.ALONG_GRAB_TWO_HANDS_AWAY_FROM_FUNCTION: ORIENT_AXIS_RULES.VEC_BOTTOM_TOP, # top
    ORIENT_CONDITIONS_EXTRINSIC_USER_OBJECT.ALONG_GRAB_ONE_HAND: ORIENT_AXIS_RULES.AXIS_BOTTOM_TOP, # top-bottom
    ORIENT_CONDITIONS_EXTRINSIC_USER_OBJECT.ALONG_GRAB_ONE_HAND_TOWARD_FUNCTION: ORIENT_AXIS_RULES.VEC_BOTTOM_TOP, # top
    ORIENT_CONDITIONS_EXTRINSIC_USER_OBJECT.ALONG_STAND_GRAB_ONE_HAND_AWAY_FROM_FUNCTION: ORIENT_AXIS_RULES.VEC_FRONT_BACK, # back
    ORIENT_CONDITIONS_EXTRINSIC_USER_OBJECT.ALONG_GRAB_TWO_FINGERS_TOWARD_FUNCTION: ORIENT_AXIS_RULES.VEC_BOTTOM_TOP, # top
    ORIENT_CONDITIONS_EXTRINSIC_USER_OBJECT.HAND_FINGERS: ORIENT_AXIS_RULES.VEC_BOTTOM_TOP, # (glove)
    ORIENT_CONDITIONS_EXTRINSIC_USER_OBJECT.HAND_BACK: ORIENT_AXIS_RULES.VEC_FRONT_BACK, # (glove)

    # USER-OBJECT-OBJECT INTERACTION (while human interaction)
    ORIENT_CONDITIONS_EXTRINSIC_USER_OBJECT_OBJECT.AWAY_FROM_OBJECT: ORIENT_AXIS_RULES.VEC_FRONT_BACK, # back 
    ORIENT_CONDITIONS_EXTRINSIC_USER_OBJECT_OBJECT.TOWARD_AND_AWAY_FROM_OBJECT: ORIENT_AXIS_RULES.AXIS_FRONT_BACK, # back-front 
    ORIENT_CONDITIONS_EXTRINSIC_USER_OBJECT_OBJECT.OBJECT_UPWARD: ORIENT_AXIS_RULES.VEC_BOTTOM_TOP, # top as object 
    ORIENT_CONDITIONS_EXTRINSIC_USER_OBJECT_OBJECT.OBJECT_SQUEEZE: ORIENT_AXIS_RULES.AXIS_FRONT_BACK, # (e.g. lemon squeezer with handle)
    ORIENT_CONDITIONS_EXTRINSIC_USER_OBJECT_OBJECT.OBJECT_SQUEEZE_BACK: ORIENT_AXIS_RULES.VEC_FRONT_BACK, # back
}

    #"stand"
    #"scrap": ORIEN
    #hit: top part of the object turns another object (e.g. bat)
    #turn: top part of the object turns another object (e.g. wrench)
    #contain: top is the containing side (e.g. spoon, ladle)
    # "containing_or_supporting": ORIENT_AXIS_RULES.VEC_BOTTOM_TOP, #  e.g. table, tray, bowl, cup # note there is also temporary containment
    #"rotating-squeeze

    # start from top, please not that only two axis must be determined to define the complete orientation.
    # some objects have zero or only one axis due to symmetry.
    # things that define orientation: 
    #  no interaction (human-object or object-object): 
    #       moving: front-back (on-its-own, also rotating motor, ventilator)
    #       growing: top-bottom
    #       containing-with-doors:      
    #       conatining_with_lid_or_cap: top-bottom (containing food or other objects, milk package, table, shelf, bed, )
    #       mounting: top-bottom, back-at-wall (fixed)
    #  with human interaction: 
    #       sit: as human
    #       lie: as human (pillow yes, but bed not?)
    #       read: front faces human so that readable
    #       type: front faces human so that readable 
    #       open: front
    #       wear: 
    #       trim/dry: front faces human hair/nails 
    #       hold: # defines one axis
    #           # note: a handle is a part without it the object would still function
    #           two stick handles: front-back
    #           two carry handles: left-right
    #           one stick handle: front-back or  
    #           one carry handle: top-bottom
    #           no handle:
    #               two finger hold: top-bottom
    #               one hand hold: top-bottom (whereever the thumb is, is top)
    #               two hands hold: left-right or right-left
    #  with object-object interaction (while human interaction)
    #       contain: top is the containing side (e.g. spoon, ladle)
    #       linear-squeeze: top-bottom, top is sometimes containing (e.g. lemon squeezer with handle)
    #       rotating-squeeze
    #       hit: top part of the object turns another object (e.g. bat)
    #       turn: top part of the object turns another object (e.g. wrench)
    #  no interaction:
    #       mounting: top-bottom, back-at-wall (temporary)
    #
    # note: deformations usually never change the pose, only in very few cases it happens: cut water melon, book
    # 
    # q1: is pin/nail mounting temporary or fixed?
    # q2: if we make all devices with handle front-back instead of top-bottom we have less problems? as there is not this top or bottom question?
    # q3: 
    
    #       interaction functions: contain, squeeze, wear, filter

    # mount, handle/grab, moving, function:wear, function:contain, function:squeeze

orientation_tree = {
"device": { 

    "fixed_always": {
        "fixed_at_wall"
        "top": "upwards as fixed",
        "front": "facing towards user",
        "categories": ["toilet", "urinal", "mixer", "scale", "microwave", "oven", "refrigerator", "colors_for_painting", "drums?"],
    },
    "fixed_when_used":{
        "top": "upwards as user",
        "front": "facing towards user",
    },
    "stand_in_front_when_used": {
        "top": "upwards as user",
        "front": "looking towards user",
        "categories": ["toilet", "urinal", "mixer", "scale", "microwave", "oven", "refrigerator", "colors_for_painting", "drums?"],
    }, # note: does not need reading in contrast to keybaord
    # note: add suitcase/trolley faces upwards when opening (handle is not important for orientation here)
    "sit_or_lie": {
        "top": "upwards as user",
        "front": "looking as user",
        "categories": ["chair", "bench", "toilet", "stool", "airplane", "car", "bicycle", "pillow"],
    },
    "squeeze": { # imagine crocodile head
        "front": "direction towards squeezing",
        "top_bottom": "direction of squeezing",
        "top": "upwards if stands on its own",
        "categories": ["stapler", "scissor", "pliers", "clothing_pin"],
    },
    "pin or nail": {
        "top": "direction of pinning or nailing",
        "categories": ["nail", "pushpin"],
    }, # note: problem if pin is flat and surface becomes larger like a jewelery pin
    "eat": {
        "top": "upwards as user for eating",
        "front": "facing user",
        "categories": ["pacifier", "kebab", "burger"]
    }, 
    "wear": {
        "ear": {}, # earring, earphone (inear, overear), earplug
        "hand": {}, # glove, ring 
        "neck": {}, # necklace, scarf, tie# # front is where theneglace, scarf or tie looks
        "armwrist": {}, # bracelet, armband, sleeve, wristband # top is on top of arm, front is where the arm goes in
        "head": {}, # cap, hat, helmet, barrette (crocodile head with top upwards), crown, hairnet
        "feet": {}, # shoes, boots, sandals, slippers,
        "mouth": {}, # pacifier, face_mask
        "eyes": {}, # glasses, sunglasses, goggles
    },
    "read": {
        "top": "upwards as user for reading", # top: keyboard looks top if laptop, remote control
        "front": "facing user",
        "categories": ["book", "laptop", "notebook", "magazine", "tablet", "clock", "smartphone"],
    },
    "hang": {}, # -> top: upwards if hangs: pad lock, towel, shawl/scarf
    "rotates": {}, # -> front: axis rotating facing user: fan, wheel, steering_wheel, motor
    "one_handle": {
        "functions_at_stand": { # note: action/handle along one direction
            "top": "upwards",
            "back": "direction of handle if handle and handle not along upwards",
            "front": "direction of clicks if computer mouse, \
                      direction of moving if scrubbing brush? perhaps not functions at stand, \
                      direction of spout if watering can, \
                      direction of first paper if toilet paper, \
                      direction of pen if sharpener \
                      direction of user if latch or handbag",
            "categories": ["sharpener", "pan", "pot", "mug", "cup", "teapot", "kettle", "can", "mouse", "latch", "glass", "soap", 
                           "fire_extinguisher", "handback", "funnel", "plate", "thread"], 
        }, 
        "not_functions_at_stand": { # note: action/handle along one direction, action handle not along one direction
            "front": "direction of action/contact",
            "bottom": "handle", # if handle is not in direction of front
            "top": "where the air comes out if whistle \
                    where the rain drops on if umbrella",
            "categories": ["hammer", "axe", "hairdryer", "toothbrush", "spoon", "fork", "knife", "whistle", "wrench", "soap"],
        }, 
        #note: differentiate between grab at stick parallel to action/contact direction (mop, light bulb, )
        #                       and  grab at stick perpendicular to action/contact direction (scrubbing brush, hair dryer)
        # note: file tool, wrench front is direction of flat sides / screwdriver? bolt? thread? toilet paper? handle? candy bar? 
        # rubber? cone? cycliner? dropper? drums? 
    },  # note: hair brush, flat hair brush? handle horizontal? hose (cylinder)? (icecream cone), iron? light bulb? masher?
        # note: matchbox? mop?
        #  note; wall socket? socket? wedding ring?
    "two_handles": { 
        "open_lid": {
            "top": "upwards as user when opening lid",
            "front": "facing user", # note: for toothpaste one flat side is front
            "categories": ["box", "facial", "bottle", "cream", "toothpaste", "jar", "canister", "pot_with_lid"],
        }, # suitcase (but rolling suitcase?)
        "open_left_right": {
            "top": "upwards as user when opening",
            "front": "facing user",
            "categories": ["purse", "wallet"], 
        }, 
        "balance": {
            "left_right": "left hand, right hand", 
            "top": "upwards as user when balancing",
            "categories": ["tray", "basket", "scale", "rolling_pin", "controller"],
        }, 
    },
},
"plants": {
    "hanging":{
        "top": "towards hanging point",
        "front": "direction of bending if banana (towards inner curve)",
        "categories": ["banana", "apple", "orange", "grape", "pear"]
    },
    "grounded": {
        "top": "towards growing direction",
        "categories": ["tree", "flower_pot", "cactus", "pineapple"],
    },
},  
"animals": {
    "front": "facing direction of main body",
    "top": "upward direction of main body",
    "categories": ["dog", "cat", "horse", "cow", "sheep", "elephant", "bird"],
},
}

# AEROSOL_CAN = "aerosol_can"
# AIR_CONDITIONER = "air_conditioner"
# AIRPLANE = "airplane"
# ALARM_CLOCK = "alarm_clock"
# ALCOHOL = "alcohol"
# ALLIGATOR = "alligator"
# ALMOND = "almond"
# AMBULANCE = "ambulance"
# AMPLIFIER = "amplifier"
# ANKLET = "anklet"
# APPLE = "apple"
# APPLESAUCE = "applesauce"
# APRICOT = "apricot"
# APRON = "apron"
# AQUARIUM = "aquarium"
# ARCTIC_TYPE_OF_SHOE = "arctic_type_of_shoe"
# ARMBAND = "armband"
# ARMCHAIR = "armchair"
# ARMOIRE = "armoire"
# ARMOR = "armor"
# ARMY_TANK = "army_tank"
# ARTICHOKE = "artichoke"
# ASHTRAY = "ashtray"
# ASPARAGUS = "asparagus"
# ATOMIZER = "atomizer"
# AUTOMATIC_WASHER = "automatic_washer"
# AVOCADO = "avocado"
# AWARD = "award"
# AWNING = "awning"
# AX = "ax"
# BABY_BUGGY = "baby_buggy"
# BACKPACK = "backpack"
# BAGEL = "bagel"
# BAGPIPE = "bagpipe"
# BAGUET = "baguet"
# BAIT = "bait"
# BALL = "ball"
# BALLET_SKIRT = "ballet_skirt"
# BALLOON = "balloon"
# BAMBOO = "bamboo"
# BANANA = "banana"
# BAND_AID = "band_aid"
# BANDAGE = "bandage"
# BANDANNA = "bandanna"
# BARBELL = "barbell"
# BARREL = "barrel"
# BARRETTE = "barrette"
# BARROW = "barrow"
# BASEBALL = "baseball"
# BASEBALL_BASE = "baseball_base"
# BASEBALL_BAT = "baseball_bat"
# BASEBALL_CAP = "baseball_cap"
# BASEBALL_GLOVE = "baseball_glove"
# BASKET = "basket"
# BASKETBALL = "basketball"
# BASKETBALL_BACKBOARD = "basketball_backboard"
# BASS_HORN = "bass_horn"
# BATH_TOWEL = "bath_towel"
# BATHTUB = "bathtub"
# BATTER_FOOD = "batter_food"
# BATTERY = "battery"
# BEACHBALL = "beachball"
# BEAD = "bead"
# BEAN_CURD = "bean_curd"
# BEANBAG = "beanbag"
# BEANIE = "beanie"
# BED = "bed"
# BEDPAN = "bedpan"
# BEDSPREAD = "bedspread"
# BEEF_FOOD = "beef_food"
# BEEPER = "beeper"
# BEER_BOTTLE = "beer_bottle"
# BEER_CAN = "beer_can"
# BELL = "bell"
# BELL_PEPPER = "bell_pepper"
# BELT = "belt"
# BELT_BUCKLE = "belt_buckle"
# BENCH = "bench"
# BERET = "beret"
# BIB = "bib"
# BIBLE = "bible"
# BICYCLE = "bicycle"
# BILLBOARD = "billboard"
# BINDER = "binder"
# BINOCULARS = "binoculars"
# BIRD = "bird"
# BIRDBATH = "birdbath"
# BIRDCAGE = "birdcage"
# BIRDFEEDER = "birdfeeder"
# BIRDHOUSE = "birdhouse"
# BIRTHDAY_CAKE = "birthday_cake"
# BIRTHDAY_CARD = "birthday_card"
# BLACK_SHEEP = "black_sheep"
# BLACKBERRY = "blackberry"
# BLANKET = "blanket"
# BLAZER = "blazer"
# BLENDER = "blender"
# BLIMP = "blimp"
# BLINDER_FOR_HORSES = "blinder_for_horses"
# BLINKER = "blinker"
# BLUEBERRY = "blueberry"
# BOAT = "boat"
# BOBBIN = "bobbin"
# BOBBY_PIN = "bobby_pin"
# BOILED_EGG = "boiled_egg"
# BOLO_TIE = "bolo_tie"
# BOLT = "bolt"
# BONNET = "bonnet"
# BOOK = "book"
# BOOKLET = "booklet"
# BOOKMARK = "bookmark"
# BOOM_MICROPHONE = "boom_microphone"
# BOOT = "boot"
# BOTTLE = "bottle"
# BOTTLE_CAP = "bottle_cap"
# BOTTLE_OPENER = "bottle_opener"
# BOUQUET = "bouquet"
# BOW_DECORATIVE_RIBBONS = "bow_decorative_ribbons"
# BOW_TIE = "bow_tie"
# BOW_WEAPON = "bow_weapon"
# BOWL = "bowl"
# BOWLING_BALL = "bowling_ball"
# BOX = "box"
# BOXING_GLOVE = "boxing_glove"
# BRACELET = "bracelet"
# BRAKE_LIGHT = "brake_light"
# BRASS_PLAQUE = "brass_plaque"
# BRASSIERE = "brassiere"
# BREAD = "bread"
# BREAD_BIN = "bread_bin"
# BREECHCLOTH = "breechcloth"
# BRIDAL_GOWN = "bridal_gown"
# BRIEFCASE = "briefcase"
# BROACH = "broach"
# BROCCOLI = "broccoli"
# BROOM = "broom"
# BROWNIE = "brownie"
# BRUSSELS_SPROUTS = "brussels_sprouts"
# BUBBLE_GUM = "bubble_gum"
# BUCKET = "bucket"
# BULL = "bull"
# BULLDOG = "bulldog"
# BULLDOZER = "bulldozer"
# BULLET_TRAIN = "bullet_train"
# BULLETPROOF_VEST = "bulletproof_vest"
# BULLHORN = "bullhorn"
# BUN = "bun"
# BUNK_BED = "bunk_bed"
# BUOY = "buoy"
# BURRITO = "burrito"
# BUS_VEHICLE = "bus_vehicle"
# BUSINESS_CARD = "business_card"
# BUTTER = "butter"
# BUTTON = "button"
# CAB_TAXI = "cab_taxi"
# CABANA = "cabana"
# CABIN_CAR = "cabin_car"
# CABINET = "cabinet"
# CAKE = "cake"
# CALCULATOR = "calculator"
# CALENDAR = "calendar"
# CALF = "calf"
# CAMCORDER = "camcorder"
# CAMEL = "camel"
# CAMERA = "camera"
# CAMERA_LENS = "camera_lens"
# CAMPER_VEHICLE = "camper_vehicle"
# CAN = "can"
# CAN_OPENER = "can_opener"
# CANDLE = "candle"
# CANDLE_HOLDER = "candle_holder"
# CANDY_BAR = "candy_bar"
# CANDY_CANE = "candy_cane"
# CANISTER = "canister"
# CANOE = "canoe"
# CANTALOUP = "cantaloup"
# CAP_HEADWEAR = "cap_headwear"
# CAPPUCCINO = "cappuccino"
# CAR_AUTOMOBILE = "car_automobile"
# CAR_BATTERY = "car_battery"
# CARD = "card"
# CARDIGAN = "cardigan"
# CARGO_SHIP = "cargo_ship"
# CARNATION = "carnation"
# CARROT = "carrot"
# CART = "cart"
# CARTON = "carton"
# CASH_REGISTER = "cash_register"
# CASSEROLE = "casserole"
# CASSETTE = "cassette"
# CAST = "cast"
# CAT = "cat"
# CAULIFLOWER = "cauliflower"
# CAYENNE_SPICE = "cayenne_spice"
# CD_PLAYER = "cd_player"
# CELERY = "celery"
# CELLULAR_TELEPHONE = "cellular_telephone"
# CHAIN_MAIL = "chain_mail"
# CHAIR = "chair"
# CHAISE_LONGUE = "chaise_longue"
# CHALICE = "chalice"
# CHANDELIER = "chandelier"
# CHAP = "chap"
# CHECKBOOK = "checkbook"
# CHECKERBOARD = "checkerboard"
# CHERRY = "cherry"
# CHICKEN_ANIMAL = "chicken_animal"
# CHICKPEA = "chickpea"
# CHILI_VEGETABLE = "chili_vegetable"
# CHIME = "chime"
# CHINAWARE = "chinaware"
# CHOCOLATE_BAR = "chocolate_bar"
# CHOCOLATE_CAKE = "chocolate_cake"
# CHOCOLATE_MILK = "chocolate_milk"
# CHOCOLATE_MOUSSE = "chocolate_mousse"
# CHOKER = "choker"
# CHOPPING_BOARD = "chopping_board"
# CHRISTMAS_TREE = "christmas_tree"
# CIDER = "cider"
# CIGAR_BOX = "cigar_box"
# CIGARETTE = "cigarette"
# CIGARETTE_CASE = "cigarette_case"
# CINCTURE = "cincture"
# CISTERN = "cistern"
# CLARINET = "clarinet"
# CLASP = "clasp"
# CLEANSING_AGENT = "cleansing_agent"
# CLEAT_FOR_SECURING_ROPE = "cleat_for_securing_rope"
# CLEMENTINE = "clementine"
# CLIP = "clip"
# CLIPBOARD = "clipboard"
# CLIPPERS_FOR_PLANTS = "clippers_for_plants"
# CLOAK = "cloak"
# CLOCK = "clock"
# CLOCK_TOWER = "clock_tower"
# CLOTHES_HAMPER = "clothes_hamper"
# CLOTHESPIN = "clothespin"
# CLUTCH_BAG = "clutch_bag"
# COASTER = "coaster"
# COAT = "coat"
# COAT_HANGER = "coat_hanger"
# COCK = "cock"
# COCOA_BEVERAGE = "cocoa_beverage"
# COCONUT = "coconut"
# COFFEE_MAKER = "coffee_maker"
# COFFEE_TABLE = "coffee_table"
# COFFEEPOT = "coffeepot"
# COIL = "coil"
# COIN = "coin"
# COLANDER = "colander"
# COLESLAW = "coleslaw"
# COLORING_MATERIAL = "coloring_material"
# COMBINATION_LOCK = "combination_lock"
# COMIC_BOOK = "comic_book"
# COMPASS = "compass"
# COMPUTER_KEYBOARD = "computer_keyboard"
# CONDIMENT = "condiment"
# CONE = "cone"
# CONTROL = "control"
# CONVERTIBLE_AUTOMOBILE = "convertible_automobile"
# COOKER = "cooker"
# COOKIE = "cookie"
# COOKING_UTENSIL = "cooking_utensil"
# COOLER_FOR_FOOD = "cooler_for_food"
# CORK_BOTTLE_PLUG = "cork_bottle_plug"
# CORKSCREW = "corkscrew"
# CORNBREAD = "cornbread"
# CORNET = "cornet"
# CORNICE = "cornice"
# CORNMEAL = "cornmeal"
# CORSET = "corset"
# COSTUME = "costume"
# COVER = "cover"
# COW = "cow"
# COWBELL = "cowbell"
# COWBOY_HAT = "cowboy_hat"
# CRAB_ANIMAL = "crab_animal"
# CRABMEAT = "crabmeat"
# CRACKER = "cracker"
# CRATE = "crate"
# CRAWFISH = "crawfish"
# CRAYON = "crayon"
# CREAM_PITCHER = "cream_pitcher"
# CRESCENT_ROLL = "crescent_roll"
# CRISP_POTATO_CHIP = "crisp_potato_chip"
# CROCK_POT = "crock_pot"
# CROSSBAR = "crossbar"
# CROUTON = "crouton"
# CROW = "crow"
# CROWBAR = "crowbar"
# CROWN = "crown"
# CRUCIFIX = "crucifix"
# CRUISE_SHIP = "cruise_ship"
# CRUMB = "crumb"
# CRUTCH = "crutch"
# CUB_ANIMAL = "cub_animal"
# CUBE = "cube"
# CUCUMBER = "cucumber"
# CUFFLINK = "cufflink"
# CUP = "cup"
# CUPBOARD = "cupboard"
# CUPCAKE = "cupcake"
# CURLING_IRON = "curling_iron"
# CURTAIN = "curtain"
# CUSHION = "cushion"
# CYLINDER = "cylinder"
# CYMBAL = "cymbal"
# DAGGER = "dagger"
# DALMATIAN = "dalmatian"
# DATE_FRUIT = "date_fruit"
# DEADBOLT = "deadbolt"
# DECK_CHAIR = "deck_chair"
# DEER = "deer"
# DENTAL_FLOSS = "dental_floss"
# DESK = "desk"
# DETERGENT = "detergent"
# DIAPER = "diaper"
# DIARY = "diary"
# DIE = "die"
# DINING_TABLE = "dining_table"
# DIRT_BIKE = "dirt_bike"
# DISH = "dish"
# DISH_ANTENNA = "dish_antenna"
# DISHRAG = "dishrag"
# DISHTOWEL = "dishtowel"
# DISHWASHER = "dishwasher"
# DISHWASHER_DETERGENT = "dishwasher_detergent"
# DISPENSER = "dispenser"
# DIVING_BOARD = "diving_board"
# DIXIE_CUP = "dixie_cup"
# DOG = "dog"
# DOG_COLLAR = "dog_collar"
# DOLL = "doll"
# DOLLAR = "dollar"
# DOLLHOUSE = "dollhouse"
# DOLPHIN = "dolphin"
# DOMESTIC_ASS = "domestic_ass"
# DOORKNOB = "doorknob"
# DOORMAT = "doormat"
# DOUGHNUT = "doughnut"
# DOVE = "dove"
# DRAWER = "drawer"
# DRESS = "dress"
# DRESS_HAT = "dress_hat"
# DRESSER = "dresser"
# DRILL = "drill"
# DRONE = "drone"
# DROPPER = "dropper"
# DRUM_MUSICAL_INSTRUMENT = "drum_musical_instrument"
# DRUMSTICK = "drumstick"
# DUCK = "duck"
# DUCKLING = "duckling"
# DUCT_TAPE = "duct_tape"
# DUFFEL_BAG = "duffel_bag"
# DUMBBELL = "dumbbell"
# DUMPSTER = "dumpster"
# DUSTPAN = "dustpan"
# EARPHONE = "earphone"
# EARPLUG = "earplug"
# EARRING = "earring"
# EASEL = "easel"
# ECLAIR = "eclair"
# EDIBLE_CORN = "edible_corn"
# EGG = "egg"
# EGG_ROLL = "egg_roll"
# EGG_YOLK = "egg_yolk"
# EGGBEATER = "eggbeater"
# EGGPLANT = "eggplant"
# ELEPHANT = "elephant"
# ELEVATOR_CAR = "elevator_car"
# ELK = "elk"
# ENVELOPE = "envelope"
# ERASER = "eraser"
# EYEPATCH = "eyepatch"
# FAN = "fan"
# FAUCET = "faucet"
# FEDORA = "fedora"
# FERRET = "ferret"
# FERRY = "ferry"
# FIG_FRUIT = "fig_fruit"
# FIGHTER_JET = "fighter_jet"
# FIGURINE = "figurine"
# FILE_CABINET = "file_cabinet"
# FILE_TOOL = "file_tool"
# FIRE_ALARM = "fire_alarm"
# FIRE_ENGINE = "fire_engine"
# FIRE_EXTINGUISHER = "fire_extinguisher"
# FIREPLACE = "fireplace"
# FIREPLUG = "fireplug"
# FIRST_AID_KIT = "first_aid_kit"
# FISH_FOOD = "fish_food"
# FISHBOWL = "fishbowl"
# FLAMINGO = "flamingo"
# FLANNEL = "flannel"
# FLASH = "flash"
# FLASHLIGHT = "flashlight"
# FLEECE = "fleece"
# FLIP_FLOP_SANDAL = "flip_flop_sandal"
# FLIPPER_FOOTWEAR = "flipper_footwear"
# FLOWER_ARRANGEMENT = "flower_arrangement"
# FLOWERPOT = "flowerpot"
# FLUTE_GLASS = "flute_glass"
# FOAL = "foal"
# FOLDING_CHAIR = "folding_chair"
# FOOD_PROCESSOR = "food_processor"
# FOOTBALL_AMERICAN = "football_american"
# FOOTBALL_HELMET = "football_helmet"
# FOOTSTOOL = "footstool"
# FORK = "fork"
# FORKLIFT = "forklift"
# FREIGHT_CAR = "freight_car"
# FRENCH_TOAST = "french_toast"
# FRESHENER = "freshener"
# FRISBEE = "frisbee"
# FROG = "frog"
# FRUIT_JUICE = "fruit_juice"
# FRYING_PAN = "frying_pan"
# FUDGE = "fudge"
# FUNNEL = "funnel"
# FUTON = "futon"
# GAG = "gag"
# GAMEBOARD = "gameboard"
# GARBAGE = "garbage"
# GARBAGE_TRUCK = "garbage_truck"
# GARDEN_HOSE = "garden_hose"
# GARGLE = "gargle"
# GARGOYLE = "gargoyle"
# GARLIC = "garlic"
# GASMASK = "gasmask"
# GAZELLE = "gazelle"
# GELATIN = "gelatin"
# GEMSTONE = "gemstone"
# GENERATOR = "generator"
# GIANT_PANDA = "giant_panda"
# GIFT_WRAP = "gift_wrap"
# GINGER = "ginger"
# GIRAFFE = "giraffe"
# GLASS_DRINK_CONTAINER = "glass_drink_container"
# GLOBE = "globe"
# GLOVE = "glove"
# GOAT = "goat"
# GOGGLES = "goggles"
# GOLDFISH = "goldfish"
# GOLF_CLUB = "golf_club"
# GOLFCART = "golfcart"
# GOOSE = "goose"
# GORILLA = "gorilla"
# GOURD = "gourd"
# GRAPE = "grape"
# GRATER = "grater"
# GRAVY_BOAT = "gravy_boat"
# GREEN_BEAN = "green_bean"
# GREEN_ONION = "green_onion"
# GRIDDLE = "griddle"
# GRILL = "grill"
# GRITS = "grits"
# GRIZZLY = "grizzly"
# GROCERY_BAG = "grocery_bag"
# GUITAR = "guitar"
# GUN = "gun"
# HAIR_CURLER = "hair_curler"
# HAIR_DRYER = "hair_dryer"
# HAIRBRUSH = "hairbrush"
# HAIRNET = "hairnet"
# HAIRPIN = "hairpin"
# HALTER_TOP = "halter_top"
# HAM = "ham"
# HAMBURGER = "hamburger"
# HAMMER = "hammer"
# HAMMOCK = "hammock"
# HAMPER = "hamper"
# HAMSTER = "hamster"
# HAND_GLASS = "hand_glass"
# HAND_TOWEL = "hand_towel"
# HANDBAG = "handbag"
# HANDCART = "handcart"
# HANDCUFF = "handcuff"
# HANDKERCHIEF = "handkerchief"
# HANDLE = "handle"
# HANDSAW = "handsaw"
# HARDBACK_BOOK = "hardback_book"
# HARMONIUM = "harmonium"
# HAT = "hat"
# HATBOX = "hatbox"
# HEADBAND = "headband"
# HEADLIGHT = "headlight"
# HEADSCARF = "headscarf"
# HEADSET = "headset"
# HEART = "heart"
# HEATER = "heater"
# HELICOPTER = "helicopter"
# HELMET = "helmet"
# HIGHCHAIR = "highchair"
# HINGE = "hinge"
# HOG = "hog"
# HONEY = "honey"
# HOOK = "hook"
# HOOKAH = "hookah"
# HORSE = "horse"
# HORSE_BUGGY = "horse_buggy"
# HOSE = "hose"
# HOT_SAUCE = "hot_sauce"
# HOTPLATE = "hotplate"
# HOURGLASS = "hourglass"
# HOUSEBOAT = "houseboat"
# HUMMUS = "hummus"
# ICE_MAKER = "ice_maker"
# ICE_PACK = "ice_pack"
# ICE_SKATE = "ice_skate"
# ICECREAM = "icecream"
# IDENTITY_CARD = "identity_card"
# IGNITER = "igniter"
# INHALER = "inhaler"
# INKPAD = "inkpad"
# IPOD = "ipod"
# IRON_FOR_CLOTHING = "iron_for_clothing"
# IRONING_BOARD = "ironing_board"
# JACKET = "jacket"
# JAM = "jam"
# JAR = "jar"
# JEAN = "jean"
# JEEP = "jeep"
# JELLY_BEAN = "jelly_bean"
# JERSEY = "jersey"
# JET_PLANE = "jet_plane"
# JEWEL = "jewel"
# JEWELRY = "jewelry"
# JOYSTICK = "joystick"
# KEG = "keg"
# KENNEL = "kennel"
# KETTLE = "kettle"
# KEY = "key"
# KEYCARD = "keycard"
# KILT = "kilt"
# KIMONO = "kimono"
# KITCHEN_SINK = "kitchen_sink"
# KITCHEN_TABLE = "kitchen_table"
# KITE = "kite"
# KITTEN = "kitten"
# KIWI_FRUIT = "kiwi_fruit"
# KNIFE = "knife"
# KNITTING_NEEDLE = "knitting_needle"
# KNOB = "knob"
# KNOCKER_ON_A_DOOR = "knocker_on_a_door"
# LAB_COAT = "lab_coat"
# LADDER = "ladder"
# LADLE = "ladle"
# LAMB_ANIMAL = "lamb_animal"
# LAMB_CHOP = "lamb_chop"
# LAMP = "lamp"
# LAMPPOST = "lamppost"
# LAMPSHADE = "lampshade"
# LANTERN = "lantern"
# LANYARD = "lanyard"
# LAPTOP_COMPUTER = "laptop_computer"
# LASAGNA = "lasagna"
# LATCH = "latch"
# LAWN_MOWER = "lawn_mower"
# LEATHER = "leather"
# LEGGING_CLOTHING = "legging_clothing"
# LEGO = "lego"
# LEGUME = "legume"
# LEMON = "lemon"
# LEMONADE = "lemonade"
# LETTUCE = "lettuce"
# LIFE_BUOY = "life_buoy"
# LIGHTBULB = "lightbulb"
# LIGHTNING_ROD = "lightning_rod"
# LIME = "lime"
# LIP_BALM = "lip_balm"
# LIQUOR = "liquor"
# LOCKER = "locker"
# LOG = "log"
# LOLLIPOP = "lollipop"
# LOVESEAT = "loveseat"
# MACHINE_GUN = "machine_gun"
# MAGNET = "magnet"
# MAILBOX_AT_HOME = "mailbox_at_home"
# MALLARD = "mallard"
# MALLET = "mallet"
# MANDARIN_ORANGE = "mandarin_orange"
# MANGER = "manger"
# MANHOLE = "manhole"
# MAP = "map"
# MARKER = "marker"
# MARTINI = "martini"
# MASCOT = "mascot"
# MASHED_POTATO = "mashed_potato"
# MASHER = "masher"
# MASK = "mask"
# MAST = "mast"
# MAT_GYM_EQUIPMENT = "mat_gym_equipment"
# MATCHBOX = "matchbox"
# MATTRESS = "mattress"
# MEASURING_CUP = "measuring_cup"
# MEATBALL = "meatball"
# MEDICINE = "medicine"
# MELON = "melon"
# MICROPHONE = "microphone"
# MICROSCOPE = "microscope"
# MICROWAVE_OVEN = "microwave_oven"
# MILESTONE = "milestone"
# MILK = "milk"
# MILK_CAN = "milk_can"
# MILKSHAKE = "milkshake"
# MINIVAN = "minivan"
# MINT_CANDY = "mint_candy"
# MIRROR = "mirror"
# MITTEN = "mitten"
# MIXER_KITCHEN_TOOL = "mixer_kitchen_tool"
# MONEY = "money"
# MONKEY = "monkey"
# MOP = "mop"
# MOTOR = "motor"
# MOTOR_SCOOTER = "motor_scooter"
# MOTOR_VEHICLE = "motor_vehicle"
# MOTORCYCLE = "motorcycle"
# MOUSE_COMPUTER_EQUIPMENT = "mouse_computer_equipment"
# MUFFIN = "muffin"
# MUG = "mug"
# MUSHROOM = "mushroom"
# MUSIC_STOOL = "music_stool"
# MUSICAL_INSTRUMENT = "musical_instrument"
# NAILFILE = "nailfile"
# NAPKIN = "napkin"
# NECKERCHIEF = "neckerchief"
# NECKLACE = "necklace"
# NECKTIE = "necktie"
# NEEDLE = "needle"
# NEST = "nest"
# NEWSSTAND = "newsstand"
# NIGHTSHIRT = "nightshirt"
# NOSEBAG_FOR_ANIMALS = "nosebag_for_animals"
# NOSEBAND_FOR_ANIMALS = "noseband_for_animals"
# NOTEBOOK = "notebook"
# NOTEPAD = "notepad"
# NUT = "nut"
# NUTCRACKER = "nutcracker"
# OAR = "oar"
# OCTOPUS_FOOD = "octopus_food"
# OIL_LAMP = "oil_lamp"
# OLIVE_OIL = "olive_oil"
# OMELET = "omelet"
# ONION = "onion"
# ORANGE_FRUIT = "orange_fruit"
# ORANGE_JUICE = "orange_juice"
# OTTOMAN = "ottoman"
# OVEN = "oven"
# OVERALLS_CLOTHING = "overalls_clothing"
# OWL = "owl"
# PACIFIER = "pacifier"
# PACKET = "packet"
# PAD = "pad"
# PADLOCK = "padlock"
# PAINTBRUSH = "paintbrush"
# PAINTING = "painting"
# PAJAMAS = "pajamas"
# PALETTE = "palette"
# PAN_FOR_COOKING = "pan_for_cooking"
# PAN_METAL_CONTAINER = "pan_metal_container"
# PANCAKE = "pancake"
# PAPAYA = "papaya"
# PAPER_PLATE = "paper_plate"
# PAPER_TOWEL = "paper_towel"
# PAPERBACK_BOOK = "paperback_book"
# PAPERWEIGHT = "paperweight"
# PARAKEET = "parakeet"
# PARASOL = "parasol"
# PARCHMENT = "parchment"
# PARKA = "parka"
# PARKING_METER = "parking_meter"
# PARROT = "parrot"
# PASSENGER_CAR_PART_OF_A_TRAIN = "passenger_car_part_of_a_train"
# PASSENGER_SHIP = "passenger_ship"
# PASSPORT = "passport"
# PASTRY = "pastry"
# PATTY_FOOD = "patty_food"
# PEACH = "peach"
# PEANUT_BUTTER = "peanut_butter"
# PEAR = "pear"
# PEELER_TOOL_FOR_FRUIT_AND_VEGETABLES = "peeler_tool_for_fruit_and_vegetables"
# PEGBOARD = "pegboard"
# PEN = "pen"
# PENCIL = "pencil"
# PENCIL_BOX = "pencil_box"
# PENCIL_SHARPENER = "pencil_sharpener"
# PENDULUM = "pendulum"
# PENGUIN = "penguin"
# PENNY_COIN = "penny_coin"
# PEPPER = "pepper"
# PEPPER_MILL = "pepper_mill"
# PERFUME = "perfume"
# PERSIMMON = "persimmon"
# PET = "pet"
# PHONEBOOK = "phonebook"
# PHONOGRAPH_RECORD = "phonograph_record"
# PIANO = "piano"
# PICKLE = "pickle"
# PICKUP_TRUCK = "pickup_truck"
# PIE = "pie"
# PIGGY_BANK = "piggy_bank"
# PILLOW = "pillow"
# PIN_NON_JEWELRY = "pin_non_jewelry"
# PINEAPPLE = "pineapple"
# PINECONE = "pinecone"
# PING_PONG_BALL = "ping_pong_ball"
# PINWHEEL = "pinwheel"
# PIPE = "pipe"
# PIPE_BOWL = "pipe_bowl"
# PISTOL = "pistol"
# PITA_BREAD = "pita_bread"
# PITCHER_VESSEL_FOR_LIQUID = "pitcher_vessel_for_liquid"
# PITCHFORK = "pitchfork"
# PIZZA = "pizza"
# PLACE_MAT = "place_mat"
# PLASTIC_BAG = "plastic_bag"
# PLATE = "plate"
# PLATTER = "platter"
# PLAYPEN = "playpen"
# PLIERS = "pliers"
# PLOW_FARM_EQUIPMENT = "plow_farm_equipment"
# PLUME = "plume"
# POCKET_WATCH = "pocket_watch"
# POCKETKNIFE = "pocketknife"
# POKER_CHIP = "poker_chip"
# POLAR_BEAR = "polar_bear"
# POLE = "pole"
# POLICE_CRUISER = "police_cruiser"
# PONCHO = "poncho"
# PONY = "pony"
# POOL_TABLE = "pool_table"
# POP_SODA = "pop_soda"
# POPSICLE = "popsicle"
# POSTBOX_PUBLIC = "postbox_public"
# POSTCARD = "postcard"
# POSTER = "poster"
# POT = "pot"
# POTATO = "potato"
# POTHOLDER = "potholder"
# POTTERY = "pottery"
# POUCH = "pouch"
# POWER_SHOVEL = "power_shovel"
# PRAWN = "prawn"
# PRETZEL = "pretzel"
# PRINTER = "printer"
# PROJECTILE_WEAPON = "projectile_weapon"
# PROJECTOR = "projector"
# PROPELLER = "propeller"
# PRUNE = "prune"
# PUDDING = "pudding"
# PUG_DOG = "pug_dog"
# PUMPKIN = "pumpkin"
# PUNCHER = "puncher"
# PUPPET = "puppet"
# PUPPY = "puppy"
# QUICHE = "quiche"
# QUILT = "quilt"
# RABBIT = "rabbit"
# RACE_CAR = "race_car"
# RACKET = "racket"
# RADIATOR = "radiator"
# RADIO_RECEIVER = "radio_receiver"
# RADISH = "radish"
# RAFT = "raft"
# RAG_DOLL = "rag_doll"
# RAILCAR_PART_OF_A_TRAIN = "railcar_part_of_a_train"
# RAINCOAT = "raincoat"
# RAM_ANIMAL = "ram_animal"
# RASPBERRY = "raspberry"
# RAZORBLADE = "razorblade"
# REAMER_JUICER = "reamer_juicer"
# REARVIEW_MIRROR = "rearview_mirror"
# RECEIPT = "receipt"
# RECLINER = "recliner"
# RECORD_PLAYER = "record_player"
# REFLECTOR = "reflector"
# REFRIGERATOR = "refrigerator"
# REMOTE_CONTROL = "remote_control"
# RIB_FOOD = "rib_food"
# RIFLE = "rifle"
# RING = "ring"
# RIVER_BOAT = "river_boat"
# ROAD_MAP = "road_map"
# ROBE = "robe"
# ROCKING_CHAIR = "rocking_chair"
# RODENT = "rodent"
# ROLLER_SKATE = "roller_skate"
# ROLLERBLADE = "rollerblade"
# ROLLING_PIN = "rolling_pin"
# ROOT_BEER = "root_beer"
# ROUTER_COMPUTER_EQUIPMENT = "router_computer_equipment"
# RUBBER_BAND = "rubber_band"
# RUNNER_CARPET = "runner_carpet"
# SADDLE_BLANKET = "saddle_blanket"
# SADDLEBAG = "saddlebag"
# SAFETY_PIN = "safety_pin"
# SAIL = "sail"
# SALAD = "salad"
# SALAD_PLATE = "salad_plate"
# SALMON_FISH = "salmon_fish"
# SALMON_FOOD = "salmon_food"
# SALSA = "salsa"
# SALTSHAKER = "saltshaker"
# SANDAL_TYPE_OF_SHOE = "sandal_type_of_shoe"
# SANDWICH = "sandwich"
# SATCHEL = "satchel"
# SAUCEPAN = "saucepan"
# SAUCER = "saucer"
# SAUSAGE = "sausage"
# SAXOPHONE = "saxophone"
# SCALE_MEASURING_INSTRUMENT = "scale_measuring_instrument"
# SCARECROW = "scarecrow"
# SCARF = "scarf"
# SCHOOL_BUS = "school_bus"
# SCISSORS = "scissors"
# SCOREBOARD = "scoreboard"
# SCRAPER = "scraper"
# SCREWDRIVER = "screwdriver"
# SCRUBBING_BRUSH = "scrubbing_brush"
# SCULPTURE = "sculpture"
# SEABIRD = "seabird"
# SEAHORSE = "seahorse"
# SEASHELL = "seashell"
# SEWING_MACHINE = "sewing_machine"
# SHAKER = "shaker"
# SHAMPOO = "shampoo"
# SHARPENER = "sharpener"
# SHARPIE = "sharpie"
# SHAVER_ELECTRIC = "shaver_electric"
# SHAVING_CREAM = "shaving_cream"
# SHAWL = "shawl"
# SHEARS = "shears"
# SHEEP = "sheep"
# SHEPHERD_DOG = "shepherd_dog"
# SHERBERT = "sherbert"
# SHIELD = "shield"
# SHIRT = "shirt"
# SHOE = "shoe"
# SHOPPING_BAG = "shopping_bag"
# SHOPPING_CART = "shopping_cart"
# SHOT_GLASS = "shot_glass"
# SHOULDER_BAG = "shoulder_bag"
# SHOVEL = "shovel"
# SHOWER_CAP = "shower_cap"
# SHOWER_CURTAIN = "shower_curtain"
# SHOWER_HEAD = "shower_head"
# SHREDDER_FOR_PAPER = "shredder_for_paper"
# SILO = "silo"
# SINK = "sink"
# SKATEBOARD = "skateboard"
# SKEWER = "skewer"
# SKI_BOOT = "ski_boot"
# SKI_PARKA = "ski_parka"
# SKULLCAP = "skullcap"
# SLEEPING_BAG = "sleeping_bag"
# SLIDE = "slide"
# SLING_BANDAGE = "sling_bandage"
# SLIPPER_FOOTWEAR = "slipper_footwear"
# SMOOTHIE = "smoothie"
# SNOWMAN = "snowman"
# SOAP = "soap"
# SOCCER_BALL = "soccer_ball"
# SOCK = "sock"
# SOFA = "sofa"
# SOFA_BED = "sofa_bed"
# SOFTBALL = "softball"
# SOLAR_ARRAY = "solar_array"
# SOMBRERO = "sombrero"
# SOUP = "soup"
# SOUP_BOWL = "soup_bowl"
# SOUPSPOON = "soupspoon"
# SOUR_CREAM = "sour_cream"
# SOYA_MILK = "soya_milk"
# SPACE_SHUTTLE = "space_shuttle"
# SPARKLER_FIREWORKS = "sparkler_fireworks"
# SPATULA = "spatula"
# SPEAKER_STERO_EQUIPMENT = "speaker_stero_equipment"
# SPEAR = "spear"
# SPECTACLES = "spectacles"
# SPICE_RACK = "spice_rack"
# SPONGE = "sponge"
# SPOON = "spoon"
# SPORTSWEAR = "sportswear"
# SPOTLIGHT = "spotlight"
# SQUID_FOOD = "squid_food"
# SQUIRREL = "squirrel"
# STAPLER_STAPLING_MACHINE = "stapler_stapling_machine"
# STARFISH = "starfish"
# STATUE_SCULPTURE = "statue_sculpture"
# STEAK_FOOD = "steak_food"
# STEAK_KNIFE = "steak_knife"
# STEERING_WHEEL = "steering_wheel"
# STEP_STOOL = "step_stool"
# STEPLADDER = "stepladder"
# STEREO_SOUND_SYSTEM = "stereo_sound_system"
# STEW = "stew"
# STIRRER = "stirrer"
# STIRRUP = "stirrup"
# STOOL = "stool"
# STOP_SIGN = "stop_sign"
# STOVE = "stove"
# STRAINER = "strainer"
# STRAW_FOR_DRINKING = "straw_for_drinking"
# STRAWBERRY = "strawberry"
# STREET_SIGN = "street_sign"
# STREETLIGHT = "streetlight"
# STRING_CHEESE = "string_cheese"
# STYLUS = "stylus"
# SUBWOOFER = "subwoofer"
# SUGAR_BOWL = "sugar_bowl"
# SUGARCANE_PLANT = "sugarcane_plant"
# SUIT_CLOTHING = "suit_clothing"
# SUITCASE = "suitcase"
# SUNFLOWER = "sunflower"
# SUNGLASSES = "sunglasses"
# SUNHAT = "sunhat"
# SUSHI = "sushi"
# SUSPENDERS = "suspenders"
# SWEATBAND = "sweatband"
# SWEATER = "sweater"
# SWEET_POTATO = "sweet_potato"
# SWIMSUIT = "swimsuit"
# SWORD = "sword"
# SYRINGE = "syringe"
# TABASCO_SAUCE = "tabasco_sauce"
# TABLE = "table"
# TABLE_LAMP = "table_lamp"
# TABLE_TENNIS_TABLE = "table_tennis_table"
# TACHOMETER = "tachometer"
# TACO = "taco"
# TAG = "tag"
# TAILLIGHT = "taillight"
# TAMBOURINE = "tambourine"
# TANK_STORAGE_VESSEL = "tank_storage_vessel"
# TAPE_MEASURE = "tape_measure"
# TAPE_STICKY_CLOTH_OR_PAPER = "tape_sticky_cloth_or_paper"
# TARP = "tarp"
# TARTAN = "tartan"
# TASSEL = "tassel"
# TEA_BAG = "tea_bag"
# TEACUP = "teacup"
# TEAKETTLE = "teakettle"
# TEAPOT = "teapot"
# TEDDY_BEAR = "teddy_bear"
# TELEPHONE = "telephone"
# TELEPHONE_BOOTH = "telephone_booth"
# TELEPHONE_POLE = "telephone_pole"
# TELEPHOTO_LENS = "telephoto_lens"
# TELEVISION_CAMERA = "television_camera"
# TELEVISION_SET = "television_set"
# TENNIS_BALL = "tennis_ball"
# TENNIS_RACKET = "tennis_racket"
# TEQUILA = "tequila"
# THERMOMETER = "thermometer"
# THERMOS_BOTTLE = "thermos_bottle"
# THERMOSTAT = "thermostat"
# THIMBLE = "thimble"
# THREAD = "thread"
# THUMBTACK = "thumbtack"
# TIARA = "tiara"
# TIGHTS_CLOTHING = "tights_clothing"
# TIMER = "timer"
# TINFOIL = "tinfoil"
# TINSEL = "tinsel"
# TOAST_FOOD = "toast_food"
# TOASTER = "toaster"
# TOASTER_OVEN = "toaster_oven"
# TOBACCO_PIPE = "tobacco_pipe"
# TOILET = "toilet"
# TOILET_TISSUE = "toilet_tissue"
# TOMATO = "tomato"
# TONGS = "tongs"
# TOOLBOX = "toolbox"
# TOOTHBRUSH = "toothbrush"
# TOOTHPASTE = "toothpaste"
# TOOTHPICK = "toothpick"
# TORTILLA = "tortilla"
# TOTE_BAG = "tote_bag"
# TOW_TRUCK = "tow_truck"
# TOWEL = "towel"
# TOWEL_RACK = "towel_rack"
# TOY = "toy"
# TRACTOR_FARM_EQUIPMENT = "tractor_farm_equipment"
# TRAFFIC_LIGHT = "traffic_light"
# TRAILER_TRUCK = "trailer_truck"
# TRAIN_RAILROAD_VEHICLE = "train_railroad_vehicle"
# TRASH_CAN = "trash_can"
# TRAY = "tray"
# TRENCH_COAT = "trench_coat"
# TRIANGLE_MUSICAL_INSTRUMENT = "triangle_musical_instrument"
# TRICYCLE = "tricycle"
# TRIPOD = "tripod"
# TROPHY_CUP = "trophy_cup"
# TRUCK = "truck"
# TRUFFLE_CHOCOLATE = "truffle_chocolate"
# TRUNK = "trunk"
# TURBAN = "turban"
# TURKEY_FOOD = "turkey_food"
# TURNIP = "turnip"
# TURTLE = "turtle"
# TURTLENECK_CLOTHING = "turtleneck_clothing"
# TUX = "tux"
# TYPEWRITER = "typewriter"
# UMBRELLA = "umbrella"
# UNDERDRAWERS = "underdrawers"
# UNDERWEAR = "underwear"
# URINAL = "urinal"
# URN = "urn"
# VACUUM_CLEANER = "vacuum_cleaner"
# VASE = "vase"
# VAT = "vat"
# VEIL = "veil"
# VENDING_MACHINE = "vending_machine"
# VEST = "vest"
# VIDEOTAPE = "videotape"
# VINEGAR = "vinegar"
# VIOLIN = "violin"
# VISOR = "visor"
# VODKA = "vodka"
# VOLLEYBALL = "volleyball"
# WAFFLE = "waffle"
# WAFFLE_IRON = "waffle_iron"
# WAGON = "wagon"
# WALKING_CANE = "walking_cane"
# WALKING_STICK = "walking_stick"
# WALL_CLOCK = "wall_clock"
# WALL_SOCKET = "wall_socket"
# WALLET = "wallet"
# WARDROBE = "wardrobe"
# WASHBASIN = "washbasin"
# WATCH = "watch"
# WATER_BOTTLE = "water_bottle"
# WATER_COOLER = "water_cooler"
# WATER_FAUCET = "water_faucet"
# WATER_GUN = "water_gun"
# WATER_HEATER = "water_heater"
# WATER_JUG = "water_jug"
# WATER_SCOOTER = "water_scooter"
# WATER_TOWER = "water_tower"
# WATERING_CAN = "watering_can"
# WATERMELON = "watermelon"
# WEATHERVANE = "weathervane"
# WEBCAM = "webcam"
# WEDDING_RING = "wedding_ring"
# WHEEL = "wheel"
# WHEELCHAIR = "wheelchair"
# WHIPPED_CREAM = "whipped_cream"
# WHISTLE = "whistle"
# WIG = "wig"
# WIND_CHIME = "wind_chime"
# WINDMILL = "windmill"
# WINDOW_BOX_FOR_PLANTS = "window_box_for_plants"
# WINDSOCK = "windsock"
# WINE_BOTTLE = "wine_bottle"
# WINE_BUCKET = "wine_bucket"
# WINEGLASS = "wineglass"
# WOK = "wok"
# WOODEN_LEG = "wooden_leg"
# WOODEN_SPOON = "wooden_spoon"
# WREATH = "wreath"
# WRENCH = "wrench"
# WRISTBAND = "wristband"
# WRISTLET = "wristlet"
# YACHT = "yacht"
# YOGURT = "yogurt"
# ZEBRA = "zebra"
# ZUCCHINI = "zucchini"