# CamVid semantic segmentation
# 32 original classes grouped into 11 classes

CLASS_NAMES = [
    "Sky",
    "Building",
    "Pole",
    "Road",
    "Pavement",
    "Tree",
    "SignSymbol",
    "Fence",
    "Car",
    "Pedestrian",
    "Bicyclist",
]

NUM_CLASSES = len(CLASS_NAMES)

# RGB color -> class ID
CLASS_COLORS = {

    # Sky
    (128, 128, 128): 0,

    # Building
    (0, 128, 64): 1,       # Bridge
    (128, 0, 0): 1,        # Building
    (64, 192, 0): 1,       # Wall
    (64, 0, 64): 1,        # Tunnel
    (192, 0, 128): 1,      # Archway

    # Pole
    (192, 192, 128): 2,    # Column_Pole
    (0, 0, 64): 2,         # TrafficCone

    # Road
    (128, 64, 128): 3,     # Road
    (128, 0, 192): 3,      # LaneMkgsDriv
    (192, 0, 64): 3,       # LaneMkgsNonDriv

    # Pavement
    (0, 0, 192): 4,       # Sidewalk
    (64, 192, 128): 4,     # ParkingBlock
    (128, 128, 192): 4,    # RoadShoulder

    # Tree
    (128, 128, 0): 5,     # Tree
    (192, 192, 0): 5,     # VegetationMisc

    # SignSymbol
    (192, 128, 128): 6,   # SignSymbol
    (128, 128, 64): 6,    # Misc_Text
    (0, 64, 64): 6,       # TrafficLight

    # Fence
    (64, 64, 128): 7,     # Fence

    # Car
    (64, 0, 128): 8,      # Car
    (64, 128, 192): 8,    # SUVPickupTruck
    (192, 128, 192): 8,   # Truck_Bus
    (192, 64, 128): 8,    # Train
    (128, 64, 64): 8,     # OtherMoving

    # Pedestrian
    (64, 64, 0): 9,       # Pedestrian
    (192, 128, 64): 9,    # Child
    (64, 0, 192): 9,      # CartLuggagePram
    (64, 128, 64): 9,     # Animal

    # Bicyclist
    (0, 128, 192): 10,    # Bicyclist
    (192, 0, 192): 10,    # MotorcycleScooter
}

# Pixels belonging to Other/Void are ignored.
IGNORE_INDEX = 255

IGNORE_COLOR = (0, 0, 0)