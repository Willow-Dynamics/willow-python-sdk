import dataclasses
import numpy as np

# Willow Standard 75-Point Topology Map
# This adheres to the Extended MediaPipe Pose Graph.
# Engineers must ensure their input arrays match this index order exactly.
#
# Index Map:
# 0-10: Face (Nose, Eyes, Ears, Mouth)
# 11-12: Shoulders (Left, Right) - Critical for Torso Normalization
# 13-16: Arms (Elbows, Wrists)
# 17-22: Hands (Palm/Pinky - often interpolated)
# 23-24: Hips (Left, Right) - Critical for Torso Normalization
# 25-32: Legs (Knees, Ankles, Heels, Toes)
# 33-53: Left Hand (21 points: Wrist, Thumb->Pinky)
# 54-74: Right Hand (21 points: Wrist, Thumb->Pinky)
ZONES = {
    "head":  (1,  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
    "torso": (2,  [11, 12, 23, 24]),
    "arms":  (4,  [13, 14, 15, 16]),
    "hands": (8,  list(range(33, 75))),
    "legs":  (16, [25, 26, 27, 28]),
    "feet":  (32, [29, 30, 31, 32])
}

@dataclasses.dataclass
class WillowConfig:
    """
    Configuration metadata derived from the 24-Byte .int8 Model Header.
    Controls the sensitivity and behavior of the Continuous DTW engine.
    """
    version: int            # Model format version (e.g., 40 = v4.0)
    zone_bitmask: int       # Which body zones (ZONES) are active in this model
    overlap_tolerance: float # NMS Threshold: Max allowed temporal overlap (0.0-1.0)
    dtw_sensitivity: float  # Scaling factor for raw DTW costs
    tempo_variance: float   # Allowed speed variation (+/- %)

@dataclasses.dataclass
class WillowModel:
    """
    The runtime representation of an Action Model.
    Contains the configuration and the decompressed Feature Matrix.
    """
    config: WillowConfig
    signature: np.ndarray  # The Float32 De-quantized RDM Matrix (Frames x Features)