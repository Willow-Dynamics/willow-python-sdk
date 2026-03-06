import dataclasses
import numpy as np

# Willow Standard 75-Point Topology Map
# Maps logical zones to their respective 3D joint indices
ZONES = {
    "head":  (1,[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
    "torso": (2,[11, 12, 23, 24]),
    "arms":  (4,[13, 14, 15, 16]),
    "hands": (8,  list(range(33, 75))),
    "legs":  (16,[25, 26, 27, 28]),
    "feet":  (32,[29, 30, 31, 32])
}

@dataclasses.dataclass
class WillowConfig:
    """Configuration derived from the 24-Byte .int8 Model Header."""
    version: int
    zone_bitmask: int
    overlap_tolerance: float
    dtw_sensitivity: float
    tempo_variance: float

@dataclasses.dataclass
class WillowModel:
    """The runtime representation of an Action Model."""
    config: WillowConfig
    signature: np.ndarray  # The Float32 De-quantized RDM Matrix