import os
import json
import struct
import numpy as np
from io import BytesIO
from typing import Union
from .types import WillowConfig, WillowModel, ZONES

def parse_int8_model(data: Union[bytes, BytesIO]) -> WillowModel:
    """
    Parses the secure Willow V4.0 .int8 binary format.
    Designed to execute in ephemeral RAM for DRM compliance.
    
    Header Format (Little-Endian, 24 Bytes):
    [0-3]   uint32 : Version (40)
    [4-7]   uint32 : Zone Bitmask
    [8-11]  float32: De-quantization Scale
    [12-15] float32: Overlap Tolerance
    [16-19] float32: DTW Sensitivity
    [20-23] float32: Tempo Variance
    """
    if isinstance(data, BytesIO):
        buffer = data.read()
    else:
        buffer = data

    if len(buffer) < 24:
        raise ValueError("Invalid Willow Binary: Header is too short (Minimum 24 bytes required).")

    # Strict Little-Endian 24-Byte Header parsing
    header = struct.unpack('<IIffff', buffer[:24])
    version, bitmask, scale, overlap, dtw_sens, tempo = header

    if version != 40:
        raise ValueError(f"Unsupported model version: {version}. This SDK supports Willow V4.0 models (Version ID 40).")

    config = WillowConfig(
        version=version,
        zone_bitmask=bitmask,
        overlap_tolerance=overlap,
        dtw_sensitivity=dtw_sens,
        tempo_variance=tempo
    )

    # Dynamically resolve feature dimension size from the bitmask
    # Dimension = N * (N-1) / 2 where N is the count of active joints
    n_joints = sum(len(indices) for bit, indices in ZONES.values() if bitmask & bit)
    dim = int(n_joints * (n_joints - 1) / 2)

    # Fast De-Quantization: Convert int8 (-128..127) back to float32 spatial distances
    raw_int8 = np.frombuffer(buffer[24:], dtype=np.int8)
    signature = (raw_int8.astype(np.float32) / 127.0) * scale
    
    # Reshape flattened array to (Frames, Features)
    # If the file is truncated, this reshape will raise a ValueError, ensuring integrity
    try:
        signature = signature.reshape(-1, dim)
    except ValueError:
        raise ValueError(f"Model data corruption: Payload size does not match calculated feature dimension ({dim}).")

    return WillowModel(config=config, signature=signature)

def parse_json_model(data: Union[str, dict]) -> WillowModel:
    """
    Fallback parser for standard web JSON signatures.
    Useful for debugging or legacy integrations.
    """
    if isinstance(data, str):
        payload = json.loads(data)
    else:
        payload = data
        
    cfg = payload.get("calibration_config", {})
    
    config = WillowConfig(
        version=int(float(cfg.get("version", 4.0)) * 10),
        zone_bitmask=int(cfg.get("zone_bitmask", 2)),
        overlap_tolerance=float(cfg.get("overlap_tolerance", 0.25)),
        dtw_sensitivity=float(cfg.get("dtw_sensitivity", 3.0)),
        tempo_variance=float(cfg.get("tempo_variance", 0.20))
    )
    
    signature = np.array(payload.get("signature", []), dtype=np.float32)
    return WillowModel(config=config, signature=signature)

def load_local_model(filepath: str) -> WillowModel:
    """
    Loads a Willow model directly from a local file.
    Use this for offline edge devices or air-gapped environments.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model file not found: {filepath}")

    if filepath.endswith('.json'):
        with open(filepath, 'r') as f:
            payload = json.load(f)
            return parse_json_model(payload)
    else:
        # Assumes .int8 or .bin optimized format
        with open(filepath, 'rb') as f:
            return parse_int8_model(f.read())