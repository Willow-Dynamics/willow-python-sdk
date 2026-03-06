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
    Executes entirely in RAM (Zero physical disk footprint).
    """
    if isinstance(data, BytesIO):
        buffer = data.read()
    else:
        buffer = data

    if len(buffer) < 24:
        raise ValueError("Invalid Willow Binary: Header too short.")

    # Strict Little-Endian 24-Byte Header parsing
    header = struct.unpack('<IIffff', buffer[:24])
    version, bitmask, scale, overlap, dtw_sens, tempo = header

    if version != 40:
        raise ValueError(f"Unsupported model version: {version}. Expected 40 (V4.0).")

    config = WillowConfig(
        version=version,
        zone_bitmask=bitmask,
        overlap_tolerance=overlap,
        dtw_sensitivity=dtw_sens,
        tempo_variance=tempo
    )

    # Resolve dimension size from bitmask to reshape flat array
    n_joints = sum(len(indices) for bit, indices in ZONES.values() if bitmask & bit)
    dim = int(n_joints * (n_joints - 1) / 2)

    # Fast De-Quantization
    raw_int8 = np.frombuffer(buffer[24:], dtype=np.int8)
    signature = (raw_int8.astype(np.float32) / 127.0) * scale
    
    # Reshape to (Frames, Features)
    signature = signature.reshape(-1, dim)

    return WillowModel(config=config, signature=signature)

def parse_json_model(data: Union[str, dict]) -> WillowModel:
    """
    Fallback parser for standard web JSON signatures.
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
    
    signature = np.array(payload.get("signature",[]), dtype=np.float32)
    return WillowModel(config=config, signature=signature)

def load_local_model(filepath: str) -> WillowModel:
    """
    Loads a Willow model directly from a local file.
    Use this if you downloaded the model manually via the Willow Web Interface.
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