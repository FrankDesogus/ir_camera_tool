from __future__ import annotations

import numpy as np


def percentile_window(frame: np.ndarray, p_min: float, p_max: float) -> tuple[float, float]:
    if p_min >= p_max:
        p_max = min(p_min + 0.1, 100.0)
    low = float(np.percentile(frame, p_min))
    high = float(np.percentile(frame, p_max))
    if high <= low:
        high = low + 1.0
    return low, high


def clipping(frame: np.ndarray, low: float, high: float) -> np.ndarray:
    frame_f = frame.astype(np.float32)
    clipped = frame_f - low
    lim_sup = high - low
    return np.clip(clipped, 0, lim_sup)


def apply_display_transform(frame: np.ndarray, mode: int, gamma: float = 1.0) -> np.ndarray:
    """
    mode: 0=linear  1=sqrt  2=log  3=asinh
    Lavora sul frame clippato (non normalizzato in [0,1]).
    """
    x = frame.astype(np.float32)
    if mode == 0:
        y = x
    elif mode == 1:
        y = np.sqrt(x)
    elif mode == 2:
        y = np.log1p(x)
    elif mode == 3:
        y = np.arcsinh(x)
    else:
        y = x
    gamma = max(gamma, 1e-6)
    y = np.power(y, gamma)
    return y


def normalize(frame: np.ndarray) -> np.ndarray:
    frame_f = frame.astype(np.float32)
    max_val = frame_f.max()
    if max_val == 0:
        return frame_f
    return frame_f / max_val


def to_uint8(norm_frame: np.ndarray) -> np.ndarray:
    return (norm_frame * 255.0).round().clip(0, 255).astype(np.uint8)


def mode_from_value(value: int) -> int:
    """Mappa un valore intero (trackbar o indice UI) a una modalità 0-3."""
    if value <= 0:
        return 0
    if value == 1:
        return 1
    if value == 2:
        return 2
    return 3


def scale_mode_from_value(value: int) -> int:
    """0=AUTO  1=GLOBAL  2=MANUAL"""
    if value <= 0:
        return 0
    if value == 1:
        return 1
    return 2


def prepare_frame_for_display(
    raw_frame: np.ndarray,
    transform_mode: int,
    scale_mode: int,
    p_min: float,
    p_max: float,
    global_low: float,
    global_high: float,
    manual_low: float,
    manual_high: float,
    gamma: float,
) -> tuple[np.ndarray, float, float]:
    """
    Pipeline: clipping -> apply_display_transform -> normalize -> to_uint8

    Restituisce (display_uint8, low_usato, high_usato).
    """
    if scale_mode == 0:  # AUTO
        low, high = percentile_window(raw_frame, p_min, p_max)
    elif scale_mode == 1:  # GLOBAL
        low, high = global_low, global_high
    else:  # MANUAL
        low, high = manual_low, manual_high
        if high <= low:
            high = low + 1.0

    clipped = clipping(raw_frame, low, high)
    transformed = apply_display_transform(clipped, transform_mode, gamma=gamma)
    norm = normalize(transformed)
    display = to_uint8(norm)

    return display, low, high
