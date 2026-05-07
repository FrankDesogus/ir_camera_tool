from __future__ import annotations

import time
import cv2
import numpy as np

from nvf_reader import import_nvf


WINDOW_NAME = "NVF Viewer"


def nothing(_: int) -> None:
    pass


def percentile_window(frame: np.ndarray, p_min: float, p_max: float) -> tuple[float, float]:
    """
    Restituisce low/high dai percentili del frame.
    """
    if p_min >= p_max:
        p_max = min(p_min + 0.1, 100.0)

    low = float(np.percentile(frame, p_min))
    high = float(np.percentile(frame, p_max))

    if high <= low:
        high = low + 1.0

    return low, high


def normalize_with_window(frame: np.ndarray, low: float, high: float) -> np.ndarray:
    """
    Porta il frame in [0,1] usando una finestra low/high fissata.
    Qui NON si fa una seconda normalizzazione dopo le trasformazioni.
    """
    frame_f = frame.astype(np.float32)
    norm = (frame_f - low) / (high - low)
    return np.clip(norm, 0.0, 1.0)


def apply_display_transform(norm_frame: np.ndarray, mode: int, gamma: float = 1.0) -> np.ndarray:
    """
    norm_frame deve essere già in [0,1].

    mode:
    0 = linear
    1 = sqrt
    2 = log
    """
    x = np.clip(norm_frame.astype(np.float32), 0.0, 1.0)

    if mode == 0:
        y = x
    elif mode == 1:
        y = np.sqrt(x)
    elif mode == 2:
        alpha = 9.0
        y = np.log1p(alpha * x) / np.log1p(alpha)
    else:
        y = x

    # Gamma opzionale come rifinitura controllata
    gamma = max(gamma, 1e-6)
    y = np.power(y, gamma)

    return np.clip(y, 0.0, 1.0)


def to_uint8(norm_frame: np.ndarray) -> np.ndarray:
    """
    Converte un frame in [0,1] a uint8 solo per il display.
    """
    return (norm_frame * 255.0).round().clip(0, 255).astype(np.uint8)


def mode_from_trackbar(value: int) -> int:
    if value <= 0:
        return 0
    if value == 1:
        return 1
    return 2


def scale_mode_from_trackbar(value: int) -> int:
    """
    0 = AUTO   (percentili sul frame corrente)
    1 = GLOBAL (percentili sul cubo intero)
    2 = MANUAL (low/high scelti dall'utente)
    """
    if value <= 0:
        return 0
    if value == 1:
        return 1
    else:
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
    Pipeline corretta per display scientificamente interpretabile:

    raw -> scelta finestra low/high -> normalizzazione [0,1]
        -> transform (linear/sqrt/log) -> gamma opzionale
        -> uint8

    Restituisce:
    - frame uint8 per display
    - low usato
    - high usato
    """
    if scale_mode == 0:  # AUTO
        low, high = percentile_window(raw_frame, p_min, p_max)
    elif scale_mode == 1:  # GLOBAL
        low, high = global_low, global_high
    else:  # MANUAL
        low, high = manual_low, manual_high
        if high <= low:
            high = low + 1.0

    norm = normalize_with_window(raw_frame, low, high)
    transformed = apply_display_transform(norm, transform_mode, gamma=gamma)
    display = to_uint8(transformed)

    return display, low, high


def main() -> None:
    nvf = import_nvf()
    data_cube = nvf.data_cube

    # Adatta qui se il cubo reale è (H, W, N) invece di (N, H, W)
    # Nel dubbio controlla bene questa parte.
    n_frames = data_cube.shape[0]

    # Statistiche globali per modalità GLOBAL
    global_min_raw = float(np.min(data_cube))
    global_max_raw = float(np.max(data_cube))

    # Percentili globali più robusti del min/max assoluto
    global_pmin_default = 1.0
    global_pmax_default = 99.0
    global_low = float(np.percentile(data_cube, global_pmin_default))
    global_high = float(np.percentile(data_cube, global_pmax_default))
    if global_high <= global_low:
        global_high = global_low + 1.0

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    cv2.createTrackbar("Frame", WINDOW_NAME, 0, max(n_frames - 1, 1), nothing)
    cv2.createTrackbar("Play", WINDOW_NAME, 0, 1, nothing)
    cv2.createTrackbar("FPS", WINDOW_NAME, 25, 100, nothing)

    cv2.createTrackbar("Mode 0L 1S 2G", WINDOW_NAME, 0, 2, nothing)
    cv2.createTrackbar("Scale 0A 1G 2M", WINDOW_NAME, 0, 2, nothing)

    cv2.createTrackbar("Min % x10", WINDOW_NAME, 10, 999, nothing)      # default 1.0%
    cv2.createTrackbar("Max % x10", WINDOW_NAME, 990, 1000, nothing)    # default 99.0%

    # Gamma x100: 100 = gamma 1.00
    cv2.createTrackbar("Gamma x100", WINDOW_NAME, 100, 300, nothing)

    # Manual low/high controllati via slider 0..1000 e mappati sul range raw globale
    cv2.createTrackbar("Manual low", WINDOW_NAME, 0, 1000, nothing)
    cv2.createTrackbar("Manual high", WINDOW_NAME, 1000, 1000, nothing)

    current_frame = 0
    last_time = time.time()

    while True:
        play = cv2.getTrackbarPos("Play", WINDOW_NAME)
        fps = max(cv2.getTrackbarPos("FPS", WINDOW_NAME), 1)

        if play:
            now = time.time()
            if now - last_time >= 1.0 / fps:
                current_frame = (current_frame + 1) % n_frames
                cv2.setTrackbarPos("Frame", WINDOW_NAME, current_frame)
                last_time = now
        else:
            current_frame = cv2.getTrackbarPos("Frame", WINDOW_NAME)

        transform_mode = mode_from_trackbar(cv2.getTrackbarPos("Mode 0L 1S 2G", WINDOW_NAME))
        scale_mode = scale_mode_from_trackbar(cv2.getTrackbarPos("Scale 0A 1G 2M", WINDOW_NAME))

        p_min = cv2.getTrackbarPos("Min % x10", WINDOW_NAME) / 10.0
        p_max = cv2.getTrackbarPos("Max % x10", WINDOW_NAME) / 10.0
        p_max = min(p_max, 100.0)

        gamma = cv2.getTrackbarPos("Gamma x100", WINDOW_NAME) / 100.0
        gamma = max(gamma, 0.01)

        # Mapping slider manuali su range raw globale
        low_slider = cv2.getTrackbarPos("Manual low", WINDOW_NAME) / 1000.0
        high_slider = cv2.getTrackbarPos("Manual high", WINDOW_NAME) / 1000.0

        manual_low = global_min_raw + low_slider * (global_max_raw - global_min_raw)
        manual_high = global_min_raw + high_slider * (global_max_raw - global_min_raw)

        raw_frame = data_cube[current_frame]
        # Se necessario:
        # raw_frame = raw_frame.T

        display_frame, used_low, used_high = prepare_frame_for_display(
            raw_frame=raw_frame,
            transform_mode=transform_mode,
            scale_mode=scale_mode,
            p_min=p_min,
            p_max=p_max,
            global_low=global_low,
            global_high=global_high,
            manual_low=manual_low,
            manual_high=manual_high,
            gamma=gamma,
        )

        mode_name = {0: "LINEAR", 1: "SQRT", 2: "LOG"}[transform_mode]
        scale_name = {0: "AUTO", 1: "GLOBAL", 2: "MANUAL"}[scale_mode]

        overlay = cv2.cvtColor(display_frame, cv2.COLOR_GRAY2BGR)

        text_1 = f"Frame {current_frame + 1}/{n_frames}"
        text_2 = f"Mode: {mode_name} | Scale: {scale_name} | Gamma: {gamma:.2f}"
        text_3 = f"Window: {used_low:.2f} -> {used_high:.2f}"

        cv2.putText(overlay, text_1, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(overlay, text_2, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(overlay, text_3, (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow(WINDOW_NAME, overlay)

        key = cv2.waitKey(1) & 0xFF

        if key == 27 or key == ord("q"):
            break
        elif key == ord(" "):
            play = 0 if play else 1
            cv2.setTrackbarPos("Play", WINDOW_NAME, play)
        elif key == ord("a"):
            current_frame = max(0, current_frame - 1)
            cv2.setTrackbarPos("Frame", WINDOW_NAME, current_frame)
        elif key == ord("d"):
            current_frame = min(n_frames - 1, current_frame + 1)
            cv2.setTrackbarPos("Frame", WINDOW_NAME, current_frame)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()