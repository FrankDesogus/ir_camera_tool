from __future__ import annotations

import time
import cv2
import numpy as np

from nvf_reader import import_nvf
from roi import ROIManager, calculate_roi_timeseries, plot_roi_timeseries


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


def _check_roi_range(rois: list, p_start: int, p_end: int) -> bool:
    """Valida le precondizioni per calcolare/plottare le timeseries.
    Stampa un messaggio e restituisce False se qualcosa non va."""
    if not rois:
        print("Nessuna ROI selezionata.")
        return False
    if p_end < p_start:
        print(f"Intervallo non valido: Plot start={p_start}, Plot end={p_end}.")
        return False
    return True


def main() -> None:
    nvf = import_nvf()
    data_cube = nvf.data_cube

    # Adatta qui se il cubo reale è (H, W, N) invece di (N, H, W)
    # Nel dubbio controlla bene questa parte.
    n_frames, frame_height, frame_width = data_cube.shape

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

    roi_manager = ROIManager(frame_width, frame_height)
    cv2.setMouseCallback(WINDOW_NAME, roi_manager.handle_mouse)

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

    # Intervallo frame per il calcolo/plot delle ROI (estremi inclusi)
    _tb_max = max(n_frames - 1, 1)
    cv2.createTrackbar("Plot start", WINDOW_NAME, 0,            _tb_max, nothing)
    cv2.createTrackbar("Plot end",   WINDOW_NAME, n_frames - 1, _tb_max, nothing)

    current_frame = 0
    last_time = time.time()
    _clear_confirm_at: float = 0.0

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

        # Clamp difensivo: evita IndexError se la trackbar supera l'indice valido
        # (può succedere con file a singolo frame dove _tb_max=1 ma n_frames=1).
        current_frame = min(current_frame, n_frames - 1)

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

        p_start = cv2.getTrackbarPos("Plot start", WINDOW_NAME)
        p_end   = cv2.getTrackbarPos("Plot end",   WINDOW_NAME)

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

        hud_info_1 = (
            f"Frame {current_frame + 1}/{n_frames}"
            f"  |  {mode_name}  |  {scale_name}  |  Gamma: {gamma:.2f}"
        )
        roi_shape = "SQ" if roi_manager.square_mode else "RECT"
        hud_info_2 = (
            f"Win: {used_low:.1f} -> {used_high:.1f}"
            f"  |  ROI: {len(roi_manager.rois)} [{roi_shape}]"
            f"  |  Plot: {p_start}..{p_end}"
        )
        hud_keys_1 = "SPACE=play/pause | A/D=prev/next | Z=undo ROI | C=clear ROI"
        hud_keys_2 = "drag/click=add ROI | R=rect | S=square | T=stats | G=plot | Q/ESC=quit"

        cv2.putText(overlay, hud_info_1, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6,  (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(overlay, hud_info_2, (10, 43), cv2.FONT_HERSHEY_SIMPLEX, 0.6,  (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(overlay, hud_keys_1, (10, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(overlay, hud_keys_2, (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1, cv2.LINE_AA)

        overlay = roi_manager.draw_on_frame(overlay)

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
        elif key == ord("z"):
            roi_manager.remove_last()
        elif key == ord("c"):
            if roi_manager.rois and time.time() - _clear_confirm_at < 2.0:
                roi_manager.clear()
                _clear_confirm_at = 0.0
                print("Tutte le ROI cancellate.")
            elif roi_manager.rois:
                _clear_confirm_at = time.time()
                print(f"Premi C ancora entro 2s per cancellare {len(roi_manager.rois)} ROI.")
        elif key == ord("r"):
            roi_manager.square_mode = False
        elif key == ord("s"):
            roi_manager.square_mode = True
        elif key == ord("t"):
            if _check_roi_range(roi_manager.rois, p_start, p_end):
                timeseries = calculate_roi_timeseries(
                    data_cube, roi_manager.rois, p_start, p_end + 1
                )
                for name, ts in timeseries.items():
                    first_vals = ", ".join(f"{v:.2f}" for v in ts[:5])
                    print(f"\n{name}  [frame {p_start}..{p_end}]")
                    print(f"  samples:      {len(ts)}")
                    print(f"  min:          {ts.min():.2f}")
                    print(f"  max:          {ts.max():.2f}")
                    print(f"  mean:         {ts.mean():.2f}")
                    print(f"  first values: [{first_vals}]")
        elif key == ord("g"):
            if _check_roi_range(roi_manager.rois, p_start, p_end):
                timeseries = calculate_roi_timeseries(
                    data_cube, roi_manager.rois, p_start, p_end + 1
                )
                colors = {roi.name: roi.color for roi in roi_manager.rois}
                plot_roi_timeseries(
                    timeseries,
                    start_frame=p_start,
                    colors=colors,
                    current_frame=current_frame,
                )

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
