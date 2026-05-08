from __future__ import annotations

import time
import cv2
import numpy as np

from nvf_reader import import_nvf
from roi import ROIManager, calculate_roi_timeseries, plot_roi_timeseries
from display_pipeline import prepare_frame_for_display, mode_from_value, scale_mode_from_value


WINDOW_NAME = "NVF Viewer"
CONTROLS_WINDOW = "Display Controls"


def nothing(_: int) -> None:
    pass


def _check_roi_range(rois: list, p_start: int, p_end: int) -> bool:
    if not rois:
        print("Nessuna ROI selezionata.")
        return False
    if p_end < p_start:
        print(f"Intervallo non valido: Plot start={p_start}, Plot end={p_end}.")
        return False
    return True


def print_help() -> None:
    print(
        "\n=== NVF Viewer — tasti ===\n"
        "  SPACE      play / pausa\n"
        "  A / D      frame precedente / successivo\n"
        "  R          modalità ROI rettangolare\n"
        "  S          modalità ROI quadrata\n"
        "  Z          annulla ultima ROI\n"
        "  C          cancella tutte le ROI (conferma premendo C di nuovo entro 2s)\n"
        "  T          stampa statistiche ROI sull'intervallo selezionato\n"
        "  G          mostra grafico ROI sull'intervallo selezionato\n"
        "  H          mostra di nuovo questo aiuto\n"
        "  Q / ESC    esci\n"
        "\nControlli display: finestra 'Display Controls'\n"
        "  Mode:  0=Linear  1=Sqrt  2=Log  3=Asinh\n"
        "  Scale: 0=Auto    1=Global  2=Manual\n"
        "=========================\n"
    )


def main() -> None:
    nvf = import_nvf()
    data_cube = nvf.data_cube

    n_frames, frame_height, frame_width = data_cube.shape

    global_min_raw = float(np.min(data_cube))
    global_max_raw = float(np.max(data_cube))

    global_low = float(np.percentile(data_cube, 1.0))
    global_high = float(np.percentile(data_cube, 99.0))
    if global_high <= global_low:
        global_high = global_low + 1.0

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.namedWindow(CONTROLS_WINDOW, cv2.WINDOW_NORMAL)

    roi_manager = ROIManager(frame_width, frame_height)
    cv2.setMouseCallback(WINDOW_NAME, roi_manager.handle_mouse)

    # --- Finestra principale: playback e plot range ---
    cv2.createTrackbar("Frame",      WINDOW_NAME, 0,            max(n_frames - 1, 1), nothing)
    cv2.createTrackbar("Play",       WINDOW_NAME, 0,            1,                    nothing)
    cv2.createTrackbar("FPS",        WINDOW_NAME, 25,           100,                  nothing)
    _tb_max = max(n_frames - 1, 1)
    cv2.createTrackbar("Plot start", WINDOW_NAME, 0,            _tb_max,              nothing)
    cv2.createTrackbar("Plot end",   WINDOW_NAME, n_frames - 1, _tb_max,              nothing)

    # --- Finestra display: parametri di rendering ---
    cv2.createTrackbar("Mode 0L 1S 2G 3A", CONTROLS_WINDOW, 0,    3,    nothing)
    cv2.createTrackbar("Scale 0A 1G 2M",   CONTROLS_WINDOW, 0,    2,    nothing)
    cv2.createTrackbar("Min % x10",        CONTROLS_WINDOW, 10,   999,  nothing)
    cv2.createTrackbar("Max % x10",        CONTROLS_WINDOW, 990,  1000, nothing)
    cv2.createTrackbar("Gamma x100",       CONTROLS_WINDOW, 100,  300,  nothing)
    cv2.createTrackbar("Manual low",       CONTROLS_WINDOW, 0,    1000, nothing)
    cv2.createTrackbar("Manual high",      CONTROLS_WINDOW, 1000, 1000, nothing)

    current_frame = 0
    last_time = time.time()
    _clear_confirm_at: float = 0.0

    print_help()

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

        current_frame = min(current_frame, n_frames - 1)

        transform_mode = mode_from_value(cv2.getTrackbarPos("Mode 0L 1S 2G 3A", CONTROLS_WINDOW))
        scale_mode = scale_mode_from_value(cv2.getTrackbarPos("Scale 0A 1G 2M", CONTROLS_WINDOW))

        p_min = cv2.getTrackbarPos("Min % x10", CONTROLS_WINDOW) / 10.0
        p_max = min(cv2.getTrackbarPos("Max % x10", CONTROLS_WINDOW) / 10.0, 100.0)

        gamma = max(cv2.getTrackbarPos("Gamma x100", CONTROLS_WINDOW) / 100.0, 0.01)

        low_slider = cv2.getTrackbarPos("Manual low", CONTROLS_WINDOW) / 1000.0
        high_slider = cv2.getTrackbarPos("Manual high", CONTROLS_WINDOW) / 1000.0
        manual_low = global_min_raw + low_slider * (global_max_raw - global_min_raw)
        manual_high = global_min_raw + high_slider * (global_max_raw - global_min_raw)

        p_start = cv2.getTrackbarPos("Plot start", WINDOW_NAME)
        p_end   = cv2.getTrackbarPos("Plot end",   WINDOW_NAME)

        raw_frame = data_cube[current_frame]

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

        mode_name  = {0: "LIN", 1: "SQRT", 2: "LOG", 3: "ASINH"}[transform_mode]
        scale_name = {0: "AUTO", 1: "GLOB", 2: "MAN"}[scale_mode]
        roi_shape  = "SQ" if roi_manager.square_mode else "RECT"

        overlay = cv2.cvtColor(display_frame, cv2.COLOR_GRAY2BGR)

        hud = (
            f"F{current_frame + 1}/{n_frames}"
            f"  {mode_name} {scale_name} g{gamma:.2f}"
            f"  {used_low:.1f}→{used_high:.1f}"
            f"  ROI:{len(roi_manager.rois)}[{roi_shape}]"
            f"  P:{p_start}-{p_end}"
        )
        cv2.putText(overlay, hud, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)

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
        elif key == ord("h"):
            print_help()
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
