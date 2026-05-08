from __future__ import annotations

from dataclasses import dataclass

import cv2
import matplotlib.pyplot as plt
import numpy as np


_COLORS: list[tuple[int, int, int]] = [
    (0,   0,   255),  # red
    (0,   255, 0  ),  # green
    (255, 0,   0  ),  # blue
    (0,   255, 255),  # yellow
    (255, 0,   255),  # magenta
    (255, 255, 0  ),  # cyan
    (0,   128, 255),  # orange
    (128, 0,   255),  # violet
]

# Riferimento all'ultima figura ROI aperta: chiusa prima di aprirne una nuova.
_last_roi_figure: plt.Figure | None = None


@dataclass
class ROI:
    name: str
    x: int
    y: int
    width: int
    height: int
    color: tuple[int, int, int]  # BGR


class ROIManager:
    def __init__(self, frame_width: int, frame_height: int) -> None:
        self._fw = frame_width
        self._fh = frame_height
        self.rois: list[ROI] = []
        self._next_id: int = 1
        self._drawing: bool = False
        self._start_pt: tuple[int, int] | None = None
        self._current_pt: tuple[int, int] | None = None
        self.square_mode: bool = True

    # ------------------------------------------------------------------
    # Mouse callback (da passare a cv2.setMouseCallback)
    # ------------------------------------------------------------------

    def handle_mouse(self, event: int, x: int, y: int, flags: int, param) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            self._drawing = True
            self._start_pt = (x, y)
            self._current_pt = (x, y)

        elif event == cv2.EVENT_MOUSEMOVE:
            if self._drawing:
                self._current_pt = (x, y)

        elif event == cv2.EVENT_LBUTTONUP:
            if self._drawing:
                self._drawing = False
                self._finalize_roi(x, y)
                self._start_pt = None
                self._current_pt = None

    # ------------------------------------------------------------------
    # Geometria: calcola la ROI da start a end
    # ------------------------------------------------------------------

    def _compute_roi(
        self, end_x: int, end_y: int
    ) -> tuple[int, int, int, int] | None:
        """
        Ritorna (tl_x, tl_y, width, height) oppure None se start_pt mancante.

        Un click senza drag produce sempre una ROI 1x1.
        Square mode: width == height == max(1, min(|dx|, |dy|)).
        Rect mode:   width == max(1, |dx|),  height == max(1, |dy|).
        Il top-left è determinato dal segno di dx/dy, poi clampato nel frame.
        """
        if self._start_pt is None:
            return None

        sx, sy = self._start_pt
        dx = end_x - sx
        dy = end_y - sy

        if self.square_mode:
            side = max(1, min(abs(dx), abs(dy)))
            w, h = side, side
        else:
            w = max(1, abs(dx))
            h = max(1, abs(dy))

        tl_x = sx if dx >= 0 else sx - w
        tl_y = sy if dy >= 0 else sy - h

        tl_x = max(0, min(tl_x, self._fw - w))
        tl_y = max(0, min(tl_y, self._fh - h))

        return tl_x, tl_y, w, h

    def _finalize_roi(self, end_x: int, end_y: int) -> None:
        result = self._compute_roi(end_x, end_y)
        if result is None:
            return

        tl_x, tl_y, w, h = result
        color = _COLORS[(self._next_id - 1) % len(_COLORS)]
        self.rois.append(ROI(
            name=f"ROI {self._next_id}",
            x=tl_x,
            y=tl_y,
            width=w,
            height=h,
            color=color,
        ))
        self._next_id += 1

    # ------------------------------------------------------------------
    # API pubblica
    # ------------------------------------------------------------------

    def remove_last(self) -> None:
        if self.rois:
            self.rois.pop()

    def clear(self) -> None:
        self.rois.clear()

    def draw_on_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Disegna le ROI confermate e l'anteprima del drag corrente su una
        copia del frame BGR. Non modifica l'array originale.
        """
        out = frame.copy()

        for roi in self.rois:
            pt1 = (roi.x, roi.y)
            pt2 = (roi.x + roi.width, roi.y + roi.height)
            cv2.rectangle(out, pt1, pt2, roi.color, 2)
            cv2.putText(
                out,
                roi.name,
                (roi.x + 4, roi.y + 16),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                roi.color,
                1,
                cv2.LINE_AA,
            )

        # Anteprima mentre l'utente sta trascinando
        if self._drawing and self._start_pt and self._current_pt:
            result = self._compute_roi(*self._current_pt)
            if result is not None:
                px, py, pw, ph = result
                preview_color = _COLORS[(self._next_id - 1) % len(_COLORS)]
                cv2.rectangle(out, (px, py), (px + pw, py + ph), preview_color, 1)

        return out


# ----------------------------------------------------------------------
# Calcolo timeseries sui valori raw originali
# ----------------------------------------------------------------------

def calculate_roi_timeseries(
    data_cube: np.ndarray,
    rois: list[ROI],
    start_frame: int = 0,
    end_frame: int | None = None,
) -> dict[str, np.ndarray]:
    """
    Per ogni ROI calcola la media dei valori raw per ogni frame dell'intervallo.

    data_cube: shape (n_frames, height, width), valori raw originali (int16).
    Ritorna dict  {roi.name: np.ndarray shape (n_selected_frames,), dtype float64}.
    Una ROI 1x1 restituisce il valore grezzo del singolo pixel per ogni frame.
    """
    if not rois:
        return {}

    n_frames, height, width = data_cube.shape

    start_frame = max(0, min(start_frame, n_frames - 1))
    if end_frame is None:
        end_frame = n_frames
    else:
        end_frame = max(0, min(end_frame, n_frames))

    if end_frame <= start_frame:
        raise ValueError(
            f"end_frame ({end_frame}) deve essere maggiore di start_frame ({start_frame})."
        )

    result: dict[str, np.ndarray] = {}

    for roi in rois:
        x1 = max(0, roi.x)
        y1 = max(0, roi.y)
        x2 = min(width,  roi.x + roi.width)
        y2 = min(height, roi.y + roi.height)

        if x2 <= x1 or y2 <= y1:
            # ROI completamente fuori dai limiti del frame, skip silenzioso
            continue

        # Slice 3D sull'intero intervallo, poi media su height e width.
        # ROI 1x1: region ha shape (n, 1, 1) → mean restituisce il pixel grezzo.
        # data_cube è int16; .mean() promuove automaticamente a float64.
        region = data_cube[start_frame:end_frame, y1:y2, x1:x2]
        result[roi.name] = region.mean(axis=(1, 2))

    return result


# ----------------------------------------------------------------------
# Grafico comparativo delle timeseries
# ----------------------------------------------------------------------

def plot_roi_timeseries(
    timeseries: dict[str, np.ndarray],
    start_frame: int = 0,
    title: str = "ROI raw mean over time",
    colors: dict[str, tuple[int, int, int]] | None = None,
    current_frame: int | None = None,
) -> None:
    """
    Mostra un grafico matplotlib con una linea per ogni ROI.

    timeseries:    dict {roi_name: np.ndarray float64} da calculate_roi_timeseries.
    colors:        opzionale, dict {roi_name: (B, G, R)} per usare gli stessi
                   colori del viewer OpenCV. Se assente usa il color cycle di matplotlib.
    current_frame: se fornito, disegna una linea verticale tratteggiata sul frame
                   corrente del viewer (coordinate assolute del video).
    """
    if not timeseries:
        return

    global _last_roi_figure
    if _last_roi_figure is not None:
        plt.close(_last_roi_figure)

    fig, ax = plt.subplots(figsize=(10, 5))
    _last_roi_figure = fig

    for name, ts in timeseries.items():
        x = np.arange(start_frame, start_frame + len(ts))

        if colors and name in colors:
            b, g, r = colors[name]
            line_color = (r / 255.0, g / 255.0, b / 255.0)
            ax.plot(x, ts, label=name, color=line_color)
        else:
            ax.plot(x, ts, label=name)

    if current_frame is not None:
        ax.axvline(
            x=current_frame,
            color="gray",
            linestyle="--",
            linewidth=1.2,
            label=f"Current frame ({current_frame})",
        )

    ax.set_xlabel("Frame")
    ax.set_ylabel("Mean raw value")
    ax.set_title(title)
    ax.legend()
    ax.grid(True)
    fig.tight_layout()

    plt.show(block=False)
    plt.pause(0.001)
