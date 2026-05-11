"""
nvf_viewer_gui.py — PySide6 frontend for NVF thermographic files.

V1: single window, image viewer, ROI panel, display controls, timeline.
Relies on nvf_reader.py and roi.py unchanged.
Run: python nvf_viewer_gui.py
"""
from __future__ import annotations

import sys

import cv2
import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QBrush, QColor, QImage, QKeySequence, QPainter, QPixmap, QShortcut
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QButtonGroup,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from nvf_reader import NVFData, import_nvf
from roi import ROI, ROIManager, calculate_roi_timeseries
from display_pipeline import prepare_frame_for_display


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _bgr_to_qpixmap(bgr: np.ndarray) -> QPixmap:
    """
    Convert a (H, W, 3) uint8 BGR numpy array to QPixmap.
    Qt expects RGB; .copy() makes the buffer contiguous so QImage can own it.
    """
    h, w = bgr.shape[:2]
    rgb = bgr[:, :, ::-1].copy()
    qi = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qi)


# ---------------------------------------------------------------------------
# ImageCanvas
# ---------------------------------------------------------------------------

class ImageCanvas(QWidget):
    """
    Central widget: renders the thermal frame letterboxed inside the widget,
    and forwards mouse events to ROIManager using OpenCV event constants.

    Coordinate mapping:
      display point  ->  image point:
        ix = (px - offset_x) / scale
        iy = (py - offset_y) / scale
      where scale = min(widget_w / img_w,  widget_h / img_h)
            offset = (widget_size - scaled_img_size) / 2  (letterbox centering)

    The base BGR frame (without ROI overlay) is stored; the overlay is
    re-composited on every paintEvent so live drag preview works without
    re-running the expensive display pipeline.
    """

    mouse_moved = Signal(int, int, int)   # image_x, image_y, raw_value
    roi_finalized = Signal()              # emitted after left-button release

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._base_bgr: np.ndarray | None = None
        self._raw_frame: np.ndarray | None = None
        self._img_w = 640
        self._img_h = 512
        self.roi_manager: ROIManager | None = None
        self.setMouseTracking(True)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumSize(320, 240)
        self.setStyleSheet("background-color: black;")

    def set_base_frame(self, bgr: np.ndarray, raw_frame: np.ndarray) -> None:
        """Update stored frame; call when frame index or display params change."""
        h, w = bgr.shape[:2]
        self._img_w, self._img_h = w, h
        self._base_bgr = bgr
        self._raw_frame = raw_frame
        self.update()

    # --- layout helpers ----------------------------------------------------

    def _layout(self) -> tuple[float, int, int]:
        """Return (scale, offset_x, offset_y) for current widget size."""
        scale = min(self.width() / self._img_w, self.height() / self._img_h)
        dw = int(self._img_w * scale)
        dh = int(self._img_h * scale)
        ox = (self.width() - dw) // 2
        oy = (self.height() - dh) // 2
        return scale, ox, oy

    def _to_image(self, px: int, py: int) -> tuple[int, int] | None:
        """Map widget coordinates to image coordinates; None if outside."""
        scale, ox, oy = self._layout()
        ix, iy = (px - ox) / scale, (py - oy) / scale
        if 0 <= ix < self._img_w and 0 <= iy < self._img_h:
            return int(ix), int(iy)
        return None

    def _clamp_to_image(self, px: int, py: int) -> tuple[int, int]:
        """Like _to_image but clamps to image boundary instead of returning None."""
        scale, ox, oy = self._layout()
        ix = int(max(0, min((px - ox) / scale, self._img_w - 1)))
        iy = int(max(0, min((py - oy) / scale, self._img_h - 1)))
        return ix, iy

    # --- paint -------------------------------------------------------------

    def paintEvent(self, event) -> None:
        if self._base_bgr is None:
            return
        scale, ox, oy = self._layout()
        dw = int(self._img_w * scale)
        dh = int(self._img_h * scale)

        # Re-composite ROI overlay on every repaint (cheap; no pipeline re-run)
        bgr = self.roi_manager.draw_on_frame(self._base_bgr) if self.roi_manager else self._base_bgr

        painter = QPainter(self)
        painter.fillRect(self.rect(), Qt.GlobalColor.black)
        painter.drawPixmap(ox, oy, dw, dh, _bgr_to_qpixmap(bgr))
        painter.end()

    # --- mouse: translate Qt events → OpenCV constants → ROIManager --------

    def mousePressEvent(self, event) -> None:
        if self.roi_manager is None:
            return
        if event.button() == Qt.MouseButton.LeftButton:
            pt = self._to_image(int(event.position().x()), int(event.position().y()))
            if pt:
                self.roi_manager.handle_mouse(cv2.EVENT_LBUTTONDOWN, pt[0], pt[1], 0, None)

    def mouseMoveEvent(self, event) -> None:
        px, py = int(event.position().x()), int(event.position().y())
        if self.roi_manager is not None:
            ix, iy = self._clamp_to_image(px, py)
            self.roi_manager.handle_mouse(cv2.EVENT_MOUSEMOVE, ix, iy, 0, None)
            if self._raw_frame is not None:
                self.mouse_moved.emit(ix, iy, int(self._raw_frame[iy, ix]))
        self.update()

    def mouseReleaseEvent(self, event) -> None:
        if self.roi_manager is None:
            return
        if event.button() == Qt.MouseButton.LeftButton:
            ix, iy = self._clamp_to_image(
                int(event.position().x()), int(event.position().y())
            )
            self.roi_manager.handle_mouse(cv2.EVENT_LBUTTONUP, ix, iy, 0, None)
            self.roi_finalized.emit()
            self.update()


# ---------------------------------------------------------------------------
# TimelinePanel
# ---------------------------------------------------------------------------

class TimelinePanel(QWidget):
    frame_changed = Signal(int)

    def __init__(self, n_frames: int, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._n_frames = max(n_frames, 1)

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._on_timer)

        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setRange(0, self._n_frames - 1)
        self._slider.valueChanged.connect(self._on_slider)

        self._label = QLabel(f"F 1 / {self._n_frames}")
        self._label.setMinimumWidth(90)
        self._label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        self._btn_prev = QPushButton("◀")
        self._btn_prev.setFixedWidth(28)
        self._btn_prev.setToolTip("Previous frame  (A / ←)")
        self._btn_prev.clicked.connect(self.prev_frame)

        self._btn_play = QPushButton("⏵")
        self._btn_play.setFixedWidth(36)
        self._btn_play.setCheckable(True)
        self._btn_play.setToolTip("Play / Pause  (Space)")
        self._btn_play.toggled.connect(self._on_play_toggled)

        self._btn_next = QPushButton("▶")
        self._btn_next.setFixedWidth(28)
        self._btn_next.setToolTip("Next frame  (D / →)")
        self._btn_next.clicked.connect(self.next_frame)

        self._fps_spin = QSpinBox()
        self._fps_spin.setRange(1, 100)
        self._fps_spin.setValue(25)
        self._fps_spin.setSuffix(" fps")
        self._fps_spin.setFixedWidth(72)
        self._fps_spin.setToolTip("Playback speed in frames per second")
        self._fps_spin.valueChanged.connect(self._on_fps_changed)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 4, 6, 4)
        layout.addWidget(self._btn_prev)
        layout.addWidget(self._btn_play)
        layout.addWidget(self._btn_next)
        layout.addWidget(self._slider, stretch=1)
        layout.addWidget(self._label)
        layout.addWidget(QLabel("FPS:"))
        layout.addWidget(self._fps_spin)

    def reinit(self, n_frames: int) -> None:
        self._n_frames = max(n_frames, 1)
        self._slider.blockSignals(True)
        self._slider.setRange(0, self._n_frames - 1)
        self._slider.setValue(0)
        self._slider.blockSignals(False)
        self._label.setText(f"F 1 / {self._n_frames}")
        if self._timer.isActive():
            self._timer.stop()
            self._btn_play.setChecked(False)

    def toggle_play(self) -> None:
        self._btn_play.setChecked(not self._btn_play.isChecked())

    def current_frame(self) -> int:
        return self._slider.value()

    def prev_frame(self) -> None:
        self._slider.setValue(max(0, self._slider.value() - 1))

    def next_frame(self) -> None:
        self._slider.setValue(min(self._n_frames - 1, self._slider.value() + 1))

    def _on_slider(self, value: int) -> None:
        self._label.setText(f"F {value + 1} / {self._n_frames}")
        self.frame_changed.emit(value)

    def _on_play_toggled(self, playing: bool) -> None:
        self._btn_play.setText("⏸" if playing else "⏵")
        if playing:
            self._timer.start(1000 // max(self._fps_spin.value(), 1))
        else:
            self._timer.stop()

    def _on_timer(self) -> None:
        self._slider.setValue((self._slider.value() + 1) % self._n_frames)

    def _on_fps_changed(self, fps: int) -> None:
        if self._timer.isActive():
            self._timer.setInterval(1000 // max(fps, 1))


# ---------------------------------------------------------------------------
# DisplayPanel
# ---------------------------------------------------------------------------

class DisplayPanel(QWidget):
    params_changed = Signal()

    def __init__(self, global_min: float, global_max: float,
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)

        # Transform mode
        self._mode_group = QButtonGroup(self)
        rb_lin  = QRadioButton("Linear")
        rb_sqrt = QRadioButton("Sqrt")
        rb_log  = QRadioButton("Log")
        rb_asinh = QRadioButton("Asinh")
        rb_lin.setChecked(True)
        for i, rb in enumerate([rb_lin, rb_sqrt, rb_log, rb_asinh]):
            self._mode_group.addButton(rb, i)
        self._mode_group.idToggled.connect(
            lambda _id, chk: self.params_changed.emit() if chk else None
        )
        mode_row = self._hrow(rb_lin, rb_sqrt, rb_log, rb_asinh)

        # Scale mode
        self._scale_group = QButtonGroup(self)
        rb_auto, rb_glob, rb_man = (
            QRadioButton("Auto"), QRadioButton("Global"), QRadioButton("Manual")
        )
        rb_auto.setChecked(True)
        for i, rb in enumerate([rb_auto, rb_glob, rb_man]):
            self._scale_group.addButton(rb, i)
        self._scale_group.idToggled.connect(self._on_scale_changed)
        scale_row = self._hrow(rb_auto, rb_glob, rb_man)

        # Percentile spinboxes
        self._pmin = self._dspin(0.0, 99.9, 1.0, 0.5, " %")
        self._pmax = self._dspin(0.1, 100.0, 99.0, 0.5, " %")

        # Gamma
        self._gamma = self._dspin(0.01, 3.0, 1.0, 0.05, "")

        # Manual low / high (shown only when Scale = Manual)
        step = max(1.0, (global_max - global_min) / 200.0)
        self._man_low = self._dspin(global_min, global_max, global_min, step, "")
        self._man_high = self._dspin(global_min, global_max, global_max, step, "")

        self._manual_box = QWidget()
        mf = QFormLayout(self._manual_box)
        mf.setContentsMargins(0, 4, 0, 0)
        mf.setVerticalSpacing(4)
        mf.addRow("Low:", self._man_low)
        mf.addRow("High:", self._man_high)
        self._manual_box.setVisible(False)

        form = QFormLayout(self)
        form.setContentsMargins(8, 8, 8, 8)
        form.setVerticalSpacing(6)
        hdr = QLabel("Display settings")
        hdr.setStyleSheet("font-weight: bold; margin-bottom: 2px;")
        form.addRow(hdr)
        form.addRow("Transform:", mode_row)
        form.addRow("Scale:", scale_row)
        form.addRow("Min %:", self._pmin)
        form.addRow("Max %:", self._pmax)
        form.addRow("Gamma:", self._gamma)
        form.addRow(self._manual_box)

    # --- helpers -----------------------------------------------------------

    @staticmethod
    def _hrow(*widgets: QWidget) -> QWidget:
        w = QWidget()
        lay = QHBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        for ww in widgets:
            lay.addWidget(ww)
        lay.addStretch()
        return w

    def _dspin(self, lo: float, hi: float, val: float,
               step: float, suffix: str) -> QDoubleSpinBox:
        sp = QDoubleSpinBox()
        sp.setRange(lo, hi)
        sp.setValue(val)
        sp.setSingleStep(step)
        sp.setSuffix(suffix)
        sp.valueChanged.connect(lambda _: self.params_changed.emit())
        return sp

    # --- public ------------------------------------------------------------

    def reinit(self, global_min: float, global_max: float) -> None:
        """Reset controls to defaults for a new file; suppress all signal emissions."""
        all_w = [self._pmin, self._pmax, self._gamma, self._man_low, self._man_high,
                 self._mode_group, self._scale_group]
        for w in all_w:
            w.blockSignals(True)

        step = max(1.0, (global_max - global_min) / 200.0)
        for sp, lo, hi, val in [
            (self._man_low,  global_min, global_max, global_min),
            (self._man_high, global_min, global_max, global_max),
        ]:
            sp.setRange(lo, hi)
            sp.setSingleStep(step)
            sp.setValue(val)
        self._pmin.setValue(1.0)
        self._pmax.setValue(99.0)
        self._gamma.setValue(1.0)
        btn = self._mode_group.button(0)
        if btn:
            btn.setChecked(True)
        btn = self._scale_group.button(0)
        if btn:
            btn.setChecked(True)

        for w in all_w:
            w.blockSignals(False)
        self._manual_box.setVisible(False)

    def get_params(self) -> dict:
        return {
            "transform_mode": max(self._mode_group.checkedId(), 0),
            "scale_mode":     max(self._scale_group.checkedId(), 0),
            "p_min":          self._pmin.value(),
            "p_max":          self._pmax.value(),
            "gamma":          self._gamma.value(),
            "manual_low":     self._man_low.value(),
            "manual_high":    self._man_high.value(),
        }

    def _on_scale_changed(self, _id: int, checked: bool) -> None:
        if not checked:
            return
        self._manual_box.setVisible(_id == 2)
        self.params_changed.emit()


# ---------------------------------------------------------------------------
# ROIPanel
# ---------------------------------------------------------------------------

class ROIPanel(QWidget):
    mode_changed = Signal()
    undo_clicked = Signal()
    clear_clicked = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self._mode_group = QButtonGroup(self)
        rb_rect = QRadioButton("Rect")
        rb_sq   = QRadioButton("Square")
        rb_sq.setChecked(True)
        self._mode_group.addButton(rb_rect, 0)
        self._mode_group.addButton(rb_sq,   1)
        self._mode_group.idToggled.connect(
            lambda _id, chk: self.mode_changed.emit() if chk else None
        )
        mode_row = QWidget()
        ml = QHBoxLayout(mode_row)
        ml.setContentsMargins(0, 0, 0, 0)
        ml.addWidget(rb_rect)
        ml.addWidget(rb_sq)
        ml.addStretch()

        self._list = QListWidget()
        self._list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)

        btn_undo  = QPushButton("Undo last")
        btn_clear = QPushButton("Clear all")
        btn_undo.setToolTip("Remove the last drawn ROI  (Z)")
        btn_clear.setToolTip("Remove all ROIs  (C)")
        btn_undo.clicked.connect(self.undo_clicked)
        btn_clear.clicked.connect(self.clear_clicked)

        btn_row = QWidget()
        bl = QHBoxLayout(btn_row)
        bl.setContentsMargins(0, 0, 0, 0)
        bl.addWidget(btn_undo)
        bl.addWidget(btn_clear)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)
        hdr = QLabel("ROI Manager")
        hdr.setStyleSheet("font-weight: bold; margin-bottom: 2px;")
        layout.addWidget(hdr)
        layout.addWidget(QLabel("Draw mode  (R = Rect, S = Square):"))
        layout.addWidget(mode_row)
        layout.addWidget(QLabel("Active ROIs:"))
        layout.addWidget(self._list, stretch=1)
        layout.addWidget(btn_row)

    def is_square(self) -> bool:
        return self._mode_group.checkedId() == 1

    def refresh_list(self, rois: list[ROI]) -> None:
        self._list.clear()
        for roi in rois:
            text = f"{roi.name}  ({roi.x},{roi.y})  {roi.width}×{roi.height}"
            item = QListWidgetItem(text)
            b, g, r = roi.color   # ROI stores BGR; Qt needs RGB
            item.setForeground(QBrush(QColor(r, g, b)))
            self._list.addItem(item)


# ---------------------------------------------------------------------------
# PlotPanel
# ---------------------------------------------------------------------------

class PlotPanel(QWidget):
    """
    Collapsible bottom panel: plots mean raw value over time per ROI.

    After PlotWidget.clear() pyqtgraph keeps the LegendItem in place but
    empties it (because removeItem() is called on each curve during clear()).
    Re-plotting with name= auto-re-populates the legend, so we never call
    addLegend() more than once.
    """

    plot_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._plot_items: list = []

        self._plot_widget = pg.PlotWidget()
        self._plot_widget.setBackground("w")
        self._plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self._plot_widget.setLabel("left", "Mean raw value")
        self._plot_widget.setLabel("bottom", "Frame")
        self._plot_widget.addLegend()

        # Vertical marker for current frame (re-added after every clear())
        self._vline = pg.InfiniteLine(
            angle=90,
            movable=False,
            pen=pg.mkPen(color=(120, 120, 120), width=1,
                         style=Qt.PenStyle.DashLine),
        )
        self._plot_widget.addItem(self._vline)

        # Range controls
        self._start_spin = QSpinBox()
        self._start_spin.setRange(0, 0)
        self._start_spin.setPrefix("From: ")
        self._start_spin.setMinimumWidth(90)
        self._start_spin.setToolTip("First frame included in the plot (inclusive)")

        self._end_spin = QSpinBox()
        self._end_spin.setRange(0, 0)
        self._end_spin.setPrefix("To: ")
        self._end_spin.setMinimumWidth(90)
        self._end_spin.setToolTip("Last frame included in the plot (inclusive)")

        self._btn_plot = QPushButton("Plot ROIs")
        self._btn_plot.setToolTip(
            "Plot mean raw value per frame for all active ROIs  (G)"
        )
        self._btn_plot.clicked.connect(self.plot_requested)

        ctrl = QWidget()
        cl = QHBoxLayout(ctrl)
        cl.setContentsMargins(6, 2, 6, 2)
        cl.addWidget(QLabel("Plot range:"))
        cl.addWidget(self._start_spin)
        cl.addWidget(self._end_spin)
        cl.addSpacing(8)
        cl.addWidget(self._btn_plot)
        cl.addStretch()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(ctrl)
        layout.addWidget(self._plot_widget, stretch=1)

    # --- public API --------------------------------------------------------

    def reinit(self, n_frames: int) -> None:
        n = max(n_frames - 1, 0)
        for sp in (self._start_spin, self._end_spin):
            sp.blockSignals(True)
        self._start_spin.setRange(0, n)
        self._start_spin.setValue(0)
        self._end_spin.setRange(0, n)
        self._end_spin.setValue(n)
        for sp in (self._start_spin, self._end_spin):
            sp.blockSignals(False)
        self._clear()

    def get_range(self) -> tuple[int, int]:
        """Return (start, end) both inclusive."""
        return self._start_spin.value(), self._end_spin.value()

    def update_frame_marker(self, frame: int) -> None:
        self._vline.setValue(frame)

    def update_plot(
        self,
        timeseries: dict[str, np.ndarray],
        start_frame: int,
        colors: dict[str, tuple[int, int, int]],
    ) -> None:
        self._clear()
        for name, ts in timeseries.items():
            x = np.arange(start_frame, start_frame + len(ts))
            b, g, r = colors.get(name, (128, 128, 128))   # BGR → RGB for Qt/pg
            item = self._plot_widget.plot(
                x, ts,
                pen=pg.mkPen(color=(r, g, b), width=2),
                name=name,
            )
            self._plot_items.append(item)

    # --- internal ----------------------------------------------------------

    def _clear(self) -> None:
        """Remove all curve items; legend is emptied automatically by pyqtgraph."""
        self._plot_widget.clear()    # removes curves + empties legend entries
        self._plot_items.clear()
        self._plot_widget.addItem(self._vline)   # re-add: clear() removed it


# ---------------------------------------------------------------------------
# MainWindow
# ---------------------------------------------------------------------------

class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("NVF Viewer")
        self.resize(1280, 720)

        self._data_cube: np.ndarray | None = None
        self._n_frames  = 0
        self._cur_frame = 0
        self._glob_low  = 0.0
        self._glob_high = 1.0
        self._glob_min  = 0.0
        self._glob_max  = 1.0
        self.roi_manager: ROIManager | None = None

        self._build_toolbar()
        self._build_ui()
        self._build_shortcuts()

    def _build_toolbar(self) -> None:
        tb = self.addToolBar("Main")
        tb.setMovable(False)
        tb.addAction("Open NVF…", self._on_open)

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(4)

        h_splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left: image canvas
        self._canvas = ImageCanvas()
        self._canvas.mouse_moved.connect(self._on_mouse_moved)
        self._canvas.roi_finalized.connect(self._on_roi_finalized)
        h_splitter.addWidget(self._canvas)

        # Right: tab widget with ROI + Display panels
        right = QWidget()
        right.setMinimumWidth(220)
        right.setMaximumWidth(300)
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        self._tabs = QTabWidget()

        self._roi_panel = ROIPanel()
        self._roi_panel.mode_changed.connect(self._on_roi_mode_changed)
        self._roi_panel.undo_clicked.connect(self._on_undo_roi)
        self._roi_panel.clear_clicked.connect(self._on_clear_rois)
        self._tabs.addTab(self._roi_panel, "ROI")

        self._display_panel = DisplayPanel(0.0, 1.0)
        self._display_panel.params_changed.connect(self._refresh)
        self._tabs.addTab(self._display_panel, "Display")

        rl.addWidget(self._tabs)
        h_splitter.addWidget(right)
        h_splitter.setStretchFactor(0, 1)
        h_splitter.setStretchFactor(1, 0)

        # Plot panel in a vertical splitter below the image area
        self._plot_panel = PlotPanel()
        self._plot_panel.plot_requested.connect(self._on_plot_requested)

        v_splitter = QSplitter(Qt.Orientation.Vertical)
        v_splitter.addWidget(h_splitter)
        v_splitter.addWidget(self._plot_panel)
        v_splitter.setStretchFactor(0, 3)
        v_splitter.setStretchFactor(1, 1)
        root.addWidget(v_splitter, stretch=1)

        # Bottom: timeline (frame marker also forwarded to plot panel)
        self._timeline = TimelinePanel(1)
        self._timeline.setEnabled(False)
        self._timeline.frame_changed.connect(self._on_frame_changed)
        self._timeline.frame_changed.connect(self._plot_panel.update_frame_marker)
        root.addWidget(self._timeline)

        # Two-zone status bar: frame info (left, stretchy) + mouse coords (right, permanent)
        self._status_frame = QLabel("Open an NVF file to begin.  (Toolbar → Open NVF…)")
        self._status_mouse = QLabel("")
        self._status_mouse.setMinimumWidth(200)
        self.statusBar().addWidget(self._status_frame, 1)
        self.statusBar().addPermanentWidget(self._status_mouse)

    # --- keyboard shortcuts ------------------------------------------------

    def _build_shortcuts(self) -> None:
        # Left/Right arrows are already handled natively by QSlider when focused.
        # A/D duplicate that for users who keep hands on keyboard without touching the slider.
        # Space, R, S, Z, C are window-wide shortcuts.
        pairs = [
            ("Space", self._timeline.toggle_play),
            ("A",     self._timeline.prev_frame),
            ("D",     self._timeline.next_frame),
            ("R",     self._set_roi_rect),
            ("S",     self._set_roi_square),
            ("Z",     self._on_undo_roi),
            ("C",     self._on_clear_rois),
            ("G",     self._on_plot_requested),
        ]
        for key, slot in pairs:
            QShortcut(QKeySequence(key), self).activated.connect(slot)

    def _set_roi_rect(self) -> None:
        btn = self._roi_panel._mode_group.button(0)
        if btn:
            btn.setChecked(True)

    def _set_roi_square(self) -> None:
        btn = self._roi_panel._mode_group.button(1)
        if btn:
            btn.setChecked(True)

    # --- file loading ------------------------------------------------------

    def _on_open(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Open NVF file", "", "All files (*.*)")
        if not path:
            return
        try:
            nvf: NVFData = import_nvf(path)
        except Exception as exc:
            QMessageBox.critical(self, "Load error", str(exc))
            return
        self._load(nvf)

    def _load(self, nvf: NVFData) -> None:
        self._data_cube = nvf.data_cube
        self._n_frames  = nvf.n_frames
        self._cur_frame = 0

        self._glob_min = float(np.min(self._data_cube))
        self._glob_max = float(np.max(self._data_cube))
        self._glob_low = float(np.percentile(self._data_cube, 1.0))
        self._glob_high = float(np.percentile(self._data_cube, 99.0))
        if self._glob_high <= self._glob_low:
            self._glob_high = self._glob_low + 1.0

        _, fh, fw = self._data_cube.shape
        self.roi_manager = ROIManager(fw, fh)
        self._canvas.roi_manager = self.roi_manager

        self._display_panel.reinit(self._glob_min, self._glob_max)
        self._timeline.reinit(self._n_frames)
        self._timeline.setEnabled(True)
        self._plot_panel.reinit(self._n_frames)
        self.setWindowTitle(f"NVF Viewer — {nvf.file_path.name}")
        self._refresh()

    # --- rendering ---------------------------------------------------------

    def _refresh(self) -> None:
        if self._data_cube is None:
            return
        p   = self._display_panel.get_params()
        raw = self._data_cube[self._cur_frame]

        uint8, used_low, used_high = prepare_frame_for_display(
            raw_frame=raw,
            transform_mode=p["transform_mode"],
            scale_mode=p["scale_mode"],
            p_min=p["p_min"],
            p_max=p["p_max"],
            global_low=self._glob_low,
            global_high=self._glob_high,
            manual_low=p["manual_low"],
            manual_high=p["manual_high"],
            gamma=p["gamma"],
        )
        # Grayscale → BGR so roi_manager.draw_on_frame can paint colored boxes
        bgr = cv2.cvtColor(uint8, cv2.COLOR_GRAY2BGR)
        self._canvas.set_base_frame(bgr, raw)

        self._roi_panel.refresh_list(self.roi_manager.rois if self.roi_manager else [])
        n_rois = len(self.roi_manager.rois) if self.roi_manager else 0
        self._status_frame.setText(
            f"F {self._cur_frame + 1} / {self._n_frames}"
            f"  |  win: {used_low:.1f} → {used_high:.1f}"
            f"  |  ROI: {n_rois}"
        )

    # --- slots -------------------------------------------------------------

    def _on_frame_changed(self, idx: int) -> None:
        self._cur_frame = idx
        self._refresh()

    def _on_roi_finalized(self) -> None:
        # Pipeline already cached in canvas; just update the list and repaint
        if self.roi_manager:
            self._roi_panel.refresh_list(self.roi_manager.rois)
        self._canvas.update()

    def _on_mouse_moved(self, ix: int, iy: int, val: int) -> None:
        self._status_mouse.setText(f"x={ix}  y={iy}  raw={val}")

    def _on_roi_mode_changed(self) -> None:
        if self.roi_manager:
            self.roi_manager.square_mode = self._roi_panel.is_square()

    def _on_plot_requested(self) -> None:
        if self._data_cube is None or not self.roi_manager or not self.roi_manager.rois:
            return
        p_start, p_end = self._plot_panel.get_range()
        if p_end < p_start:
            return
        timeseries = calculate_roi_timeseries(
            self._data_cube, self.roi_manager.rois, p_start, p_end + 1
        )
        colors = {roi.name: roi.color for roi in self.roi_manager.rois}
        self._plot_panel.update_plot(timeseries, p_start, colors)

    def _on_undo_roi(self) -> None:
        if self.roi_manager:
            self.roi_manager.remove_last()
            self._roi_panel.refresh_list(self.roi_manager.rois)
            self._canvas.update()

    def _on_clear_rois(self) -> None:
        if self.roi_manager:
            self.roi_manager.clear()
            self._roi_panel.refresh_list([])
            self._canvas.update()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
