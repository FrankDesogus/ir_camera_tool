"""
nvf_viewer_gui.py — interfaccia grafica PySide6 per i file NVF termografici.

Struttura dell'interfaccia:
  ┌─────────────────────────────────────────────────────────┐
  │  Toolbar:  [Open NVF…]                                  │
  ├──────────────────────────────┬──────────────────────────┤
  │                              │  Tab "ROI":              │
  │   ImageCanvas                │    ROIPanel              │
  │   (immagine termica +        │  Tab "Display":          │
  │    rettangoli ROI)           │    DisplayPanel          │
  ├──────────────────────────────┴──────────────────────────┤
  │  PlotPanel (grafico serie temporali ROI)                │
  ├─────────────────────────────────────────────────────────┤
  │  TimelinePanel (slider frame + pulsanti play)           │
  ├─────────────────────────────────────────────────────────┤
  │  Status bar: info frame a sinistra, posizione mouse a   │
  │              destra                                     │
  └─────────────────────────────────────────────────────────┘

Flusso dati:
  1. L'utente apre un file NVF → import_nvf() → data_cube numpy int16
  2. Ad ogni cambio frame o parametro display → prepare_frame_for_display()
     produce un array uint8 → ImageCanvas lo visualizza letterboxed
  3. L'utente disegna ROI sull'immagine → ROIManager raccoglie i rettangoli
  4. Premendo "Plot ROIs" → calculate_roi_timeseries() → PlotPanel

Avvio:
  python nvf_viewer_gui.py
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
# Funzione di utilità: conversione array NumPy → QPixmap
# ---------------------------------------------------------------------------

def _bgr_to_qpixmap(bgr: np.ndarray) -> QPixmap:
    """
    Converte un array NumPy BGR uint8 di forma (H, W, 3) in un QPixmap Qt.

    Qt si aspetta i dati in formato RGB, non BGR. Il rovesciamento degli assi
    [::-1] inverte l'ordine dei canali: BGR → RGB.
    .copy() rende il buffer contiguo in memoria, necessario perché QImage
    acceda ai dati senza copiarli internamente in modo errato.
    """
    h, w = bgr.shape[:2]
    rgb = bgr[:, :, ::-1].copy()   # BGR → RGB, buffer contiguo
    qi = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qi)


# ---------------------------------------------------------------------------
# ImageCanvas — widget centrale che mostra il frame termico
# ---------------------------------------------------------------------------

class ImageCanvas(QWidget):
    """
    Widget che visualizza il frame termico letterboxed (con bande nere sui lati)
    e traduce gli eventi mouse Qt → ROIManager.

    "Letterboxed" significa che l'immagine viene scalata uniformemente per
    occupare il massimo spazio disponibile nel widget mantenendo le proporzioni
    originali (640×512). Le bande nere riempiono lo spazio rimanente.

    Architettura di rendering:
      - _base_bgr: frame BGR senza overlay ROI, aggiornato quando cambia il frame
        o i parametri display (operazione costosa: pipeline completa)
      - Ad ogni paintEvent: ROIManager.draw_on_frame() sovrappone i rettangoli
        ROI sul _base_bgr (operazione economica, solo disegno)
      Questo schema evita di rieseguire la pipeline display ad ogni movimento
      del mouse (es. anteprima drag ROI).

    Segnali emessi:
      mouse_moved(ix, iy, raw_value) — posizione immagine e valore grezzo al cursore
      roi_finalized()                — dopo il rilascio del pulsante sinistro
    """

    mouse_moved  = Signal(int, int, int)   # x_immagine, y_immagine, valore_grezzo
    roi_finalized = Signal()               # emesso dopo il completamento di una ROI

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._base_bgr: np.ndarray | None = None   # frame corrente in BGR
        self._raw_frame: np.ndarray | None = None  # frame grezzo (per leggere il valore al cursore)
        self._img_w = 640
        self._img_h = 512
        self.roi_manager: ROIManager | None = None
        self.setMouseTracking(True)   # abilita MOUSEMOVE anche senza pulsanti premuti
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumSize(320, 240)
        self.setStyleSheet("background-color: black;")

    def set_base_frame(self, bgr: np.ndarray, raw_frame: np.ndarray) -> None:
        """
        Aggiorna il frame da visualizzare. Va chiamato ogni volta che cambia
        il frame corrente o i parametri della pipeline display.
        Scatena un repaint (update()).
        """
        h, w = bgr.shape[:2]
        self._img_w, self._img_h = w, h
        self._base_bgr = bgr
        self._raw_frame = raw_frame
        self.update()   # chiede a Qt di ridisegnare il widget

    # --- calcolo della geometria letterbox ---

    def _layout(self) -> tuple[float, int, int]:
        """
        Calcola i parametri di layout letterbox per il widget corrente.

        Restituisce (scala, offset_x, offset_y):
          scala    — fattore di scala uniforme (< 1 se il widget è più piccolo del frame)
          offset_x — spazio orizzontale libero a sinistra dell'immagine scalata
          offset_y — spazio verticale libero sopra l'immagine scalata
        """
        scale = min(self.width() / self._img_w, self.height() / self._img_h)
        dw = int(self._img_w * scale)
        dh = int(self._img_h * scale)
        ox = (self.width()  - dw) // 2
        oy = (self.height() - dh) // 2
        return scale, ox, oy

    def _to_image(self, px: int, py: int) -> tuple[int, int] | None:
        """
        Converte coordinate widget → coordinate immagine.
        Restituisce None se il punto cade fuori dall'area dell'immagine scalata.
        """
        scale, ox, oy = self._layout()
        ix, iy = (px - ox) / scale, (py - oy) / scale
        if 0 <= ix < self._img_w and 0 <= iy < self._img_h:
            return int(ix), int(iy)
        return None

    def _clamp_to_image(self, px: int, py: int) -> tuple[int, int]:
        """
        Come _to_image ma clampa il punto ai bordi dell'immagine invece di
        restituire None. Utile per il drag ROI: anche se il mouse esce
        dall'immagine, la ROI rimane dentro i limiti.
        """
        scale, ox, oy = self._layout()
        ix = int(max(0, min((px - ox) / scale, self._img_w - 1)))
        iy = int(max(0, min((py - oy) / scale, self._img_h - 1)))
        return ix, iy

    # --- rendering ---

    def paintEvent(self, event) -> None:
        """
        Disegna il frame + overlay ROI ad ogni repaint.
        Viene chiamato automaticamente da Qt quando il widget deve essere ridisegnato.
        """
        if self._base_bgr is None:
            return
        scale, ox, oy = self._layout()
        dw = int(self._img_w * scale)
        dh = int(self._img_h * scale)

        # Sovrappone i rettangoli ROI al frame base (operazione economica)
        bgr = self.roi_manager.draw_on_frame(self._base_bgr) if self.roi_manager else self._base_bgr

        painter = QPainter(self)
        painter.fillRect(self.rect(), Qt.GlobalColor.black)   # sfondo nero (letterbox)
        painter.drawPixmap(ox, oy, dw, dh, _bgr_to_qpixmap(bgr))
        painter.end()

    # --- eventi mouse: traduzione Qt → costanti OpenCV → ROIManager ---

    def mousePressEvent(self, event) -> None:
        """
        Tasto sinistro premuto → invia EVENT_LBUTTONDOWN al ROIManager
        con le coordinate convertite in spazio immagine.
        Ignora i clic fuori dall'area dell'immagine scalata.
        """
        if self.roi_manager is None:
            return
        if event.button() == Qt.MouseButton.LeftButton:
            pt = self._to_image(int(event.position().x()), int(event.position().y()))
            if pt:
                self.roi_manager.handle_mouse(cv2.EVENT_LBUTTONDOWN, pt[0], pt[1], 0, None)

    def mouseMoveEvent(self, event) -> None:
        """
        Movimento mouse → aggiorna l'anteprima drag nel ROIManager e
        emette il segnale mouse_moved con la posizione e il valore grezzo.
        Il clamping permette di trascinare fuori bordo senza perdere l'evento.
        """
        px, py = int(event.position().x()), int(event.position().y())
        if self.roi_manager is not None:
            ix, iy = self._clamp_to_image(px, py)
            self.roi_manager.handle_mouse(cv2.EVENT_MOUSEMOVE, ix, iy, 0, None)
            if self._raw_frame is not None:
                self.mouse_moved.emit(ix, iy, int(self._raw_frame[iy, ix]))
        self.update()   # forza il repaint per mostrare l'anteprima drag

    def mouseReleaseEvent(self, event) -> None:
        """
        Tasto sinistro rilasciato → finalizza la ROI nel ROIManager e
        emette il segnale roi_finalized per aggiornare la lista ROI nella GUI.
        """
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
# TimelinePanel — controlli di playback
# ---------------------------------------------------------------------------

class TimelinePanel(QWidget):
    """
    Barra inferiore con slider frame, pulsanti play/pausa/prev/next e
    spinbox FPS.

    Il QTimer interno scatta ogni (1000 / fps) millisecondi durante il play
    e avanza lo slider di un frame. Lo slider emette frame_changed che
    propaga il cambio al resto della GUI.
    """

    frame_changed = Signal(int)   # emesso con il nuovo indice frame (0-based)

    def __init__(self, n_frames: int, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._n_frames = max(n_frames, 1)

        # Timer per la riproduzione automatica
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._on_timer)

        # Slider principale: un'unità = un frame
        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setRange(0, self._n_frames - 1)
        self._slider.valueChanged.connect(self._on_slider)

        # Etichetta "F X / N"
        self._label = QLabel(f"F 1 / {self._n_frames}")
        self._label.setMinimumWidth(90)
        self._label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        # Pulsanti di navigazione frame
        self._btn_prev = QPushButton("◀")
        self._btn_prev.setFixedWidth(28)
        self._btn_prev.setToolTip("Frame precedente  (A / ←)")
        self._btn_prev.clicked.connect(self.prev_frame)

        self._btn_play = QPushButton("⏵")
        self._btn_play.setFixedWidth(36)
        self._btn_play.setCheckable(True)   # rimane premuto durante il play
        self._btn_play.setToolTip("Play / Pausa  (Space)")
        self._btn_play.toggled.connect(self._on_play_toggled)

        self._btn_next = QPushButton("▶")
        self._btn_next.setFixedWidth(28)
        self._btn_next.setToolTip("Frame successivo  (D / →)")
        self._btn_next.clicked.connect(self.next_frame)

        # Spinbox velocità di riproduzione
        self._fps_spin = QSpinBox()
        self._fps_spin.setRange(1, 100)
        self._fps_spin.setValue(25)
        self._fps_spin.setSuffix(" fps")
        self._fps_spin.setFixedWidth(72)
        self._fps_spin.setToolTip("Velocità di riproduzione in frame al secondo")
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
        """
        Reimposta il pannello per un nuovo file: aggiorna il range dello slider,
        torna al frame 0 e ferma la riproduzione se attiva.
        blockSignals(True/False) evita notifiche spurie durante il reset.
        """
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
        """Alterna play/pausa (usato dalla shortcut da tastiera Space)."""
        self._btn_play.setChecked(not self._btn_play.isChecked())

    def current_frame(self) -> int:
        """Restituisce l'indice del frame correntemente selezionato."""
        return self._slider.value()

    def prev_frame(self) -> None:
        """Va al frame precedente (clampa a 0 se già al primo)."""
        self._slider.setValue(max(0, self._slider.value() - 1))

    def next_frame(self) -> None:
        """Va al frame successivo (clampa all'ultimo frame se già all'ultimo)."""
        self._slider.setValue(min(self._n_frames - 1, self._slider.value() + 1))

    def _on_slider(self, value: int) -> None:
        """Aggiorna l'etichetta e propaga il cambio frame al resto della GUI."""
        self._label.setText(f"F {value + 1} / {self._n_frames}")
        self.frame_changed.emit(value)

    def _on_play_toggled(self, playing: bool) -> None:
        """Avvia o ferma il timer quando il pulsante play viene premuto/rilasciato."""
        self._btn_play.setText("⏸" if playing else "⏵")
        if playing:
            self._timer.start(1000 // max(self._fps_spin.value(), 1))
        else:
            self._timer.stop()

    def _on_timer(self) -> None:
        """Scatta ogni (1000/fps) ms durante il play: avanza di un frame."""
        self._slider.setValue((self._slider.value() + 1) % self._n_frames)

    def _on_fps_changed(self, fps: int) -> None:
        """Aggiorna l'intervallo del timer se il play è attivo."""
        if self._timer.isActive():
            self._timer.setInterval(1000 // max(fps, 1))


# ---------------------------------------------------------------------------
# DisplayPanel — controlli della pipeline di rendering
# ---------------------------------------------------------------------------

class DisplayPanel(QWidget):
    """
    Pannello dei parametri display: trasformazione, scala, percentili, gamma,
    low/high manuali.

    Tutti i controlli emettono params_changed quando cambiano, così
    MainWindow._refresh() può aggiornare l'immagine immediatamente.

    Modalità di trasformazione (radio button):
      Linear / Sqrt / Log / Asinh — vedere display_pipeline.py per la descrizione

    Modalità di scala (radio button):
      Auto   — percentili calcolati su ogni frame individualmente
      Global — percentili calcolati sull'intero video al caricamento
      Manual — l'utente imposta direttamente i valori low/high

    Controlli aggiuntivi:
      Min % / Max % — percentili per la modalità Auto (es. 1.0 e 99.0)
      Gamma         — correzione gamma (1.0 = nessun effetto)
      Low / High    — valori manuali (visibili solo in modalità Manual)
    """

    params_changed = Signal()   # emesso ogni volta che un parametro cambia

    def __init__(self, global_min: float, global_max: float,
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)

        # --- Gruppo radio: modalità di trasformazione ---
        self._mode_group = QButtonGroup(self)
        rb_lin   = QRadioButton("Linear")
        rb_sqrt  = QRadioButton("Sqrt")
        rb_log   = QRadioButton("Log")
        rb_asinh = QRadioButton("Asinh")
        rb_lin.setChecked(True)
        for i, rb in enumerate([rb_lin, rb_sqrt, rb_log, rb_asinh]):
            self._mode_group.addButton(rb, i)
        # idToggled emette (id, checked): emette params_changed solo quando un bottone
        # passa a checked=True (non quando viene deselezionato)
        self._mode_group.idToggled.connect(
            lambda _id, chk: self.params_changed.emit() if chk else None
        )
        mode_row = self._hrow(rb_lin, rb_sqrt, rb_log, rb_asinh)

        # --- Gruppo radio: modalità di scala ---
        self._scale_group = QButtonGroup(self)
        rb_auto, rb_glob, rb_man = (
            QRadioButton("Auto"), QRadioButton("Global"), QRadioButton("Manual")
        )
        rb_auto.setChecked(True)
        for i, rb in enumerate([rb_auto, rb_glob, rb_man]):
            self._scale_group.addButton(rb, i)
        self._scale_group.idToggled.connect(self._on_scale_changed)
        scale_row = self._hrow(rb_auto, rb_glob, rb_man)

        # --- Spinbox percentili per modalità Auto ---
        self._pmin = self._dspin(0.0, 99.9,  1.0,  0.5, " %")   # percentile inferiore
        self._pmax = self._dspin(0.1, 100.0, 99.0, 0.5, " %")   # percentile superiore

        # --- Spinbox gamma ---
        self._gamma = self._dspin(0.01, 3.0, 1.0, 0.05, "")

        # --- Spinbox low/high manuali ---
        # Il passo viene scelto come 1/200 del range globale per una regolazione comoda
        step = max(1.0, (global_max - global_min) / 200.0)
        self._man_low  = self._dspin(global_min, global_max, global_min, step, "")
        self._man_high = self._dspin(global_min, global_max, global_max, step, "")

        # Contenitore nascosto per i controlli manuali (visibile solo in modo Manual)
        self._manual_box = QWidget()
        mf = QFormLayout(self._manual_box)
        mf.setContentsMargins(0, 4, 0, 0)
        mf.setVerticalSpacing(4)
        mf.addRow("Low:",  self._man_low)
        mf.addRow("High:", self._man_high)
        self._manual_box.setVisible(False)   # nascosto di default

        # --- Layout a form ---
        form = QFormLayout(self)
        form.setContentsMargins(8, 8, 8, 8)
        form.setVerticalSpacing(6)
        hdr = QLabel("Display settings")
        hdr.setStyleSheet("font-weight: bold; margin-bottom: 2px;")
        form.addRow(hdr)
        form.addRow("Transform:", mode_row)
        form.addRow("Scale:",     scale_row)
        form.addRow("Min %:",     self._pmin)
        form.addRow("Max %:",     self._pmax)
        form.addRow("Gamma:",     self._gamma)
        form.addRow(self._manual_box)

    # --- metodi helper privati ---

    @staticmethod
    def _hrow(*widgets: QWidget) -> QWidget:
        """
        Crea un contenitore orizzontale con i widget dati affiancati.
        Usato per costruire le righe di radio button.
        """
        w = QWidget()
        lay = QHBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        for ww in widgets:
            lay.addWidget(ww)
        lay.addStretch()
        return w

    def _dspin(self, lo: float, hi: float, val: float,
               step: float, suffix: str) -> QDoubleSpinBox:
        """
        Crea un QDoubleSpinBox già configurato e collegato a params_changed.
        """
        sp = QDoubleSpinBox()
        sp.setRange(lo, hi)
        sp.setValue(val)
        sp.setSingleStep(step)
        sp.setSuffix(suffix)
        sp.valueChanged.connect(lambda _: self.params_changed.emit())
        return sp

    # --- API pubblica ---

    def reinit(self, global_min: float, global_max: float) -> None:
        """
        Reimposta tutti i controlli ai valori di default per un nuovo file.
        blockSignals(True/False) evita che i reset scatenino un refresh prematuro
        mentre i range non sono ancora coerenti.
        """
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
        """
        Restituisce un dizionario con tutti i parametri correnti.
        Usato da MainWindow._refresh() per passare i dati alla pipeline display.
        """
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
        """
        Mostra/nasconde i controlli manuali low/high in base alla modalità
        di scala selezionata. I controlli manuali sono visibili solo quando
        scale_mode = 2 (Manual).
        """
        if not checked:
            return
        self._manual_box.setVisible(_id == 2)
        self.params_changed.emit()


# ---------------------------------------------------------------------------
# ROIPanel — gestione visuale delle ROI
# ---------------------------------------------------------------------------

class ROIPanel(QWidget):
    """
    Pannello laterale per la gestione delle ROI:
      - scelta modalità di disegno (rettangolo o quadrato)
      - lista delle ROI attive con colori corrispondenti
      - pulsanti undo (rimuovi ultima) e clear (rimuovi tutte)

    Non gestisce direttamente il ROIManager: emette segnali che
    MainWindow gestisce per mantenere separata la logica dalla UI.
    """

    mode_changed  = Signal()   # la modalità di disegno è cambiata
    undo_clicked  = Signal()   # pulsante "Undo last" premuto
    clear_clicked = Signal()   # pulsante "Clear all" premuto

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        # --- Gruppo radio: modalità di disegno ---
        self._mode_group = QButtonGroup(self)
        rb_rect = QRadioButton("Rect")
        rb_sq   = QRadioButton("Square")
        rb_sq.setChecked(True)   # quadrato di default
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

        # --- Lista ROI attive ---
        self._list = QListWidget()
        self._list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)

        # --- Pulsanti ---
        btn_undo  = QPushButton("Undo last")
        btn_clear = QPushButton("Clear all")
        btn_undo.setToolTip("Rimuove l'ultima ROI disegnata  (Z)")
        btn_clear.setToolTip("Rimuove tutte le ROI  (C)")
        btn_undo.clicked.connect(self.undo_clicked)
        btn_clear.clicked.connect(self.clear_clicked)

        btn_row = QWidget()
        bl = QHBoxLayout(btn_row)
        bl.setContentsMargins(0, 0, 0, 0)
        bl.addWidget(btn_undo)
        bl.addWidget(btn_clear)

        # --- Layout verticale del pannello ---
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)
        hdr = QLabel("ROI Manager")
        hdr.setStyleSheet("font-weight: bold; margin-bottom: 2px;")
        layout.addWidget(hdr)
        layout.addWidget(QLabel("Modalità di disegno  (R = Rect, S = Square):"))
        layout.addWidget(mode_row)
        layout.addWidget(QLabel("ROI attive:"))
        layout.addWidget(self._list, stretch=1)
        layout.addWidget(btn_row)

    def is_square(self) -> bool:
        """Restituisce True se la modalità quadrato è selezionata."""
        return self._mode_group.checkedId() == 1

    def refresh_list(self, rois: list[ROI]) -> None:
        """
        Aggiorna la lista visuale delle ROI: cancella le voci precedenti
        e ricrea una riga per ogni ROI, colorata con il colore della ROI stessa.
        I colori ROI sono in BGR; Qt vuole RGB, quindi si inverte l'ordine.
        """
        self._list.clear()
        for roi in rois:
            text = f"{roi.name}  ({roi.x},{roi.y})  {roi.width}×{roi.height}"
            item = QListWidgetItem(text)
            b, g, r = roi.color   # BGR → RGB
            item.setForeground(QBrush(QColor(r, g, b)))
            self._list.addItem(item)


# ---------------------------------------------------------------------------
# PlotPanel — grafico interattivo delle serie temporali ROI
# ---------------------------------------------------------------------------

class PlotPanel(QWidget):
    """
    Pannello inferiore con il grafico pyqtgraph dei valori medi per ROI nel tempo.

    Contenuto:
      - Riga di controllo: spinbox "From" e "To" per l'intervallo di frame,
        pulsante "Plot ROIs" (shortcut G)
      - PlotWidget pyqtgraph: una linea per ROI + linea verticale frame corrente

    Note tecniche pyqtgraph:
      - Dopo PlotWidget.clear(), pyqtgraph svuota la LegendItem automaticamente
        (rimuove ogni curva dalla legenda). Ri-aggiungere curve con name=
        ripopola la legenda senza dover chiamare addLegend() di nuovo.
      - La linea verticale (InfiniteLine) viene rimossa da clear() e deve essere
        ri-aggiunta dopo ogni clear().
    """

    plot_requested = Signal()   # pulsante "Plot ROIs" premuto o shortcut G

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._plot_items: list = []   # riferimenti alle curve, per eventuale rimozione selettiva

        # Crea il widget pyqtgraph
        self._plot_widget = pg.PlotWidget()
        self._plot_widget.setBackground("w")
        self._plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self._plot_widget.setLabel("left",   "Mean raw value")
        self._plot_widget.setLabel("bottom", "Frame")
        self._plot_widget.addLegend()

        # Linea verticale tratteggiata che segue il frame corrente del viewer
        self._vline = pg.InfiniteLine(
            angle=90,
            movable=False,
            pen=pg.mkPen(color=(120, 120, 120), width=1,
                         style=Qt.PenStyle.DashLine),
        )
        self._plot_widget.addItem(self._vline)

        # --- Controlli range plot ---
        self._start_spin = QSpinBox()
        self._start_spin.setRange(0, 0)
        self._start_spin.setPrefix("From: ")
        self._start_spin.setMinimumWidth(90)
        self._start_spin.setToolTip("Primo frame incluso nel grafico (incluso)")

        self._end_spin = QSpinBox()
        self._end_spin.setRange(0, 0)
        self._end_spin.setPrefix("To: ")
        self._end_spin.setMinimumWidth(90)
        self._end_spin.setToolTip("Ultimo frame incluso nel grafico (incluso)")

        self._btn_plot = QPushButton("Plot ROIs")
        self._btn_plot.setToolTip(
            "Traccia il valore medio per frame per tutte le ROI attive  (G)"
        )
        self._btn_plot.clicked.connect(self.plot_requested)

        # Riga di controllo orizzontale
        ctrl = QWidget()
        cl = QHBoxLayout(ctrl)
        cl.setContentsMargins(6, 2, 6, 2)
        cl.addWidget(QLabel("Intervallo plot:"))
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

    # --- API pubblica ---

    def reinit(self, n_frames: int) -> None:
        """
        Reimposta il pannello per un nuovo file: aggiorna i range degli spinbox
        e cancella il grafico precedente.
        """
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
        """
        Restituisce (start, end) entrambi inclusi, come impostati dagli spinbox.
        """
        return self._start_spin.value(), self._end_spin.value()

    def update_frame_marker(self, frame: int) -> None:
        """
        Aggiorna la posizione della linea verticale al frame corrente.
        Viene chiamata ad ogni cambio frame durante la riproduzione.
        """
        self._vline.setValue(frame)

    def update_plot(
        self,
        timeseries: dict[str, np.ndarray],
        start_frame: int,
        colors: dict[str, tuple[int, int, int]],
    ) -> None:
        """
        Ridisegna il grafico con le nuove serie temporali.

        Per ogni ROI traccia una linea: asse X = indice frame assoluto,
        asse Y = valore medio grezzo nella regione.
        I colori sono in BGR (come nella ROI), convertiti in RGB per pyqtgraph.
        """
        self._clear()
        for name, ts in timeseries.items():
            x = np.arange(start_frame, start_frame + len(ts))
            b, g, r = colors.get(name, (128, 128, 128))   # default grigio se colore mancante
            item = self._plot_widget.plot(
                x, ts,
                pen=pg.mkPen(color=(r, g, b), width=2),
                name=name,
            )
            self._plot_items.append(item)

    # --- metodo interno ---

    def _clear(self) -> None:
        """
        Rimuove tutte le curve dal grafico. clear() di pyqtgraph rimuove anche
        la InfiniteLine, quindi va ri-aggiunta manualmente.
        """
        self._plot_widget.clear()       # rimuove curve e svuota la legenda
        self._plot_items.clear()
        self._plot_widget.addItem(self._vline)   # re-inserisce la linea frame


# ---------------------------------------------------------------------------
# MainWindow — finestra principale, assembla tutti i pannelli
# ---------------------------------------------------------------------------

class MainWindow(QMainWindow):
    """
    Finestra principale dell'applicazione.

    Responsabilità:
      - Crea e assembla tutti i sotto-pannelli (canvas, timeline, display, ROI, plot)
      - Gestisce il caricamento del file NVF
      - Coordina i segnali tra i pannelli:
          cambio frame → _refresh() → canvas
          parametri display → _refresh() → canvas
          ROI disegnata → aggiorna lista ROI
          richiesta plot → calcola timeseries → aggiorna plot
      - Registra le scorciatoie da tastiera globali
    """

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("NVF Viewer")
        self.resize(1280, 720)

        # --- Stato interno ---
        self._data_cube: np.ndarray | None = None   # cubo 3D int16, shape (N, 512, 640)
        self._n_frames  = 0
        self._cur_frame = 0
        self._glob_low  = 0.0    # percentile 1% sull'intero video (per scala Global)
        self._glob_high = 1.0    # percentile 99% sull'intero video
        self._glob_min  = 0.0    # valore minimo assoluto (per range spinbox manuali)
        self._glob_max  = 1.0    # valore massimo assoluto
        self.roi_manager: ROIManager | None = None

        self._build_toolbar()
        self._build_ui()
        self._build_shortcuts()

    def _build_toolbar(self) -> None:
        """Crea la toolbar con il pulsante 'Open NVF…'."""
        tb = self.addToolBar("Main")
        tb.setMovable(False)
        tb.addAction("Open NVF…", self._on_open)

    def _build_ui(self) -> None:
        """
        Costruisce il layout principale dell'interfaccia:

          QVBoxLayout (root)
          ├─ QSplitter verticale
          │   ├─ QSplitter orizzontale
          │   │   ├─ ImageCanvas (stretch=1)
          │   │   └─ QWidget right (tab ROI + Display, width fissa ~300px)
          │   └─ PlotPanel (stretch=1)
          └─ TimelinePanel (altezza fissa)

        I QSplitter permettono all'utente di ridimensionare le aree
        trascinando il separatore.
        """
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(4)

        h_splitter = QSplitter(Qt.Orientation.Horizontal)

        # Area sinistra: canvas immagine
        self._canvas = ImageCanvas()
        self._canvas.mouse_moved.connect(self._on_mouse_moved)
        self._canvas.roi_finalized.connect(self._on_roi_finalized)
        h_splitter.addWidget(self._canvas)

        # Area destra: tab ROI + Display
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
        h_splitter.setStretchFactor(0, 1)   # canvas si allarga
        h_splitter.setStretchFactor(1, 0)   # pannello dx rimane fisso

        # Splitter verticale: area immagine + grafico
        self._plot_panel = PlotPanel()
        self._plot_panel.plot_requested.connect(self._on_plot_requested)

        v_splitter = QSplitter(Qt.Orientation.Vertical)
        v_splitter.addWidget(h_splitter)
        v_splitter.addWidget(self._plot_panel)
        v_splitter.setStretchFactor(0, 3)   # immagine occupa più spazio
        v_splitter.setStretchFactor(1, 1)
        root.addWidget(v_splitter, stretch=1)

        # Timeline in fondo, fuori dallo splitter (altezza fissa)
        self._timeline = TimelinePanel(1)
        self._timeline.setEnabled(False)   # disabilitata finché non si carica un file
        self._timeline.frame_changed.connect(self._on_frame_changed)
        self._timeline.frame_changed.connect(self._plot_panel.update_frame_marker)
        root.addWidget(self._timeline)

        # Barra di stato con due zone:
        #   _status_frame (sinistra, elastica): info frame corrente
        #   _status_mouse (destra, fissa):      coordinate e valore al cursore
        self._status_frame = QLabel("Apri un file NVF per iniziare.  (Toolbar → Open NVF…)")
        self._status_mouse = QLabel("")
        self._status_mouse.setMinimumWidth(200)
        self.statusBar().addWidget(self._status_frame, 1)
        self.statusBar().addPermanentWidget(self._status_mouse)

    def _build_shortcuts(self) -> None:
        """
        Registra le scorciatoie da tastiera globali della finestra.

        Space — play/pausa
        A / D — frame precedente / successivo
        R / S — modalità ROI rettangolo / quadrato
        Z     — annulla ultima ROI
        C     — cancella tutte le ROI
        G     — mostra grafico ROI
        """
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
        """Imposta la modalità ROI rettangolo (shortcut R)."""
        btn = self._roi_panel._mode_group.button(0)
        if btn:
            btn.setChecked(True)

    def _set_roi_square(self) -> None:
        """Imposta la modalità ROI quadrato (shortcut S)."""
        btn = self._roi_panel._mode_group.button(1)
        if btn:
            btn.setChecked(True)

    # --- caricamento file ---

    def _on_open(self) -> None:
        """
        Apre una finestra di dialogo per selezionare un file NVF,
        poi lo carica con import_nvf(). In caso di errore mostra un messaggio.
        """
        path, _ = QFileDialog.getOpenFileName(self, "Apri file NVF", "", "All files (*.*)")
        if not path:
            return
        try:
            nvf: NVFData = import_nvf(path)
        except Exception as exc:
            QMessageBox.critical(self, "Errore di caricamento", str(exc))
            return
        self._load(nvf)

    def _load(self, nvf: NVFData) -> None:
        """
        Inizializza lo stato interno con i dati del file appena caricato:
          - aggiorna data_cube e n_frames
          - calcola statistiche globali (min, max, percentili 1% e 99%)
          - crea un nuovo ROIManager per le dimensioni del frame
          - reimposta tutti i pannelli al loro stato iniziale
          - lancia il primo refresh dell'immagine
        """
        self._data_cube = nvf.data_cube
        self._n_frames  = nvf.n_frames
        self._cur_frame = 0

        # Statistiche globali calcolate una sola volta sul cubo intero
        self._glob_min  = float(np.min(self._data_cube))
        self._glob_max  = float(np.max(self._data_cube))
        self._glob_low  = float(np.percentile(self._data_cube, 1.0))
        self._glob_high = float(np.percentile(self._data_cube, 99.0))
        if self._glob_high <= self._glob_low:
            self._glob_high = self._glob_low + 1.0

        # Crea il gestore ROI per le dimensioni di questo file
        _, fh, fw = self._data_cube.shape
        self.roi_manager = ROIManager(fw, fh)
        self._canvas.roi_manager = self.roi_manager

        # Reimposta i pannelli
        self._display_panel.reinit(self._glob_min, self._glob_max)
        self._timeline.reinit(self._n_frames)
        self._timeline.setEnabled(True)
        self._plot_panel.reinit(self._n_frames)
        self.setWindowTitle(f"NVF Viewer — {nvf.file_path.name}")
        self._refresh()

    # --- rendering ---

    def _refresh(self) -> None:
        """
        Riesegue la pipeline display per il frame corrente e aggiorna il canvas.

        Viene chiamata ogni volta che:
          - cambia il frame corrente (_on_frame_changed)
          - cambiano i parametri display (params_changed di DisplayPanel)

        Legge i parametri dal DisplayPanel, chiama prepare_frame_for_display()
        dal modulo display_pipeline, converte il risultato in BGR e lo passa
        al canvas. Aggiorna anche la barra di stato con le informazioni
        sulla finestra di visualizzazione.
        """
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
        # Converte grayscale → BGR in modo che ROIManager possa disegnare rettangoli colorati
        bgr = cv2.cvtColor(uint8, cv2.COLOR_GRAY2BGR)
        self._canvas.set_base_frame(bgr, raw)

        # Aggiorna lista ROI e barra di stato
        self._roi_panel.refresh_list(self.roi_manager.rois if self.roi_manager else [])
        n_rois = len(self.roi_manager.rois) if self.roi_manager else 0
        self._status_frame.setText(
            f"F {self._cur_frame + 1} / {self._n_frames}"
            f"  |  win: {used_low:.1f} → {used_high:.1f}"
            f"  |  ROI: {n_rois}"
        )

    # --- slot ---

    def _on_frame_changed(self, idx: int) -> None:
        """Slot collegato a TimelinePanel.frame_changed: aggiorna il frame corrente."""
        self._cur_frame = idx
        self._refresh()

    def _on_roi_finalized(self) -> None:
        """
        Dopo che l'utente completa una ROI (rilascio mouse), aggiorna la lista
        nel ROIPanel e forza un repaint del canvas per mostrarla subito.
        Non serve rieseguire la pipeline: il canvas ridisegna l'overlay in paintEvent.
        """
        if self.roi_manager:
            self._roi_panel.refresh_list(self.roi_manager.rois)
        self._canvas.update()

    def _on_mouse_moved(self, ix: int, iy: int, val: int) -> None:
        """Aggiorna la zona destra della barra di stato con posizione e valore grezzo."""
        self._status_mouse.setText(f"x={ix}  y={iy}  raw={val}")

    def _on_roi_mode_changed(self) -> None:
        """Propaga la modalità di disegno (quadrato/rettangolo) al ROIManager."""
        if self.roi_manager:
            self.roi_manager.square_mode = self._roi_panel.is_square()

    def _on_plot_requested(self) -> None:
        """
        Calcola le serie temporali per le ROI attive nell'intervallo selezionato
        e aggiorna il grafico nel PlotPanel. Non fa nulla se non c'è un file
        caricato, non ci sono ROI o l'intervallo è invalido.
        """
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
        """Rimuove l'ultima ROI e aggiorna lista + canvas."""
        if self.roi_manager:
            self.roi_manager.remove_last()
            self._roi_panel.refresh_list(self.roi_manager.rois)
            self._canvas.update()

    def _on_clear_rois(self) -> None:
        """Cancella tutte le ROI e aggiorna lista + canvas."""
        if self.roi_manager:
            self.roi_manager.clear()
            self._roi_panel.refresh_list([])
            self._canvas.update()


# ---------------------------------------------------------------------------
# Punto di ingresso
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Crea l'applicazione Qt, la finestra principale e avvia il loop degli eventi.
    sys.exit() restituisce il codice di uscita di Qt al sistema operativo.
    """
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
