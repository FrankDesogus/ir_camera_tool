"""
roi.py — gestione delle Region Of Interest (ROI) termografiche.

Una ROI è un rettangolo disegnato dall'utente sull'immagine termica.
Per ogni ROI il programma calcola la media dei valori grezzi di temperatura
pixel-per-pixel, frame-per-frame, producendo una serie temporale.

Questo modulo gestisce tre aspetti distinti:
  1. Struttura dati ROI (dataclass ROI)
  2. Interazione mouse e disegno (classe ROIManager)
  3. Calcolo e visualizzazione delle serie temporali (funzioni standalone)
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Palette di colori per le ROI
# ---------------------------------------------------------------------------

# I colori sono in formato BGR (Blue, Green, Red) come richiesto da OpenCV.
# Vengono assegnati in sequenza ciclica alle ROI man mano che vengono disegnate.
_COLORS: list[tuple[int, int, int]] = [
    (0,   0,   255),   # rosso
    (0,   255, 0  ),   # verde
    (255, 0,   0  ),   # blu
    (0,   255, 255),   # giallo
    (255, 0,   255),   # magenta
    (255, 255, 0  ),   # ciano
    (0,   128, 255),   # arancione
    (128, 0,   255),   # viola
]

# Riferimento all'ultima figura matplotlib aperta per i grafici ROI.
# Viene chiusa prima di aprirne una nuova, così non si accumulano finestre.
_last_roi_figure: plt.Figure | None = None


# ---------------------------------------------------------------------------
# Struttura dati di una singola ROI
# ---------------------------------------------------------------------------

@dataclass
class ROI:
    """
    Rappresenta una singola Region Of Interest rettangolare.

    Campi:
      name   — etichetta testuale (es. "ROI 1"), mostrata sull'immagine.
      x      — coordinata orizzontale del vertice in alto a sinistra (pixel).
      y      — coordinata verticale del vertice in alto a sinistra (pixel).
      width  — larghezza del rettangolo in pixel.
      height — altezza del rettangolo in pixel.
      color  — colore BGR per il disegno su frame OpenCV.
    """
    name: str
    x: int
    y: int
    width: int
    height: int
    color: tuple[int, int, int]   # BGR


# ---------------------------------------------------------------------------
# Gestore delle ROI: interazione mouse e disegno
# ---------------------------------------------------------------------------

class ROIManager:
    """
    Gestisce il ciclo di vita delle ROI: creazione tramite drag del mouse,
    annullamento dell'ultima, cancellazione totale e rendering sull'immagine.

    Viene usata sia dal viewer OpenCV (tramite setMouseCallback) sia dalla GUI
    PySide6 (il canvas traduce gli eventi Qt negli stessi costanti cv2).

    Attributi pubblici:
      rois        — lista delle ROI confermate (in ordine di creazione).
      square_mode — se True il drag produce sempre un quadrato; se False un rettangolo.
    """

    def __init__(self, frame_width: int, frame_height: int) -> None:
        """
        Inizializza il gestore con le dimensioni del frame.
        Le dimensioni servono a clampare le ROI dentro i bordi dell'immagine.
        """
        self._fw = frame_width    # larghezza frame in pixel
        self._fh = frame_height   # altezza frame in pixel
        self.rois: list[ROI] = []
        self._next_id: int = 1            # contatore per assegnare nomi univoci
        self._drawing: bool = False       # True mentre l'utente sta trascinando
        self._start_pt: tuple[int, int] | None = None    # punto di inizio drag
        self._current_pt: tuple[int, int] | None = None  # posizione corrente del mouse
        self.square_mode: bool = True     # default: quadrato

    # ------------------------------------------------------------------
    # Callback mouse — compatibile con cv2.setMouseCallback
    # ------------------------------------------------------------------

    def handle_mouse(self, event: int, x: int, y: int, flags: int, param) -> None:
        """
        Riceve gli eventi del mouse e aggiorna lo stato interno.

        Ciclo di vita di una ROI:
          LBUTTONDOWN → inizia il drag, registra il punto di partenza
          MOUSEMOVE   → aggiorna il punto corrente (per l'anteprima)
          LBUTTONUP   → finalizza la ROI e la aggiunge alla lista
        """
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
    # Geometria interna
    # ------------------------------------------------------------------

    def _compute_roi(
        self, end_x: int, end_y: int
    ) -> tuple[int, int, int, int] | None:
        """
        Calcola la geometria del rettangolo dal punto di inizio al punto finale.

        Gestisce tutti e quattro i possibili versi di drag (↗ ↘ ↙ ↖).
        Il vertice in alto a sinistra (top-left) viene calcolato in base al segno
        della differenza tra punto finale e punto iniziale.

        In modalità quadrato: width = height = min(|dx|, |dy|) (almeno 1 px).
        In modalità rettangolo: width = |dx|, height = |dy| (almeno 1 px ciascuno).

        Il rettangolo viene poi clampato dentro i limiti del frame.
        Restituisce (tl_x, tl_y, width, height) oppure None se manca il punto iniziale.
        """
        if self._start_pt is None:
            return None

        sx, sy = self._start_pt
        dx = end_x - sx
        dy = end_y - sy

        if self.square_mode:
            # Lato = il più piccolo dei due spostamenti → quadrato inscritto nel drag
            side = max(1, min(abs(dx), abs(dy)))
            w, h = side, side
        else:
            # Rettangolo libero, minimo 1 pixel per lato
            w = max(1, abs(dx))
            h = max(1, abs(dy))

        # Il top-left dipende dalla direzione del drag
        tl_x = sx if dx >= 0 else sx - w
        tl_y = sy if dy >= 0 else sy - h

        # Clampa il rettangolo dentro i limiti dell'immagine
        tl_x = max(0, min(tl_x, self._fw - w))
        tl_y = max(0, min(tl_y, self._fh - h))

        return tl_x, tl_y, w, h

    def _finalize_roi(self, end_x: int, end_y: int) -> None:
        """
        Crea l'oggetto ROI dal drag completato e lo aggiunge alla lista.
        Il colore viene scelto in modo ciclico dalla palette _COLORS.
        """
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
        """Rimuove l'ultima ROI aggiunta (undo)."""
        if self.rois:
            self.rois.pop()

    def clear(self) -> None:
        """Cancella tutte le ROI presenti."""
        self.rois.clear()

    def draw_on_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Disegna tutte le ROI confermate e l'anteprima del drag in corso
        su una COPIA del frame BGR. Il frame originale non viene modificato.

        Per ogni ROI confermata:
          - disegna il rettangolo colorato (spessore 2)
          - scrive il nome della ROI in alto a sinistra del rettangolo

        Se l'utente sta trascinando, mostra un rettangolo in anteprima
        (spessore 1) con il colore che avrebbe la prossima ROI.
        """
        out = frame.copy()

        # Disegna le ROI già salvate
        for roi in self.rois:
            pt1 = (roi.x, roi.y)
            pt2 = (roi.x + roi.width, roi.y + roi.height)
            cv2.rectangle(out, pt1, pt2, roi.color, 2)
            cv2.putText(
                out,
                roi.name,
                (roi.x + 4, roi.y + 16),   # piccolo offset per non sovrapporsi al bordo
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                roi.color,
                1,
                cv2.LINE_AA,
            )

        # Anteprima del drag in corso (rettangolo fantasma)
        if self._drawing and self._start_pt and self._current_pt:
            result = self._compute_roi(*self._current_pt)
            if result is not None:
                px, py, pw, ph = result
                preview_color = _COLORS[(self._next_id - 1) % len(_COLORS)]
                cv2.rectangle(out, (px, py), (px + pw, py + ph), preview_color, 1)

        return out


# ---------------------------------------------------------------------------
# Calcolo serie temporali sui valori raw originali
# ---------------------------------------------------------------------------

def calculate_roi_timeseries(
    data_cube: np.ndarray,
    rois: list[ROI],
    start_frame: int = 0,
    end_frame: int | None = None,
) -> dict[str, np.ndarray]:
    """
    Calcola la serie temporale della media dei valori grezzi per ogni ROI.

    Per ogni ROI e per ogni frame nell'intervallo [start_frame, end_frame),
    estrae la regione rettangolare dal cubo 3D e calcola la media di tutti
    i pixel dentro quella regione. Il risultato è un array 1D di lunghezza
    (end_frame - start_frame): un valore medio per frame.

    Nota importante: lavora sui valori GREZZI (int16), non sui valori
    trasformati per la visualizzazione. Questo permette di confrontare
    misure fisiche reali, indipendenti dai parametri di display.

    Parametri:
      data_cube   — cubo 3D shape (n_frames, H, W), dtype int16
      rois        — lista delle ROI attive
      start_frame — primo frame da includere (incluso, indice 0-based)
      end_frame   — frame finale ESCLUSO; None = fino all'ultimo frame

    Restituisce:
      dict {roi.name: np.ndarray float64 shape (n_frames_selezionati,)}
      Le ROI completamente fuori dai limiti dell'immagine vengono silenziosamente saltate.
    """
    if not rois:
        return {}

    n_frames, height, width = data_cube.shape

    # Clampa gli indici di frame ai limiti validi
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
        # Calcola i limiti della regione clampati dentro il frame
        x1 = max(0, roi.x)
        y1 = max(0, roi.y)
        x2 = min(width,  roi.x + roi.width)
        y2 = min(height, roi.y + roi.height)

        # ROI completamente fuori dai bordi: niente da calcolare
        if x2 <= x1 or y2 <= y1:
            continue

        # Estrae la sub-regione per tutti i frame dell'intervallo in un solo slice 3D
        # Forma di 'region': (n_frame_selezionati, y2-y1, x2-x1)
        # .mean(axis=(1,2)) calcola la media su riga e colonna → un valore per frame
        # int16 viene automaticamente promosso a float64 da NumPy durante .mean()
        region = data_cube[start_frame:end_frame, y1:y2, x1:x2]
        result[roi.name] = region.mean(axis=(1, 2))

    return result


# ---------------------------------------------------------------------------
# Visualizzazione grafico comparativo delle serie temporali
# ---------------------------------------------------------------------------

def plot_roi_timeseries(
    timeseries: dict[str, np.ndarray],
    start_frame: int = 0,
    title: str = "ROI raw mean over time",
    colors: dict[str, tuple[int, int, int]] | None = None,
    current_frame: int | None = None,
) -> None:
    """
    Mostra un grafico matplotlib con una linea per ogni ROI attiva.

    L'asse X è il numero di frame (assoluto, non relativo all'intervallo).
    L'asse Y è il valore medio grezzo di temperatura nella ROI.

    Parametri:
      timeseries    — dict {nome_roi: array float64} da calculate_roi_timeseries
      start_frame   — indice del primo frame nell'intervallo (per l'asse X)
      title         — titolo del grafico
      colors        — opzionale, dict {nome_roi: (B,G,R)} per usare gli stessi
                      colori del viewer. Se assente usa il color cycle di matplotlib.
                      Nota: i colori ROI sono in BGR (OpenCV); per matplotlib servono RGB.
      current_frame — se fornito, disegna una linea verticale tratteggiata grigia
                      che indica il frame attualmente visualizzato nel viewer.

    La figura precedente viene chiusa automaticamente prima di aprirne una nuova
    per non accumulare finestre matplotlib aperte.
    """
    if not timeseries:
        return

    # Chiude la figura precedente se ancora aperta
    global _last_roi_figure
    if _last_roi_figure is not None:
        plt.close(_last_roi_figure)

    fig, ax = plt.subplots(figsize=(10, 5))
    _last_roi_figure = fig

    for name, ts in timeseries.items():
        # L'asse X parte da start_frame, non da 0, così le coordinate corrispondono
        # all'indice di frame del video originale
        x = np.arange(start_frame, start_frame + len(ts))

        if colors and name in colors:
            # Converte BGR (OpenCV) → RGB normalizzato in [0,1] (matplotlib)
            b, g, r = colors[name]
            line_color = (r / 255.0, g / 255.0, b / 255.0)
            ax.plot(x, ts, label=name, color=line_color)
        else:
            ax.plot(x, ts, label=name)

    # Linea verticale che indica il frame corrente nel viewer
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

    # block=False → il grafico appare senza bloccare il viewer
    plt.show(block=False)
    plt.pause(0.001)   # necessario per aggiornare il rendering su alcune piattaforme
