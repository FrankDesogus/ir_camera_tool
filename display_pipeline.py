"""
display_pipeline.py — pipeline di conversione da valori grezzi a immagine visualizzabile.

Questo modulo è il cuore del rendering termografico. Prende i valori int16 grezzi
letti dal file NVF e li trasforma in un'immagine uint8 (0–255) pronta per essere
mostrata a schermo.

Pipeline applicata in ordine:
  1. clipping       — taglia i valori fuori dalla finestra [low, high] e sottrae 'low'
  2. apply_display_transform — applica la trasformazione non-lineare scelta (sqrt, log, ecc.)
  3. normalize      — porta i valori nell'intervallo [0, 1]
  4. to_uint8       — converte in interi 0–255

Modalità di trasformazione (transform_mode):
  0 = Linear  — nessuna trasformazione, utile quando il contrasto è già buono
  1 = Sqrt    — radice quadrata, schiaccia i picchi e alza i dettagli nelle zone scure
  2 = Log     — logaritmo, molto efficace su segnali con range dinamico molto ampio
  3 = Asinh   — arcosenho iperbolico, simile al log ma funziona anche su valori negativi

Modalità di scala (scale_mode):
  0 = Auto   — la finestra [low, high] è calcolata percentile-per-frame (adattiva)
  1 = Global — usa percentili calcolati sull'intero cubo 3D al caricamento del file
  2 = Manual — l'utente imposta manualmente low e high tramite i controlli della GUI
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Funzioni elementari della pipeline
# ---------------------------------------------------------------------------

def percentile_window(frame: np.ndarray, p_min: float, p_max: float) -> tuple[float, float]:
    """
    Calcola la finestra di visualizzazione [low, high] per un singolo frame
    usando i percentili p_min e p_max dei suoi valori.

    I percentili ignorano i pixel estremi (ad es. pixel caldi o freddi anomali),
    producendo una finestra più robusta rispetto al min/max assoluto.

    Parametri:
      frame — array 2D dei valori grezzi del frame corrente
      p_min — percentile inferiore (es. 1.0 = 1°)
      p_max — percentile superiore (es. 99.0 = 99°)

    Restituisce (low, high) con high > low garantito.
    """
    # Assicura che p_max sia sempre maggiore di p_min per evitare una finestra nulla
    if p_min >= p_max:
        p_max = min(p_min + 0.1, 100.0)

    low  = float(np.percentile(frame, p_min))
    high = float(np.percentile(frame, p_max))

    # Caso degenere: tutti i pixel hanno lo stesso valore → aggiunge 1 per evitare /0
    if high <= low:
        high = low + 1.0

    return low, high


def clipping(frame: np.ndarray, low: float, high: float) -> np.ndarray:
    """
    Taglia i valori del frame alla finestra [low, high] e sposta l'origine.

    Dopo questa operazione:
      - i valori sotto 'low' diventano 0
      - i valori sopra 'high' diventano (high - low)
      - i valori interni sono traslati così che 'low' corrisponda a 0

    Restituisce un array float32. Non normalizza in [0, 1]: questo è compito
    della funzione normalize() successiva.
    """
    frame_f = frame.astype(np.float32)
    clipped = frame_f - low           # sposta: low → 0
    lim_sup = high - low              # il massimo possibile dopo lo spostamento
    return np.clip(clipped, 0, lim_sup)   # taglia i valori fuori intervallo


def apply_display_transform(frame: np.ndarray, mode: int, gamma: float = 1.0) -> np.ndarray:
    """
    Applica la trasformazione non-lineare per migliorare il contrasto visivo.

    Lavora su valori già clippati (NON normalizzati in [0,1]).
    Le trasformazioni non-lineari schiacciano i valori alti e amplificano i bassi,
    rendendo visibili i dettagli nelle zone più scure dell'immagine.

    Modalità disponibili:
      mode=0  Linear — y = x               (nessuna modifica)
      mode=1  Sqrt   — y = √x              (attenuazione moderata dei picchi)
      mode=2  Log    — y = log(1+x)        (attenuazione forte, ottima per range ampi)
      mode=3  Asinh  — y = arcsinh(x)      (simile al log, funziona con valori negativi)

    Dopo la trasformazione viene applicato il gamma: y = y^gamma.
      gamma < 1 → schiarisce l'immagine (alza i mezzitoni)
      gamma > 1 → scurisce l'immagine (abbassa i mezzitoni)
      gamma = 1 → nessun effetto aggiuntivo
    """
    x = frame.astype(np.float32)

    if mode == 0:
        y = x
    elif mode == 1:
        y = np.sqrt(x)
    elif mode == 2:
        y = np.log1p(x)      # log(1+x): evita log(0) quando x=0
    elif mode == 3:
        y = np.arcsinh(x)    # comportamento logaritmico per x grandi, lineare vicino a 0
    else:
        y = x

    # Applica correzione gamma (esponenziale sui valori trasformati)
    gamma = max(gamma, 1e-6)   # evita potenza con esponente 0
    y = np.power(y, gamma)

    return y


def normalize(frame: np.ndarray) -> np.ndarray:
    """
    Normalizza i valori nell'intervallo [0.0, 1.0] dividendo per il massimo.

    Viene chiamata DOPO apply_display_transform, quando i valori non sono
    più nella scala originale. La divisione per il massimo garantisce che
    il pixel più luminoso diventi 1.0, indipendentemente dalla trasformazione
    applicata in precedenza.

    Se il frame è tutto zero (massimo = 0), restituisce il frame invariato
    per evitare la divisione per zero.
    """
    frame_f = frame.astype(np.float32)
    max_val = frame_f.max()
    if max_val == 0:
        return frame_f
    return frame_f / max_val


def to_uint8(norm_frame: np.ndarray) -> np.ndarray:
    """
    Converte un frame normalizzato in [0.0, 1.0] in un array uint8 in [0, 255].

    Moltiplica per 255, arrotonda all'intero più vicino, taglia eventuali
    valori fuori range per sicurezza, e converte il tipo di dato.
    Il risultato è compatibile con OpenCV e Qt per la visualizzazione a schermo.
    """
    return (norm_frame * 255.0).round().clip(0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Funzioni di mappatura indice → costante (usate dai controlli UI)
# ---------------------------------------------------------------------------

def mode_from_value(value: int) -> int:
    """
    Mappa un indice intero (da trackbar o radio button) alla costante di modalità
    di trasformazione (0=Linear, 1=Sqrt, 2=Log, 3=Asinh).

    Valori fuori range vengono clampati al più vicino estremo valido.
    """
    if value <= 0:
        return 0
    if value == 1:
        return 1
    if value == 2:
        return 2
    return 3


def scale_mode_from_value(value: int) -> int:
    """
    Mappa un indice intero alla costante di modalità di scala
    (0=Auto, 1=Global, 2=Manual).
    """
    if value <= 0:
        return 0
    if value == 1:
        return 1
    return 2


# ---------------------------------------------------------------------------
# Funzione principale: applica l'intera pipeline in un unico passaggio
# ---------------------------------------------------------------------------

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
    Esegue la pipeline completa di rendering su un singolo frame grezzo.

    Passaggi (nell'ordine):
      1. Sceglie la finestra [low, high] in base alla modalità di scala:
           scale_mode=0 (Auto)   → percentili calcolati su questo frame
           scale_mode=1 (Global) → percentili pre-calcolati sull'intero video
           scale_mode=2 (Manual) → valori impostati dall'utente
      2. clipping(raw_frame, low, high)
      3. apply_display_transform(clipped, transform_mode, gamma)
      4. normalize(transformed)
      5. to_uint8(norm)

    Parametri:
      raw_frame     — frame grezzo int16, shape (H, W)
      transform_mode— 0=Lin, 1=Sqrt, 2=Log, 3=Asinh
      scale_mode    — 0=Auto, 1=Global, 2=Manual
      p_min, p_max  — percentili per la modalità Auto (es. 1.0, 99.0)
      global_low/high — finestra pre-calcolata per la modalità Global
      manual_low/high — finestra impostata dall'utente per la modalità Manual
      gamma         — esponente di correzione del gamma

    Restituisce:
      (display_uint8, low_usato, high_usato)
        display_uint8 — array uint8 shape (H, W) pronto per la visualizzazione
        low_usato     — valore grezzo che corrisponde al nero (0)
        high_usato    — valore grezzo che corrisponde al bianco (255)
    """
    # Passo 1: seleziona la finestra di visualizzazione in base alla modalità
    if scale_mode == 0:   # Auto: adattiva per ogni frame
        low, high = percentile_window(raw_frame, p_min, p_max)
    elif scale_mode == 1: # Global: calcolata una volta sola sull'intero video
        low, high = global_low, global_high
    else:                 # Manual: impostata dall'utente
        low, high = manual_low, manual_high
        if high <= low:
            high = low + 1.0   # evita finestra degenerata

    # Passi 2–5: applicazione sequenziale della pipeline
    clipped     = clipping(raw_frame, low, high)
    transformed = apply_display_transform(clipped, transform_mode, gamma=gamma)
    norm        = normalize(transformed)
    display     = to_uint8(norm)

    return display, low, high
