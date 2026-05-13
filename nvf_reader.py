"""
nvf_reader.py — lettura di file NVF termografici.

Struttura di un file NVF:
  [header: 3072 int16]
  per ogni frame:
    [blocco nascosto: 512 int16]   ← metadati interni, non usati dal viewer
    [dati frame: 640×512 int16]    ← temperatura grezza di ogni pixel

Questa struttura è ricavata dal codice IDL originale della camera.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from config import (
    FILE_DTYPE,
    FRAME_WIDTH,
    FRAME_HEIGHT,
    FRAME_ELEMENTS,
    HIDDEN_BLOCK_ELEMENTS,
    HEADER_ELEMENTS,
    HEADER_BYTES,
    RECORD_BYTES,
)


# ---------------------------------------------------------------------------
# Struttura dati restituita al chiamante
# ---------------------------------------------------------------------------

@dataclass
class NVFData:
    """
    Contenitore con tutti i dati letti dal file NVF.

    Campi:
      header       — array grezzo dell'intestazione (3072 int16).
      hidden_blocks— blocchi nascosti di ogni frame (n_frames × 512 int16).
                     Sono metadati interni della camera, non servono al viewer.
      data_cube    — cubo 3D dei valori grezzi di temperatura:
                     shape = (n_frames, 512, 640), dtype int16.
                     L'asse 0 è il tempo (frame), l'asse 1 le righe (y), l'asse 2 le colonne (x).
      n_frames     — numero totale di frame nel file.
      file_path    — percorso assoluto del file letto.
    """
    header: np.ndarray
    hidden_blocks: np.ndarray
    data_cube: np.ndarray   # shape: (n_frames, 512, 640)
    n_frames: int
    file_path: Path


# ---------------------------------------------------------------------------
# Funzioni interne di parsing
# ---------------------------------------------------------------------------

def estimate_frame_count(file_size_bytes: int) -> int:
    """
    Calcola quanti frame ci sono nel file dalla sua dimensione in byte.

    Logica:
      dimensione totale  =  header  +  N × record
      N = (dimensione totale − header_bytes) / record_bytes

    Lancia ValueError se il file è troppo piccolo o ha una dimensione non divisibile.
    """
    # Quanto spazio rimane dopo l'header
    payload_bytes = file_size_bytes - HEADER_BYTES

    if payload_bytes < 0:
        raise ValueError(
            f"File troppo piccolo: {file_size_bytes} byte, "
            f"header richiesto: {HEADER_BYTES} byte."
        )

    # Il payload deve essere un multiplo esatto della dimensione di un record
    if payload_bytes % RECORD_BYTES != 0:
        raise ValueError(
            "La dimensione del file non è compatibile con la struttura NVF attesa.\n"
            f"payload_bytes = {payload_bytes}\n"
            f"record_bytes  = {RECORD_BYTES}\n"
            f"resto         = {payload_bytes % RECORD_BYTES}"
        )

    return payload_bytes // RECORD_BYTES


def read_nvf(file_obj, filename: str | Path) -> NVFData:
    """
    Legge il contenuto binario di un file NVF già aperto e lo struttura.

    Passaggi:
      1. Calcola il numero di frame dalla dimensione del file.
      2. Legge tutti i byte come array int16 piatto (raw).
      3. Verifica che il numero di elementi corrisponda a quanto atteso.
      4. Estrae header, blocchi nascosti e frame dati.
      5. Rimodella i dati frame in un cubo (n_frames, 512, 640).
    """
    file_path = Path(filename)

    # Legge la dimensione del file spostandosi alla fine (seek 0 = inizio, 2 = fine)
    file_obj.seek(0, 2)
    file_size = file_obj.tell()
    file_obj.seek(0)   # ritorna all'inizio per la lettura

    n_frames = estimate_frame_count(file_size)

    # Legge l'intero file come sequenza continua di int16
    raw = np.fromfile(file_obj, dtype=FILE_DTYPE)

    # Verifica di integrità: il numero totale di elementi deve coincidere esattamente
    expected_total_elements = HEADER_ELEMENTS + n_frames * (
        HIDDEN_BLOCK_ELEMENTS + FRAME_ELEMENTS
    )
    if raw.size != expected_total_elements:
        raise ValueError(
            "Numero totale di elementi nel file diverso da quello atteso.\n"
            f"Elementi letti   = {raw.size}\n"
            f"Elementi attesi  = {expected_total_elements}"
        )

    # --- Estrazione delle parti del file ------------------------------------

    # I primi HEADER_ELEMENTS int16 sono l'intestazione globale del file
    header = raw[:HEADER_ELEMENTS]

    # Il resto del file è il payload: una sequenza di record (uno per frame)
    payload = raw[HEADER_ELEMENTS:]

    # Ogni record ha dimensione fissa: (HIDDEN_BLOCK_ELEMENTS + FRAME_ELEMENTS) int16
    records = payload.reshape(n_frames, HIDDEN_BLOCK_ELEMENTS + FRAME_ELEMENTS)

    # Prima parte di ogni record: blocco nascosto (512 int16 di metadati interni)
    hidden_blocks = records[:, :HIDDEN_BLOCK_ELEMENTS]

    # Seconda parte di ogni record: dati grezzi del frame (640×512 int16)
    frame_data = records[:, HIDDEN_BLOCK_ELEMENTS:]

    # Ridimensiona i frame in (n_frames, altezza=512, larghezza=640)
    # Così data_cube[i] è l'immagine termica del frame i-esimo
    data_cube = frame_data.reshape(n_frames, FRAME_HEIGHT, FRAME_WIDTH)

    return NVFData(
        header=header,
        hidden_blocks=hidden_blocks,
        data_cube=data_cube,
        n_frames=n_frames,
        file_path=file_path,
    )


# ---------------------------------------------------------------------------
# Funzione pubblica principale
# ---------------------------------------------------------------------------

def import_nvf(filename: str | Path) -> NVFData:
    """
    Punto di ingresso principale: apre il file NVF al percorso indicato e
    restituisce un oggetto NVFData con tutti i dati strutturati.

    Il file viene aperto in modalità binaria ('rb') e chiuso automaticamente
    al termine grazie al blocco 'with'.

    Lancia ValueError se il file è malformato o ha dimensioni incompatibili.
    """
    path = Path(filename)
    with open(path, "rb") as f:
        return read_nvf(f, path)


# ---------------------------------------------------------------------------
# Esecuzione diretta (test rapido da riga di comando)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Chiede il percorso del file da riga di comando o input interattivo
    import sys
    if len(sys.argv) > 1:
        nvf = import_nvf(sys.argv[1])
    else:
        path_input = input("Percorso file NVF: ").strip()
        nvf = import_nvf(path_input)
    print(f"Letti {nvf.n_frames} frame da '{nvf.file_path}'")
    print(f"Cubo dati: shape={nvf.data_cube.shape}, dtype={nvf.data_cube.dtype}")
    print(f"Min={nvf.data_cube.min()}, Max={nvf.data_cube.max()}")
