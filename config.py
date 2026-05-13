"""
config.py — costanti strutturali per il formato file NVF.

Il formato NVF è il formato binario proprietario della camera termica.
Tutte le costanti qui definite sono ricavate dal codice IDL originale
fornito con la camera. Le dimensioni sono in NUMERO DI ELEMENTI (int16),
non in byte, salvo dove esplicitamente indicato.

Struttura del file NVF:
  ┌──────────────────────────────────────────────────────┐
  │  Header globale:  3072 × int16  (6144 byte)          │
  ├──────────────────────────────────────────────────────┤
  │  Record 0:                                           │
  │    Blocco nascosto:  512 × int16                     │
  │    Dati frame:       640 × 512 × int16               │
  ├──────────────────────────────────────────────────────┤
  │  Record 1: (stessa struttura di Record 0)            │
  │  ...                                                 │
  └──────────────────────────────────────────────────────┘

Nota sul tipo di dato:
  Il codice IDL usa 'intarr', che corrisponde a interi con segno a 16 bit
  (numpy.int16). I valori di temperatura sono codificati in questo formato.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Dimensioni del fotogramma (in pixel)
# ---------------------------------------------------------------------------

# Larghezza dell'immagine termica in pixel (asse x / colonne)
FRAME_WIDTH = 640

# Altezza dell'immagine termica in pixel (asse y / righe)
FRAME_HEIGHT = 512


# ---------------------------------------------------------------------------
# Struttura del blocco nascosto
# ---------------------------------------------------------------------------

# Ogni record contiene un "blocco nascosto" di 512 interi prima del frame vero e proprio.
# Nel codice IDL originale: READU, 1, s  con  s = intarr(512)
# Questi dati sono metadati interni della camera e non vengono usati dal viewer.
HIDDEN_BLOCK_ELEMENTS = 512


# ---------------------------------------------------------------------------
# Struttura dell'header globale
# ---------------------------------------------------------------------------

# L'header del file è calcolato così (dal codice IDL):
#   h = intarr(3511 + 72 + 1 - 512)
# Il risultato è 3072 elementi int16.
# Contiene informazioni globali sul file (data, configurazione camera, ecc.)
# Non ci serve per il rendering, ma va saltato durante la lettura.
HEADER_ELEMENTS = 3511 + 72 + 1 - 512   # = 3072 elementi int16


# ---------------------------------------------------------------------------
# Tipo di dato e dimensioni in byte
# ---------------------------------------------------------------------------

# Tipo numpy usato per leggere il file: int16 a 16 bit con segno.
# Corrisponde a 'intarr' del codice IDL.
FILE_DTYPE = np.int16

# Quanti byte occupa un singolo elemento del tipo FILE_DTYPE (di solito 2)
BYTES_PER_ELEMENT = np.dtype(FILE_DTYPE).itemsize

# Dimensione dell'header in byte (usata per calcolare il numero di frame)
HEADER_BYTES = HEADER_ELEMENTS * BYTES_PER_ELEMENT


# ---------------------------------------------------------------------------
# Dimensioni di un frame (in elementi e in byte)
# ---------------------------------------------------------------------------

# Numero totale di pixel in un frame: 640 colonne × 512 righe
FRAME_ELEMENTS = FRAME_WIDTH * FRAME_HEIGHT

# Dimensione di un frame in byte (solo i dati pixel, senza blocco nascosto)
FRAME_BYTES = FRAME_ELEMENTS * BYTES_PER_ELEMENT


# ---------------------------------------------------------------------------
# Dimensioni di un record (blocco nascosto + frame)
# ---------------------------------------------------------------------------

# Un record è la struttura letta ad ogni iterazione del loop IDL:
#   READU, 1, s     ← blocco nascosto (512 elementi)
#   READU, 1, frame ← dati del frame (640×512 elementi)
RECORD_ELEMENTS = HIDDEN_BLOCK_ELEMENTS + FRAME_ELEMENTS

# Dimensione di un record in byte (usata per verificare la divisibilità del file)
RECORD_BYTES = RECORD_ELEMENTS * BYTES_PER_ELEMENT
