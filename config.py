"""
Configurazione e costanti per il parser NVF.

Nota importante:
- Nel file IDL, INTARR crea array di interi a 16 bit.
- Le dimensioni dell'header e dei blocchi nel codice IDL sono espresse
  in NUMERO DI ELEMENTI, non in byte.
"""

from __future__ import annotations

import numpy as np

# ----------------------------------------------------------------------
# Costanti strutturali ricavate dal codice IDL
# ----------------------------------------------------------------------

FRAME_WIDTH = 640
FRAME_HEIGHT = 512

# Nel loop IDL: READU,1, s  con s=intarr(512)
HIDDEN_BLOCK_ELEMENTS = 512

# Dal codice IDL:
# h = intarr(3511 + 72 + 1 - 512)
HEADER_ELEMENTS = 3511 + 72 + 1 - 512  # = 3072 elementi, NON byte

# Tipo dati:
# IDL usa intarr -> int16 signed.
# Tuttavia il file potrebbe contenere dati che in pratica andrebbero
# interpretati come uint16. Lo lasciamo configurabile.
FILE_DTYPE = np.int16

# Bytes per elemento, coerente col dtype
BYTES_PER_ELEMENT = np.dtype(FILE_DTYPE).itemsize

# Header espresso anche in byte, ma sempre derivato dagli elementi
HEADER_BYTES = HEADER_ELEMENTS * BYTES_PER_ELEMENT

# Un frame contiene 640 x 512 elementi
FRAME_ELEMENTS = FRAME_WIDTH * FRAME_HEIGHT
FRAME_BYTES = FRAME_ELEMENTS * BYTES_PER_ELEMENT

# Un record del loop contiene:
# - 512 elementi "hidden"
# - 640*512 elementi di frame
RECORD_ELEMENTS = HIDDEN_BLOCK_ELEMENTS + FRAME_ELEMENTS
RECORD_BYTES = RECORD_ELEMENTS * BYTES_PER_ELEMENT