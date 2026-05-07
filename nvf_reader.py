from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from tkinter import Tk, filedialog

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


@dataclass
class NVFData:
    header: np.ndarray
    hidden_blocks: np.ndarray
    data_cube: np.ndarray   # shape: (n_frames, 512, 640)
    n_frames: int
    file_path: Path


def pickfile() -> str:
    root = Tk()
    root.withdraw()
    filename = filedialog.askopenfilename(
        title="Seleziona file NVF",
        filetypes=[("NVF files", "*.*"), ("All files", "*.*")]
    )
    root.destroy()
    return filename


def open_nvf_file(filename: str):
    return open(filename, "rb")


def close_nvf_file(file_obj) -> None:
    file_obj.close()


def estimate_frame_count(file_size_bytes: int) -> int:
    payload_bytes = file_size_bytes - HEADER_BYTES

    if payload_bytes < 0:
        raise ValueError(
            f"File troppo piccolo: {file_size_bytes} byte, "
            f"header richiesto: {HEADER_BYTES} byte."
        )

    if payload_bytes % RECORD_BYTES != 0:
        raise ValueError(
            "La dimensione del file non è compatibile con la struttura NVF attesa.\n"
            f"payload_bytes = {payload_bytes}\n"
            f"record_bytes = {RECORD_BYTES}\n"
            f"resto = {payload_bytes % RECORD_BYTES}"
        )

    return payload_bytes // RECORD_BYTES


def read_nvf(file_obj, filename: str | Path) -> NVFData:
    file_path = Path(filename)

    file_obj.seek(0, 2)
    file_size = file_obj.tell()
    file_obj.seek(0)

    n_frames = estimate_frame_count(file_size)

    raw = np.fromfile(file_obj, dtype=FILE_DTYPE)

    expected_total_elements = HEADER_ELEMENTS + n_frames * (
        HIDDEN_BLOCK_ELEMENTS + FRAME_ELEMENTS
    )

    if raw.size != expected_total_elements:
        raise ValueError(
            "Numero totale di elementi nel file diverso da quello atteso.\n"
            f"Elementi letti   = {raw.size}\n"
            f"Elementi attesi  = {expected_total_elements}"
        )

    header = raw[:HEADER_ELEMENTS]

    payload = raw[HEADER_ELEMENTS:]
    records = payload.reshape(n_frames, HIDDEN_BLOCK_ELEMENTS + FRAME_ELEMENTS)

    hidden_blocks = records[:, :HIDDEN_BLOCK_ELEMENTS]
    frame_data = records[:, HIDDEN_BLOCK_ELEMENTS:]

    # Questa è la parte importante:
    # ogni frame viene interpretato come (height, width) = (512, 640)
    data_cube = frame_data.reshape(n_frames, FRAME_HEIGHT, FRAME_WIDTH)
    for i in range(n_frames):
        print(f"\n--- Frame {i} ---")
        print(data_cube[i])

    return NVFData(
        header=header,
        hidden_blocks=hidden_blocks,
        data_cube=data_cube,
        n_frames=n_frames,
        file_path=file_path,
    )


def import_nvf(filename: str | None = None) -> NVFData:
    if filename is None:
        filename = pickfile()

    if not filename:
        raise ValueError("Nessun file selezionato.")

    file_obj = open_nvf_file(filename)
    try:
        nvf_data = read_nvf(file_obj, filename)
    finally:
        close_nvf_file(file_obj)

    return nvf_data


if __name__ == "__main__":
    nvf = import_nvf()
