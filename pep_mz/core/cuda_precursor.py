"""Load the PepMZ CUDA precursor kernel when an NVIDIA GPU is present.

The kernel source is ``native/precursor_kernel.cu``. ``native/build.sh``
compiles it with nvcc. This module returns no result when that library is
absent, so the caller keeps the CPU path.
"""

from __future__ import annotations

import ctypes
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_LIB_NAME = "libpepmz_cuda.so"
_LIB: ctypes.CDLL | None | bool = None


@dataclass(frozen=True)
class CudaStatus:
    available: bool
    device_count: int
    library: str
    reason: str


def library_path() -> Path:
    return Path(__file__).resolve().parents[2] / "native" / _LIB_NAME


def _load() -> ctypes.CDLL | None:
    global _LIB
    if _LIB is False:
        return None
    if _LIB is not None:
        return _LIB
    path = library_path()
    if not path.is_file():
        _LIB = False
        return None
    lib = ctypes.CDLL(str(path))
    lib.pepmz_device_count.restype = ctypes.c_int
    lib.pepmz_precursor_launch.argtypes = [
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_double,
        ctypes.c_double,
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
    ]
    lib.pepmz_precursor_launch.restype = ctypes.c_int
    _LIB = lib
    return lib


def cuda_status() -> CudaStatus:
    path = library_path()
    lib = _load()
    if lib is None:
        return CudaStatus(False, 0, str(path), "CUDA library is not built. Run native/build.sh on an NVIDIA machine.")
    count = int(lib.pepmz_device_count())
    if count < 1:
        return CudaStatus(False, 0, str(path), "The CUDA library loaded, but no NVIDIA GPU is visible.")
    return CudaStatus(True, count, str(path), "CUDA precursor kernel is ready.")


def launch_precursor(
    sequences: list[str],
    charges: list[int],
    residue_mass: list[float],
    water: float,
    proton: float,
) -> tuple[list[float], list[list[float]], float]:
    """Run one peptide per CUDA thread. Returns mono masses, m/z rows, and milliseconds."""
    status = cuda_status()
    if not status.available:
        raise RuntimeError(status.reason)
    if len(residue_mass) != 27:
        raise RuntimeError("Residue mass table must have 27 entries (index 0 is padding).")

    lib = _load()
    assert lib is not None
    width = max(len(seq) for seq in sequences)
    encoded = np.zeros((len(sequences), width), dtype=np.int32)
    lengths = np.zeros(len(sequences), dtype=np.int32)
    for i, seq in enumerate(sequences):
        lengths[i] = len(seq)
        for j, aa in enumerate(seq):
            encoded[i, j] = ord(aa) - ord("A") + 1
    charge_arr = np.asarray(charges, dtype=np.int32)
    mass_arr = np.asarray(residue_mass, dtype=np.float64)
    mono = np.zeros(len(sequences), dtype=np.float64)
    mz = np.zeros((len(sequences), len(charges)), dtype=np.float64)

    started = time.perf_counter()
    code = lib.pepmz_precursor_launch(
        encoded.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        lengths.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        len(sequences),
        width,
        mass_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        float(water),
        float(proton),
        charge_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        len(charges),
        mono.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        mz.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    )
    elapsed_ms = (time.perf_counter() - started) * 1000
    if code != 0:
        raise RuntimeError(f"CUDA precursor kernel failed with code {code}.")
    return mono.tolist(), mz.tolist(), elapsed_ms
