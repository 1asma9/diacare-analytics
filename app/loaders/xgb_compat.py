"""
Cross-platform XGBoost joblib loader.

XGBoost >= 3.3 serialises its Booster's internal RNG state as a text string
using the C++ STL stream operator.  On Linux (libstdc++) the operator writes
decimal numbers; on Windows (MSVC STL) it writes hex numbers.  Loading a
Linux-trained bundle on Windows therefore can corrupt the stream.

Work-around (XGBoost issue #12459):
  - Try standard joblib.load first.
  - If it fails, extract the raw Booster byte-buffer and replace the Linux-formatted
    rng_state chunk with a Windows-native one that uses an equivalent RNG format
    serialised by the current runtime.
  - Reconstruct the Booster handle via XGBoosterUnserializeFromBuffer.

No model weights, trees, or parameters are modified.
"""

import re
import warnings
import ctypes

import joblib
import xgboost as xgb
from joblib.numpy_pickle import NumpyUnpickler
from xgboost.core import _LIB, _check_call, c_bst_ulong, c_array

# Build a Windows-native rng_state chunk once
def _build_win_rng_chunk() -> bytes:
    import numpy as np
    clf = xgb.XGBClassifier(n_estimators=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf.fit(np.zeros((2, 1)), [0, 1])
    buf = clf._Booster.__getstate__()["handle"]
    m = re.search(rb"\x09rng_stateSL\x00\x00\x00\x00\x00\x00(..)", buf)
    if m is None:
        return b""
    length = int.from_bytes(buf[m.end() - 2 : m.end()], "big")
    return bytes(buf[m.start() : m.end() + length])


_WIN_RNG_CHUNK: bytes = _build_win_rng_chunk()


def _patch_rng_buffer(buf: bytearray) -> bytearray:
    if not _WIN_RNG_CHUNK:
        return buf
    m = re.search(rb"\x09rng_stateSL\x00\x00\x00\x00\x00\x00(..)", buf)
    if m is None:
        return buf
    length = int.from_bytes(buf[m.end() - 2 : m.end()], "big")
    return bytearray(buf[: m.start()]) + bytearray(_WIN_RNG_CHUNK) + bytearray(buf[m.end() + length :])


class _WindowsCompatBooster(xgb.core.Booster):
    def __setstate__(self, state: dict) -> None:
        handle = state.get("handle")
        if handle is not None:
            state = dict(state)
            state["handle"] = _patch_rng_buffer(bytearray(handle))
        super().__setstate__(state)


class _CompatUnpickler(NumpyUnpickler):
    def find_class(self, module: str, name: str):
        if module == "xgboost.core" and name == "Booster":
            return _WindowsCompatBooster
        return super().find_class(module, name)


def load(path: str):
    """Load a joblib file that may contain Linux-serialised XGBoost Boosters."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            return joblib.load(path)
        except Exception:
            with open(path, "rb") as fh:
                unpickler = _CompatUnpickler(path, fh, True)
                return unpickler.load()
