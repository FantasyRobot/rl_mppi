#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
from typing import Any

import numpy as np


def _sanitize_key(k: str) -> str:
    # MATLAB variable names: start with letter, then letters/digits/underscore.
    # We'll keep it minimal: replace invalid chars with underscore.
    k = str(k)
    out = []
    for i, ch in enumerate(k):
        if (ch.isalpha() or ch == "_") if i == 0 else (ch.isalnum() or ch == "_"):
            out.append(ch)
        else:
            out.append("_")
    kk = "".join(out)
    if not kk or not (kk[0].isalpha() or kk[0] == "_"):
        kk = "v_" + kk
    return kk


def _npz_to_dict(npz_path: str) -> dict[str, Any]:
    data = np.load(npz_path, allow_pickle=True)
    out: dict[str, Any] = {}
    for k in data.files:
        v = data[k]
        # Convert object arrays conservatively (MATLAB can't represent arbitrary Python objects).
        if isinstance(v, np.ndarray) and v.dtype == object:
            if v.size == 1:
                try:
                    out[_sanitize_key(k)] = np.asarray(v.item())
                except Exception:
                    out[_sanitize_key(k)] = np.asarray(str(v.item()), dtype=object)
            else:
                out[_sanitize_key(k)] = np.asarray([str(x) for x in v.reshape(-1)], dtype=object).reshape(v.shape)
        else:
            out[_sanitize_key(k)] = v
    return out


def save_mat(npz_path: str, out_path: str) -> str:
    out_path = os.path.expanduser(os.path.expandvars(out_path))
    if not out_path.lower().endswith(".mat"):
        out_path += ".mat"
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    d = _npz_to_dict(npz_path)

    try:
        from scipy.io import savemat  # type: ignore
    except ModuleNotFoundError as e:
        raise SystemExit(
            "scipy is required to write .mat. Install with: pip install scipy\n"
            "Or use --format h5 to write an HDF5 file MATLAB can read."
        ) from e

    savemat(out_path, d, do_compression=True)
    return out_path


def save_h5(npz_path: str, out_path: str) -> str:
    out_path = os.path.expanduser(os.path.expandvars(out_path))
    if not (out_path.lower().endswith(".h5") or out_path.lower().endswith(".hdf5")):
        out_path += ".h5"
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    d = _npz_to_dict(npz_path)

    try:
        import h5py  # type: ignore
    except ModuleNotFoundError as e:
        raise SystemExit("h5py is required to write .h5. Install with: pip install h5py") from e

    with h5py.File(out_path, "w") as f:
        for k, v in d.items():
            vv = v
            if isinstance(vv, np.ndarray) and vv.dtype.kind in {"U", "S"}:
                # store strings as variable-length UTF-8
                dt = h5py.string_dtype(encoding="utf-8")
                f.create_dataset(k, data=vv.astype(object), dtype=dt)
            elif isinstance(vv, np.ndarray) and vv.dtype == object:
                dt = h5py.string_dtype(encoding="utf-8")
                f.create_dataset(k, data=np.asarray(vv, dtype=str), dtype=dt)
            else:
                f.create_dataset(k, data=vv)

    return out_path


def main() -> None:
    p = argparse.ArgumentParser(description="Convert .npz logs to MATLAB-readable .mat or .h5")
    p.add_argument("--in", dest="inp", required=True, help="Input .npz path")
    p.add_argument("--out", dest="out", default="", help="Output path (optional)")
    p.add_argument("--format", dest="fmt", choices=["mat", "h5"], default="mat")

    args = p.parse_args()

    inp = os.path.abspath(str(args.inp))
    if not os.path.exists(inp):
        raise SystemExit(f"Input not found: {inp}")

    if args.out:
        out = str(args.out)
    else:
        root, _ = os.path.splitext(inp)
        out = root

    if str(args.fmt) == "mat":
        out_path = save_mat(inp, out)
    else:
        out_path = save_h5(inp, out)

    print("[DONE] saved:", os.path.abspath(out_path))


if __name__ == "__main__":
    main()
