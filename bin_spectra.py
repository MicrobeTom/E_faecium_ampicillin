# ---------------------------------------------------------------------------
# Adapted from the **maldi_amr** project by the Borgwardt Lab
# (https://github.com/BorgwardtLab/maldi_amr) licensed under the
# BSD 3‑Clause "New" License.  Copyright (c) 2020, Caroline V. Weis, Bastian Rieck, Aline Cuénod
#
# This derivative work retains the BSD 3‑Clause license; redistribution of this
# file must include the above notice and the full license text.
# ---------------------------------------------------------------------------

import os
import sys
import pandas as pd
from tools_final import MaldiTofSpectrum, BinningVectorizer


RAW_DIR = r""    # folder containing raw *.txt spectra
OUT_DIR = r""    # destination folder for binning (created if missing)

if not os.path.isdir(RAW_DIR):
    sys.exit(f"Input directory not found: {RAW_DIR}")

os.makedirs(OUT_DIR, exist_ok=True)


def load_spectrum(filepath):
    try:
        data = pd.read_csv(filepath, sep=" ", comment="#", engine="c").values
        return MaldiTofSpectrum(data)
    except Exception as exc:
        print(f"ERROR reading {filepath}: {exc}")
        return None

spectra_files = {}
for root, _, files in os.walk(RAW_DIR):
    for fname in files:
        if fname.endswith(".txt"):
            spectra_files[os.path.join(root, fname)] = fname

spectra = []
for path, name in spectra_files.items():
    spec = load_spectrum(path)
    if spec is not None:
        spectra.append((spec, name))

bv = BinningVectorizer(6000, min_bin=2000, max_bin=20000)

for spectrum, fname in spectra:
    try:
        binned = bv.fit_transform([spectrum])[0]
    except Exception as exc:
        print(f"ERROR binning {fname}: {exc}")
        continue

    out_file = os.path.join(OUT_DIR, fname.replace(".txt", "_binned.txt"))
    df = pd.DataFrame({"binned_intensity": binned})
    df.index.name = "bin_index"
    try:
        df.to_csv(out_file, sep=" ")
        print(f"Saved: {out_file}")
    except Exception as exc:
        print(f"ERROR writing {out_file}: {exc}")