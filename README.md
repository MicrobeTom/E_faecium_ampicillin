# Early Detection of Ampicillin Susceptibility in *Enterococcus faecium* with MALDI-TOF MS and machine learning

This repository accompanies the publication “Early detection of ampicillin susceptibility in *Enterococcus faecium* with MALDI-TOF MS and machine learning.” We developed logistic regression and LightGBM models to detect ampicillin-susceptible *Enterococcus faecium* isolates from MALDI-TOF mass spectra obtained from two clinical datasets (TUM & MS‑UMG).

## Repository Overview

| Path                                                | Purpose                                                                             |
| --------------------------------------------------- | ----------------------------------------------------------------------------------- |
| `align_spectra.R`                                   | Align raw spectra to dataset‑specific reference peaks (TUM or MS‑UMG).              |
| `ms-umg_dataextraction.py`                              | Extract ampicillin-susceptible and resistant *E. faecium* spectra from the MS-UMG dataset.                      |
| `bin_spectra.py`                                    | Convert each aligned spectrum to a fixed‑length (6 000‑dim) vector.       |
| `tools_final.py`                                    | Common helpers for I/O, preprocessing, SHAP, visualisation, CSV exports, etc.       |
| `internal_nested_lgbm.py` / `internal_nested_lr.py` | **Nested CV** inside one dataset (hyper‑parameter tuning via `RandomizedSearchCV`). |
| `external_lgbm.py` / `external_lr.py`               | Train on dataset *A*, test on dataset *B* (external validation)               |
| `requirements.txt`                                  | Python dependencies.                                                                |
| `LICENCE`                                           | BSD 3‑Clause     |
| `results/` (generated)                              | Run‑specific sub‑folders with CSVs, PR curves, and SHAP plots.                      |


---

## Installation

### 1 . Clone & create a virtual environment

```bash
# Clone
$ git clone https://github.com/MicrobeTom/E_faecium_ampicillin.git
$ cd E_faecium_ampicillin

# (Recommended) create venv
$ python -m venv .venv
$ source .venv/bin/activate
```

### 2 . Install Python requirements

```bash
$ pip install -r requirements.txt
```

### 3 . Install R dependencies (for alignment)

The alignment script expects R (≥ 4.0) with packages:

```R
install.packages(c("MALDIquant", "MALDIquantForeign", "tidyverse"))
```

---

## Data access

| Dataset                                     | Link                                                                               |
| ------------------------------------------- | ---------------------------------------------------------------------------------- |
| **TUM spectra** (choose pre‑processed)       | [https://doi.org/10.5281/zenodo.15769315] |
| **MS‑UMG spectra** (choose pre‑processed) | [https://doi.org/10.5281/zenodo.13911744] |

Both DOIs resolve to Zenodo records. For this project, only regular (non-screening) spectra were obtained from the MS-UMG dataset.


## Data Preparation

### 1 . Obtain the spectra

The TUM dataset provides spectra according to ampicillin resistance status.

The file ms-umg_dataextraction.py provides the code to extract ampicillin-susceptible and ampicillin-resistant *Enterococcus faecium* spectra from the MS-UMG dataset of regular MALDI-TOF mass spectra.

### 2 . Align spectra

For internal nested models, align spectra according to each dataset's reference peaks.

For external models, align TUM spectra according to the MS-UMG reference peak list and vice versa.


```bash
Rscript align_spectra.R \
        --input  preprocessed_spectra/ \
        --output aligned_spectra/ \
        --reference peaks_reference.csv
```

Note: First, create a new reference peak list when aligning spectra for internal validation (i.e. align each dataset using its own peaks).
Then, use the existing reference peak list for external validation (e.g. align TUM spectra using the MS‑UMG reference peak list and vice versa).

### 3 . Bin spectra

The models in this project were developed with 6000-dimensional feature (bin) vectors representing the spectra.

```bash
python bin_spectra.py \
       --raw_dir  aligned_spectra/ \
       --out_dir  binned_spectra/
```

### 4 . Tell the pipeline where your data live

Open tools_final.py and edit the folder constants (empty by default):

```python
# example paths for internal (nested) models 
tum_ecfmar_warped_tum     = "/data/TUM/warped_TUM/resistant/"
tum_ecfmas_warped_tum     = "/data/TUM/warped_TUM/susceptible/"
msumg_ecfmar_warped_msumg = "/data/MSUMG/warped_MSUMG/resistant/"
msumg_ecfmas_warped_msumg = "/data/MSUMG/warped_MSUMG/susceptible/"
```

```python
# example paths for external models
tum_ecfmar_warped_msumg     = "/data/TUM/warped_MSUMG/resistant/"
tum_ecfmas_warped_msumg     = "/data/TUM/warped_MSUMG/susceptible/"
msumg_ecfmar_warped_tum = "/data/MSUMG/warped_TUM/resistant/"
msumg_ecfmas_warped_tum = "/data/MSUMG/warped_TUM/susceptible/"
```

---

## Running the models

### 1 . Internal, nested cross‑validation (single dataset)

```bash
# LightGBM (default: MCC scoring)
python internal_nested_lgbm.py

# Logistic regression
python internal_nested_lr.py
```

### 2 . External validation (train ↔ test across sites)

```bash
# Train on TUM, test on MS‑UMG (edit header of script for the opposite direction)
python external_lgbm.py
python external_lr.py
```

Results are written to `./results/train<TRAIN>test<TEST>/` and include:

* `allresults_*.csv` – every hyper‑parameter / threshold combination.
* `readable_*.csv`  – concise summary of best runs.
* `summarystats_*.csv`, `meta_summary_*.csv`.
* `plots/`           – SHAP summaries & precision‑recall curves (PDF/PNG).

### 3 . SHAP values

Each run creates SHAP bar/summary plots showing the most relevant m/z ranges. 


## Citation

This section will be updated shortly.


## Licence

This project is licensed under the **BSD 3‑Clause** licence (see `LICENCE`).
Portions of the preprocessing pipeline are adapted from the [**maldi\_amr**](https://github.com/BorgwardtLab/maldi_amr) project and therefore retain the same license terms.


## Disclaimer

This software is **for research use only**.  It is **not** a certified medical device and must **not** be used for clinical decision making without confirmatory tests.


## Contact

Thomas Pichl  ·   Institute of Medical Microbiology, Immunology and Hygiene at the Technical University of Munich
[thomas.pichl@tum.de](mailto:thomas.pichl@tum.de)  
[https://github.com/MicrobeTom](https://github.com/MicrobeTom)  


