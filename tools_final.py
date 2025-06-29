import os
import numpy as np
import pandas as pd
import csv
from datetime import datetime
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import json
from collections import defaultdict

# ---------------------------------------------------------------------------
# Paths to folders containing preprocessed spectra
# efmar = Enterococcus faecium, ampicillin resistant
# efcmas = Enterococcus faecium, ampicillin susceptible
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# NESTED CROSS-VALIDATION
# ---------------------------------------------------------------------------
# For nested cross-validation, the TUM and MS-UMG datasets are analysed separately.
# - TUM raw spectra must be aligned with TUM reference peaks (see preprocessTUM.R).
# - MS-UMG spectra are already preprocessed and must be aligned with MS-UMG reference peaks (see align_MS-UMG.R).
# After alignment, spectra are binned (see binning_efaecium.py).
# For more information, see the README.md file.
# ---------------------------------------------------------------------------


###TUM dataset folder paths containing warped (TUM reference peaks, see preprocessing.R), binned spectra files
tum_ecfmar_warped_tum = r"" # ampicillin RESISTANT
tum_ecfmas_warped_tum = r"" # ampicillin SUSCEPTIBLE

### MS-UMG dataset folder paths containing warped (MS-UMG reference peaks, see preprocessing.R), binned spectra files
msumg_ecfmar_warped_msumg = r"" # ampicillin RESISTANT  
msumg_ecfmas_warped_msumg = r"" # ampicillin SUSCEPTIBLE

# ---------------------------------------------------------------------------
# EXTERNAL VALIDATION
# ---------------------------------------------------------------------------
# For external validation, spectra must be aligned according to the training dataset:
# - If training on MS-UMG and testing on TUM: TUM spectra must be aligned to MS-UMG reference peaks.
# - If training on TUM and testing on MS-UMG: MS-UMG spectra must be aligned to TUM reference peaks.
# After alignment, spectra are binned (see binning_efaecium.py).
# For more information, see the README.md file.
# ---------------------------------------------------------------------------

### TUM dataset folder paths containing aligned-to-MS-UMG binned spectra files
tum_ecfmar_warped_msumg = r"" # ampicillin RESISTANT
tum_ecfmas_warped_msumg = r"" # ampicillin SUSCEPTIBLE

### MS-UMG dataset folder paths containing aligned-to-TUM binned spectra files
msumg_ecfmar_warped_tum = r"" # ampicillin RESISTANT
msumg_ecfmas_warped_tum = r"" #ampicillin SUSCEPTIBLE


def load_data_nested(seed, testdataset, traindataset):

    np.random.seed(seed)

    if testdataset=="TUM_warped_TUM" and traindataset=="TUM_warped_TUM":
        ecfmarpath = tum_ecfmar_warped_tum
        ecfmaspath = tum_ecfmas_warped_tum
    elif testdataset=="MS-UMG_warped_MS-UMG" and traindataset=="MS-UMG_warped_MS-UMG":
        ecfmarpath = msumg_ecfmar_warped_msumg
        ecfmaspath = msumg_ecfmas_warped_msumg
    else:
        print("INCONSISTENT DATASETS!!")
        exit()

    # Load ECFMAR (resistant) spectra data
    print("Loading ECFMAR frame...")
    ecfmarspectralist = [os.path.join(ecfmarpath, file) for file in os.listdir(ecfmarpath) if file.endswith(".txt")]
    ecfmardflist = [pd.read_csv(spectrum, index_col=0, sep=" ") for spectrum in ecfmarspectralist]

    # Load ECFMAS (susceptible) spectral data
    print("Loading ECFMAS frame...")
    ecfmasspectralist = [os.path.join(ecfmaspath, file) for file in os.listdir(ecfmaspath) if file.endswith(".txt")]
    ecfmasdflist = [pd.read_csv(spectrum, index_col=0, sep=" ") for spectrum in ecfmasspectralist]

    # Combine the dataframes
    ecfmarframe = pd.concat(ecfmardflist, axis=1, ignore_index=True)
    print("Done loading ECFMAR frame...")
    ecfmasframe = pd.concat(ecfmasdflist, axis=1, ignore_index=True)
    print("Done loading ECFMAS frame...")

    # Transpose the dataframes to have samples as rows and features as columns
    print("Transposing dataframes...")
    ecfmarframeI = ecfmarframe.T
    ecfmasframeI = ecfmasframe.T
    print("Done transposing dataframes...")

    ecfmarframeI['target'] = 0
    ecfmasframeI['target'] = 1

    # Combine the two dataframes into one
    combined_df = pd.concat([ecfmarframeI, ecfmasframeI], axis=0)

    # Reset index
    combined_df.reset_index(drop=True, inplace=True)

    # Separate features and target
    X = combined_df.drop('target', axis=1)
    y = combined_df['target']

    return X, y

# Load datasets for external validation (5-fold cross validation; TUM dataset for training, MS-UMG for testing or vice versa)

def load_data_comb_5f(seed, testdataset, traindataset):
    # Set random seeds
    np.random.seed(seed)

    if traindataset == "TUM-warped_TUM" and testdataset == "MS-UMG_warped_TUM":
        ecfmarpath_train = tum_ecfmar_warped_tum
        ecfmaspath_train = tum_ecfmas_warped_msumg
        ecfmarpath_test = msumg_ecfmar_warped_tum
        ecfmaspath_test = msumg_ecfmas_warped_tum
    elif traindataset == "MS-UMG_warped_MS-UMG" and testdataset == "TUM-warped_MS-UMG":
        ecfmarpath_train = msumg_ecfmar_warped_msumg
        ecfmaspath_train = msumg_ecfmas_warped_msumg
        ecfmarpath_test = tum_ecfmar_warped_msumg
        ecfmaspath_test = tum_ecfmas_warped_msumg
    else:
        print("INCONSISTENT DATASETS!!")
        exit()

    # Load ECFMAR (resistant) and ECFMAS (sensitive) data for training
    def load_spectrum_data(path):
        return [pd.read_csv(os.path.join(path, file), index_col=0, sep=" ")
                for file in os.listdir(path) if file.endswith(".txt")]

    ecfmardflist_train = load_spectrum_data(ecfmarpath_train)
    ecfmasdflist_train = load_spectrum_data(ecfmaspath_train)

    # Combine the training dataframes
    ecfmarframe_train = pd.concat(ecfmardflist_train, axis=1, ignore_index=True)
    ecfmasframe_train = pd.concat(ecfmasdflist_train, axis=1, ignore_index=True)

    # Transpose and label training data
    ecfmarframeI_train = ecfmarframe_train.T
    ecfmasframeI_train = ecfmasframe_train.T
    ecfmarframeI_train['target'] = 0
    ecfmasframeI_train['target'] = 1
    train_df = pd.concat([ecfmarframeI_train, ecfmasframeI_train], axis=0).reset_index(drop=True)

    # Load and process external test data
    ecfmardflist_test = load_spectrum_data(ecfmarpath_test)
    ecfmasdflist_test = load_spectrum_data(ecfmaspath_test)

    ecfmarframe_test = pd.concat(ecfmardflist_test, axis=1, ignore_index=True)
    ecfmasframe_test = pd.concat(ecfmasdflist_test, axis=1, ignore_index=True)

    ecfmarframeI_test = ecfmarframe_test.T
    ecfmasframeI_test = ecfmasframe_test.T
    ecfmarframeI_test['target'] = 0
    ecfmasframeI_test['target'] = 1
    test_df = pd.concat([ecfmarframeI_test, ecfmasframeI_test], axis=0).reset_index(drop=True)

    # Separate features and target
    X_train, y_train = train_df.drop('target', axis=1), train_df['target']
    X_test, y_test = test_df.drop('target', axis=1), test_df['target']

    return X_train, y_train, X_test, y_test

# Get SHAP values for logistic regression models (overall for test and train, and separated positive and negative class for test and train)

def get_separated_shap_values_log(model, X_train, X_test):
    explainer = shap.LinearExplainer(model, X_train)
    
    # Get SHAP values directly from the explanation object
    shap_values_train = explainer.shap_values(X_train)
    shap_values_test = explainer.shap_values(X_test)
    
    # Separate positive and negative contributions
    positive_shap_train = np.where(shap_values_train > 0, shap_values_train, 0)
    negative_shap_train = np.where(shap_values_train < 0, shap_values_train, 0)
    
    positive_shap_test = np.where(shap_values_test > 0, shap_values_test, 0)
    negative_shap_test = np.where(shap_values_test < 0, shap_values_test, 0)
    
    return (positive_shap_train, negative_shap_train, 
            positive_shap_test, negative_shap_test, 
            shap_values_train, shap_values_test)

# Get top SHAP indices according to SHAP values of feature (bin)
def get_top_shap_indices(shap_values, top_n=10):
    shap_mean_abs = np.abs(shap_values).mean(axis=0)
    top_indices = np.argsort(shap_mean_abs)[-top_n:]
    top_indices = top_indices[np.argsort(shap_mean_abs[top_indices])[::-1]]
    return top_indices

# Get matching top SHAP features for testing and training data (adjustable buffer, default=1)

def match_shap_bins_with_buffer(train_indices, test_indices, buffer=1):
    matched_test_indices = set()
    matches = 0

    for train_idx in train_indices:
        for test_idx in test_indices:
            if test_idx not in matched_test_indices and abs(train_idx - test_idx) <= buffer:
                matches += 1
                matched_test_indices.add(test_idx)
                break

    return matches

# LightGBM work around to get SHAP array in correct order

def extract_class_shap(shap_out, class_idx: int, n_classes: int, X_ref):
    """
    Return SHAP array for the requested class with shape (n_samples, n_features),
    whatever format `explainer.shap_values()` gives us.

    Works with:
      • list-of-arrays         -> typical TreeExplainer new API
      • (n_classes, n_samples, n_features)
      • (n_samples, n_classes, n_features)
      • (n_samples, n_features, n_classes)
    """

    # Newer SHAP: list where each entry is (n_samples, n_features)
    if isinstance(shap_out, list):
        shap_cls = shap_out[class_idx]

    else:
        arr = np.asarray(shap_out)

        if arr.ndim != 3:
            # binary problems often already give (n_samples, n_features)
            shap_cls = arr
        else:
            # find which axis matches n_classes
            if arr.shape[0] == n_classes:
                shap_cls = arr[class_idx]
            elif arr.shape[1] == n_classes:
                shap_cls = arr[:, class_idx, :]
            elif arr.shape[2] == n_classes:
                shap_cls = arr[:, :, class_idx]
            else:
                raise ValueError(
                    f"Can't locate class axis in SHAP array shape {arr.shape}"
                )

    # Make sure the first axis is the samples axis
    if shap_cls.shape[0] != X_ref.shape[0] and shap_cls.shape[1] == X_ref.shape[0]:
        shap_cls = shap_cls.T

    return shap_cls

def get_separated_shap_values_tree(model, X_train, X_test, pos_class_idx: int = 1):
    explainer = shap.TreeExplainer(model)
    n_classes = getattr(model, "n_classes_", 2)

    shap_train_raw = explainer.shap_values(X_train)
    shap_test_raw  = explainer.shap_values(X_test)

    shap_values_train = extract_class_shap(shap_train_raw, pos_class_idx, n_classes, X_train)
    shap_values_test  = extract_class_shap(shap_test_raw,  pos_class_idx, n_classes, X_test)

    # split into positive/negative contributions
    positive_shap_train = np.where(shap_values_train > 0, shap_values_train, 0)
    negative_shap_train = np.where(shap_values_train < 0, shap_values_train, 0)
    positive_shap_test  = np.where(shap_values_test  > 0, shap_values_test,  0)
    negative_shap_test  = np.where(shap_values_test  < 0, shap_values_test,  0)


    return (positive_shap_train, negative_shap_train,
            positive_shap_test,  negative_shap_test,
            shap_values_train,   shap_values_test)


# Helper function to calculate m/z ranges from index ("bin number")

def get_dalton_range_labels(column_indices, da_start=2000, bin_size=3):
    """
    Convert each bin index (0..5999) to a string like '2000–2003 m/z'.

    Args:
        column_indices (list or pd.Index): The numeric bin indices (0..5999).
        da_start (int): The starting mass in Da (2000 by default).
        bin_size (int): The Da width per bin (3 by default).

    Returns:
        list of str, e.g.: ["2000–2003 m/z", "2003–2006 m/z", ...].
    """
    labels = []
    for col in column_indices:
        bin_idx = int(col)  # in case col is string or similar
        start_da = da_start + bin_idx * bin_size
        end_da   = da_start + (bin_idx + 1) * bin_size
        labels.append(f"{start_da}–{end_da} m/z")
    return labels

# Create plots for nested cross-validation

def create_plots_nested(seed=None, X_test=None, X_train=None, fold=None, plotdir=None, fold_dir="shap_plots", positive_shap_train=None, negative_shap_train=None, 
         positive_shap_test=None, negative_shap_test=None, 
         shap_values_train=None, shap_values_test=None):

            # Create or ensure the shap_plots directory for saving
            fold_dir = os.path.join(plotdir,fold_dir)
            os.makedirs(fold_dir, exist_ok=True)

            train_feature_names = get_dalton_range_labels(X_train.columns)
            test_feature_names  = get_dalton_range_labels(X_test.columns)

            # ------------------ TEST SET: Positive Only ------------------
            plt.figure()
            shap.summary_plot(positive_shap_test, X_test, plot_type="dot", max_display=10, show=False, feature_names=test_feature_names)
            plt.title(f"SHAP Beeswarm (Test Positive) - Seed {seed}, Fold {fold}")
            plt.savefig(os.path.join(fold_dir, f"shap_beeswarm_test_positive_{seed}_{fold}.png"), 
                        dpi=300, bbox_inches='tight')
            plt.close()

            # ------------------ TEST SET: Negative Only ------------------
            plt.figure()
            shap.summary_plot(negative_shap_test, X_test, plot_type="dot", max_display=10,show=False, feature_names=test_feature_names)
            plt.title(f"SHAP Beeswarm (Test Negative) - Seed {seed}, Fold {fold}")
            plt.savefig(os.path.join(fold_dir, f"shap_beeswarm_test_negative_{seed}_{fold}.png"), 
                        dpi=300, bbox_inches='tight')
            plt.close()

            # ------------------ TEST SET: Overall ------------------------
            plt.figure()
            shap.summary_plot(shap_values_test, X_test, plot_type="dot", max_display=10, show=False, feature_names=test_feature_names)
            #plt.title(f"SHAP beeswarm plot")
            plt.savefig(os.path.join(fold_dir, f"shap_beeswarm_test_overall_{seed}_{fold}.png"), 
                        dpi=300, bbox_inches='tight')
            plt.close()

            # ------------------ TRAIN SET: Positive Only -----------------
            plt.figure()
            shap.summary_plot(positive_shap_train, X_train, plot_type="dot", max_display=10, show=False, feature_names=test_feature_names)
            plt.title(f"SHAP Beeswarm (Train Positive) - Seed {seed}, Fold {fold}")
            plt.savefig(os.path.join(fold_dir, f"shap_beeswarm_train_positive_{seed}_{fold}.png"), 
                        dpi=300, bbox_inches='tight')
            plt.close()

            # ------------------ TRAIN SET: Negative Only -----------------
            plt.figure()
            shap.summary_plot(negative_shap_train, X_train, plot_type="dot", max_display=10, show=False, feature_names=test_feature_names)
            plt.title(f"SHAP Beeswarm (Train Negative) - Seed {seed}, Fold {fold}")
            plt.savefig(os.path.join(fold_dir, f"shap_beeswarm_train_negative_{seed}_{fold}.png"), 
                        dpi=300, bbox_inches='tight')
            plt.close()

            # ------------------ TRAIN SET: Overall -----------------------
            plt.figure()
            shap.summary_plot(shap_values_train, X_train, plot_type="dot", max_display=10, show=False, feature_names=train_feature_names)
            #plt.title(f"SHAP Beeswarm")
            plt.savefig(os.path.join(fold_dir, f"shap_beeswarm_train_overall_{seed}_{fold}.png"), 
                        dpi=300, bbox_inches='tight')
            plt.close()


# Create precision-recall curves for nested cross-validation:

def plot_pr_folds(fold_curves, macro_curve, seed, output_path):
    """
    Draws the Precision–Recall curves for each outer-CV fold
    and the *macro*-average curve (mean of the five folds).
    """
    plt.figure()

    for f_idx, (fold_prec, fold_rec) in enumerate(fold_curves, start=1):
        plt.plot(
            fold_rec,
            fold_prec,
            color="gray",
            alpha=0.6,
            linestyle="--",
            label="Folds 1-5" if f_idx == 1 else None,
        )

    macro_prec, macro_rec, macro_ap = macro_curve
    plt.plot(
        macro_rec,
        macro_prec,
        linewidth=3,
        label=f"Average across folds (AUPCR = {macro_ap:.3f})"
    )

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    # plt.title("Precision-Recall – nested CV")
    major = np.arange(0.0, 1.01, 0.2)
    minor = np.arange(0.0, 1.01, 0.1)
    plt.xticks(major)
    plt.yticks(major)
    plt.gca().set_xticks(minor, minor=True)
    plt.gca().set_yticks(minor, minor=True)
    plt.grid(True, which="both", linewidth=0.5, alpha=0.4)
    plt.grid(True, which="minor", linewidth=0.5, alpha=0.4)
    plt.legend(loc="lower left")

    plot_filename = os.path.join(
        output_path, f"precision_recall_seed{seed}.png"
    )
    plt.savefig(plot_filename, dpi=300, bbox_inches="tight")
    plt.close()

# Create plots for external validation

def create_plots_5f(
    *,
    seed,
    X_train,
    X_test,
    plotdir,
    positive_shap_train,
    negative_shap_train,
    positive_shap_test,
    negative_shap_test,
    shap_values_train,
    shap_values_test,
    max_display=10,
):
    """
    Writes six SHAP beeswarm plots into
        {plotdir}/shap_plots/
            shap_train_pos_seed<seed>.png
            shap_train_neg_seed<seed>.png
            shap_train_overall_seed<seed>.png
            shap_test_pos_seed<seed>.png
            shap_test_neg_seed<seed>.png
            shap_test_overall_seed<seed>.png
    """

    fold_dir = os.path.join(plotdir, "shap_plots")
    os.makedirs(fold_dir, exist_ok=True)

    train_feature_names = get_dalton_range_labels(X_train.columns)
    test_feature_names  = get_dalton_range_labels(X_test.columns)

    def _save(beeswarm, X, f_names, tag):
        plt.figure()
        shap.summary_plot(
            beeswarm,
            X,
            plot_type="dot",
            max_display=max_display,
            show=False,
            feature_names=f_names,
        )
        out_path = os.path.join(fold_dir, f"{tag}_seed{seed}.png")
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()

    # TEST
    _save(positive_shap_test,  X_test, test_feature_names,  "shap_test_positive")
    _save(negative_shap_test,  X_test, test_feature_names,  "shap_test_negative")
    _save(shap_values_test,    X_test, test_feature_names,  "shap_test_overall")

    # TRAIN
    _save(positive_shap_train, X_train, train_feature_names, "shap_train_positive")
    _save(negative_shap_train, X_train, train_feature_names, "shap_train_negative")
    _save(shap_values_train,   X_train, train_feature_names, "shap_train_overall")

def plot_external_test_curve(testdataset, sub_dir, rec_test, prec_test, seed, test_ap):

        if testdataset == "MS-UMG_reg_warpedTUM":
            labelset = "MS-UMG"
        else:
            labelset = "TUM"

        plot_path = os.path.join(sub_dir, f"precision_recall_seed{seed}.png")
        plt.figure()
        plt.plot(rec_test, prec_test, label=f"PR curve test set ({labelset}, AUPRC = {test_ap:.3f})")
        #(AP={test_ap:.3f})
        # plt.plot(rec_train, prec_train, label=f"PR curve train set")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        major_ticks = np.arange(0.0, 1.01, 0.2)
        minor_ticks = np.arange(0.0, 1.01, 0.1)
        plt.xticks(major_ticks); plt.yticks(major_ticks)
        plt.gca().set_xticks(minor_ticks, minor=True)
        plt.gca().set_yticks(minor_ticks, minor=True)

        # Grid lines: major and minor
        plt.grid(True, which='both', linewidth=0.5, alpha=0.4)
        plt.grid(True, which='minor', linewidth=0.5, alpha=0.4)
        plt.legend(loc="lower left")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

# Output files
## Nested cross validation
### Write all data to machine-readable file

def write_to_csv_nested(run_id, results, scoring_method, csv_file):
    write_header = not os.path.exists(csv_file)
    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file,quoting=csv.QUOTE_ALL)
        if write_header:
            header = [
                "Run ID", 
                "Seed", 
                "Fold", 
                "Threshold", 
                "Model Details",
                "Train dataset", 
                "Test dataset", 
                "Scoring Method",
                "F1 Score 0", 
                "F1 Score 1",
                "mcc",
                "Bin Score",
                "Positive Bin Score", 
                "Negative Bin Score",
                "True Negatives", 
                "False Positives", 
                "False Negatives", 
                "True Positives",
                "Top10_Positive_SHAP_Train", 
                "Top10_Negative_SHAP_Train",
                "Top10_Positive_SHAP_Test", 
                "Top10_Negative_SHAP_Test",
                "Top_10_overall_shap_train",
                "Top_10_overall_shap_test",
                "Evaluated Params",
                "Best Params"
            ]
            writer.writerow(header)

        for result in results:
            row = [
                run_id,
                result['seed'],
                result['fold'],
                result['threshold'],
                str(result['model']),
                result['traindataset'],
                result['testdataset'],
                scoring_method,
                result['f1_score_0'],
                result['f1_score_1'],
                result['mcc'],
                result['bin_score'],
                result['bin_score_pos'],
                result['bin_score_neg'],
                result['conf_matrix'][0][0],  # True Negatives
                result['conf_matrix'][0][1],  # False Positives
                result['conf_matrix'][1][0],  # False Negatives
                result['conf_matrix'][1][1],  # True Positives
                "; ".join(map(str, result['top_10_positive_shap_train'])),
                "; ".join(map(str, result['top_10_negative_shap_train'])),
                "; ".join(map(str, result['top_10_positive_shap_test'])),
                "; ".join(map(str, result['top_10_negative_shap_test'])),
                "; ".join(map(str, result['top_10_overall_shap_train'])),
                "; ".join(map(str, result['top_10_overall_shap_test'])),
                result['evaluated_params'],
                result['best_params']
            ]
            writer.writerow(row)

### Write a human-readable summary file

def write_to_readable_file_nested(run_id, results, scoring_method, readable_file):
    """
    Writes a readable summary file grouped by (seed, threshold).
    Typically used with 'best_mcc_each_seed' data, meaning you have
    exactly one threshold per fold. Hence you end up with a single
    item in each (seed, threshold) group. 'evaluated_params' is still
    a JSON string of *all* parameters tested during RandomizedSearchCV
    in that fold.
    """

    # Group results by seed and threshold
    grouped = defaultdict(lambda: defaultdict(list))
    for res in results:
        seed = res['seed']
        fold = res['fold']
        grouped[seed][fold].append(res)
        
    # Sort by fold inside each (seed, threshold)
    for seed in grouped:
        for fold in grouped[seed]:
            grouped[seed][fold].sort(key=lambda x: x['fold'])

    with open(readable_file, mode='w') as file:
        file.write(f"Run ID: {run_id}\n")
        file.write(f"Scoring Method: {scoring_method}\n")
        file.write("-" * 80 + "\n")
        file.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        file.write("=" * 80 + "\n\n")

        for seed in sorted(grouped.keys()):
            file.write(f"SEED: {seed}\n")
            file.write("-" * 80 + "\n")

            for fold in sorted(grouped[seed].keys()):
                file.write(f"FOLD: {fold}\n")
                file.write("-" * 80 + "\n")

                for fold_res in grouped[seed][fold]:
                    threshold = fold_res['threshold']

                    file.write(f"Threshold: {threshold}\n")
                    file.write(f"Train Dataset: {fold_res['traindataset']}, Test Dataset: {fold_res['testdataset']}\n")
                    file.write(f"Best params: {fold_res['best_params']}\n")
                    file.write(f"Time Taken: {fold_res['time_taken']:.2f} seconds\n")
                    file.write("=" * 80 + "\n")


                    file.write("Metrics:\n")
                    file.write(f"  F1 Score (Class 0): {fold_res['f1_score_0']:.4f}\n")
                    file.write(f"  F1 Score (Class 1): {fold_res['f1_score_1']:.4f}\n")
                    file.write(f"  F1 Score (Overall): {fold_res['f1_score overall']:.4f}\n")
                    file.write(f"  Matthews Correlation Coefficient (MCC): {fold_res['mcc']:.4f}\n")
                    file.write(f"  Overall Bin Score: {fold_res['bin_score']}\n")
                    file.write(f"  Positive Bin Score: {fold_res['bin_score_pos']}\n")
                    file.write(f"  Negative Bin Score: {fold_res['bin_score_neg']}\n")
                    if fold_res['fold_ap']:
                        file.write(f"  Area under PR curve (inverted): {fold_res['fold_ap']}\n")
                    file.write("=" * 80 + "\n")

                    cm = fold_res['conf_matrix']
                    file.write("Confusion Matrix:\n")
                    file.write(f"  True Negatives: {cm[0][0]}, False Positives: {cm[0][1]}\n")
                    file.write(f"  False Negatives: {cm[1][0]}, True Positives: {cm[1][1]}\n")
                    file.write(f"  NPV: {fold_res['NPV']:.4f}\n")
                    file.write(f"  PPV: {fold_res['PPV']:.4f}\n")
                    file.write(f"  Sensitivity: {fold_res['sensitivity']:.4f}\n")
                    file.write(f"  Specificity: {fold_res['specificity']:.4f}\n")
                    file.write("=" * 80 + "\n")


                    file.write("Top 10 SHAP Values (Training):\n")
                    file.write(f"  Positive SHAP: {fold_res['top_10_positive_shap_train']}\n")
                    file.write(f"  Negative SHAP: {fold_res['top_10_negative_shap_train']}\n")
                    file.write("Top 10 SHAP Values (Testing):\n")
                    file.write(f"  Positive SHAP: {fold_res['top_10_positive_shap_test']}\n")
                    file.write(f"  Negative SHAP: {fold_res['top_10_negative_shap_test']}\n")
                    file.write("=" * 80 + "\n\n")

            file.write("#" * 80 + "\n\n")

# Summary file for nested cross-validation

def save_micro_macro_summary_nested(all_results, summary_file, all_aps, threshold=0.5):
    """
    Create a long-format CSV with columns:
        seed, metric, micro, macro_mean, macro_std, macro_min, macro_max

    `all_results`  – list of per-threshold rows (one row per fold & threshold)
    `all_aps`      – list of dicts produced by nested_cross_validation, contains
                     the five fold APs + the 'micro' and 'macro' entries
    `threshold`    – which decision threshold to keep when computing confusion-
                     matrix metrics (e.g. 0.5)
    """

    df = pd.DataFrame(all_results)

    # keep only rows at the chosen threshold and add a plain 'AP' column
    df_thresh = df[df["threshold"] == threshold].copy()
    df_thresh["AP"] = df_thresh["fold_ap"]

    metrics = [
        "PPV", "sensitivity", "NPV", "specificity", "mcc",
        "AP",               
        "TN", "FP", "FN", "TP",
        "f1_score_0", "f1_score_1",
    ]

    summary_rows = []

    for seed_value, seed_df in df_thresh.groupby("seed"):

        seed_df = seed_df.copy() 

        folds_df = seed_df[seed_df["fold"].apply(lambda f: str(f).isdigit())]

        # ---------- macro statistics (mean / std / min / max) -----------------
        macro_means = folds_df[metrics].mean()
        macro_stds  = folds_df[metrics].std()
        macro_mins  = folds_df[metrics].min()
        macro_maxes = folds_df[metrics].max()

        # ---------- micro statistics ------------------------------------------
        total_TN, total_FP = folds_df[["TN", "FP"]].sum()
        total_FN, total_TP = folds_df[["FN", "TP"]].sum()

        specificity_micro = total_TN / (total_TN + total_FP) if (total_TN + total_FP) else 0
        sensitivity_micro = total_TP / (total_TP + total_FN) if (total_TP + total_FN) else 0
        npv_micro         = total_TN / (total_TN + total_FN) if (total_TN + total_FN) else 0
        ppv_micro         = total_TP / (total_TP + total_FP) if (total_TP + total_FP) else 0

        f1_1_micro = (2 * total_TP) / (2*total_TP + total_FP + total_FN) if (2*total_TP + total_FP + total_FN) else 0
        f1_0_micro = (2 * total_TN) / (2*total_TN + total_FP + total_FN) if (2*total_TN + total_FP + total_FN) else 0

        denom = np.sqrt((total_TP+total_FP)*(total_TP+total_FN)*(total_TN+total_FP)*(total_TN+total_FN))
        mcc_micro = ((total_TP * total_TN) - (total_FP * total_FN)) / denom if denom else 0

        micro_ap_value = next(
            d["pos_class_ap"] for d in all_aps
            if d["seed"] == seed_value and d["fold"] == "micro"
        )

        micro_vals = {
            "TN": total_TN,
            "FP": total_FP,
            "FN": total_FN,
            "TP": total_TP,
            "specificity": specificity_micro,
            "sensitivity": sensitivity_micro,
            "NPV": npv_micro,
            "PPV": ppv_micro,
            "f1_score_1": f1_1_micro,
            "f1_score_0": f1_0_micro,
            "mcc": mcc_micro,
            "AP": micro_ap_value,
        }


        for m in metrics:
            summary_rows.append({
                "seed":        seed_value,
                "metric":      m,
                "micro":       micro_vals[m],
                "macro_mean":  macro_means[m],
                "macro_std":   macro_stds[m],
                "macro_min":   macro_mins[m],
                "macro_max":   macro_maxes[m],
            })

    pd.DataFrame(summary_rows).to_csv(summary_file, index=False)
    print(f"[INFO] Micro & macro summary saved to {summary_file}")

# Write all results for external validation

def write_to_csv_5fcomb(run_id, results, scoring_method, csv_file):

    write_header = not os.path.exists(csv_file)
    
    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file, quoting=csv.QUOTE_ALL)
        
        if write_header:
            header = [
                "run_id", 
                "seed",
                "threshold",
                "best_params",      
                "cv_best_score",
                "model_details",
                "traindataset",
                "testdataset",
                
                # train metrics
                "train_TN", 
                "train_FP", 
                "train_FN", 
                "train_TP",
                "train_f1_pos",
                "train_f1_neg",
                "train_specificity",
                "train_sensitivity",
                "train_npv",
                "train_ppv",
                "train_mcc",
                "train_bal_acc",
                
                # test metrics
                "test_TN",
                "test_FP",
                "test_FN",
                "test_TP",
                "test_f1_pos",
                "test_f1_neg",
                "test_specificity",
                "test_sensitivity",
                "test_npv",
                "test_ppv",
                "test_mcc",
                "test_bal_acc",
                
                # SHAP / bin scores
                "bin_score",
                "bin_score_pos",
                "bin_score_neg",
                "top_10_positive_shap_train",
                "top_10_negative_shap_train",
                "top_10_positive_shap_test",
                "top_10_negative_shap_test",
                "top_10_overall_shap_train",
                "top_10_overall_shap_test",
                
                # All evaluated params
                "all_evaluated_params",
                
                # Additional info
                "scoring_method",
            ]
            writer.writerow(header)

        for r in results:
            best_params_str = str(r['best_params'])
            
            row = [
                run_id,
                r['seed'],
                r['threshold'],
                best_params_str,
                r['cv_best_score'],
                r['model_details'],
                r['traindataset'],
                r['testdataset'],
                
                # train metrics
                r['train_TN'],
                r['train_FP'],
                r['train_FN'],
                r['train_TP'],
                r['train_f1_pos'],
                r['train_f1_neg'],
                r['train_specificity'],
                r['train_sensitivity'],
                r['train_npv'],
                r['train_ppv'],
                r['train_mcc'],
                r['train_bal_acc'],

                # test metrics
                r['test_TN'],
                r['test_FP'],
                r['test_FN'],
                r['test_TP'],
                r['test_f1_pos'],
                r['test_f1_neg'],
                r['test_specificity'],
                r['test_sensitivity'],
                r['test_npv'],
                r['test_ppv'],
                r['test_mcc'],
                r['test_bal_acc'],

                # SHAP / bn scores
                r['bin_score'],
                r['bin_score_pos'],
                r['bin_score_neg'],
                ";".join(map(str, r['top_10_positive_shap_train'])),
                ";".join(map(str, r['top_10_negative_shap_train'])),
                ";".join(map(str, r['top_10_positive_shap_test'])),
                ";".join(map(str, r['top_10_negative_shap_test'])),
                "; ".join(map(str, r['top_10_overall_shap_train'])),
                "; ".join(map(str, r['top_10_overall_shap_test'])),

                r['evaluated_params'],

                # Additional info
                scoring_method
            ]
            
            writer.writerow(row)

# Write to readable file for external validation

def write_to_readable_file_5fcomb_inverted(run_id, results, best_rows, auc_results, scoring_method, readable_file):

    grouped_by_seed = defaultdict(list)
    for r in results:
        grouped_by_seed[r['seed']].append(r)
    
    with open(readable_file, 'w') as f:
        f.write(f"Run ID: {run_id}\n")
        f.write(f"Scoring Method: {scoring_method}\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")

        for seed in sorted(grouped_by_seed.keys()):
            seed_results = grouped_by_seed[seed]
            first_result = seed_results[0]

            traindataset = first_result.get('traindataset', 'N/A')
            testdataset  = first_result.get('testdataset', 'N/A')
            best_params = first_result.get('best_params', {})
            cv_best_score = first_result.get('cv_best_score', 0.0)

            f.write(f"SEED: {seed}\n")
            f.write(f"Train Dataset: {traindataset}\n")
            f.write(f"Test Dataset: {testdataset}\n")
            for auc_entry in auc_results:
                if auc_entry['seed'] == seed:
                    f.write(f"Area under PR curve (TRAIN): {auc_entry['train_ap']:.3f}\n")
                    f.write(f"Area under PR curve (TEST): {auc_entry['test_ap']:.3f}\n")
                    break
            f.write(f"Best Hyperparameters: {json.dumps(best_params, indent=2)}\n")
            f.write(f"Cross-Validation Best Score: {cv_best_score:.4f}\n")
            f.write("=" * 80 + "\n")

            best_row = best_rows[best_rows["seed"] == seed].iloc[0]

            threshold = best_row['threshold']
            f.write(f"Best Threshold: {threshold:.3f}\n\n")

            # ---- Train Metrics ----
            f.write("TRAIN METRICS at best TEST threshold:\n")
            f.write(f"  MCC: {best_row['train_mcc']:.4f}\n")
            f.write(f"  Specificity (TNR): {best_row['train_specificity']:.4f}\n")
            f.write(f"  Sensitivity (TPR): {best_row['train_sensitivity']:.4f}\n")
            f.write(f"  PPV: {best_row['train_ppv']:.4f}\n")
            f.write(f"  Confusion Matrix (TN, FP, FN, TP): ({best_row['train_TN']}, {best_row['train_FP']}, {best_row['train_FN']}, {best_row['train_TP']})\n")
            f.write("\n")

            # ---- Test Metrics ----
            f.write("TEST METRICS at best TEST threshold:\n")
            f.write(f"  MCC: {best_row['test_mcc']:.4f}\n")
            f.write(f"  Specificity (TNR): {best_row['test_specificity']:.4f}\n")
            f.write(f"  Sensitivity (TPR): {best_row['test_sensitivity']:.4f}\n")
            f.write(f"  PPV: {best_row['test_ppv']:.4f}\n")
            f.write(f"  Confusion Matrix (TN, FP, FN, TP): ({best_row['test_TN']}, {best_row['test_FP']}, {best_row['test_FN']}, {best_row['test_TP']})\n")
            f.write("=" * 80 + "\n\n")

            # ---- SHAP & Feature Importance ----
            bin_score = first_result.get('bin_score', 0)
            bin_score_pos = first_result.get('bin_score_pos', 0)
            bin_score_neg = first_result.get('bin_score_neg', 0)
            shap_pos_train = first_result.get('top_10_positive_shap_train', [])[:10]
            shap_neg_train = first_result.get('top_10_negative_shap_train', [])[:10]
            shap_pos_test  = first_result.get('top_10_positive_shap_test', [])[:10]
            shap_neg_test  = first_result.get('top_10_negative_shap_test', [])[:10]

            f.write("SHAP Feature Importance (Top 10 per category):\n")
            f.write(f"  Training (Positive): {shap_pos_train}\n")
            f.write(f"  Training (Negative): {shap_neg_train}\n")
            f.write(f"  Testing  (Positive): {shap_pos_test}\n")
            f.write(f"  Testing  (Negative): {shap_neg_test}\n")
            f.write(f"  Bin Score: {bin_score} (Pos: {bin_score_pos}, Neg: {bin_score_neg})\n")
            f.write("=" * 80 + "\n\n")

        f.write("END OF REPORT\n")
        f.write("#" * 80 + "\n")

# Summary for external evaluation

def write_summary_stats_file5fcomb(all_results, filename, auc_lookup=None):

    rows = []

    for res in all_results:
        seed_num = res["seed"] 
        threshold = res["threshold"]
        train_auc = test_auc = ""
        if auc_lookup and seed_num in auc_lookup:
            train_auc = auc_lookup[seed_num]["train_ap"]
            test_auc  = auc_lookup[seed_num]["test_ap"]

        # train row
        rows.append({
            "seed": f"{seed_num}_train",
            "threshold": threshold,
            "specificity": res["train_specificity"],
            "sensitivity": res["train_sensitivity"],
            "f1_score_0": res["train_f1_neg"],
            "f1_score_1": res["train_f1_pos"], 
            "mcc": res["train_mcc"],
            "balanced_accuracy": res["train_bal_acc"],
            "NPV": res["train_npv"],
            "PPV": res["train_ppv"],
            "AUC_PR": train_auc,
            "TN": res["train_TN"],
            "FP": res["train_FP"],
            "FN": res["train_FN"],
            "TP": res["train_TP"]
            
        })

        # test row
        rows.append({
            "seed": f"{seed_num}_test",
            "threshold": threshold,
            "specificity": res["test_specificity"],
            "sensitivity": res["test_sensitivity"],
            "f1_score_0": res["test_f1_neg"],
            "f1_score_1": res["test_f1_pos"],
            "mcc": res["test_mcc"],
            "balanced_accuracy": res["test_bal_acc"],
            "NPV": res["test_npv"],
            "PPV": res["test_ppv"],
            "AUC_PR": test_auc,
            "TN": res["test_TN"],
            "FP": res["test_FP"],
            "FN": res["test_FN"],
            "TP": res["test_TP"]
        })
    

    def custom_sort(row):
            seed_str, mode = row["seed"].split("_")
            seed_val = int(seed_str)
            mode_val = 0 if mode == "train" else 1
            return (seed_val, row["threshold"],mode_val)

    rows.sort(key=custom_sort)

    fieldnames = [
        "seed",
        "threshold",
        "specificity",
        "sensitivity",
        "f1_score_0",
        "f1_score_1",
        "mcc",
        "balanced_accuracy",
        "NPV",
        "PPV",
        "AUC_PR",
        "TN",
        "FP",
        "FN",
        "TP",
    ]

    # Write to CSV
    with open(filename, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


#Calculate mean ± SD for testing set across seeds at selected threshold from the per-seed summary file that save_micro_macro_summary_nested has produced 
def write_meta_summary_from_summary(summary_file: str,
                                    out_file: str,
                                    threshold: float = 0.5) -> None:
    td_metrics = ['PPV', 'sensitivity', 'NPV', 'specificity', 'mcc']
    df = pd.read_csv(summary_file)

    df = df[df['seed'].str.endswith('_test')].copy()
    df['base_seed'] = df['seed'].str.extract(r'(\d+)').astype(int)


    by_seed = (df[df['threshold'] == threshold]
               .groupby('base_seed')[td_metrics]
               .mean())
    meta_td = by_seed.agg(['mean', 'std']).T.reset_index() \
               .rename(columns={'index': 'metric',
                                'mean': 'mean',
                                'std':  'sd'})

    auprc = df.groupby('base_seed')['AUC_PR'].mean() 
    meta_auprc = pd.DataFrame({
        'metric': ['auprc'],
        'mean':   [auprc.mean()],
        'sd':     [auprc.std(ddof=1)]
    })

    # ── 4. final table ────────────────────────────────────────────────
    meta = pd.concat([meta_td, meta_auprc], ignore_index=True)
    meta.to_csv(out_file, index=False)


# ---------------------------------------------------------------------------
# The code below has been adapted from the **maldi_amr** project by the Borgwardt Lab
# (https://github.com/BorgwardtLab/maldi_amr) licensed under the
# BSD 3‑Clause "New" License.  Copyright (c) 2020, Caroline V. Weis, Bastian Rieck, Aline Cuénod
#
# This derivative work retains the BSD 3‑Clause license; redistribution of this
# file must include the above notice and the full license text.
# ---------------------------------------------------------------------------


class MaldiTofSpectrum(np.ndarray):
    """Numpy NDArray subclass representing a MALDI-TOF Spectrum."""

    def __new__(cls, peaks):
        """Create a MaldiTofSpectrum.

        Args:
            peaks: 2d array or list of tuples or list of list containing pairs
                of mass/charge to intensity.

        Raises:
            ValueError: If the input data is not in the correct format.

        """
        peaks = np.asarray(peaks).view(cls)
        if peaks.ndim != 2 or peaks.shape[1] != 2:
            raise ValueError(
                f'Input shape of {peaks.shape} does not match expected shape '
                'for spectrum [n_peaks, 2].'
            )
        return peaks

    @property
    def n_peaks(self):
        """Get number of peaks of the spectrum."""
        return self.shape[0]

    @property
    def intensities(self):
        """Get the intensities of the spectrum."""
        return self[:, 1]

    @property
    def mass_to_charge_ratios(self):
        """Get mass-t0-charge ratios of spectrum."""
        return self[:, 0]

import joblib
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin

class BinningVectorizer(BaseEstimator, TransformerMixin):
    """Vectorizer based on binning MALDI-TOF spectra.

    Attributes:
        bin_edges_: Edges of the bins derived after fitting the transformer.

    """

    _required_parameters = ['n_bins']

    def __init__(
        self,
        n_bins,
        min_bin=float('inf'),
        max_bin=float('-inf'),
        n_jobs=None
    ):
        """Initialize BinningVectorizer.

        Args:
            n_bins: Number of bins to bin the inputs spectra into.
            min_bin: Smallest possible bin edge.
            max_bin: Largest possible bin edge.
            n_jobs: If set, uses parallel processing with `n_jobs` jobs
        """
        self.n_bins = n_bins
        self.min_bin = min_bin
        self.max_bin = max_bin
        self.bin_edges_ = None
        self.n_jobs = n_jobs

    def fit(self, X, y=None):
        """Fit transformer, derives bins used to bin spectra."""
        # Find the smallest and largest time values in the dataset
        # It should be that the first/last time value is the smallest/biggest
        # but we call min/max to be safe.
        min_range = min(spectrum[:, 0].min() for spectrum in X)
        min_range = min(min_range, self.min_bin)
        max_range = max(spectrum[:, 0].max() for spectrum in X)
        max_range = max(max_range, self.max_bin)
        self.bin_edges_ = np.linspace(min_range, max_range, self.n_bins + 1)
        return self

    def transform(self, X):
        """Transform list of spectra into vector using bins.

        Args:
            X: List of MALDI-TOF spectra

        Returns:
            2D numpy array with shape [n_instances x n_bins]

        """
        if self.n_jobs is None:
            output = [self._transform(spectrum) for spectrum in X]
        else:
            output = joblib.Parallel(n_jobs=self.n_jobs)(
                joblib.delayed(self._transform)(s) for s in X
            )

        return np.stack(output, axis=0)

    def _transform(self, spectrum):
        times = spectrum[:, 0]

        valid = (times > self.bin_edges_[0]) & (times <= self.bin_edges_[-1])
        spectrum = spectrum[valid]

        vec = np.histogram(spectrum[:, 0], bins=self.bin_edges_, weights=spectrum[:, 1])[0]

        return vec


