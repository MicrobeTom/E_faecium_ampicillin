import json
import os
from datetime import datetime

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from lightgbm.callback import early_stopping
from scipy.stats import randint, uniform
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    make_scorer,
    matthews_corrcoef,
    precision_recall_curve,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    train_test_split,
)

from tools_final import (
    load_data_nested_inverted,
    get_separated_shap_values_tree,
    get_top_shap_indices,
    match_shap_bins_with_buffer,
    create_plots_nested,
    plot_pr_folds,
    write_to_csv_nested,
    write_to_readable_file_nested,
    save_micro_macro_summary_nested
    
)
#########################
method = "RandomCV"
testdataset = "MS-UMG_reg_warpedSELF"
traindataset = "MS-UMG_reg_warpedSELF"
# testdataset = "TUM-warped"
# traindataset = "TUM-warped"
scoring_method = "mcc"  # choices: "f1-score susceptible", "mcc"
use_outer_early_stop = True
earlystop_n = 50
n_iter_call = 1
#########################
# List of seeds for multiple runs
seeds = list(range(42, 142, 100))

thresholds_list = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
auto_threshold = True

# Hyper‑parameter search space for LightGBM
param_grid = {
"bagging_freq": randint(1, 8),
"learning_rate"     : uniform(0.02, 0.13),
"num_leaves"        : randint(15, 128),
"max_depth"         : [-1, 4, 6, 8, 10],
"min_child_samples" : randint(10, 80),
"bagging_fraction"  : uniform(0.7, 0.3),
"feature_fraction"  : uniform(0.7, 0.3),
"reg_lambda"        : uniform(0.5, 4.5),
"reg_alpha"         : uniform(0.0, 3.0),
"n_estimators"      : randint(100, 300),
"class_weight"      : [None, "balanced", {0: 1, 1: 2}, {0: 1, 1: 3}, {0: 1, 1: 5}],
}

#  define scoring function
if scoring_method == "f1-score susceptible":
    scoring = make_scorer(f1_score, pos_label=1)
elif scoring_method == "mcc":
    scoring = make_scorer(matthews_corrcoef)
else:
    raise ValueError("Invalid scoring method. Choose 'f1-score susceptible' or 'mcc'.")


def nested_cross_validation(X, y,
                            param_distributions,
                            scoring,
                            n_splits=5,
                            n_iter=50,
                            seed=None,
                            thresholds=None,
                            auto_threshold=False,
                            plotdir=".",
                            ):
    outer_cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    results = []
    fold_curves = []
    best_per_fold = []

    # These will store all out-of-fold predictions and labels for micro-average
    all_probs: list[float] = []
    all_labels: list[int] = []
    ap_results = []

    for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X, y), start=1):
        print(f"Seed: {seed}, Outer Fold: {fold}")
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # Inner cross‑validation for hyper‑parameter tuning
        inner_cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=seed)

        lgb_base = LGBMClassifier(
            objective    = "binary",
            n_jobs       = -1,
            random_state = seed,
            verbosity=-1,
            n_estimators = 3000,
        )


        search = RandomizedSearchCV(
        estimator           = lgb_base,
        param_distributions = param_distributions,
        n_iter              = n_iter,
        scoring             = scoring,
        cv                  = inner_cv,
        random_state        = seed,
        n_jobs              = 1,
        verbose             = 0,
        )

        search.fit(X_train, y_train,  
           eval_metric="binary_logloss")
        best_params = search.best_params_


        # ----------------------- OUTER refit ----------------------------
        if use_outer_early_stop:
            ceiling = 3000
        else:
            ceiling = best_params.get("n_estimators", 500)
        params_refit = {k: v for k, v in best_params.items()
                        if k != "n_estimators"}

        #Build the model
        best_model = LGBMClassifier(
            objective="binary",
            n_jobs=-1,
            random_state=seed,
            n_estimators=ceiling,
            verbosity=-1,
            **params_refit
        )

        #Fit the model
        if use_outer_early_stop:
            X_tr, X_val, y_tr, y_val = train_test_split(
                X_train, y_train,
                test_size=0.15,
                stratify=y_train,
                random_state=seed
            )
            best_model.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                eval_metric="binary_logloss",
                callbacks=[early_stopping(earlystop_n, verbose=False)]
            )
        else:
            best_model.fit(X_train, y_train, eval_metric="binary_logloss")

        # Evaluate on the outer test set for all thresholds
        y_proba = best_model.predict_proba(X_test)[:, 1]

        # PR curve (per‑fold)
        fold_prec, fold_rec, _ = precision_recall_curve(y_test, y_proba, pos_label=1)
        fold_curves.append((fold_prec, fold_rec))
        ap = average_precision_score(y_test, y_proba)
        ap_results.append({"seed": seed, "fold": fold, "pos_class_ap": ap})

        # Accumulate for micro‑average later
        all_probs.extend(y_proba)
        all_labels.extend(y_test)

        (
            positive_shap_train,
            negative_shap_train,
            positive_shap_test,
            negative_shap_test,
            shap_values_train,
            shap_values_test,
        ) = get_separated_shap_values_tree(best_model, X_train, X_test)

        fold_dir = os.path.join(plotdir, "shap_plots")
        os.makedirs(fold_dir, exist_ok=True)

        create_plots_nested(
            seed=seed,
            X_test=X_test,
            X_train=X_train,
            fold=fold,
            plotdir=plotdir,
            positive_shap_train=positive_shap_train,
            negative_shap_train=negative_shap_train,
            positive_shap_test=positive_shap_test,
            negative_shap_test=negative_shap_test,
            shap_values_train=shap_values_train,
            shap_values_test=shap_values_test,
        )

        # Calculate positive / negative bin scores separately
        bin_score_pos = match_shap_bins_with_buffer(
            get_top_shap_indices(positive_shap_train),
            get_top_shap_indices(positive_shap_test),
            buffer=1,
        )
        bin_score_neg = match_shap_bins_with_buffer(
            get_top_shap_indices(negative_shap_train),
            get_top_shap_indices(negative_shap_test),
            buffer=1,
        )

        # Identify top‑10 indices (for logging/readability)
        top_10_positive_shap_train = get_top_shap_indices(positive_shap_train)[:10]
        top_10_negative_shap_train = get_top_shap_indices(negative_shap_train)[:10]
        top_10_positive_shap_test = get_top_shap_indices(positive_shap_test)[:10]
        top_10_negative_shap_test = get_top_shap_indices(negative_shap_test)[:10]
        top_10_overall_shap_test = get_top_shap_indices(shap_values_test)[:10]
        top_10_overall_shap_train = get_top_shap_indices(shap_values_train)[:10]

        # Determine thresholds to evaluate
        if auto_threshold:
            unique_test_probs = np.unique(y_proba)
            thresholds_to_use = np.sort(
                np.unique(np.concatenate([unique_test_probs, [0.5]]))
            )
        else:
            thresholds_to_use = thresholds or [0.5]

        all_evaluated_params_json = json.dumps(search.cv_results_["params"], default=str)
        best_params_json = json.dumps(search.best_params_,default=str)

        fold_results_local = []
        for threshold in thresholds_to_use:
            y_pred = (y_proba >= threshold).astype(int)
            conf_matrix = confusion_matrix(y_test, y_pred,  labels=[0, 1])
            tn, fp, fn, tp = conf_matrix.ravel()
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            npv = tn / (tn + fn) if (tn + fn) > 0 else 0
            ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
            overall_f1 = f1_score(y_test, y_pred)

            fold_entry = {
                "seed": seed,
                "fold": fold,
                "threshold": threshold,
                "model": best_model,
                "testdataset": testdataset,
                "traindataset": traindataset,
                "best_params": best_params_json,
                "evaluated_params": all_evaluated_params_json,
                "time_taken": search.refit_time_,
                "NPV": npv,
                "PPV": ppv,
                "TN": tn,
                "FP": fp,
                "FN": fn,
                "TP": tp,
                "conf_matrix": conf_matrix.tolist(),
                "f1_score_0": f1_score(y_test, y_pred, pos_label=0),
                "f1_score_1": f1_score(y_test, y_pred, pos_label=1),
                "f1_score overall": overall_f1,
                "specificity": specificity,
                "sensitivity": sensitivity,
                "mcc": matthews_corrcoef(y_test, y_pred),
                "bin_score_pos": bin_score_pos,
                "bin_score_neg": bin_score_neg,
                "bin_score": bin_score_neg + bin_score_pos,
                "top_10_positive_shap_train": top_10_positive_shap_train,
                "top_10_negative_shap_train": top_10_negative_shap_train,
                "top_10_positive_shap_test": top_10_positive_shap_test,
                "top_10_negative_shap_test": top_10_negative_shap_test,
                'top_10_overall_shap_train': top_10_overall_shap_train,
                'top_10_overall_shap_test': top_10_overall_shap_test,
                "fold_ap": ap
            }

            fold_results_local.append(fold_entry)

        # Best fold result (maximizing MCC) for readability file
        best_for_fold = max(fold_results_local, key=lambda r: r["mcc"])
        best_per_fold.append(best_for_fold)

        results.extend(fold_results_local)

    # After looping over all outer folds
    micro_prec, micro_rec, _ = precision_recall_curve(all_labels, all_probs, pos_label=1)
    micro_ap = average_precision_score(all_labels, all_probs)
    ap_results.append({"seed": seed, "fold": "micro", "pos_class_ap": micro_ap})

    # interpolate every fold onto a common recall grid
    recall_grid = np.linspace(0, 1, 101)
    interp_list = []
    for prec, rec in fold_curves:
        interp = np.interp(recall_grid, rec[::-1], prec[::-1])
        interp_list.append(interp)

    macro_prec = np.mean(interp_list, axis=0)


    # arithmetic mean of the five fold-wise AP values
    macro_ap = np.mean([d["pos_class_ap"] for d in ap_results if isinstance(d["fold"], int)])
    ap_results.append({"seed": seed, "fold": "macro", "pos_class_ap": macro_ap})

    macro_curve = (macro_prec, recall_grid, macro_ap)

    return results, fold_curves, ap_results, macro_curve, best_per_fold

#Create filepaths, ID
run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

tag = f"earlystop{earlystop_n}" if use_outer_early_stop else "noearlystop"

nested_sub_dir = os.path.join(
    f"./results/train{traindataset}test{testdataset}/",
    f"lgbm_nested_{run_id}_{method}_iter{n_iter_call}_{tag}", 
)
os.makedirs(nested_sub_dir, exist_ok=True)

results_file = os.path.join(nested_sub_dir, f"allresults_nested{method}_{run_id}_{scoring_method}.csv")
readable_file = os.path.join(nested_sub_dir, f"readable_nested{method}_{run_id}_{scoring_method}.csv")
summary_file = os.path.join(nested_sub_dir, f"summarystats_nested{method}_{run_id}_{scoring_method}.csv")
auc_file = os.path.join(nested_sub_dir, f"auc_nested{method}_{run_id}_{scoring_method}.csv")

all_results = []
all_aps = []
best_mcc_each_seed = []

for seed in seeds:
    X, y = load_data_nested_inverted(seed=seed, testdataset=testdataset, traindataset=traindataset)

    seed_results,fold_curves,ap_results,macro_curve,best_per_fold = nested_cross_validation(
        X,y,param_grid,
        scoring,
        n_splits=5,
        n_iter=n_iter_call,
        seed=seed,
        thresholds=thresholds_list,
        auto_threshold=auto_threshold,
        plotdir=nested_sub_dir,
        )

    all_results.extend(seed_results)
    best_mcc_each_seed.extend(best_per_fold)
    all_aps.extend(ap_results)

    plot_pr_folds(
        fold_curves=fold_curves,
        macro_curve=macro_curve,
        seed=seed,
        output_path=nested_sub_dir,
    )

write_to_csv_nested(run_id, all_results, scoring_method, results_file)
write_to_readable_file_nested(run_id, best_mcc_each_seed, scoring_method, readable_file)
save_micro_macro_summary_nested(
    all_results=all_results,
    all_aps=all_aps,
    summary_file=summary_file,
    threshold=0.5,
)
pd.DataFrame(all_aps).to_csv(auc_file, index=False)

print("Nested CV complete!")
print(f"Results and plots saved in: {nested_sub_dir}")
