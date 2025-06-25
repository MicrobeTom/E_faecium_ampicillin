from datetime import datetime
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import randint, uniform
from lightgbm import LGBMClassifier
from lightgbm.callback import early_stopping
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    balanced_accuracy_score,
    precision_recall_curve,
    average_precision_score,
    make_scorer,
)
from sklearn.model_selection import (
    StratifiedKFold,
    RandomizedSearchCV,
    train_test_split,
)

from tools_final import (
    load_data_comb_5f_inverted,
    get_separated_shap_values_tree,
    get_top_shap_indices,
    match_shap_bins_with_buffer,
    write_to_csv_5fcomb,
    write_to_readable_file_5fcomb_inverted,
    write_summary_stats_file5fcomb,
    write_meta_summary_from_summary,
    create_plots_5f,
    plot_external_test_curve
)

#########################
method = "RandomCV"
# traindataset = "MS-UMG_reg_warpedSELF"
# testdataset = "TUM-warpedMS-UMG"
traindataset = "TUM-warped"
testdataset = "MS-UMG_reg_warpedTUM"
scoring_method = "mcc"
n_iter_call = 50
use_early_stop = True
early_stop_rounds = 50
#########################
# List of seeds for multiple runs
seeds = list(range(42, 142, 20))

thresholds = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

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

# Define scoring based on user's choice
if scoring_method == "f1-score positive":
    scoring = make_scorer(f1_score, pos_label=1)
elif scoring_method == "mcc":
    scoring = make_scorer(matthews_corrcoef)
else:
    raise ValueError("Invalid scoring method. Choose 'f1-score positive' or 'mcc'.")

def tune_and_evaluate(
    X_train, y_train, X_test, y_test,
    param_distributions, scoring, n_iter=100, seed=None,
    thresholds=None, get_shap_values_fn=get_separated_shap_values_tree, auto_threshold=False
):
    # Hyperparameter tuning
    lgb_base = LGBMClassifier(
        objective    = "binary",
        n_jobs       = -1,
        random_state = seed,
        n_estimators = 3000,
        verbosity    = -1,
    )
    cv_object = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    random_search = RandomizedSearchCV(
        estimator           = lgb_base,
        param_distributions = param_distributions,
        n_iter              = n_iter,
        scoring             = scoring,
        cv                  = cv_object,
        random_state        = seed,
        n_jobs              = -1,
        verbose             = 0,
    )
    print("RandomSearchCV launched…")

    random_search.fit(X_train, y_train, eval_metric="binary_logloss")

    best_params = random_search.best_params_
    best_score  = random_search.best_score_
    print(f"[Seed={seed}] Best params: {best_params}")
    print(f"[Seed={seed}] CV score   : {best_score:.4f}")
    
    if use_early_stop:
        # keep 10 % of TRAIN as validation for the callback
        X_sub_tr, X_val, y_sub_tr, y_val = train_test_split(
            X_train, y_train,
            test_size=0.10, stratify=y_train, random_state=seed
        )
        ceiling = 3000
        params_refit = {k:v for k,v in best_params.items() if k != "n_estimators"}
        best_params_original = best_params.copy()
        final_model = LGBMClassifier(
            objective="binary",
            n_jobs=-1,
            verbosity=-1,
            random_state=seed,
            n_estimators=ceiling,
            **params_refit
        )
        final_model.fit(
            X_sub_tr, y_sub_tr,
            eval_set=[(X_val, y_val)],
            eval_metric="binary_logloss",
            callbacks=[early_stopping(early_stop_rounds, verbose=False)]
        )
    else:
        # train the exact forest size chosen by the search
        best_params_original = best_params
        final_model = LGBMClassifier(
            objective="binary",
            n_jobs=-1,
            verbosity=-1,
            random_state=seed,
            **best_params
        )
        final_model.fit(X_train, y_train, eval_metric="binary_logloss")
    
    # Probabilities + SHAP (tree version)
    y_proba_train = final_model.predict_proba(X_train)[:, 1]
    y_proba_test  = final_model.predict_proba(X_test)[:, 1]

    (positive_shap_train, negative_shap_train,
     positive_shap_test,  negative_shap_test,
     shap_values_train,   shap_values_test) = get_shap_values_fn(
        final_model, X_train, X_test
    )

   
    create_plots_5f(
        seed               = seed,
        X_train            = X_train,
        X_test             = X_test,
        plotdir            = sub_dir,
        positive_shap_train= positive_shap_train,
        negative_shap_train= negative_shap_train,
        positive_shap_test = positive_shap_test,
        negative_shap_test = negative_shap_test,
        shap_values_train  = shap_values_train,
        shap_values_test   = shap_values_test,
        max_display        = 10
    )

    bin_score_pos = match_shap_bins_with_buffer(
        get_top_shap_indices(positive_shap_train),
        get_top_shap_indices(positive_shap_test),
        buffer=1
    )
    bin_score_neg = match_shap_bins_with_buffer(
        get_top_shap_indices(negative_shap_train),
        get_top_shap_indices(negative_shap_test),
        buffer=1
    )

    top_10_positive_shap_train = get_top_shap_indices(positive_shap_train)[:10]
    top_10_negative_shap_train = get_top_shap_indices(negative_shap_train)[:10]
    top_10_positive_shap_test = get_top_shap_indices(positive_shap_test)[:10]
    top_10_negative_shap_test = get_top_shap_indices(negative_shap_test)[:10]
    top_10_overall_shap_test = get_top_shap_indices(shap_values_test)[:10]
    top_10_overall_shap_train = get_top_shap_indices(shap_values_train)[:10]


    # Evaluate across thresholds
    if auto_threshold:
            unique_test_probs = np.unique(y_proba_test)
            thresholds_to_use = np.sort(np.unique(np.concatenate([unique_test_probs, [0.5]])))
            print(f"Auto-derived thresholds from test set: {len(thresholds_to_use)} unique values")
    elif thresholds is not None:
        thresholds_to_use = thresholds
    else:
        thresholds_to_use = [0.5]
    results = []

    for threshold in thresholds_to_use:
        # TRAIN metrics at threshold
        y_pred_train = (y_proba_train >= threshold).astype(int)
        conf_mat_train = confusion_matrix(y_train, y_pred_train)
        train_tn, train_fp, train_fn, train_tp = conf_mat_train.ravel()

        train_specificity = train_tn / (train_tn + train_fp) if (train_tn + train_fp) > 0 else 0
        train_sensitivity = train_tp / (train_tp + train_fn) if (train_tp + train_fn) > 0 else 0
        train_f1_pos = f1_score(y_train, y_pred_train, pos_label=1)
        train_f1_neg = f1_score(y_train, y_pred_train, pos_label=0)
        train_mcc = matthews_corrcoef(y_train, y_pred_train)
        train_bal_acc = balanced_accuracy_score(y_train, y_pred_train)
        train_ppv = train_tp / (train_tp + train_fp) if (train_tp + train_fp) > 0 else 0
        train_npv = train_tn / (train_tn + train_fn) if (train_tn + train_fn) > 0 else 0

        # TEST metrics at threshold
        y_pred_test = (y_proba_test >= threshold).astype(int)
        conf_mat_test = confusion_matrix(y_test, y_pred_test)
        test_tn, test_fp, test_fn, test_tp = conf_mat_test.ravel()

        test_specificity = test_tn / (test_tn + test_fp) if (test_tn + test_fp) > 0 else 0
        test_sensitivity = test_tp / (test_tp + test_fn) if (test_tp + test_fn) > 0 else 0
        test_f1_pos = f1_score(y_test, y_pred_test, pos_label=1)
        test_f1_neg = f1_score(y_test, y_pred_test, pos_label=0)
        test_mcc = matthews_corrcoef(y_test, y_pred_test)
        test_bal_acc  = balanced_accuracy_score(y_test, y_pred_test)
        test_ppv = test_tp / (test_tp + test_fp) if (test_tp + test_fp) > 0 else 0
        test_npv = test_tn / (test_tn + test_fn) if (test_tn + test_fn) > 0 else 0

        #Save all evaluated params
        all_evaluated_params = random_search.cv_results_["params"]

        # Append combined result
        results.append({
            'seed': seed,
            'threshold': threshold,
            'best_params': best_params_original,
            'cv_best_score': best_score,
            'model_details': str(final_model),
            'traindataset': traindataset,
            'testdataset': testdataset,
            'evaluated_params': json.dumps(all_evaluated_params),


            # ----- TRAIN METRICS -----
            'train_conf_matrix': conf_mat_train,
            'train_TN': train_tn,
            'train_FP': train_fp,
            'train_FN': train_fn,
            'train_TP': train_tp,
            'train_f1_pos': train_f1_pos,
            'train_f1_neg': train_f1_neg,
            'train_specificity': train_specificity,
            'train_sensitivity': train_sensitivity,
            'train_mcc': train_mcc,
            'train_bal_acc':train_bal_acc,
            'train_ppv': train_ppv,
            'train_npv': train_npv,

            # ----- TEST METRICS -----
            'test_conf_matrix': conf_mat_test,
            'test_TN': test_tn,
            'test_FP': test_fp,
            'test_FN': test_fn,
            'test_TP': test_tp,
            'test_f1_pos': test_f1_pos,
            'test_f1_neg': test_f1_neg,
            'test_specificity': test_specificity,
            'test_sensitivity': test_sensitivity,
            'test_mcc': test_mcc,
            'test_bal_acc':test_bal_acc,
            'test_ppv': test_ppv,
            'test_npv': test_npv,

            # ----- SHAP/BIN SCORES -----
            'bin_score_pos': bin_score_pos,
            'bin_score_neg': bin_score_neg,
            'bin_score': bin_score_pos + bin_score_neg,
            'top_10_positive_shap_train': top_10_positive_shap_train,
            'top_10_negative_shap_train': top_10_negative_shap_train,
            'top_10_positive_shap_test': top_10_positive_shap_test,
            'top_10_negative_shap_test': top_10_negative_shap_test,
            'top_10_overall_shap_train': top_10_overall_shap_train,
            'top_10_overall_shap_test': top_10_overall_shap_test
        })

    return results, final_model



# main script
if __name__ == "__main__":

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    if use_early_stop:
        earlystop = f"early_stop{early_stop_rounds}"
    else:
        earlystop = "noearlystop"

    sub_dir = os.path.join(f"./results/train{traindataset}test{testdataset}/", f"lgbm_{run_id}_{method}_itercall{n_iter_call}_{earlystop}")
    os.makedirs(sub_dir, exist_ok=True)

    results_file = os.path.join(sub_dir, f"allresults_{method}_{run_id}_{scoring_method}.csv")
    readable_file = os.path.join(sub_dir, f"readable_{method}_{run_id}_{scoring_method}.csv")
    summary_file  = os.path.join(sub_dir, f"summarystats_{method}_{run_id}_{scoring_method}.csv")
    meta_file = os.path.join(sub_dir, f"meta_summary_{method}_{run_id}_{scoring_method}.csv")

    all_results = []
    auc_results = []

    for seed in seeds:
        X_train, y_train, X_test, y_test = load_data_comb_5f_inverted(
            seed=seed,
            testdataset=testdataset,
            traindataset=traindataset
        )
        print(f"Running model for seed: {seed}")

        # Tune on TRAIN set, then evaluate on TEST set
        results_for_seed, final_model = tune_and_evaluate(
            X_train, y_train,
            X_test,  y_test,
            param_distributions = param_grid,
            scoring             = scoring,
            n_iter              = n_iter_call,
            seed                = seed,
            thresholds          = thresholds,
            get_shap_values_fn  = get_separated_shap_values_tree,   # ← NEW
            auto_threshold      = True,
        )

        all_results.extend(results_for_seed)

        y_proba_test = final_model.predict_proba(X_test)[:, 1]
        y_proba_train = final_model.predict_proba(X_train)[:, 1]

        # precision_recall_curve
        prec_test, rec_test, thr_test = precision_recall_curve(y_test, y_proba_test, pos_label=1)
        test_ap = average_precision_score(y_test, y_proba_test)

        prec_train, rec_train, thr_train = precision_recall_curve(y_train, y_proba_train, pos_label=1)
        train_ap = average_precision_score(y_train, y_proba_train)

        print(f"[Seed={seed}] Positive-class area under Precision-Recall (TEST): {test_ap:.3f}")
        print(f"[Seed={seed}] Positive-class area under Precision recall (TRAIN): {train_ap:.3f}")

        auc_results.append({
            'seed': seed,
            'train_ap': train_ap,
            'test_ap': test_ap
        })

        auc_lookup = {d["seed"]: d for d in auc_results}


        plot_external_test_curve(testdataset=testdataset,
                                sub_dir=sub_dir,
                                rec_test=rec_test,
                                prec_test=prec_test,
                                seed=seed,
                                test_ap=test_ap)

        

    # Save results to CSV and other files

    df_all = pd.DataFrame(all_results)
    best_rows = df_all.loc[df_all.groupby('seed')['test_mcc'].idxmax()].copy()
    df_threshold05 = df_all[df_all["threshold"] == 0.5]
    df_combined = pd.concat([df_threshold05, best_rows], ignore_index=True)
    threshold_05_andbest = df_combined.to_dict(orient="records")

    write_to_csv_5fcomb(run_id, all_results,scoring_method, results_file)
    write_to_readable_file_5fcomb_inverted(run_id, all_results, best_rows, auc_results, scoring_method, readable_file)
    write_summary_stats_file5fcomb(threshold_05_andbest, summary_file, auc_lookup)
    write_meta_summary_from_summary(summary_file=summary_file, out_file=meta_file, threshold=0.5)

    print("Done!")


