import pandas as pd
import numpy as np
import os
import sys
import random
import torch

sys.path.append("..")
import causal_discovery, data_load, models_nnets, utils


# PYTHONHASHSEED=0 python main_nets_loop_multimethod.py
os.environ["PYTHONHASHSEED"] = "0"
seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
print(torch.set_default_dtype(torch.float32))


# Data  ------------------------------------------------------------------------


def run_method(method, data, dag_edges):
    PEHES_RF = []
    ATES_RF = []
    PEHES_VANILLA = []
    ATES_VANILLA = []
    PEHES_CONSTR = []
    ATES_CONSTR = []
    R_RISK_RF = []
    R_RISK_VANILLA = []
    R_RISK_CONSTR = []

    # Modelling --------------------------------------------------------------------
    model_trainer = models_nnets.modelTrainer(
        data.train_df.copy(),
        data.X_features.copy(),
        data.Y_feature.copy(),
        test_df=data.test_df.copy(),
        dag_edges=dag_edges,
        cv_indexes=data.cv_indexes,
        task="regression",
        learner=method,
    )

    # PRED ----------------
    # unconstrained
    model_trainer.train_model_nn()
    Y_pred, pred_ITE_unconstr, t_pred = model_trainer.predict_model_nn()
    ## Acc by T ##### --------
    df = model_trainer.test_df[["T", "Y"]].copy()
    df["Y_pred"] = Y_pred
    df["Y_abs_diff"] = np.abs(df["Y"] - df["Y_pred"])
    # ------------------------

    # constrained
    model_trainer.train_model_nn_constr()
    Y_pred, pred_ITE_constr, t_pred = model_trainer.predict_model_nn()
    ## Acc by T ##### --------
    df = model_trainer.test_df[["T", "Y"]].copy()
    df["Y_pred"] = Y_pred
    df["Y_abs_diff"] = np.abs(df["Y"] - df["Y_pred"])
    # ------------------------

    # CI ----------------
    real_ITE = data.test_ite

    model_trainer.metrics_dict.clear()
    model_trainer.metrics_dict["ATE"] = utils.compute_ATE
    model_trainer.metrics_dict["PEHE"] = utils.compute_PEHE
    model_trainer.metrics_dict["R_RISK"] = utils.r_risk_syn

    # Real ITE from the test values
    pred_ITE, rf = model_trainer.slearn_rf(T="T")

    metric_values = dict((k, []) for k in model_trainer.metrics_dict.keys())
    for k, v in model_trainer.metrics_dict.items():
        metric_values[k] = np.round(v(real_ITE, pred_ITE, data), 4)

    PEHES_RF.append(metric_values["PEHE"])
    ATES_RF.append(metric_values["ATE"])
    R_RISK_RF.append(metric_values["R_RISK"])

    # Vanilla NNET ---------- CI
    # Real ITE from the test values
    pred_ITE = pred_ITE_unconstr
    metric_values = dict((k, []) for k in model_trainer.metrics_dict.keys())
    for k, v in model_trainer.metrics_dict.items():
        metric_values[k] = np.round(v(real_ITE, pred_ITE, data), 4)

    PEHES_VANILLA.append(metric_values["PEHE"])
    ATES_VANILLA.append(metric_values["ATE"])
    R_RISK_VANILLA.append(metric_values["R_RISK"])

    # Constrained NNET ----------
    # Real ITE from the test values
    pred_ITE = pred_ITE_constr
    metric_values = dict((k, []) for k in model_trainer.metrics_dict.keys())
    for k, v in model_trainer.metrics_dict.items():
        metric_values[k] = np.round(v(real_ITE, pred_ITE, data), 4)

    PEHES_CONSTR.append(metric_values["PEHE"])
    ATES_CONSTR.append(metric_values["ATE"])
    R_RISK_CONSTR.append(metric_values["R_RISK"])

    return [
        PEHES_RF,
        ATES_RF,
        PEHES_VANILLA,
        ATES_VANILLA,
        PEHES_CONSTR,
        ATES_CONSTR,
        R_RISK_RF,
        R_RISK_VANILLA,
        R_RISK_CONSTR,
    ]


# Start script ----------------------------------------
def basic_run(dataset_name, n_instances, n_vars=5, Y_noise=1):
    results = {
        m: {
            "PEHES_RF": [],
            "ATES_RF": [],
            "PEHES_VANILLA": [],
            "ATES_VANILLA": [],
            "PEHES_CONSTR": [],
            "ATES_CONSTR": [],
            "R_RISK_RF": [],
            "R_RISK_VANILLA": [],
            "R_RISK_CONSTR": [],
        }
        for m in methods
    }
    for i in selected_range:
        # print(i)
        # n ∈ {500, 1000}, d ∈ {6, 12} and σ ∈ {0.5, 1, 2, 3},
        data = data_load.dataLoader(
            data_path="../../../data/",
            n_instances=n_instances,
            n_vars=n_vars,
            X_noise=0,
            Y_noise=Y_noise,
            test_perc=0.2,
            cv_splits=5,
            rep_i=i,
        )
        data.load_dataset(dataset_name)

        # Extract DAG ------------------------------------------------------------------
        dag_creator = causal_discovery.DAGCreator(
            data.train_df,
            data.X_features,
            data.Y_feature,
            method="icalingam",
            max_samples=1000,
        )

        dag_edges = dag_creator.predefined_dags[dataset_name]()

        # Run methods ------------------------
        for method in methods:
            l = run_method(method, data, dag_edges)
            results[method]["PEHES_RF"].append(l[0][0])
            results[method]["ATES_RF"].append(l[1][0])
            results[method]["PEHES_VANILLA"].append(l[2][0])
            results[method]["ATES_VANILLA"].append(l[3][0])
            results[method]["PEHES_CONSTR"].append(l[4][0])
            results[method]["ATES_CONSTR"].append(l[5][0])
            results[method]["R_RISK_RF"].append(l[6][0])
            results[method]["R_RISK_VANILLA"].append(l[7][0])
            results[method]["R_RISK_CONSTR"].append(l[8][0])

    results_df = pd.DataFrame(utils.nested_dict_to_series(results))

    return results_df


# PRINTSS
methods = ["BCAUSS", "Dragonnet", "Tarnet"]
selected_range = list(range(100))

dataset_name_list = [
    "causalml_mode_1",
    "causalml_mode_2",
    "causalml_mode_3",
    "causalml_mode_4",
]
n_instances_list = [1000, 500]
n_vars_list = [6, 12]
Y_noise_list = [0.5, 1, 2, 4]

exp_results_pd = pd.DataFrame()
for n_vars in n_vars_list:
    for n_instances in n_instances_list:
        for Y_noise in Y_noise_list:
            for dataset_name in dataset_name_list:

                print(f"Starting {dataset_name}, {Y_noise}, {n_instances}, {n_vars}")
                results_df = basic_run(dataset_name, n_instances, n_vars, Y_noise)
                results_df_loop = pd.DataFrame(results_df.to_dict()[0])
                results_df_loop["dataset_name"] = dataset_name
                results_df_loop["n_instances"] = n_instances
                results_df_loop["Y_noise"] = Y_noise
                results_df_loop["n_vars"] = n_vars

                exp_results_pd = pd.concat(
                    [exp_results_pd, results_df_loop], axis=0, ignore_index=True
                )

            exp_results_pd.to_csv("main_loop_results_syn.csv")
