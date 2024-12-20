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
def run_method(method, data, dag_edges, filter_T=False):
    PEHES_RF = []
    ATES_RF = []
    PEHES_VANILLA = []
    ATES_VANILLA = []
    PEHES_CONSTR = []
    ATES_CONSTR = []

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
    if filter_T:
        model_trainer.train_model_nn_constr({"groups_to_t": [0, 1, 2]})
    else:
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

    # Real ITE from the test values
    pred_ITE, rf = model_trainer.slearn_rf(T="T")

    metric_values = dict((k, []) for k in model_trainer.metrics_dict.keys())
    for k, v in model_trainer.metrics_dict.items():
        metric_values[k] = np.round(v(real_ITE, pred_ITE, data), 4)

    PEHES_RF.append(metric_values["PEHE"])
    ATES_RF.append(metric_values["ATE"])

    # Vanilla NNET ---------- CI
    # Real ITE from the test values
    pred_ITE = pred_ITE_unconstr
    metric_values = dict((k, []) for k in model_trainer.metrics_dict.keys())
    for k, v in model_trainer.metrics_dict.items():
        metric_values[k] = np.round(v(real_ITE, pred_ITE, data), 4)

    PEHES_VANILLA.append(metric_values["PEHE"])
    ATES_VANILLA.append(metric_values["ATE"])

    # Constrained NNET ----------
    # Real ITE from the test values
    pred_ITE = pred_ITE_constr
    metric_values = dict((k, []) for k in model_trainer.metrics_dict.keys())
    for k, v in model_trainer.metrics_dict.items():
        metric_values[k] = np.round(v(real_ITE, pred_ITE, data), 4)

    PEHES_CONSTR.append(metric_values["PEHE"])
    ATES_CONSTR.append(metric_values["ATE"])

    return [
        PEHES_RF,
        ATES_RF,
        PEHES_VANILLA,
        ATES_VANILLA,
        PEHES_CONSTR,
        ATES_CONSTR,
    ]


# Start script ----------------------------------------
def basic_run(dataset_name, n_instances, de, Y_noise=1):
    results = {
        m: {
            "PEHES_RF": [],
            "ATES_RF": [],
            "PEHES_VANILLA": [],
            "ATES_VANILLA": [],
            "PEHES_CONSTR": [],
            "ATES_CONSTR": [],
        }
        for m in methods
    }
    for i in selected_range:
        # print(i)
        # n ∈ {500, 1000}, d ∈ {6, 12} and σ ∈ {0.5, 1, 2, 3},
        data = data_load.dataLoader(
            data_path="../../../data/",
            n_instances=n_instances,
            n_vars=-1,
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

        filter_T = False
        if "filter_t" in de:
            filter_T = True
            de = de.replace("filter_t", "")

        dag_edges = dag_creator.predefined_dags[dataset_name + de[:-1]]()

        # Run methods ------------------------
        for method in methods:
            l = run_method(method, data, dag_edges, filter_T)
            results[method]["PEHES_RF"].append(l[0][0])
            results[method]["ATES_RF"].append(l[1][0])
            results[method]["PEHES_VANILLA"].append(l[2][0])
            results[method]["ATES_VANILLA"].append(l[3][0])
            results[method]["PEHES_CONSTR"].append(l[4][0])
            results[method]["ATES_CONSTR"].append(l[5][0])

    results_df = pd.DataFrame(utils.nested_dict_to_series(results))

    return results_df


# PRINTSS
methods = ["Dragonnet"]
selected_range = list(range(100))

dataset_name_list = [
    "toy_example_2",
]
n_instances_list = [500]
dag_edges_list = ["-", "_single-", "_no_individual-", "_no_neutral-"]
Y_noise_list = [1]

exp_results_pd = pd.DataFrame()
for de in dag_edges_list:
    for n_instances in n_instances_list:
        for Y_noise in Y_noise_list:
            for dataset_name in dataset_name_list:

                if (dataset_name == "toy_example") & (de == "_no_neutral-"):
                    continue
                if (dataset_name == "toy_example") & (de == "-filter_t-"):
                    continue

                print(f"Starting {dataset_name}, {Y_noise}, {n_instances}, {de}")
                results_df = basic_run(dataset_name, n_instances, de, Y_noise)
                results_df_loop = pd.DataFrame(results_df.to_dict()[0])
                results_df_loop["dataset_name"] = dataset_name
                results_df_loop["n_instances"] = n_instances
                results_df_loop["Y_noise"] = Y_noise
                results_df_loop["n_vars"] = de

                exp_results_pd = pd.concat(
                    [exp_results_pd, results_df_loop], axis=0, ignore_index=True
                )

            exp_results_pd.to_csv("main_loop_results_toy.csv")
