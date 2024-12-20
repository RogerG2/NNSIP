import pandas as pd
import numpy as np
import os
import sys
import random
import torch
import argparse

sys.path.append("..")
import causal_discovery, data_load, models_nnets, utils


# PYTHONHASHSEED=0 python main_nets_loop_multimethod.py
os.environ["PYTHONHASHSEED"] = "0"
seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
print(torch.set_default_dtype(torch.float32))


# ARGS  ------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default='data/', help='The path to the data folder')
parser.add_argument('--dataset_name', type=str, default='ihdp', choices=['ihdp', 'jobs'], help='The name of the dataset')
parser.add_argument('--pred_test', action='store_true', help='Whether to predict on the test set')


args = parser.parse_args()
data_path = args.data_path
dataset_name = args.dataset_name
pred_test = args.pred_test

methods = ["BCAUSS", "Dragonnet", "Tarnet"]
selected_range = list(range(1000)) if dataset_name == "ihdp" else list(range(10)) * 10


# F ------------------------------------------------------------------------------
def run_method(method, data, dag_edges):
    PEHES_RF = []
    ATES_RF = []
    PEHES_VANILLA = []
    ATES_VANILLA = []
    PEHES_CONSTR = []
    ATES_CONSTR = []

    # Modelling ----------------------------------------------------------------
    model_trainer = models_nnets.modelTrainer(
        data.train_df.copy(),
        data.X_features.copy(),
        data.Y_feature.copy(),
        test_df=data.test_df.copy(),
        dag_edges=dag_edges,
        cv_indexes=data.cv_indexes,
        task="regression",
        learner=method,
        data_loader=data if dataset_name == "ihdp" else None,
        y_scale=dataset_name != "jobs",
    )

    # PRED ----------------
    # unconstrained
    model_trainer.train_model_nn()
    Y_pred, pred_ITE_unconstr, t_pred = model_trainer.predict_model_nn(pred_test)

    # constrained
    model_trainer.train_model_nn_constr()
    Y_pred, pred_ITE_constr, t_pred = model_trainer.predict_model_nn(pred_test)

    # CI ----------------
    real_ITE = data.test_ite if pred_test else data.train_ite

    model_trainer.metrics_dict.clear()
    model_trainer.metrics_dict["ATE"] = utils.compute_ATE
    model_trainer.metrics_dict["PEHE"] = utils.compute_PEHE

    if dataset_name == "jobs":
        model_trainer.metrics_dict["PEHE"] = utils.risk_jobs
        model_trainer.metrics_dict["ATE"] = utils.att_jobs

    # Real ITE from the test values
    pred_ITE, rf = model_trainer.slearn_rf(T="T", pred_test=pred_test)

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

    print(i)
    data = data_load.dataLoader(
        data_path=data_path,
        n_vars=5,
        X_noise=0,
        test_perc=0.2,
        cv_splits=5,
        rep_i=i,
    )
    data.load_dataset(dataset_name)
    data.pred_test = pred_test

    # Extract DAG ------------------------------------------------------------------
    dag_creator = causal_discovery.DAGCreator(
        data.train_df,
        data.X_features,
        data.Y_feature,
        method="icalingam",
        max_samples=1000,
    )

    dag_edges = dag_creator.create_dag_edges()
    if len(dag_edges) == 0:
        dag_edges = dag_creator.return_predefined_dag(dataset_name)

    nodes_in_edges = [item for sublist in dag_edges for item in sublist]
    nodes_not_in_edges = [x for x in dag_creator.X_features if x not in nodes_in_edges]
    dag_edges.append(nodes_not_in_edges)

    # Run methods ------------------------
    for method in methods:

        l = run_method(method, data, dag_edges)
        results[method]["PEHES_RF"].append(l[0][0])
        results[method]["ATES_RF"].append(l[1][0])
        results[method]["PEHES_VANILLA"].append(l[2][0])
        results[method]["ATES_VANILLA"].append(l[3][0])
        results[method]["PEHES_CONSTR"].append(l[4][0])
        results[method]["ATES_CONSTR"].append(l[5][0])

    if i % 50 == 0:
        results_df = pd.DataFrame(utils.nested_dict_to_series(results))
        results_df.to_csv(f"main_loop_results_{dataset_name}.csv")

results_df = pd.DataFrame(utils.nested_dict_to_series(results))
results_df.to_csv(f"main_loop_results_{dataset_name}.csv")
