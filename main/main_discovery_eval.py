import pandas as pd
import numpy as np
import os
import sys
import random

sys.path.append("..")
import causal_discovery, data_load, utils


# PYTHONHASHSEED=0 python main_nets_loop_multimethod.py
os.environ["PYTHONHASHSEED"] = "0"
seed = 0
random.seed(seed)
np.random.seed(seed)


G_real_dict = {
    "causalml_mode_1": [
        ("col_0", "Y"),
        ("col_1", "Y"),
        ("col_2", "Y"),
        ("col_3", "Y"),
        ("col_4", "Y"),
        ("T", "Y"),
        ("col_0", "T"),
        ("col_1", "T"),
    ],
    "causalml_mode_2": [
        ("col_0", "Y"),
        ("col_1", "Y"),
        ("col_2", "Y"),
        ("col_3", "Y"),
        ("col_4", "Y"),
        ("T", "Y"),
    ],
    "causalml_mode_3": [
        ("col_0", "Y"),
        ("col_1", "Y"),
        ("col_2", "Y"),
        ("T", "Y"),
        ("col_1", "T"),
        ("col_2", "T"),
    ],
    "causalml_mode_4": [
        ("col_0", "Y"),
        ("col_1", "Y"),
        ("col_2", "Y"),
        ("col_3", "Y"),
        ("col_4", "Y"),
        ("T", "Y"),
        ("col_0", "T"),
        ("col_1", "T"),
    ],
}


def basic_run(dataset_name, n_instances, n_vars, Y_noise):
    data = data_load.dataLoader(
        data_path="../../../data/",
        n_instances=n_instances,
        n_vars=n_vars,
        X_noise=0,
        Y_noise=Y_noise,
        test_perc=0.2,
        cv_splits=5,
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
    dag_creator.run_gcastle(dag_creator.method)
    G_discovered = dag_creator.dag_edges
    G_real = G_real_dict[dataset_name]

    return utils.graph_edit_distance(G_discovered, G_real)


dataset_name_list = [
    "causalml_mode_1",
    "causalml_mode_2",
    "causalml_mode_3",
    "causalml_mode_4",
]
n_instances_list = [1000, 500]
n_vars_list = [6, 12]
Y_noise_list = [0.5, 1, 2, 4]

df = pd.DataFrame()
for n_vars in n_vars_list:
    for n_instances in n_instances_list:
        for Y_noise in Y_noise_list:
            for dataset_name in dataset_name_list:

                for i in range(10):

                    rr = {
                        "n_vars": n_vars,
                        "n_instances": n_instances,
                        "Y_noise": Y_noise,
                        "dataset_name": dataset_name,
                        "GED": basic_run(dataset_name, n_instances, n_vars, Y_noise),
                    }

                    rr_series = pd.Series(rr).to_frame().T
                    df = pd.concat([df, rr_series], ignore_index=True)

            #     print(f"Starting {dataset_name}, {Y_noise}, {n_instances}, {n_vars}")
            #     results_df = basic_run(dataset_name, n_instances, n_vars, Y_noise)
            #     results_df_loop = pd.DataFrame(results_df.to_dict()[0])
            #     results_df_loop["dataset_name"] = dataset_name
            #     results_df_loop["n_instances"] = n_instances
            #     results_df_loop["Y_noise"] = Y_noise
            #     results_df_loop["n_vars"] = n_vars

            #     exp_results_pd = pd.concat(
            #         [exp_results_pd, results_df_loop], axis=0, ignore_index=True
            #     )

            # exp_results_pd.to_csv("main_loop_results_syn.csv")


print(df[["Y_noise", "GED"]].groupby("Y_noise").mean())
