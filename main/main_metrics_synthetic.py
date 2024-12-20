import pandas as pd
import numpy as np
import scipy

pd.set_option("display.max_rows", 200)
pd.set_option("display.max_columns", None)
df = pd.read_csv("main_loop_results_syn.csv")
PEHES_VARS = [x for x in df.columns if "PEHES" in x]
ATES_VARS = [x for x in df.columns if "ATE" in x]
grp_vars = ["dataset_name", "Y_noise", "n_instances", "n_vars"]
# grp_vars = ["dataset_name", "Y_noise"]
# grp_vars = ["dataset_name"]
# grp_vars = ["Y_noise"]
# grp_vars = ["dataset_name", "Y_noise", "n_vars"]
BCAUSS_Vars = [x for x in df.columns if "BCAUSS" in x]
dragonnet_vars = [x for x in df.columns if "Dragonnet" in x]
tarnet_Vars = [x for x in df.columns if "Tarnet" in x]


def print_results(df, metric_vars, method_vars):
    df_g = df.groupby(grp_vars)[
        list(set(metric_vars).intersection(set(method_vars)))
    ].mean()
    # ].std()
    c_col = [x for x in df_g.columns if "CONSTR" in x][0]
    v_col = [x for x in df_g.columns if "VANILLA" in x][0]
    df_g["comp"] = df_g[c_col] / df_g[v_col]
    # print(df_g)
    print(df_g.columns)
    print(df_g["comp"].round(2))
    return df_g["comp"].round(2)


g1 = print_results(df, PEHES_VARS, BCAUSS_Vars)
g2 = print_results(df, PEHES_VARS, dragonnet_vars)
g3 = print_results(df, PEHES_VARS, tarnet_Vars)

g1 = print_results(df, ATES_VARS, BCAUSS_Vars)
g2 = print_results(df, ATES_VARS, dragonnet_vars)
g3 = print_results(df, ATES_VARS, tarnet_Vars)


metric_vars = PEHES_VARS + ATES_VARS
method_vars = BCAUSS_Vars + dragonnet_vars + tarnet_Vars
grp_vars = ["dataset_name", "Y_noise", "n_instances", "n_vars"]
df_g = df.groupby(grp_vars)[
    list(set(metric_vars).intersection(set(method_vars)))
].agg(lambda x: scipy.stats.sem(x, axis=0, nan_policy='omit'))


grp_vars = ["dataset_name"]
df_g2 = df_g.groupby(grp_vars)[
    list(set(metric_vars).intersection(set(method_vars)))
].mean()
