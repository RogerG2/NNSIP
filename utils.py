import numpy as np
import pandas as pd


# AUX FUNC -----------------------------
def compute_ATE(real_ITE, pred_ITE, *args):
    return np.abs((np.mean(real_ITE) - np.mean(pred_ITE)))


def compute_PEHE(real_ITE, pred_ITE, *args):
    return np.sqrt(np.mean((real_ITE - pred_ITE) ** 2))


def att_jobs(real, pred, data):

    df = data.test_df if data.pred_test else data.train_df

    e = df["e"].values == 1
    t = df["T"].values == 1
    c = df["T"].values == 0  # Control group
    Y = df[data.Y_feature[0]].values

    att = np.mean(Y[t]) - np.mean(Y[c & e])
    pred_att = np.mean(np.array(pred)[t])

    return np.abs(att - pred_att)


def risk_jobs(real, pred, data):

    df = data.test_df if data.pred_test else data.train_df

    e = df["e"].values == 1
    t = df["T"].values == 1
    c = df["T"].values == 0  # Control group
    y = df[data.Y_feature[0]].values

    # policy = ypred1 > ypred0
    policy = np.array(pred) > 0
    if sum(policy) < 1:
        return 9999
    ppolicy = policy.mean()

    return 1 - (
        y[e & policy & t].mean() * ppolicy + y[e & ~policy & ~t].mean() * (1 - ppolicy)
    )


def nested_dict_to_series(nested_dict):
    def flatten_dict(d, parent_key="", sep=";"):
        items = []
        for k, v in d.items():
            k = str(k)
            new_key = parent_key + sep + k if parent_key else k
            if isinstance(v, dict):
                items.extend(flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)

    flat_dict = flatten_dict(nested_dict)
    series = pd.Series(flat_dict)
    # series.index = series.index.str.replace('_', '.')
    return series


def r_risk_syn(real, pred, data):
    hat_m = data.m[-data.n_instances :]
    hat_e = data.e[-data.n_instances :]
    a = data.test_df["T"]
    y = data.test_df[data.Y_feature[0]]
    hat_tau = pred

    return np.mean(((y - hat_m) - (a - hat_e) * hat_tau) ** 2)


def graph_edit_distance(edges1, edges2):
    # Convert edge lists to sets for easier comparison
    set_edges1 = set(edges1)
    set_edges2 = set(edges2)
    
    # Calculate the symmetric difference of the edge sets
    sym_diff_edges = set_edges1.symmetric_difference(set_edges2)
    
    # The edit distance is the size of the symmetric difference
    # since each edge in the symmetric difference needs to be added or removed
    distance = len(sym_diff_edges)
    
    return distance

# # Example usage:
# # Define two graphs by their edge lists
# G1_edges = [(1, 2), (2, 3)]
# G2_edges = [(4, 5), (5, 6)]

# # Calculate Graph Edit Distance
# ged = graph_edit_distance(G1_edges, G2_edges)
# print(f"The Graph Edit Distance between G1 and G2 is: {ged}")