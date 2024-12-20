import pandas as pd
import numpy as np
import scipy


dataset_name="ihdp"
file = f"main_loop_results_{dataset_name}.csv"

results_df = pd.read_csv(file).reset_index(drop=True)
results_df.index = results_df.iloc[:, 0]
results_df = results_df[["PO" not in x for x in results_df.index]]
results_df.drop(results_df.columns[0], axis=1, inplace=True)
results_df["metric_mean"] = [
    np.mean(eval(x.replace("nan", "9999"))) for x in results_df.iloc[:, 0]
]

results_df["metric_sem"] = [
    scipy.stats.sem(eval(x.replace("nan", "9999"))) for x in results_df.iloc[:, 0]
]

results_df = results_df.loc[[x for x in results_df.index if "_RF" not in x]]
# print(results_df["metric_mean"].loc[[x for x in results_df.index if "PEHE" in x]])
# print(results_df["metric_mean"].loc[[x for x in results_df.index if "ATE" in x]])
# print(len(results_df["0"].iloc[0].split(",")))
# print(results_df["metric_sem"].loc[[x for x in results_df.index if "PEHE" in x]])
# print(results_df["metric_sem"].loc[[x for x in results_df.index if "ATE" in x]])

results_df.reset_index(inplace=True)
results_df.columns = ["id", "values", "metric_mean", "metric_sem"]
results_df["tmp"] = results_df['id'].str.split('_')
results_df['model'] = results_df['tmp'].apply(lambda t: t[0])
results_df['type'] = results_df['tmp'].apply(lambda t: t[1])
results_df.sort_values(['model', 'type'], inplace=True)
results_df = results_df.pivot(index=['model'], columns=['type'], values=['metric_mean', 'metric_sem'])
for col in ['metric_mean', 'metric_sem']:
    results_df[(col, 'comp')] = (results_df[(col, 'CONSTR')]) / (results_df[(col, 'VANILLA')])

print(results_df)
