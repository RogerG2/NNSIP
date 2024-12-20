import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from learners.bcauss import BCAUSSTrainer
from learners.bcauss_constr import BCAUSSConstrainedTrainer
from learners.dragonnet import DragonNetTrainer, TARNetTrainer
from learners.dragonnet_constr import (
    DragonNetConstrainedTrainer,
    TARNetConstrainedTrainer,
)


class modelTrainer:
    def __init__(
        self,
        train_df,
        X_features,
        Y_feature,
        test_df=None,
        dag_edges=None,
        model_params=None,
        cv_indexes=None,
        task="regression",
        t="T",
        method="Default",
        learner="BCAUSS",
        data_loader=None,
        remove_t=True,
        y_scale=True,
    ) -> None:

        # todel, but done to compare apples to apples with bcaus
        yscaler = np.array(
            list(train_df[Y_feature].values) + list(test_df[Y_feature].values)
        )
        if y_scale:
            y_scaler = StandardScaler().fit(yscaler)
        else:
            y_scaler = MinMaxScaler().fit(yscaler)

        # y_scaler = StandardScaler().fit(train_df[Y_feature].to_numpy())
        train_df[Y_feature[0]] = y_scaler.transform(train_df[Y_feature].to_numpy())
        test_df[Y_feature[0]] = y_scaler.transform(test_df[Y_feature].to_numpy())

        self.learner = learner
        if learner == "BCAUSS":
            self.trainer = BCAUSSTrainer
            self.trainer_constr = BCAUSSConstrainedTrainer

        elif learner == "Dragonnet":
            self.trainer = DragonNetTrainer
            self.trainer_constr = DragonNetConstrainedTrainer

        elif learner == "Tarnet":
            self.trainer = TARNetTrainer
            self.trainer_constr = TARNetConstrainedTrainer

        if remove_t:
            # Remove T from dag edges
            dag_edges = [list(set(d) - set([t])) for d in dag_edges]
            dag_edges = [d for d in dag_edges if len(d) > 0]

            xf = [x for x in X_features if x != t]
            self.df = train_df[xf + [t] + Y_feature]
            self.test_df = test_df[xf + [t] + Y_feature]

        else:
            xf = X_features
            self.df = train_df[xf + Y_feature]
            self.test_df = test_df[xf + Y_feature]

        self.t = t
        self.y_scaler = y_scaler
        self.X_features = xf
        self.Y_feature = Y_feature
        self.dag_edges = dag_edges
        nodes_in_dag_edges = list(
            set([item for sublist in self.dag_edges for item in sublist])
        )
        self.X_features = [c for c in self.X_features if c in nodes_in_dag_edges]
        self.dag_edges_idx = [
            [self.X_features.index(col) for col in sublist] for sublist in dag_edges
        ]
        self.model_params = model_params
        self.cv_indexes = cv_indexes
        self.task = task
        if data_loader is not None:
            self.mu_0_tr = data_loader.mu0_train
            self.mu_1_tr = data_loader.mu1_train
            self.mu_0_te = data_loader.mu0_test
            self.mu_1_te = data_loader.mu1_test
            self.ihdp_idx = data_loader.rep_i

        self.metrics_dict_regression = {
            "MSE": metrics.mean_squared_error,
            "MAE": metrics.mean_absolute_error,
            "MAPE": metrics.mean_absolute_percentage_error,
        }
        self.metrics_dict_classification = {
            "Acc": metrics.accuracy_score,
        }
        self.metrics_dict = (
            self.metrics_dict_regression
            if task == "regression"
            else self.metrics_dict_classification
        )

    def train_model_nn(self, params=None):

        X = self.df[self.X_features].to_numpy()
        t = self.df[self.t].to_numpy()
        y = self.df[self.Y_feature[0]].to_numpy()
        self.model = self.trainer(X, t, y, params=params)
        self.model.train()
        self.method = "Vanilla"

    def train_model_nn_constr(self, params=None):

        X = self.df[self.X_features].to_numpy()
        t = self.df[self.t].to_numpy()
        y = self.df[self.Y_feature[0]].to_numpy()
        self.model = self.trainer_constr(X, t, y, self.dag_edges_idx, params=params)
        self.model.train()
        self.method = "Constr"

    def predict_model_nn(self, pred_test=True):

        if pred_test:
            X = self.test_df[self.X_features].copy()
            T_var = self.test_df[self.t].to_numpy()
        else:
            X = self.df[self.X_features].copy()
            T_var = self.df[self.t].to_numpy()

        pred = self.model.predict(X.to_numpy())
        pred[:, 0] = self.y_scaler.inverse_transform(pred[:, 0].reshape(-1, 1)).reshape(
            -1,
        )
        pred[:, 1] = self.y_scaler.inverse_transform(pred[:, 1].reshape(-1, 1)).reshape(
            -1,
        )

        final_pred = [x[0] if t == 0 else x[1] for x, t in zip(pred, T_var)]
        ite_pred = [x[1] - x[0] for x in pred]

        return final_pred, ite_pred, pred[:, 2]

    @staticmethod
    def plot_ite_distribution(ites):
        ate = np.mean(ites)
        sns.set_style("darkgrid")
        g = sns.displot(
            pd.DataFrame(ites, columns=["ITE"]),
            x="ITE",
            kind="kde",
            fill=True,
            common_norm=False,
        )
        g.fig.set_size_inches(10, 7)
        g.set_axis_labels("ITE", "Density")
        plt.axvline(ate, color="red")
        plt.text(ate, 0, "ATE", rotation=0, color="red")
        return g

    def slearn_rf(self, T="T", pred_test=True):

        df = self.df.copy()
        if pred_test:
            test_df = self.test_df.copy()
        else:
            test_df = self.df.copy()

        X = df[self.X_features].to_numpy()
        y = df[self.Y_feature].to_numpy().reshape(-1)
        m = RandomForestRegressor()
        m.fit(X, y)

        test_df[T] = 0
        preds0 = m.predict(test_df[self.X_features].to_numpy())

        test_df[T] = 1
        preds1 = m.predict(test_df[self.X_features].to_numpy())

        return preds1 - preds0, m
