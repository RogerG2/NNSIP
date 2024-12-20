import pandas as pd
import numpy as np
from scipy.special import expit, logit
from sklearn.model_selection import KFold


class dataLoader:
    def __init__(
        self,
        data_path="../../../data/",
        n_instances=1000,
        n_vars=5,
        X_noise=0,
        Y_noise=1,
        test_perc=0.2,
        cv_splits=5,
        experiment_params=None,
        rep_i=0,
    ) -> None:

        self.data_path = data_path

        self.datasets = {
            "ihdp": self.load_ihdp_data,
            "jobs": self.load_jobs_data,
            "causalml_mode_1": self.load_causalml_mode_1,
            "causalml_mode_2": self.load_causalml_mode_2,
            "causalml_mode_3": self.load_causalml_mode_3,
            "causalml_mode_4": self.load_causalml_mode_4,
            "toy_example": self.load_toy_example,
            "toy_example_2": self.load_toy_example_2,
        }
        self.X_noise = X_noise
        self.Y_noise = Y_noise
        self.n_instances = n_instances
        self.n_vars = n_vars
        self.test_perc = test_perc
        self.cv_splits = cv_splits
        self.experiment_params = experiment_params
        if experiment_params is not None:
            if "cv_splits" in experiment_params.keys():
                self.cv_splits = experiment_params["cv_splits"]
            if "n_instances" in experiment_params.keys():
                self.n_instances = experiment_params["n_instances"]
            if "n_vars" in experiment_params.keys():
                self.n_vars = experiment_params["n_vars"]
            if "X_noise" in experiment_params.keys():
                self.X_noise = experiment_params["X_noise"]
            if "Y_noise" in experiment_params.keys():
                self.Y_noise = experiment_params["Y_noise"]
            if "test_perc" in experiment_params.keys():
                self.test_perc = experiment_params["test_perc"]

        self.test_weights = None
        self.rep_i = rep_i
        self.train_df_orig = None

    def load_dataset(self, dataset_name):
        self.datasets[dataset_name]()
        kf = KFold(
            n_splits=self.cv_splits,
            # shuffle=self.random_state is not None,
            shuffle=True,
            # random_state=self.random_state,
        )
        self.cv_indexes = list(kf.split(self.train_df))

    def load_ihdp_data(self):

        i = self.rep_i
        data = np.load(self.data_path + "ihdp/ihdp_npci_1-1000.train.npz")
        data_test = np.load(self.data_path + "ihdp/ihdp_npci_1-1000.test.npz")

        train_df = pd.DataFrame(data["x"][:, :, i])
        colnames = [f"x{c}" for c in range(train_df.shape[1])]
        train_df.columns = colnames
        train_df["Y"] = data["yf"][:, i]
        train_df["T"] = data["t"][:, i]

        test_df = pd.DataFrame(data_test["x"][:, :, i])
        test_df.columns = colnames
        test_df["Y"] = data_test["yf"][:, i]
        test_df["T"] = data_test["t"][:, i]

        self.X_features = colnames + ["T"]
        self.Y_feature = ["Y"]

        self.train_df = train_df
        self.test_df = test_df
        self.train_ite = data["mu1"][:, i] - data["mu0"][:, i]
        self.test_ite = data_test["mu1"][:, i] - data_test["mu0"][:, i]
        self.mu0_train = data["mu0"][:, i]
        self.mu1_train = data["mu1"][:, i]
        self.mu0_test = data_test["mu0"][:, i]
        self.mu1_test = data_test["mu1"][:, i]

    def load_jobs_data(self):

        i = self.rep_i
        data = np.load(self.data_path + "jobs/train.npz")
        data_test = np.load(self.data_path + "jobs/test.npz")

        train_df = pd.DataFrame(data["x"][:, :, i])
        colnames = [f"x{c}" for c in range(train_df.shape[1])]
        train_df.columns = colnames
        train_df["Y"] = data["yf"][:, i]
        train_df["T"] = data["t"][:, i]
        train_df["e"] = data["e"][:, i]

        test_df = pd.DataFrame(data_test["x"][:, :, i])
        test_df.columns = colnames
        test_df["Y"] = data_test["yf"][:, i]
        test_df["T"] = data_test["t"][:, i]
        test_df["e"] = data_test["e"][:, i]

        self.X_features = colnames + ["T"]
        self.Y_feature = ["Y"]

        self.train_df = train_df
        self.test_df = test_df

        self.real_ate = data["ate"]
        self.test_ite = np.array([99999] * test_df.shape[0])
        self.train_ite = np.array([99999] * train_df.shape[0])

    def load_causalml_mode_1(self, adj=0.0, shift_test=False):
        # def simulate_nuisance_and_easy_treatment(n=1000, p=5, sigma=1.0, adj=0.0):
        """Synthetic data with a difficult nuisance components and an easy treatment effect
            From Setup A in Nie X. and Wager S. (2018) 'Quasi-Oracle Estimation of Heterogeneous Treatment Effects'
        Args:
            n (int, optional): number of observations
            p (int optional): number of covariates (>=5)
            Y_noise (float): standard deviation of the error term
            adj (float): adjustment term for the distribution of propensity, e. Higher values shift the distribution to 0.
        Returns:
            (tuple): Synthetically generated samples with the following outputs:
                - y ((n,)-array): outcome variable.
                - X ((n,p)-ndarray): independent variables.
                - w ((n,)-array): treatment flag with value 0 or 1.
                - tau ((n,)-array): individual treatment effect.
                - b ((n,)-array): expected outcome.
                - e ((n,)-array): propensity of receiving treatment.
        """
        n = self.n_instances * 2
        p = self.n_vars
        X_noise = self.X_noise
        Y_noise = self.Y_noise

        X = (
            np.random.uniform(size=n * p) + np.random.normal(0, X_noise, size=n * p)
        ).reshape((n, -1))

        b = (
            np.sin(np.pi * X[:, 0] * X[:, 1])
            + 2 * (X[:, 2] - 0.5) ** 2
            + X[:, 3]
            + 0.5 * X[:, 4]
        )
        eta = 0.1
        e = np.maximum(
            np.repeat(eta, n),
            np.minimum(np.sin(np.pi * X[:, 0] * X[:, 1]), np.repeat(1 - eta, n)),
        )
        e = expit(logit(e) - adj)
        tau = (X[:, 0] + X[:, 1]) / 2

        w = np.random.binomial(1, e, size=n)
        y = b + (w - 0.5) * tau + Y_noise * np.random.normal(size=n)
        y_exp = b + (e - 0.5) * tau

        self.e = e
        self.m = y_exp

        # return y, X, w, tau, b, e

        train_df = pd.DataFrame(X, columns=[f"col_{i}" for i in range(X.shape[1])])
        train_df["T"] = w
        train_df["Y"] = y

        self.X_features = [f"col_{i}" for i in range(X.shape[1])] + ["T"]
        self.Y_feature = ["Y"]
        self.train_df = train_df.iloc[: self.n_instances, :].reset_index()
        self.test_df = train_df.iloc[self.n_instances :, :].reset_index()
        self.train_ite = tau[: self.n_instances]
        self.test_ite = tau[self.n_instances :]

    def load_causalml_mode_2(self, adj=0.0):
        # def simulate_randomized_trial(n=1000, p=5, sigma=1.0, adj=0.0):
        """Synthetic data of a randomized trial
            From Setup B in Nie X. and Wager S. (2018) 'Quasi-Oracle Estimation of Heterogeneous Treatment Effects'

        Args:
            n (int, optional): number of observations
            p (int optional): number of covariates (>=5)
            sigma (float): standard deviation of the error term
            adj (float): no effect. added for consistency


        Returns:
            (tuple): Synthetically generated samples with the following outputs:

                - y ((n,)-array): outcome variable.
                - X ((n,p)-ndarray): independent variables.
                - w ((n,)-array): treatment flag with value 0 or 1.
                - tau ((n,)-array): individual treatment effect.
                - b ((n,)-array): expected outcome.
                - e ((n,)-array): propensity of receiving treatment.
        """
        n = self.n_instances * 2
        p = self.n_vars
        X_noise = self.X_noise
        Y_noise = self.Y_noise

        X = (
            np.random.uniform(size=n * p) + np.random.normal(0, X_noise, size=n * p)
        ).reshape((n, -1))

        b = np.maximum(np.repeat(0.0, n), X[:, 0] + X[:, 1], X[:, 2]) + np.maximum(
            np.repeat(0.0, n), X[:, 3] + X[:, 4]
        )
        e = np.repeat(0.5, n)
        tau = X[:, 0] + np.log1p(np.exp(X[:, 1]))

        w = np.random.binomial(1, e, size=n)
        y = b + (w - 0.5) * tau + Y_noise * np.random.normal(size=n)
        y_exp = b + (e - 0.5) * tau
        self.e = e
        self.m = y_exp

        # return y, X, w, tau, b, e
        train_df = pd.DataFrame(X, columns=[f"col_{i}" for i in range(X.shape[1])])
        train_df["T"] = w
        train_df["Y"] = y

        self.X_features = [f"col_{i}" for i in range(X.shape[1])] + ["T"]
        self.Y_feature = ["Y"]
        self.train_df = train_df.iloc[: self.n_instances, :].reset_index()
        self.test_df = train_df.iloc[self.n_instances :, :].reset_index()
        self.train_ite = tau[: self.n_instances]
        self.test_ite = tau[self.n_instances :]

    def load_causalml_mode_3(self, adj=0.0):
        """Synthetic data with easy propensity and a difficult baseline
            From Setup C in Nie X. and Wager S. (2018) 'Quasi-Oracle Estimation of Heterogeneous Treatment Effects'

        Args:
            n (int, optional): number of observations
            p (int optional): number of covariates (>=3)
            sigma (float): standard deviation of the error term
            adj (float): no effect. added for consistency

        Returns:
            (tuple): Synthetically generated samples with the following outputs:

                - y ((n,)-array): outcome variable.
                - X ((n,p)-ndarray): independent variables.
                - w ((n,)-array): treatment flag with value 0 or 1.
                - tau ((n,)-array): individual treatment effect.
                - b ((n,)-array): expected outcome.
                - e ((n,)-array): propensity of receiving treatment.
        """

        n = self.n_instances * 2
        p = self.n_vars
        X_noise = self.X_noise
        Y_noise = self.Y_noise

        X = (
            np.random.uniform(size=n * p) + np.random.normal(0, X_noise, size=n * p)
        ).reshape((n, -1))

        b = 2 * np.log1p(np.exp(X[:, 0] + X[:, 1] + X[:, 2]))
        e = 1 / (1 + np.exp(X[:, 1] + X[:, 2]))
        tau = np.repeat(1.0, n)

        w = np.random.binomial(1, e, size=n)
        y = b + (w - 0.5) * tau + Y_noise * np.random.normal(size=n)

        y_exp = b + (e - 0.5) * tau
        self.e = e
        self.m = y_exp

        # return y, X, w, tau, b, e
        train_df = pd.DataFrame(X, columns=[f"col_{i}" for i in range(X.shape[1])])
        train_df["T"] = w
        train_df["Y"] = y

        self.X_features = [f"col_{i}" for i in range(X.shape[1])] + ["T"]
        self.Y_feature = ["Y"]
        self.train_df = train_df.iloc[: self.n_instances, :].reset_index()
        self.test_df = train_df.iloc[self.n_instances :, :].reset_index()
        self.train_ite = tau[: self.n_instances]
        self.test_ite = tau[self.n_instances :]

    def load_causalml_mode_4(self, adj=0.0):
        # def simulate_unrelated_treatment_control(n=1000, p=5, sigma=1.0, adj=0.0):
        """Synthetic data with unrelated treatment and control groups.
            From Setup D in Nie X. and Wager S. (2018) 'Quasi-Oracle Estimation of Heterogeneous Treatment Effects'

        Args:
            n (int, optional): number of observations
            p (int optional): number of covariates (>=3)
            sigma (float): standard deviation of the error term
            adj (float): adjustment term for the distribution of propensity, e. Higher values shift the distribution to 0.

        Returns:
            (tuple): Synthetically generated samples with the following outputs:

                - y ((n,)-array): outcome variable.
                - X ((n,p)-ndarray): independent variables.
                - w ((n,)-array): treatment flag with value 0 or 1.
                - tau ((n,)-array): individual treatment effect.
                - b ((n,)-array): expected outcome.
                - e ((n,)-array): propensity of receiving treatment.
        """

        n = self.n_instances * 2
        p = self.n_vars
        X_noise = self.X_noise
        Y_noise = self.Y_noise

        X = (
            np.random.uniform(size=n * p) + np.random.normal(0, X_noise, size=n * p)
        ).reshape((n, -1))

        b = (
            np.maximum(np.repeat(0.0, n), X[:, 0] + X[:, 1] + X[:, 2])
            + np.maximum(np.repeat(0.0, n), X[:, 3] + X[:, 4])
        ) / 2
        e = 1 / (1 + np.exp(-X[:, 0]) + np.exp(-X[:, 1]))
        e = expit(logit(e) - adj)
        tau = np.maximum(np.repeat(0.0, n), X[:, 0] + X[:, 1] + X[:, 2]) - np.maximum(
            np.repeat(0.0, n), X[:, 3] + X[:, 4]
        )

        w = np.random.binomial(1, e, size=n)
        y = b + (w - 0.5) * tau + Y_noise * np.random.normal(size=n)

        y_exp = b + (e - 0.5) * tau
        self.e = e
        self.m = y_exp

        # return y, X, w, tau, b, e
        train_df = pd.DataFrame(X, columns=[f"col_{i}" for i in range(X.shape[1])])
        train_df["T"] = w
        train_df["Y"] = y

        self.X_features = [f"col_{i}" for i in range(X.shape[1])] + ["T"]
        self.Y_feature = ["Y"]
        self.train_df = train_df.iloc[: self.n_instances, :].reset_index()
        self.test_df = train_df.iloc[self.n_instances :, :].reset_index()
        self.train_ite = tau[: self.n_instances]
        self.test_ite = tau[self.n_instances :]

    def load_toy_example(self):
        """Synthetic data with easy propensity and a difficult baseline"""

        n = self.n_instances * 2
        # p = 1
        p = 2
        Y_noise = self.Y_noise

        X = (np.random.uniform(size=n * p)).reshape((n, -1))
        eta = 0.1
        e = np.maximum(
            np.repeat(eta, n),
            np.minimum(np.sin(np.pi * X[:, 0] * X[:, 1]), np.repeat(1 - eta, n)),
            # np.minimum(np.sin(np.pi * X[:, 0]), np.repeat(1 - eta, n)),
        )
        e = expit(logit(e))

        tau = np.random.normal(1, 1, n)

        w = np.random.binomial(1, e, size=n)
        y = 2 * np.log1p(
            np.exp(X[:, 0] + X[:, 1] + (w - 0.5) * tau)
            # np.exp(X[:, 0] + (w - 0.5) * tau)
        ) + Y_noise * np.random.normal(size=n)
        w_cf = np.abs(w - 1)
        y_cf = 2 * np.log1p(np.exp(X[:, 0] + X[:, 1] + (w_cf - 0.5) * tau))
        # y_cf = 2 * np.log1p(np.exp(X[:, 0] + (w_cf - 0.5) * tau))
        ite = y - y_cf

        # return y, X, w, tau, b, e
        train_df = pd.DataFrame(X, columns=[f"col_{i}" for i in range(X.shape[1])])
        train_df["T"] = w
        train_df["Y"] = y

        self.X_features = [f"col_{i}" for i in range(X.shape[1])] + ["T"]
        self.Y_feature = ["Y"]
        self.train_df = train_df.iloc[: self.n_instances, :].reset_index()
        self.test_df = train_df.iloc[self.n_instances :, :].reset_index()
        self.train_ite = ite[: self.n_instances]
        self.test_ite = ite[self.n_instances :]

    def load_toy_example_2(self):
        """Synthetic data with easy propensity and a difficult baseline"""

        n = self.n_instances * 2
        p = 4
        Y_noise = self.Y_noise

        X = (np.random.uniform(size=n * p)).reshape((n, -1))
        eta = 0.1
        e = np.maximum(
            np.repeat(eta, n),
            np.minimum(np.sin(np.pi * X[:, 0] * X[:, 1]), np.repeat(1 - eta, n)),
        )
        e = expit(logit(e))

        e2 = np.maximum(
            np.repeat(eta, n),
            np.minimum(np.sin(np.pi * X[:, 2] + X[:, 3]), np.repeat(1 - eta, n)),
        )
        e2 = expit(logit(e2)) + np.random.uniform(size=n)
        X = np.column_stack((X, e2))

        tau = np.random.normal(1, 1, n)

        w = np.random.binomial(1, e, size=n)
        y = 2 * np.log1p(
            np.exp(X[:, 0] + X[:, 1] + (w - 0.5) * tau + e2)
        ) + Y_noise * np.random.normal(size=n)
        w_cf = np.abs(w - 1)
        y_cf = 2 * np.log1p(np.exp(X[:, 0] + X[:, 1] + (w_cf - 0.5) * tau + e2))
        ite = y - y_cf

        # return y, X, w, tau, b, e
        train_df = pd.DataFrame(X, columns=[f"col_{i}" for i in range(X.shape[1])])
        train_df["T"] = w
        train_df["Y"] = y

        self.X_features = [f"col_{i}" for i in range(X.shape[1])] + ["T"]
        self.Y_feature = ["Y"]
        self.train_df = train_df.iloc[: self.n_instances, :].reset_index()
        self.test_df = train_df.iloc[self.n_instances :, :].reset_index()
        self.train_ite = ite[: self.n_instances]
        self.test_ite = ite[self.n_instances :]
