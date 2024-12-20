import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def BCAUSS_loss(
    pred,
    t_true,
    y_true,
    inputs,
    norm_bal_term=True,
    b_ratio=1.0,
):
    y0_predictions = pred[:, 0]
    y1_predictions = pred[:, 1]
    t_predictions = pred[:, 2]

    t_pred = (t_predictions + 0.001) / 1.002

    # regression_loss
    loss0 = torch.sum((1.0 - t_true) * torch.square(y_true - y0_predictions))
    loss1 = torch.sum(t_true * torch.square(y_true - y1_predictions))
    vanilla_loss = loss0 + loss1

    # auto-balancing self-supervised objective
    ones_to_sum = (
        torch.repeat_interleave(
            (t_true / t_pred).unsqueeze(1), repeats=inputs.shape[1], dim=1
        )
        * inputs
    )
    zeros_to_sum = (
        torch.repeat_interleave(
            ((1 - t_true) / (1 - t_pred)).unsqueeze(1), repeats=inputs.shape[1], dim=1
        )
        * inputs
    )

    if norm_bal_term:
        ones_mean = torch.sum(ones_to_sum, 0) / torch.sum(t_true / t_pred, 0)
        zeros_mean = torch.sum(zeros_to_sum, 0) / torch.sum(
            (1 - t_true) / (1 - t_pred), 0
        )
    else:
        ones_mean = torch.sum(ones_to_sum, 0)
        zeros_mean = torch.sum(zeros_to_sum, 0)

    # # final loss
    loss = vanilla_loss + b_ratio * F.mse_loss(zeros_mean, ones_mean)

    return loss


class BCAUSSConstrained(nn.Module):
    # Initialize the network layers
    def __init__(self, p, structure_params):
        """p=params"""
        super(BCAUSSConstrained, self).__init__()
        torch.set_default_dtype(torch.float32)

        # representation
        self.input_layers = nn.ParameterDict()
        # self.input_layers = {}
        n_output_nodes = 0
        for key, value in structure_params.items():
            self.input_layers[key] = nn.ModuleList()
            self.input_layers[key].append(
                nn.Linear(value["input_size"], value["hidden_size"])
            )
            for i in range(value["hidden_layers"] - 1):
                self.input_layers[key].append(
                    nn.Linear(value["hidden_size"], value["hidden_size"])
                )
            self.input_layers[key].append(
                nn.Linear(value["hidden_size"], value["output_size"])
            )
            # This is prediction layer for groups
            self.input_layers[key].append(nn.Linear(value["hidden_size"], 1))

            n_output_nodes += value["output_size"]

        self.merge_layer = nn.Linear(n_output_nodes, p["neurons_per_layer"])

        self.t_layer = nn.Linear(p["neurons_per_layer"], 1)

        # Hypothesis
        self.y0layers = nn.ModuleList()
        self.y0layers.append(
            nn.Linear(p["neurons_per_layer"], p["neurons_per_layerYs"])
        )
        self.y0layers.append(
            nn.Linear(p["neurons_per_layerYs"], p["neurons_per_layerYs"])
        )
        self.y0layers.append(nn.Linear(p["neurons_per_layerYs"], 1))

        self.y1layers = nn.ModuleList()
        self.y1layers.append(
            nn.Linear(p["neurons_per_layer"], p["neurons_per_layerYs"])
        )
        self.y1layers.append(
            nn.Linear(p["neurons_per_layerYs"], p["neurons_per_layerYs"])
        )
        self.y1layers.append(nn.Linear(p["neurons_per_layerYs"], 1))

    # Define the forward pass
    def forward(self, Xs):
        """Xs is a dict with keys and values for each input group"""

        def evaluate_input_layer(input_layer, x):
            # -1 because last layer to predict directly

            for l in input_layer[:-2]:
                x = l(x)
                x = torch.nn.ReLU()(x)

            representation = input_layer[-2](x)
            representation = torch.nn.ReLU()(representation)
            direct_output = input_layer[-1](x)


            return representation, direct_output

        output_of_input_layers = [
            evaluate_input_layer(self.input_layers[k], Xs[k]) for k in sorted(Xs.keys())
        ]
        representation_input_layers = [x[0] for x in output_of_input_layers]
        pred_input_layers = [x[1] for x in output_of_input_layers]

        x = torch.cat(representation_input_layers, dim=1)
        x = self.merge_layer(x)

        t_pred = torch.nn.Sigmoid()(self.t_layer(x))

        y0 = torch.nn.ReLU()(self.y0layers[0](x))
        y0 = torch.nn.ReLU()(self.y0layers[1](y0))
        y0_pred = self.y0layers[2](y0)

        y1 = torch.nn.ReLU()(self.y1layers[0](x))
        y1 = torch.nn.ReLU()(self.y1layers[1](y1))
        y1_pred = self.y1layers[2](y1)

        out = torch.cat([y0_pred, y1_pred, t_pred] + pred_input_layers, dim=1)

        return out


class BCAUSSConstrainedTrainer:
    def __init__(self, train_X, train_t, train_y, dag_edges, params=None):
        """Dag edges should be indices"""
        # Hyperparameters --
        default_params = {
            "neurons_per_layer": 200,  # Original = 200,
            "reg_l2": 0.01,
            "val_split": 0.22,
            "batch_size_ratio": 1,
            "batch_size": None,  # 64, # None
            "epochs": 500,
            "learning_rate": 1e-5,
            "momentum": 0.9,
        }
        if params is not None:
            for key, value in params.items():
                default_params[key] = value

        params = default_params
        self.batch_size = params["batch_size"]
        if params["batch_size"] is None:
            self.batch_size = int(
                train_X.shape[0] * (params["batch_size_ratio"] - params["val_split"])
            )
        self.num_epochs = params["epochs"]
        self.early_stopping_value = 40
        params["input_dim"] = train_X.shape[1]
        params["neurons_per_layerYs"] = int(params["neurons_per_layer"] / 2)

        # ------------------
        self.device = "cpu"
        self.early_stopping = True
        self.print_every = 800

        self.dag_edges = dag_edges

        self.structure_params = {}
        self.train_losses = []
        self.val_losses = []

        train_ids = np.random.choice(
            range(train_X.shape[0]),
            int(train_X.shape[0] * (1 - params["val_split"])),
            replace=False,
        )
        val_ids = [i for i in range(train_X.shape[0]) if i not in train_ids]

        self.Xs_train = {}
        self.Xs_val = {}
        for i in range(len(self.dag_edges)):
            self.structure_params[f"CG{i}"] = {
                "input_size": len(self.dag_edges[i]),
                "hidden_size": params["neurons_per_layer"],
                "output_size": int(params["neurons_per_layer"] / len(self.dag_edges)),
                "hidden_layers": 1,
            }
            self.Xs_train[f"CG{i}"] = (
                torch.from_numpy(
                    self.array_to_batch(
                        train_X[train_ids][:, self.dag_edges[i]], self.batch_size
                    )
                )
                .to(torch.float32)
                .to(self.device)
            )
            self.Xs_val[f"CG{i}"] = (
                torch.from_numpy(
                    self.array_to_batch(
                        train_X[val_ids][:, self.dag_edges[i]], self.batch_size
                    )
                )
                .to(torch.float32)
                .to(self.device)
            )
        self.batched_train_t = (
            torch.from_numpy(self.array_to_batch(train_t[train_ids], self.batch_size))
            .to(torch.float32)
            .to(self.device)
        )
        self.batched_train_y = (
            torch.from_numpy(self.array_to_batch(train_y[train_ids], self.batch_size))
            .to(torch.float32)
            .to(self.device)
        )
        # Normal X required only for the loss function (check)
        self.batched_train_X = (
            torch.from_numpy(self.array_to_batch(train_X[train_ids], self.batch_size))
            .to(torch.float32)
            .to(self.device)
        )
        self.batched_val_t = (
            torch.from_numpy(self.array_to_batch(train_t[val_ids], self.batch_size))
            .to(torch.float32)
            .to(self.device)
        )
        self.batched_val_y = (
            torch.from_numpy(self.array_to_batch(train_y[val_ids], self.batch_size))
            .to(torch.float32)
            .to(self.device)
        )
        self.batched_val_X = (
            torch.from_numpy(self.array_to_batch(train_X[val_ids], self.batch_size))
            .to(torch.float32)
            .to(self.device)
        )

        self.model = BCAUSSConstrained(p=params, structure_params=self.structure_params)
        self.optimizer = torch.optim.SGD(
            [
                {"params": self.model.input_layers.parameters(), "weight_decay": 0},
                {"params": self.model.merge_layer.parameters(), "weight_decay": 0},
                {"params": self.model.t_layer.parameters(), "weight_decay": 0},
                {
                    "params": self.model.y0layers.parameters(),
                    "weight_decay": params["reg_l2"],
                },
                {
                    "params": self.model.y1layers.parameters(),
                    "weight_decay": params["reg_l2"],
                },
            ],
            lr=params["learning_rate"],
            momentum=params["momentum"],
            nesterov=True,
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            factor=0.5,
            patience=5,
            eps=1e-8,
            cooldown=0,
            min_lr=0,
        )

        self.criterion = BCAUSS_loss

    def train(self):
        best_val_loss = float("inf")
        loss_counter = 0
        self.reporting = []
        for epoch in range(1, self.num_epochs + 1):
            losses = []
            for i in range(len(self.batched_train_y)):

                y_batch = self.batched_train_y[i]
                t_batch = self.batched_train_t[i]
                Xs_batch = {k: v[i] for k, v in self.Xs_train.items()}
                X_batch = self.batched_train_X[i]

                self.optimizer.zero_grad()
                pred = self.model(Xs_batch)
                if torch.isnan(pred[:, 2]).any():
                    print("pred is nan")
                    break
                loss = self.criterion(pred, t_batch, y_batch, X_batch)
                if torch.isnan(loss):
                    print("loss is nan")
                    break
                losses.append(loss.item())
                loss.backward()
                # self.scheduler.step(loss)
                self.optimizer.step()

            if torch.isnan(pred[:, 2]).any():
                print("pred is nan _ epoch loop")
                break
            if torch.isnan(loss):
                print("loss is nan _ epoch loop")
                break

            self.scheduler.step(np.mean(losses))
            self.train_losses.append(np.mean(losses))

            # Validation
            losses = []
            for i in range(len(self.batched_val_y)):

                y_batch = self.batched_val_y[i]
                t_batch = self.batched_val_t[i]
                Xs_batch = {k: v[i] for k, v in self.Xs_val.items()}
                X_batch = self.batched_val_X[i]

                self.optimizer.zero_grad()
                pred = self.model(Xs_batch)
                if torch.isnan(pred[:, 2]).any():
                    print("pred is nan")
                    break
                loss = self.criterion(pred, t_batch, y_batch, X_batch)
                # self.scheduler.step(loss)
                losses.append(loss.item())

            if torch.isnan(pred[:, 2]).any():
                print("pred is nan _epoch loop")
                break
            self.val_losses.append(np.mean(losses))

            # Save best model
            val_loss = self.val_losses[-1]
            if val_loss < best_val_loss:
                torch.save(self.model.state_dict(), "best_model.pth")
                best_val_loss = val_loss


            # Early stopping
            if self.early_stopping and epoch > 1:
                if self.val_losses[-1] > self.val_losses[-2]:

                    loss_counter += 1
                    if loss_counter == self.early_stopping_value:
                        break

                else:
                    loss_counter = 0

            if epoch % self.print_every == 0:
                print(
                    f"Epoch {epoch}, Train Loss: {self.train_losses[-1]}, , Val Loss: {self.val_losses[-1]}"
                )
        self.model.load_state_dict(torch.load("best_model.pth"))

    def predict(self, test_X):

        test_Xs = {}
        for i in range(len(self.dag_edges)):
            test_Xs[f"CG{i}"] = (
                torch.from_numpy(test_X[:, self.dag_edges[i]])
                .to(torch.float32)
                .to(self.device)
            )

        y_test = self.model(test_Xs)
        return y_test.detach().numpy()

    @staticmethod
    def array_to_batch(data, batch_size):

        num_batches = np.floor(len(data) / batch_size)

        # Case when batch size is greater than num samples
        if num_batches == 0:
            return np.array([data])

        if len(data) % batch_size == 0:
            batches = np.array_split(data, num_batches)
        else:
            batches = np.array_split(data[: -(len(data) % batch_size)], num_batches)

        return np.array(batches)

