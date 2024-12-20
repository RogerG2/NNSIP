
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def dragonnet_loss(pred, t_true, y_true, tarnet_bin=1.0, is_tarnet=False):
    """
    Generic loss function for dragonnet

    Parameters
    ----------
    -------
    loss: torch.Tensor
    """
    y0_pred = pred[:, 0]
    y1_pred = pred[:, 1]
    t_pred = pred[:, 2]

    t_pred = (t_pred + 0.001) / 1.002
    loss_t = torch.sum(F.binary_cross_entropy_with_logits(t_pred, t_true))

    loss0 = torch.sum((1.0 - t_true) * torch.square(y_true - y0_pred))
    loss1 = torch.sum(t_true * torch.square(y_true - y1_pred))
    loss_y = loss0 + loss1

    if is_tarnet:
        tarnet_bin = 0

    loss = loss_y + tarnet_bin * loss_t
    return loss


class DragonNetConstrained(nn.Module):
    # Initialize the network layers
    def __init__(self, p, structure_params):
        """p=params"""
        super(DragonNetConstrained, self).__init__()
        torch.set_default_dtype(torch.float32)
        self.is_tarnet = p["is_tarnet"]

        # representation
        self.input_layers = nn.ParameterDict()
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
            for l in input_layer[:-1]:
                x = l(x)
                x = torch.nn.ELU()(x)
            return x

        output_of_input_layers = [
            evaluate_input_layer(self.input_layers[k], Xs[k]) for k in Xs.keys()
        ]

        x = torch.cat(output_of_input_layers, dim=1)
        x = self.merge_layer(x)
        # x = torch.nn.ELU()(x)

        if self.is_tarnet:
            t_pred = torch.nn.Sigmoid()(self.t_layer(torch.zeros_like(x)))
        else:
            t_pred = torch.nn.Sigmoid()(self.t_layer(x))

        y0 = torch.nn.ELU()(self.y0layers[0](x))
        y0 = torch.nn.ELU()(self.y0layers[1](y0))
        y0_pred = self.y0layers[2](y0)

        y1 = torch.nn.ELU()(self.y1layers[0](x))
        y1 = torch.nn.ELU()(self.y1layers[1](y1))
        y1_pred = self.y1layers[2](y1)

        out = torch.cat([y0_pred, y1_pred, t_pred], dim=1)

        return out


class DragonNetConstrainedTrainer:
    def __init__(
        self, train_X, train_t, train_y, dag_edges, params=None, is_tarnet=False
    ):
        """Dag edges should be indices"""

        self.is_tarnet = is_tarnet
        # Hyperparameters --
        params_default = {
            "neurons_per_layer": 200,
            "reg_l2": 0.01,
            "val_split": 0.2,
            "batch_size": 64,
            "epochs": 200,
            "adam_learning_rate": 1e-3,
            "learning_rate": 1e-5,
            "momentum": 0.9,
        }
        if params is not None:
            for key, value in params.items():
                params_default[key] = value

        params = params_default

        self.batch_size = params["batch_size"]
        self.num_epochs = params["epochs"]
        self.early_stopping_value = 40
        params["input_dim"] = train_X.shape[1]
        params["neurons_per_layerYs"] = int(params["neurons_per_layer"] / 2)
        params["is_tarnet"] = is_tarnet

        # ------------------
        self.device = "cpu"
        self.early_stopping = True
        self.print_every = 400

        self.dag_edges = dag_edges

        self.structure_params = {}
        self.Xs = {}
        self.train_losses = []
        self.val_losses = []

        for i in range(len(self.dag_edges)):
            self.structure_params[f"CG{i}"] = {
                "input_size": len(self.dag_edges[i]),
                "hidden_size": params["neurons_per_layer"],
                "output_size": params["neurons_per_layer"],
                "hidden_layers": 1,
            }
            self.Xs[f"CG{i}"] = (
                torch.from_numpy(
                    self.array_to_batch(train_X[:, self.dag_edges[i]], self.batch_size)
                )
                .to(torch.float32)
                .to(self.device)
            )

        self.batched_train_y = (
            torch.from_numpy(self.array_to_batch(train_y, self.batch_size))
            .to(torch.float32)
            .to(self.device)
        )
        self.batched_train_t = (
            torch.from_numpy(self.array_to_batch(train_t, self.batch_size))
            .to(torch.float32)
            .to(self.device)
        )
        self.train_batch_ids = np.random.choice(
            range(len(self.batched_train_y)), int(len(self.batched_train_y) * 0.8)
        )
        self.val_batch_ids = [
            i for i in range(len(self.batched_train_y)) if i not in self.train_batch_ids
        ]

        self.model = DragonNetConstrained(
            p=params, structure_params=self.structure_params
        )
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

        self.criterion = dragonnet_loss

    @staticmethod
    def array_to_batch(data, batch_size):

        num_batches = np.floor(len(data) / batch_size)

        if len(data) % batch_size == 0:
            batches = np.array_split(data, num_batches)
        else:
            batches = np.array_split(data[: -(len(data) % batch_size)], num_batches)

        return np.array(batches)

    def train(self):
        self.reporting = []
        best_val_loss = float("inf")
        loss_counter = 0
        for epoch in range(1, self.num_epochs + 1):
            losses = []
            for i in self.train_batch_ids:

                y_batch = self.batched_train_y[i]
                t_batch = self.batched_train_t[i]
                Xs_batch = {k: v[i] for k, v in self.Xs.items()}

                self.optimizer.zero_grad()
                pred = self.model(Xs_batch)
                if torch.isnan(pred[:, 2]).any():
                    print("pred is nan")
                    break
                loss = self.criterion(pred, t_batch, y_batch, is_tarnet=self.is_tarnet)
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
            for i in self.val_batch_ids:

                y_batch = self.batched_train_y[i]
                t_batch = self.batched_train_t[i]
                Xs_batch = {k: v[i] for k, v in self.Xs.items()}

                self.optimizer.zero_grad()
                pred = self.model(Xs_batch)
                if torch.isnan(pred[:, 2]).any():
                    print("pred is nan")
                    break
                loss = self.criterion(pred, t_batch, y_batch, is_tarnet=self.is_tarnet)
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
                if self.val_losses[-1] > best_val_loss:

                    loss_counter += 1
                    if loss_counter == self.early_stopping_value:
                        break

                else:
                    loss_counter = 0

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
    

class TARNetConstrainedTrainer(DragonNetConstrainedTrainer):
    def __init__(
        self, train_X, train_t, train_y, dag_edges=None, params=None, is_tarnet=True
    ):
        super().__init__(train_X, train_t, train_y, dag_edges, params, is_tarnet)
