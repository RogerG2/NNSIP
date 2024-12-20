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


# Define the network class
class BCAUSS(nn.Module):
    # Initialize the network layers
    def __init__(self, p):
        """p=params"""
        super(BCAUSS, self).__init__()
        torch.set_default_dtype(torch.float32)

        # Representation
        self.commonlayers = nn.ModuleList()
        self.commonlayers.append(nn.Linear(p["input_dim"], p["neurons_per_layer"]))
        self.commonlayers.append(
            nn.Linear(p["neurons_per_layer"], p["neurons_per_layer"])
        )
        self.commonlayers.append(
            nn.Linear(p["neurons_per_layer"], p["neurons_per_layer"])
        )

        for l in self.commonlayers:
            nn.init.normal_(l.weight, mean=0, std=0.05)

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
    def forward(self, x):

        for layer in self.commonlayers:
            x = torch.nn.ReLU()(layer(x))

        t_pred = torch.nn.Sigmoid()(self.t_layer(x))
        # t_pred = self.t_layer(x)

        y0 = torch.nn.ReLU()(self.y0layers[0](x))
        y0 = torch.nn.ReLU()(self.y0layers[1](y0))
        y0_pred = self.y0layers[2](y0)

        y1 = torch.nn.ReLU()(self.y1layers[0](x))
        y1 = torch.nn.ReLU()(self.y1layers[1](y1))
        y1_pred = self.y1layers[2](y1)

        out = torch.cat([y0_pred, y1_pred, t_pred], dim=1)

        return out



class BCAUSSTrainer:
    def __init__(self, train_X, train_t, train_y, dag_edges=None, params=None):

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
        self.params = params
        # ------------------
        self.device = "cpu"
        self.early_stopping = True
        self.print_every = 800

        self.dag_edges = dag_edges
        self.train_losses = []
        self.val_losses = []

        train_ids = np.random.choice(
            range(train_X.shape[0]),
            int(train_X.shape[0] * (1 - params["val_split"])),
            replace=False,
        )
        val_ids = [i for i in range(train_X.shape[0]) if i not in train_ids]

        self.batched_train_X = (
            torch.from_numpy(self.array_to_batch(train_X[train_ids], self.batch_size))
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

        self.batched_val_X = (
            torch.from_numpy(self.array_to_batch(train_X[val_ids], self.batch_size))
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

        self.model = BCAUSS(p=params)
        self.optimizer = torch.optim.SGD(
            [
                {"params": self.model.commonlayers.parameters(), "weight_decay": 0},
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
        # self.criterion = dragonnet_loss

    def train(self):
        best_val_loss = float("inf")
        loss_counter = 0
        for epoch in range(1, self.num_epochs + 1):
            losses = []
            # i= 0
            # aa =  zip(
            #     self.batched_train_X, self.batched_train_t, self.batched_train_y
            # )
            # X_batch, t_batch, y_batch = next(aa)
            for X_batch, t_batch, y_batch in zip(
                self.batched_train_X, self.batched_train_t, self.batched_train_y
            ):
                # print(i)
                # i += 1
                self.optimizer.zero_grad()
                pred = self.model(X_batch)
                # print("outside")
                # print(pred[:, 2])
                if torch.isnan(pred[:, 2]).any():
                    print("pred is nan")
                    break

                loss = self.criterion(pred, t_batch, y_batch, X_batch)
                if torch.isnan(loss):
                    print("loss is nan")
                    break
                    # print(4 + "h")
                loss.backward()
                # self.scheduler.step(loss)
                losses.append(loss.item())
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
            for X_batch, t_batch, y_batch in zip(
                self.batched_val_X, self.batched_val_t, self.batched_val_y
            ):
                self.optimizer.zero_grad()
                pred = self.model(X_batch)
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
                # print("saving new best model")
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
        # print("load_new_model")
        self.model.load_state_dict(torch.load("best_model.pth"))

    def predict(self, test_X):
        test_X = torch.from_numpy(test_X).to(torch.float32).to(self.device)
        y_test = self.model(test_X)
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
