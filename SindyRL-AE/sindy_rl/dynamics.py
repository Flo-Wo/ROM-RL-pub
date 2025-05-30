# import gym
import numpy as np
import pysindy as ps
import warnings
from tqdm import tqdm
from datetime import datetime
import pickle
import logging

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader, Dataset, random_split

from scipy.integrate import solve_ivp

from sindy_rl.sindy_utils import build_optimizer, build_feature_library
from sindy_rl import dynamics_callbacks

from src.autoencoder.sindy_autoencoder import SindySurrogate
from src.autoencoder.loss_function import GradFreeFullLoss
from src.autoencoder.logger_AE import LoggerAE
import torch.nn as nn


class BaseDynamicsModel:
    """
    Parent class for dynamics models. Each subclass must implement it's own
        predict and _ functions.
    """

    def __init__(self, config):
        raise NotImplementedError

    def predict(self, x, u):
        raise NotImplementedError

    def fit(self, X, U) -> tuple[object, dict]:
        raise NotImplementedError


class AutoEncoderDynamicsModel(BaseDynamicsModel):
    def __init__(self, config):
        self.config = config

        self.feature_library = self.config.get(
            "feature_library", ps.PolynomialLibrary(library_ensemble=None)
        )
        if isinstance(self.feature_library, dict):
            self.feature_library = build_feature_library(self.feature_library)

        self.n_epochs = config.get("n_epochs", 100)
        self.batch_size = config.get("batch_size", 64)

        # compute the output size of the feature library -> not "really" a fit
        x_test = np.zeros(
            (
                self.batch_size,
                config.get("state_latent_dim") + config.get("control_latent_dim"),
            )
        )

        self.feature_library.fit(x_test)

        # this value is computed on the fly
        self.config["dict_dim"] = self.feature_library.size

        self.model = SindySurrogate(**config, act_func=nn.Sigmoid)
        # if isinstance(self.feature_library, dict):
        #     self.feature_library = build_feature_library(self.feature_library)

        # build the loss function
        loss_arguments = self.config.get(
            "loss_function", {"lambda_0": 1.0, "lambda_1": 1.0, "lambda_2": 1.0}
        )
        self.loss_fn = GradFreeFullLoss(
            theta=self.feature_library.transform, **loss_arguments
        )

        # init the optimizer
        self.optimizer_kwargs = config.get("optimizer_kwargs", {})
        self.optimizer = optim.Adam(self.model.parameters(), **self.optimizer_kwargs)

        # internal logging and plotting of the training process
        self.log_dir = config.get("log_dir")
        self.plot_dir = config.get("plot_dir")
        self.logger = LoggerAE(self.log_dir, dummy_logger=config.get("dummy_logger"))
        self.logger.info("Plot dir: {}".format(self.plot_dir))
        self.logger.info("Log dir: {}".format(self.log_dir))

    def forward(self, xu: np.ndarray) -> np.ndarray:
        """Wrapper around the forward method to debug the AE in the evaluation part."""
        vec_latent, x_dec, u_dec = self.model.forward(
            torch.as_tensor(xu, dtype=torch.float32)[None, :]
        )
        return (
            vec_latent.detach().numpy(),
            x_dec.detach().numpy(),
            u_dec.detach().numpy(),
        )

    def predict(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Predict the next state, x_next given the current state and an action.

        Parameters
        ----------
        x : np.ndarray
            Current state of the system.
        u : np.ndarray
            Action of the policy.

        Returns
        -------
        np.ndarray
            Next state, x_next.
        """
        # use dim 0, since the prediction function does not work on batches
        xu = torch.cat(
            (
                torch.as_tensor(x, dtype=torch.float32),
                torch.as_tensor(u, dtype=torch.float32),
            ),
            dim=0,
        )
        return (
            self.model.predict(
                xu,
                self.feature_library.transform,
            )
            .detach()
            .numpy()
        )

    def fit(
        self,
        observations: list[np.ndarray],
        actions: list[np.ndarray],
        dtype=torch.float32,
    ) -> tuple[object, dict]:
        """Fit the dynamics model based on new available (online) data.

        Parameters
        ----------
        observations : list[np.ndarray]
            List of observations, will be internally correctly converted to (s, a, s_next).
        actions : list[np.ndarray]
            List of actions, will be internally correctly converted to (s, a, s_next).
        dtype : torch.datatype, optional
            We convert numpy arrays to torch tensors, internally torch.nn uses torch.float32,
            by default torch.float32.

        Returns
        -------
        tuple[object, dict]
            The trained model and a dict containing information about the training
            process.
        """
        XU_in, X_out = _reshape_data(observations, actions, dtype=dtype)

        dataset = TrajDataset(XU_in, X_out)
        n_pts = len(dataset)
        n_train = int(0.8 * n_pts)
        n_val = n_pts - n_train

        train_dataset, val_dataset = random_split(dataset, [n_train, n_val])

        train_dataloader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )

        val_loss_prev = np.inf

        train_loss_list = []
        val_loss_list = []

        # debug info
        pred_loss_list = []
        AE_loss_list = []
        reg_loss_list = []

        pred_loss_list_val = []
        AE_loss_list_val = []
        reg_loss_list_val = []

        start_time = datetime.now()
        for epoch in range(self.n_epochs):
            train_loss_epoch = 0.0

            # debugging information
            pred_loss_epoch = 0.0
            AE_loss_epoch = 0.0
            reg_loss_epoch = 0.0

            for xu_in, x_out in train_dataloader:

                self.optimizer.zero_grad()
                loss, pred_loss_batch, AE_loss_batch, reg_loss_batch = self.loss_fn(
                    xu_in, None, x_out, self.model
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

                train_loss_epoch += loss.item()
                pred_loss_epoch += pred_loss_batch.item()
                AE_loss_epoch += AE_loss_batch.item()
                reg_loss_epoch += reg_loss_batch.item()

            train_loss_list.append(train_loss_epoch / n_train)

            # debug information
            pred_loss_list.append(pred_loss_epoch / n_train)
            AE_loss_list.append(AE_loss_epoch / n_train)
            reg_loss_list.append(reg_loss_epoch / n_train)

            # compare previous validation loss. If we have an increase, stop training
            val_loss_epoch, pred_loss_val, AE_loss_val, reg_loss_val = (
                self._get_val_loss(val_dataset)
            )
            val_loss_list.append(val_loss_epoch / n_val)

            # debug information
            pred_loss_list_val.append(pred_loss_val / n_val)
            AE_loss_list_val.append(AE_loss_val / n_val)
            reg_loss_list_val.append(reg_loss_val / n_val)

        # logging.info("AUTOENCODER: Trained Dynamics for {} Epochs".format(epoch))
        train_info = {
            "AE_epoch": epoch,
            "AE_val_loss": val_loss_epoch / n_val,
            "AE_train_loss": train_loss_epoch / n_train,
            "AE_train_time_s": (datetime.now() - start_time).total_seconds(),
            "AE_n_train_samples": n_train,
            "AE_n_val_samples": n_val,
            "xi_matrix": self.model.xi.weight.data,
        }
        self.logger.log(train_info)
        # visualize the internal training process
        """
        self._plot_train_val_loss(
            train_loss_list=train_loss_list,
            val_loss_list=val_loss_list,
            title="total_loss",
        )
        # debug information
        self._plot_train_val_loss(
            train_loss_list=pred_loss_list,
            val_loss_list=pred_loss_list_val,
            title="prediction_loss",
        )
        self._plot_train_val_loss(
            train_loss_list=AE_loss_list,
            val_loss_list=AE_loss_list_val,
            title="AE_loss",
        )
        self._plot_train_val_loss(
            train_loss_list=reg_loss_list,
            val_loss_list=reg_loss_list_val,
            title="regularization_loss",
        )
        """
        return self.model, train_info

    def _plot_train_val_loss(
        self,
        train_loss_list: list[float],
        val_loss_list: list[float],
        title: str = "",
    ) -> None:
        """Plot the internal train and validation loss of the AE.

        Parameters
        ----------
        train_loss_list : list[float]
            List with training losses.
        val_loss_list : list[float]
            List with validation losses.
        """
        # self.logger.info("Plotting train and validation loss.")
        ts = datetime.now()
        fig = plt.figure(figsize=(8, 6))
        plt.plot(train_loss_list, label="Train Loss")
        plt.plot(val_loss_list, label="Val Loss")
        plt.xlabel("AE epoch")
        plt.legend(loc="best")
        plt.title(title)
        plt.savefig(
            self.plot_dir
            + "/AE_train_val_{}_{}.pdf".format(title, ts.strftime("%Y-%m-%d_%H-%M-%S"))
        )
        plt.close(fig)

    def _get_val_loss(self, val_dataset: torch.tensor) -> torch.tensor:
        """Compute the entire validation loss at once."""
        xu, x_out = val_dataset[:]

        with torch.no_grad():
            val_loss, pred_loss, AE_loss, reg_loss = self.loss_fn(
                xu, None, x_out, self.model
            )
        return val_loss, pred_loss, AE_loss, reg_loss

    # functions to match the SindyRL API
    def set_median_coef_(self, **kargs):
        """Need analogous function to conform to EnsembleSINDyDynamicsModel API"""
        pass

    def get_coef_list(self):
        return self.model.state_dict()

    def set_ensemble_coefs_(self, state_dict):
        self.model.load_state_dict(state_dict)

    def save(self, save_path):
        """EXPERIMENTAL"""
        with open(save_path, "wb") as f:
            pickle.dump(self.get_coef_list(), f)

    def load(self, load_path):
        """EXPERIMENTAL"""
        with open(load_path, "rb") as f:
            state_dicts = pickle.load(f)

        self.set_ensemble_coefs_(state_dicts)


class EnsembleSINDyDynamicsModel(BaseDynamicsModel):
    """
    An ensemble SINDy Dynamcis Model. Note that all SINDy models can be an ensemble
        of just 1 model.
    """

    def __init__(self, config):
        self.config = config or {}
        self.dt = self.config.get("dt", 1)
        self.discrete = self.config.get("discrete", True)
        self.callbacks = self.config.get("callbacks", None)

        if self.callbacks is not None:
            self.callbacks = getattr(dynamics_callbacks, self.callbacks)

        # init optimizer
        optimizer = self.config.get("optimizer")
        if isinstance(optimizer, ps.BaseOptimizer):
            self.optimizer = optimizer
        else:
            self.optimizer = build_optimizer(optimizer)

        # init feature library
        self.feature_library = self.config.get(
            "feature_library", ps.PolynomialLibrary(library_ensemble=None)
        )

        if isinstance(self.feature_library, dict):
            self.feature_library = build_feature_library(self.feature_library)

        self.model = ps.SINDy(
            discrete_time=self.discrete,
            optimizer=self.optimizer,
            feature_library=self.feature_library,
        )
        logging.info("SINDy model is created.")
        self.n_models = self.model.optimizer.n_models

        # once initialized, `safe_idx` can be used to determine which members of the
        # ensemble can be trusted, since some might blow up quickly in finite time.
        self.safe_idx = None

    def reset_safe_list(self):
        """Reset this to list all models as safe"""
        self.safe_idx = np.ones(self.n_models, dtype=bool)

    def fit(self, observations, actions, **sindy_fit_kwargs) -> tuple[object, dict]:
        """
        Inputs:
            `observations': list of observations (state variables) of the form [traj_1, traj_2, ... traj_N]
                where each trajectory is an (n_time, n_state) dimensional array
                (this is in the form of pysindy's `multiple_trajecotries=True` keyword)
            `actions': list of actions (control inputs) of the form [traj_1, traj_2, ... traj_N]
                where each trajectory is an (n_time, n_inputs) dimensionla array
                (this is in the form of pysindy's `multiple_trajecotries=True` keyword)
            `sindy_fit_kwargs': keyword arguments to be passed to pysindy's `model.fit' function.
        Returns:
            `model': Fitted PySINDy model
        """
        self.optimizer.coef_list = []
        self.model.fit(
            observations,
            u=actions,
            multiple_trajectories=True,
            t=self.dt,
            **sindy_fit_kwargs
        )
        self.safe_idx = np.ones(self.n_models, dtype=bool)

        self.n_state = observations[0][0].shape[0]
        self.n_control = actions[0][0].shape[0]
        return self.model, {}

    def validate_ensemble(
        self, x0, u_data, targets, thresh=None, verbose=True, **sim_kwargs
    ):
        """
        Validates which members of the ensemble are usuable with respect to a given test data.

        Inputs:
            `x0': initial state for a trajectory `ndarray' of shape (n_state,)
            `u_data': the associated control inputs (actions) to apply during the simulation.
            `targets`: the associated expected predictions
            `thresh`: (float) scalar threshold for MSE. If MSE exceeds threshold, report that the
                dynamics model is not safe.
            `verbose': whether to use a tqdm progress bar.
        Ouptus:
            `pred_list': ndarray of shape (n_ensmble, n_time, n_state) -ish
                The set of predictions. A given ensemble index reports none if that member fails
                to be safe.
            `safe_idx': bool ndarray of shape (n_ensemble,). If an index is `True`: the corresponding
                model is assumed to be safe. If it is `False`, the corresponding model is assumed to be
                not safe.
        """
        coef_list = self.get_coef_list()
        pred_list = []
        for coef_idx, coef in enumerate(
            tqdm(coef_list, disable=(not verbose), position=0, leave=True)
        ):
            try:
                self.model.optimizer.coef_ = coef
                preds = self.simulate(x0, u=u_data, t=len(u_data), **sim_kwargs)
                mse = np.mean((targets - preds) ** 2)
                pred_list.append(preds)
                if thresh and mse >= thresh:
                    self.safe_idx[coef_idx] = False
            except Exception as e:
                warnings.warn(str(e))
                self.safe_idx[coef_idx] = False
                pred_list.append(None)
        return np.array(pred_list), self.safe_idx

    def set_mean_coef_(self, valid=False):
        """
        Set the model coefficients to be the ensemble mean.

        Inputs:
            `valid': (bool) whether to only perform this on validated models.
        Outputs:
            `coef_`: the ensemble mean coefficients
        """
        coef_list = np.array(self.get_coef_list())
        if valid:
            coef_list = coef_list[self.safe_idx]
        self.model.optimizer.coef_ = np.mean(coef_list, axis=0)
        return self.model.optimizer.coef_

    def set_median_coef_(self, valid=False):
        """
        Set the model coefficients to be the ensemble median.

        Inputs:
            `valid': (bool) whether to only perform this on validated models.
        Outputs:
            `coef_`: the ensemble median coefficients
        """
        coef_list = np.array(self.get_coef_list())
        if valid:
            coef_list = coef_list[self.safe_idx]
        self.model.optimizer.coef_ = np.median(coef_list, axis=0)

        return self.model.optimizer.coef_

    def set_idx_coef_(self, idx):
        """
        Set the model coefficients to be the `idx`-th ensemble coefficient

        Inputs:
            `valid': (bool) whether to only perform this on validated models.
        Outputs:
            `coef_`: the ensemble `idx`-th ensemble coefficient
        """
        self.model.optimizer.coef_ = self.model.optimizer.coef_list[idx]
        return self.model.optimizer.coef_

    def set_rand_coef_(self, valid=True):
        """
        Set the model coefficients to be a random element of the ensemble.

        Inputs:
            `valid': (bool) whether to only perform this on validated models.
        Outputs:
            `coef_`: the random ensemble coefficient
        """
        idx_list = np.arange(self.n_models)
        if valid:
            idx_list = idx_list[self.safe_idx]

        idx = np.random.choice(idx_list)

        return self.set_idx_coef_(idx)

    def simulate(self, x0, u, t=None, upper_bound=1e4, **kwargs):
        """
        Discrete:
            Wrapper for pysindy model simulator
        Continuous:
            faster implementation using zero-hold control
            (reduced accuracy, increased chance of divergence)

        Inputs:
            x0: ndarray with shape (n_state,) or (1, n_state)
                initial state to simulate forward
            u: ndarray with shape (n_time, n_inputs)
                control inputs (actions) to simulate forward
            kwargs: key word arguments for pysindy model.simulate

        Note:
            must pass a `t` kwarg <= len(u).
        Returns:
            ndarray of states with shape (n_time, n_state)
        """
        if self.discrete:
            x_list = self.model.simulate(x0, u=u, t=t, **kwargs)
            if np.any(np.abs(x_list) > upper_bound):
                logging.warning("Bound exceeded: DISCRETE. Likely integration blowup")
                x_list = np.clip(x_list, a_min=-upper_bound, a_max=upper_bound)
        else:
            # zero-hold control
            x_list = [x0]
            for i, ui in enumerate(u):
                update = solve_ivp(
                    self._dyn_fn,
                    y0=x_list[-1],
                    t_span=[0, self.dt],
                    args=(np.array([ui]),),
                    **kwargs
                ).y.T[-1]
                if np.any(np.abs(update) > upper_bound):
                    # np.save("blowup_error_x0.npy", x_list)
                    # np.save("blowup_error.npy", x_list)
                    # np.save("blowup_error_control.npy", u)
                    logging.warning(
                        "Bound exceeded: ZERO-HOLD. Likely integration blowup"
                    )
                    update = np.clip(update, a_min=-upper_bound, a_max=upper_bound)
                    # raise ValueError(
                    #     "Bound exceeded: ZERO-HOLD. Likely integration blowup"
                    # )
                x_list.append(update)
            x_list.pop(-1)
        return np.array(x_list)
        # return NotImplementedError

    def _dyn_fn(self, t, x, u=None):
        return self.model.predict(x.reshape(1, -1), u=u.reshape(1, -1))

    def get_coef_list(self):
        """'
        Get list of model coefficients.

        (Wrapper for pysindy optimizer `coef_list` attribute.)
        """
        return self.model.optimizer.coef_list

    def predict(self, x, u):
        """
        The one-step predictor (wrapper for pysindy simulator)
        """
        if self.discrete:
            update = (self.simulate(x, np.array([u]), t=2))[-1]
        else:
            update = solve_ivp(
                self._dyn_fn, y0=x, t_span=[0, self.dt], args=(np.array([u]),)
            ).y.T[-1]
        if self.callbacks is not None:
            update = self.callbacks(update)
        return update

    def print(self):
        """wrapper for pysindy print"""
        self.model.print()

    def set_ensemble_coefs_(self, weight_list):
        self.model.optimizer.coef_list = weight_list
        self.model.optimizer.coef_ = self.set_idx_coef_(0)

    def save(self, save_path):
        """
        ~~EXPERIMENTAL~~
        Save dynamics model as a pickle file
        """
        with open(save_path, "wb") as f:
            pickle.dump(self, f)

    def load(self, load_path):
        """
        ~~EXPERIMENTAL~~
        Load dynamics model from a pickle file
        """

        with open(load_path, "rb") as f:
            model = pickle.load(f)
            config = model.config
            self.__init__(config)

            # some silly bookkeeping to make pysindy happy
            x_tmp = np.ones((10, model.n_state))
            u_tmp = np.ones((10, model.n_control))
            self.fit([x_tmp], [u_tmp])

            self.model.optimizer.coef_list = model.optimizer.coef_list
            self.model.optimizer.coef_ = model.optimizer.coef_


class FCNet(nn.Module):
    """Simple Torch Fully Connected Neural Network"""

    def __init__(self, n_input, n_output, hidden_size=64):
        self.n_input = n_input
        self.n_output = n_output
        self.hidden_size = hidden_size
        super().__init__()

        self.activation = nn.Tanh()

        self.linear_in = nn.Linear(self.n_input, hidden_size)
        self.linear_hidden = nn.Linear(hidden_size, hidden_size)
        self.linear_out = nn.Linear(hidden_size, self.n_output)

        layers = [self.linear_in, self.linear_hidden, self.linear_out]

        # TO-DO: Figure out appropriate place to set seeds
        bias_init = 0.0
        initializer = nn.init.xavier_uniform_
        for layer in layers:
            initializer(layer.weight)
            nn.init.constant_(layer.bias, bias_init)

    def forward(self, x):
        x = self.linear_in(x)
        x = self.activation(x)
        x = self.linear_hidden(x)
        x = self.activation(x)
        x = self.linear_out(x)
        return x


def _reshape_data(obs_trajs, u_trajs, dtype=torch.float32):
    """helper function for reshaping data to conform to a similar API to sindy"""
    X_in = [obs_traj[:-1] for obs_traj in obs_trajs]
    X_out = [obs_traj[1:] for obs_traj in obs_trajs]
    U_in = [u_traj[:-1] for u_traj in u_trajs]

    X_in = torch.tensor(np.concatenate(X_in, axis=0), dtype=dtype)
    X_out = torch.tensor(np.concatenate(X_out, axis=0), dtype=dtype)
    U_in = torch.tensor(np.concatenate(U_in, axis=0), dtype=dtype)

    XU_in = torch.concat((X_in, U_in), dim=-1)
    return XU_in, X_out


class TrajDataset(Dataset):
    """Helper class for bookkepping the neural network training"""

    def __init__(self, XU, X_out):
        self.XU = XU
        self.X_out = X_out

    def __len__(self):
        return len(self.XU)

    def __getitem__(self, idx):
        xu = self.XU[idx]
        x_out = self.X_out[idx]
        return xu, x_out


class SingleNetDynamicsModel(BaseDynamicsModel):
    """A single-neural network dynamics model of the form
     x_{n+1} = NN(x_n, u_n)
    Where NN is a fully connected neural net (FCNet class)
    """

    def __init__(self, config):
        self.config = config
        self.nn_kwargs = config.get("nn_kwargs", {})
        self.callbacks = config.get("callbacks", None)
        self.optimizer_kwargs = config.get("optimizer_kwargs", {})
        self.n_epochs = config.get("n_epochs", 100)
        self.batch_size = config.get("batch_size", 500)

        self.callbacks = config.get("callbacks", None)

        self.model = FCNet(**self.nn_kwargs)
        self.optimizer = optim.Adam(self.model.parameters(), **self.optimizer_kwargs)
        # self.optimizer = optim.LBFGS(self.model.parameters(),
        #                             **self.optimizer_kwargs)
        self.loss_fn = torch.nn.MSELoss()

        if self.callbacks is not None:
            self.callbacks = getattr(dynamics_callbacks, self.callbacks)

    def predict(self, x, u, dtype=torch.float32):
        """The one-step predictor"""
        x_in = torch.tensor(np.array(x), dtype=dtype)
        u_in = torch.tensor(np.array(u), dtype=dtype)
        xu_in = torch.concat((x_in, u_in), dim=-1)

        update = self.model(xu_in)

        if isinstance(update, torch.Tensor):
            update = update.detach().numpy()

        if self.callbacks is not None:
            update = self.callbacks(update)
        return update

    def set_weights_(self, state_dict):
        self.model.load_state_dict(state_dict)

    def get_coef_list(self):
        return self.model.state_dict()

    def _get_val_loss(self, val_dataset):
        xu, x_out = val_dataset[:]

        with torch.no_grad():
            output = self.model(xu)
            val_loss = self.loss_fn(output, x_out)
        return val_loss

    def fit(self, XU_in, X_out) -> tuple[object, dict]:
        """We assume XU_in and X_out are outputs of `_reshape_data`
        and ready to be passed into the neural network.

        vvvv
        We also
        assume that we're dealing with small enough batch sizes to make
        use of fully-batch optimization with the L-BFGS optimizer
        ^^^^

        Because we don't actually expect to use a single model by itself,
        we'll let the ensemble dispatch the proper training sets

        TO-DO: figure out if callbacks need to be here...?
        Probably not because we'd have to have the callbacks
        be differentialbe?
        """
        n_pts = len(XU_in)

        dataset = TrajDataset(XU_in, X_out)
        n_train = int(0.8 * n_pts)
        n_val = n_pts - n_train

        train_dataset, val_dataset = random_split(dataset, [n_train, n_val])
        train_dataloader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )

        val_loss_prev = np.inf

        for epoch in range(self.n_epochs):
            for xu, x_out in train_dataloader:

                def closure():
                    self.optimizer.zero_grad()
                    output = self.model(xu)
                    loss = self.loss_fn(output, x_out)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    return loss

                self.optimizer.step(closure)

            # compare previous validation loss. If we have an increase, stop training
            val_loss = self._get_val_loss(val_dataset)
            if val_loss_prev < val_loss:
                break
            val_loss_prev = val_loss

        train_info = {
            "epoch": epoch,
            "val_loss": val_loss,
        }

        return self.model, train_info


class EnsembleNetDynamicsModel(BaseDynamicsModel):
    """
    An ensemble wrapper for `n_models` identical SingleNetDynamicsModel objects.
    When multiple trajectories are provided, their (x_n, u_n), x_{n+1} pairs are generated
    and then the all samples are split among the
    """

    def __init__(self, config):
        self.config = config
        self.n_models = config.get("n_models", 5)
        self.single_model_config = config.get("single_model_config", None)

        self.frac_subset = config.get(
            "frac_subset", 0.6
        )  # fraction of samples to provide each model
        self.resample = config.get("resample", True)  # Whether to resample

        assert (
            self.single_model_config is not None
        ), "Must include single model configuration"

        self.ensemble = [
            SingleNetDynamicsModel(self.single_model_config)
            for n in range(self.n_models)
        ]

        self.callbacks = self.ensemble[0].callbacks

    def split_data(self, X, Y):
        n_pts = len(X)
        splits = []
        for n in range(self.n_models):
            idx = np.random.choice(
                n_pts, size=int(n_pts * self.frac_subset), replace=False
            )
            splits.append((X[idx], Y[idx]))
        return splits

    def fit(self, observations, actions, dtype=torch.float32) -> tuple[object, dict]:
        XU_in, X_out = _reshape_data(observations, actions, dtype=dtype)

        # split among members of the ensemble
        datasets = self.split_data(XU_in, X_out)  # IMPLEMENT

        val_losses = []
        epochs = []
        for (xu_in, x_out), net_model in zip(datasets, self.ensemble):

            _, train_info = net_model.fit(xu_in, x_out)
            val_losses.append(train_info["val_loss"])

            epochs.append(train_info["epoch"])

        total_train_info = {"epochs": epochs, "val_losses": val_losses}
        return None, total_train_info

    def get_coef_list(self):
        return [net_model.get_coef_list() for net_model in self.ensemble]

    def set_ensemble_coefs_(self, state_dicts):
        """
        TO-DO: set optimizer values, too...?
        """
        for net_model, state_dict in zip(self.ensemble, state_dicts):
            net_model.set_weights_(state_dict)

    def predict(self, x, u):
        preds = np.array([net_model.predict(x, u) for net_model in self.ensemble])
        update = np.mean(preds, axis=0)

        if self.callbacks is not None:
            update = self.callbacks(update)

        return update

    def set_mean_coef_(self, **kwargs):
        """Need analogous function to conform to EnsembleSINDyDynamicsModel API"""
        pass

    def save(self, save_path):
        """EXPERIMENTAL"""

        with open(save_path, "wb") as f:
            pickle.dump(self.get_coef_list(), f)

    def load(self, load_path):
        """EXPERIMENTAL"""

        with open(load_path, "rb") as f:
            state_dicts = pickle.load(f)

        self.set_ensemble_coefs_(state_dicts)
