import numpy as np
import matplotlib.pyplot as plt
from controlgym.envs import PDE
from pathlib import Path
import yaml
import os
import logging

from pprint import pprint


def load_config(exp_folder: str, verbose: bool = True):
    filename = exp_folder / "exp_config.yml"
    with open(filename, "r") as f:
        dyna_config = yaml.load(f, Loader=yaml.SafeLoader)
    if verbose:
        pprint(dyna_config)
    return dyna_config


def get_global_paths(baseline: bool, config: dict):
    """Return:
    - filename, name of the model
    - prefix, surrogate_ or baseline_ depending on the baseline flag
    - global_path, where to find the model
    - global_plot_path, where to save the plots
    - global_checkpoint_path, where to find the checkpoint to load from
    """
    if baseline:
        filename = "dyna_burgers_baseline.yml"
        model_prefix = "ppo_baseline_"
    else:
        filename = "dyna_burgers.yml"
        model_prefix = "ppo_surrogate_"

    global_path = config["global_dir"] + config["folder_name"]

    global_plot_path = global_path + config["plot_dir"]
    global_log_path = global_path + config["log_dir"]
    global_checkpoint_path = global_path + config["checkpoint_dir"]
    return (
        filename,
        model_prefix,
        global_path,
        global_plot_path,
        global_log_path,
        global_checkpoint_path,
    )


def plot_reward(
    full_reward_real_env: np.ndarray,
    projected_reward_real_env: np.ndarray,
    proj_reward_surrogate: np.ndarray = None,
):
    plt.figure()
    plt.plot(full_reward_real_env, label="Rew")
    plt.plot(projected_reward_real_env, label="Rew Projected")
    if proj_reward_surrogate is not None:
        plt.plot(proj_reward_surrogate, label="Rew SURROGATE")
    plt.legend(loc="best")


# plot actions/controls
def plot_action_trajectory(
    action_traj: np.ndarray[float],
    global_plot_path: Path,
    model_name: str,
    save: bool = True,
    surrogate: bool = False,
    n_steps_extrapolation: int = 20,
):
    _plot_actions(
        action_traj,
        global_plot_path,
        model_name,
        "all_actions",
        save,
        surrogate=surrogate,
        n_steps_extrapolation=n_steps_extrapolation,
    )
    _plot_actions(
        action_traj,
        global_plot_path,
        model_name,
        "partial",
        save,
        idx_list=[1, 4, 6],
        n_steps_extrapolation=n_steps_extrapolation,
    )


def _plot_actions(
    action_traj: np.ndarray[float],
    global_plot_path: Path,
    model_name: str,
    suffix: str,
    save: bool = True,
    idx_list: list[int] = None,
    surrogate: bool = False,
    n_steps_extrapolation: int = 20,
):
    n_actions = action_traj.shape[1]
    x_axis = np.arange(0, n_actions)
    max_action = np.max(np.abs(action_traj))
    plt.figure()
    if idx_list is None:
        idx_list = list(range(action_traj.shape[0]))
    for idx in idx_list:
        plt.plot(x_axis, action_traj[idx, :], label="idx {}".format(idx))
    plt.axvline(n_steps_extrapolation, label="Extrapolation", color="black")
    plt.hlines(0.0, 0, n_actions, linestyles="--", colors="red", label="zero control")
    surrogate_str = "_surrogate" if surrogate else ""
    plt.title(
        "{}{}:\nTrajectories of ALL actions.\n max(abs(actions)) = {}".format(
            model_name, surrogate_str, max_action
        )
    )
    plt.legend(loc="upper right")
    if save:
        plt.savefig(
            global_plot_path
            + "/traj_actions{}_{}_{}.pdf".format(surrogate_str, suffix, model_name)
        )


# plot the state evolution over time
def plot_state_traj(
    env: PDE,
    start_idx_non_observed: int,
    global_plot_path: Path,
    model_name: str,
    save: bool = True,
    surrogate_model: bool = False,
    avg_reward: float = None,
    n_steps_extrapolation: int = 20,
):
    """Analyze the state trajectory of a controller in an env.

    Parameters
    ----------
    env : PDE
        PDE-like environment.
    start_idx_non_observed : int
        Index of the first non-observed states
    global_plot_path : str
        Path to save the plots.
    model_name : str
        Name of the model, i.e. suffix for the filenames.
    save : bool, optional
        Save the plots, by default True.
    """
    state_traj = env.state_traj
    n_states = state_traj.shape[1]
    x_axis = np.arange(0, n_states)

    _state_heatmap(
        state_traj,
        env.C @ state_traj if not surrogate_model else None,
        global_plot_path,
        model_name,
        surrogate_model=False,
        n_steps_extrapolation=n_steps_extrapolation,
    )

    # plot the trajectory of states directly observed by the controller
    fig = plt.figure()
    # C.shape = (n_obs, n_states) with evenly distributed state observers
    if not surrogate_model:
        observations = env.C @ state_traj
        offset = env.n_observation
    else:
        observations = state_traj
    for idx in range(observations.shape[0]):
        plt.plot(x_axis, observations[idx, :], label="idx {}".format(idx))
    plt.axvline(n_steps_extrapolation, label="Extrapolation", color="gray")
    plt.hlines(0.0, 0, n_states, linestyles="--", colors="red", label="target_state")
    plt.title(
        "{}{}:\nTrajectories of DIRECTLY observed states\nTotal avg reward: {}".format(
            model_name, "-Surrogate" if surrogate_model else "", avg_reward
        )
    )
    plt.legend(loc="upper right")
    if save:
        plt.savefig(
            global_plot_path
            + "/traj_directly_observed_states{}_{}.pdf".format(
                "_surrogate" if surrogate_model else "", model_name
            )
        )
    plt.close(fig)

    if surrogate_model:
        return

    # plot trajectory of non-observed states
    fig_non_observed = plt.figure()
    start = start_idx_non_observed
    for idx in range(start, state_traj.shape[0], offset):
        plt.plot(x_axis, state_traj[idx, :], label="idx {}".format(idx))
    plt.axvline(n_steps_extrapolation, label="Extrapolation", color="gray")
    plt.hlines(0.0, 0, 100, linestyles="--", colors="red", label="target_state")
    plt.title(
        "{}:\nTrajectories of NON-observed states\nTotal avg reward: {}".format(
            model_name, avg_reward
        )
    )
    plt.legend(loc="upper right")

    if save:
        plt.savefig(
            global_plot_path + "/traj_non_observed_states_{}.pdf".format(model_name)
        )
    plt.close(fig_non_observed)


def _state_heatmap(
    state_traj,
    observations,
    global_plot_path,
    model_name,
    surrogate_model=False,
    n_steps_extrapolation: int = 20,
):
    """
    Plot two heatmaps side by side: one for state_traj (all states) and the other
    for (directly observed states) observations.

    Parameters:
        state_traj (numpy.ndarray): Array representing the state trajectory.
        observations (numpy.ndarray): Array representing the observations.
        global_plot_path (str): Path to the directory where the plot will be saved.
        model_name (str): Name of the model.
        surrogate_model (bool, optional): Whether the model is a surrogate model. Defaults to False.
    """

    # Create a figure and two subplots side by side
    fig, axs = plt.subplots(1, 2, figsize=(15, 6))

    # Plot the heatmap for state_traj
    axs[0].imshow(state_traj, aspect="auto")
    axs[0].set_title("trajectory over time (all state)")
    axs[0].set_xlabel("Time")
    axs[0].set_ylabel("State")
    axs[0].set_aspect("auto")
    axs[0].grid(False)
    fig.colorbar(axs[0].imshow(state_traj), ax=axs[0])

    # Plot the heatmap for observations
    if observations is not None:
        axs[1].imshow(observations, aspect="auto")
        axs[1].set_title("trajectory over time (directly observed states)")
        axs[1].set_xlabel("Time")
        axs[1].set_ylabel("State")
        axs[1].set_aspect("auto")
        axs[1].grid(False)
        fig.colorbar(axs[1].imshow(observations), ax=axs[1])

    # extrapolation line
    for ax in axs:
        ax.axvline(x=n_steps_extrapolation, color="red", linestyle="--")
    # Adjust layout
    plt.tight_layout()

    # Save the plot
    plt.savefig(
        global_plot_path
        + "/heatmap_states{}_{}.pdf".format(
            "_surrogate" if surrogate_model else "", model_name
        )
    )

    # Show the plot
    plt.close(fig)


def _parse_log_file(log_file_path, keys_to_find: list[str]) -> dict:
    """Parse a generic log-file to find NUMERICAL values."""
    logging.info("Parsing the log file {}".format(log_file_path))
    if not os.path.exists(log_file_path):
        print("Log file {} does NOT exist, aborting.".format(log_file_path))
        return
    results = {key: [] for key in keys_to_find}
    with open(log_file_path, "r") as file:
        print(file)
        for line in file:
            for key in keys_to_find:
                if key in line:
                    key_value = float(line.split(":")[-1].strip())
                    results[key].append(key_value)
    return results


def _get_log_files(global_log_path: str) -> list[str]:
    """Get the logging files in a directory, we know by construction
    that each model has at most 2 log files (SINDyRL + optional AE).

    Parameters
    ----------
    global_log_path : str
        Path to the logging folder.

    Returns
    -------
    list[str]
        List with names of the logging files.
    """
    files = os.listdir(global_log_path)
    # Filter out the .log file -> there are only exactly two files (AE training and the DRL training)
    log_files = [
        file
        for file in files
        if file.endswith(".log") and os.path.isfile(os.path.join(global_log_path, file))
    ]
    print(log_files)
    return log_files


def plot_training_kpis(
    global_plot_path: str,
    global_log_path: str,
    model_name: str,
    save: bool = True,
):
    """Higher level wrapper to perform the following analysis:
        - find all log files
        - check if the model has an AE
        - plot the rewards
        - [optionally] plot the training of the AE

    Parameters
    ----------
    global_plot_path : str
        Path to all plots, used to save the results.
    global_log_path : str
        Path to the <= 2 log files, where the data is stored.
    model_name : str
        Name of the model, names of the plots are adjusted accordingly.
    save : bool, optional
        Save the analysis plots, by default True
    """

    log_files = _get_log_files(global_log_path)
    if len(log_files) == 2:
        if log_files[0].startswith("dynamics"):
            AE_log_file, drl_log_file = log_files
        else:
            drl_log_file, AE_log_file = log_files
        _plot_AE_training(
            global_plot_path, global_log_path, AE_log_file, model_name, save
        )
    else:
        drl_log_file = _get_log_files(global_log_path)[0]

    _plot_rewards(global_plot_path, global_log_path, drl_log_file, model_name, save)


def _plot_AE_training(
    global_plot_path: str,
    global_log_path: str,
    AE_log_file: str,
    model_name: str,
    save: bool = True,
):
    """Plot the AE training, i.e. number of internal epochs, validation and
    training loss and internal training time.

    Parameters
    ----------
    global_plot_path : str
        Path to all plots.
    AE_log_file : str
        Name of the AE log file.
    model_name : str
        Name of the higher level mode.
    save : bool, optional
        Save the resulting plots, by default True.
    """
    train_AE_res = _parse_log_file(
        global_log_path + "/" + AE_log_file,
        keys_to_find=["AE_epoch", "AE_val_loss", "AE_train_loss", "AE_train_time_s"],
    )
    print(train_AE_res)
    for key, value in train_AE_res.items():
        plt.figure()
        plt.plot(value)
        plt.title("{}:\n{}".format(model_name, key))
        plt.legend(loc="upper right")
        if save:
            plt.savefig(global_plot_path + "/{}_{}.pdf".format(key, model_name))


def _plot_rewards(
    global_plot_path: str,
    global_log_path: str,
    drl_log_file: str,
    model_name: str,
    save: bool = True,
):
    """Plot the reward of the highest level RL agent.

    Parameters
    ----------
    global_plot_path : str
        Path to all plots.
    global_log_path : str
        Path to all logs, we load the rewards from the logs.
    drl_log_file : str
        Name of the DRL log file (not the AE one).
    model_name : str
        Name of the model.
    save : bool, optional
        Save the resulting plots, by default True.
    """
    reward_res = _parse_log_file(
        global_log_path + "/" + drl_log_file,
        keys_to_find=["episode_reward_max", "episode_reward_mean"],
    )
    rewards_max = np.array(reward_res["episode_reward_max"][::2])[:200]
    rewards_mean = np.array(reward_res["episode_reward_mean"][::2])[:200]
    plt.figure()
    plt.plot(rewards_max)
    plt.title("{}:\nMaximum Rewards".format(model_name))
    plt.legend(loc="upper right")
    if save:
        plt.savefig(global_plot_path + "/traj_rewards_max_{}.pdf".format(model_name))

    plt.figure()
    plt.plot(rewards_mean)
    plt.title("{}:\nMean Rewards".format(model_name))
    plt.legend(loc="upper right")
    if save:
        plt.savefig(global_plot_path + "/traj_rewards_mean_{}.pdf".format(model_name))
