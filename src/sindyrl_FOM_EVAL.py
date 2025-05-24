from sindy_rl.file_helpers import subfolder_to_full_path
from sindy_rl.utils.parse_cli_args import parse_cli_args
from sindyrl_LOAD import _load_dyna_policy
import matplotlib.pyplot as plt

from sindy_rl.policy import RandomPolicy
import logging
from utils import get_global_paths, load_config
from pathlib import Path
import numpy as np
import random
import torch
import os
import re

# main script to analyze the sample efficiency for the paper experiments
# for each checkpoint we evalute the model on 5 new evaluation seeds,
# count the number of interactions with the full order model and
# save the results to create a combined plot

global_test_seed = 0
np.random.seed(global_test_seed)
random.seed(global_test_seed)
torch.manual_seed(global_test_seed)

test_seeds = [42, 123, 789, 2024, 5678]

# enable latex formatting
plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        # "font.serif": ["Palatino"],
        "font.size": 14,
    }
)


def _save_results(name_data: dict, prefix: str, path: str):
    for filename, arr in name_data.items():
        np.save(path + prefix + filename, arr)


def eval_single_checkpoint(config: str, model_checkpoint_number: str):
    dyna, policy, surrogate_env = _load_dyna_policy(
        config,
        model_checkpoint_number,
        global_checkpoint_path=global_checkpoint_path,
        only_real_env=True,
    )
    avg_rew_over_time_eval = []
    avg_rew_over_time_extrapolate = []

    for seed in test_seeds:
        # run the policy on the real environment -> use custom seed
        rew_list, _, _ = policy.run(dyna.real_env, seed=seed)
        eval = rew_list[1:20]
        extrapolate = rew_list[20:]
        avg_rew_over_time_eval.append(np.average(eval))
        avg_rew_over_time_extrapolate.append(np.average(extrapolate))
    return (
        np.average(avg_rew_over_time_eval),
        np.average(avg_rew_over_time_extrapolate),
        np.std(avg_rew_over_time_eval),
        np.std(avg_rew_over_time_extrapolate),
    )


def eval_single_checkpoint_NavierStokes(config: str, model_checkpoint_number: str):
    dyna, policy, surrogate_env = _load_dyna_policy(
        config,
        model_checkpoint_number,
        global_checkpoint_path=global_checkpoint_path,
        only_real_env=True,
    )
    avg_rew_over_time_eval = []
    avg_rew_over_time_extrapolate = []

    for seed in test_seeds:
        # run the policy on the real environment -> use custom seed
        # rew_list, _, _ = policy.run(dyna.real_env, seed=seed)
        rewards_rollout, action_traj = policy.run_navierStokes(
            dyna.real_env,
            seed=seed,
        )
        avg_rew_over_time_eval.append(np.average(rewards_rollout))
        avg_rew_over_time_extrapolate.append(np.average(rewards_rollout))
    return (
        np.average(avg_rew_over_time_eval),
        np.average(avg_rew_over_time_extrapolate),
        0,
        0,
    )


def num_FOM_interaction_NavierStokes(model_checkpoint_number: str):
    pattern = r"\d+"
    match = re.search(pattern, model_checkpoint_number)
    num_interact = int(match.group())
    return num_interact * 128


def num_FOM_interactions(config: str, model_checkpoint_number: str):
    episodes_per_iter = 10
    episode_len = 20

    pattern = r"\d+"
    match = re.search(pattern, model_checkpoint_number)
    num_interact = int(match.group()) * episodes_per_iter * episode_len

    # working with the full order model
    if config["drl"]["config"]["environment"]["env_config"]["use_real_env"]:
        return num_interact
    fit_freq = config["dyn_fit_freq"]
    # divide by the fitting frequency
    num_interact /= fit_freq
    # add off-policy samples from the beginning
    off_policy_steps = config["off_policy_buffer"]["init"]["kwargs"]["n_steps"]
    return num_interact + off_policy_steps


def performance_per_FOM_interactions(
    config: dict,
    global_checkpoint_path: str,
    upper_bound: int,
    surrogate: bool = False,
    prefix_save_data: str = "",
    path_save_data: str = "",
):
    try:
        checkpoint_names = os.listdir(global_checkpoint_path)
        checkpoint_names = [
            check for check in checkpoint_names if check.startswith("checkpoint")
        ]
        checkpoint_names.sort()
        print(checkpoint_names)
    except:
        print("No checkpoints found")
        return
    # kpis we want to plot: for the internal time horizon and the extrapolation
    # with length factor 4 of the trained time interval
    rewards_eval = []
    stds_eval = []
    rewards_extra = []
    stds_extra = []

    fom_interactions = []
    # cutoff, i.e. simulate early stopping
    checkpoint_names = checkpoint_names[:upper_bound]
    # correct for higher fit frequency, for FOM we can use a finer grid
    if surrogate:
        checkpoint_names = checkpoint_names[::5]  # [::10]
    else:
        # for Navier-Stokes FOM, we use all checkpoints (already saved in the correct way)
        checkpoint_names = checkpoint_names
    print("checkpoints to be evaluated: ")
    print(checkpoint_names)
    for checkpoint in checkpoint_names:
        print("EVALUTION of {}".format(checkpoint))
        # rew_extra, rew_eval, std_extra, std_eval = eval_single_checkpoint(
        #     config, checkpoint
        # )
        rew_extra, rew_eval, std_extra, std_eval = eval_single_checkpoint_NavierStokes(
            config, checkpoint
        )
        # num_interact = num_FOM_interactions(config, checkpoint)
        num_interact = num_FOM_interaction_NavierStokes(checkpoint)

        # append the computed values
        rewards_eval.append(rew_eval)
        rewards_extra.append(rew_extra)

        stds_eval.append(std_eval)
        stds_extra.append(std_extra)

        fom_interactions.append(num_interact)

    # transform to numpy to have numerical operations
    rewards_eval = np.array(rewards_eval)
    rewards_extra = np.array(rewards_extra)
    stds_extra = np.array(stds_extra)
    stds_eval = np.array(stds_eval)

    fom_interactions = np.array(fom_interactions)

    name_data = {
        "rewards_evaluation": rewards_eval,
        "rewards_extrapolation": rewards_extra,
        "stds_evaluation": stds_eval,
        "stds_extrapolation": stds_extra,
        "fom_interactions": fom_interactions,
    }
    _save_results(name_data, prefix_save_data, path_save_data)

    _plot_fom_kpis(
        fom_interactions,
        rewards_eval,
        stds_eval,
        title="Evaluation",
        prefix=prefix_save_data,
    )
    _plot_fom_kpis(
        fom_interactions,
        rewards_extra,
        stds_extra,
        title="Extrapolation",
        prefix=prefix_save_data,
    )
    # actual plot
    plt.show()


def plot_state_control_distribution(
    dyna_config: dict,
    global_checkpoint_path: str,
    model_checkpoint_number: str,
    prefix_save_data: str,
    title: str,
    path_save_data="../data/paper_evaluation/burgers/fom_interact/",
    num_zero_controls: int = 0,
    bell_shape_init: bool = False,
):
    dyna, policy, surrogate_env = _load_dyna_policy(
        dyna_config,
        model_checkpoint_number,
        global_checkpoint_path=global_checkpoint_path,
        only_real_env=True,
    )
    vertical_line_style = "--"
    init_state = None
    # bell shape case
    if num_zero_controls > 0:
        pass
    if bell_shape_init:
        pde_domain = dyna.real_env.domain_coordinates  # spatial domain
        pde_domain_length = dyna.real_env.domain_length  # spatial domain

        # controlgym init
        rand_offset = 0
        # rand_offset = 0.5 * np.random.random(1)  # random offset [0, 0.5)
        init_state = 5 * (
            np.cosh(10 * (pde_domain - 1 * pde_domain_length / 2 + rand_offset)) ** (-1)
        )
        dyna.real_env.diffusivity_constant = 0.01
        vertical_line_style = "-"

        # cosine
        # init_state = 0.75 * np.cos(4 * np.pi * pde_domain + rand_offset)

    print("RUNNING THE REAL ENVIRONMENT")
    total_rew, proj_total_rew, action_traj = policy.run(
        dyna.real_env,
        seed=global_test_seed,
        state=init_state,
        num_zero_controls=num_zero_controls,
    )
    # plot the heatmap for the controls and the state over time
    _heatmap(
        matrix=dyna.real_env.state_traj,
        path=path_save_data,
        prefix=prefix_save_data,
        postfix="states",
        title=title,
        ylabel="States",
        vertical_line_style=vertical_line_style,
    )
    _heatmap(
        matrix=action_traj,
        path=path_save_data,
        prefix=prefix_save_data,
        postfix="controls",
        title=title,
        ylabel="Controls",
        vertical_line_style=vertical_line_style,
    )


def _heatmap(
    matrix: torch.tensor,
    path: str,
    prefix: str,
    postfix: str,
    title: str,
    ylabel: str,
    save: bool = True,
    vertical_line=20,
    vertical_line_style: str = "--",
) -> None:
    fig, ax = plt.subplots()
    ax.axvline(x=vertical_line, color="black", linestyle=vertical_line_style)
    cax = plt.imshow(matrix, cmap="RdBu_r", aspect="auto")
    # Set the x-axis labels
    n_steps = matrix.shape[1]
    tick_interval = 20
    displayed_ticks = range(0, n_steps, tick_interval)
    time_labels = [f"{step * 0.05:.2f}" for step in displayed_ticks]
    ax.set_xticks(displayed_ticks)
    ax.set_xticklabels(time_labels)

    # Add colorbar
    fig.colorbar(cax)
    plt.title(title)
    plt.xlabel("time [s]")
    plt.ylabel(ylabel)
    if save:
        plt.savefig(path + prefix + "_{}_heatmap.pdf".format(postfix))
    plt.close(fig)


def _plot_fom_kpis(
    fom_interactions: np.ndarray,
    rewards: np.ndarray,
    stds: np.ndarray,
    title: str = "",
    prefix: str = "",
) -> None:
    fig, ax = plt.subplots()
    ax.plot(fom_interactions, rewards)
    ax.fill_between(
        fom_interactions, (rewards - stds), (rewards + stds), color="b", alpha=0.1
    )
    ax.set_xlabel("Number of FOM interactions")
    ax.set_ylabel("Average reward on 5 random policies")
    ax.set_title(
        "{}\nEvaluation of full order samples usage (5 diff seeds)".format(title)
    )
    path = "../data/paper_evaluation/burgers/fom_interact/plots/"
    plt.savefig(path + prefix + title + "_fom_interactions.pdf")


if __name__ == "__main__":

    baseline, filename, _, checkpoint, _ = parse_cli_args()
    folder_to_analyze = (
        Path(__file__).parent.parent
        # / "data/sindyrl_data/surrogate/clipping_reset_cap_reward"
        / ("data/sindyrl_data/" + filename)
    )
    dyna_config = load_config(folder_to_analyze, "exp_config.yaml")
    (
        model_filename,
        model_prefix,
        global_path,
        global_plot_path,
        global_log_path,
        global_checkpoint_path,
    ) = get_global_paths(folder_to_analyze, dyna_config)

    dyna_config = subfolder_to_full_path(dyna_config)

    dyna_config["dummy_logger"] = True

    # Setup logger, will not be used
    logging.basicConfig()
    logger = logging.getLogger("dyna-sindy")
    logger.setLevel(logging.INFO)

    # Initialize default off-policy, needed to construct the Dyna element
    n_control = dyna_config["drl"]["config"]["environment"]["env_config"]["act_dim"]
    print("creating the random policy")
    dyna_config["off_policy_pi"] = RandomPolicy(
        low=-1 * np.ones(n_control), high=np.ones(n_control), seed=global_test_seed
    )

    # COMPUTE THE PERFORMANCE AND THE NUMBER OF FO MODEL INTERACTIONS
    performance_per_FOM_interactions(
        dyna_config,
        global_checkpoint_path,
        upper_bound=5000,
        surrogate=(not baseline),
        # prefix_save_data="baseline_",
        prefix_save_data="AE_5",
        # prefix_save_data="baseline_FO_",
        # prefix_save_data="baseline_PO_",
        # prefix_save_data="AE_FO_",
        # prefix_save_data="AE_PO_",
        # prefix_save_data="AE_PO_5_",
        # prefix_save_data="AE_FO_5_",
        # prefix_save_data="AE_PO_10_",
        path_save_data="../data/paper_evaluation/navier_stokes/fom_interact/",
    )
