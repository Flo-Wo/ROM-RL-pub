from sindy_rl.file_helpers import subfolder_to_full_path
from sindy_rl.utils.parse_cli_args import parse_cli_args
from analysis.helper_sindyrl_train import (
    eval_navierStokes_policy,
    plot_navStok_controls,
    plot_navStok_vector_field,
)
from sindyrl_LOAD import _load_dyna_policy
import matplotlib.pyplot as plt

from sindy_rl.policy import RandomPolicy
import logging
from utils import get_global_paths, load_config
from pathlib import Path
import numpy as np
import random
import torch

# main script to analyze the sample efficiency for the paper experiments
# for each checkpoint we evalute the model on 5 new evaluation seeds,
# count the number of interactions with the full order model and
# save the results to create a combined plot

plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        # "font.serif": ["Palatino"],
        "font.size": 14,
    }
)

global_test_seed = 0
np.random.seed(global_test_seed)
random.seed(global_test_seed)
torch.manual_seed(global_test_seed)

test_seeds = [42, 123, 789, 2024, 5678]


def plot_NavStokes(
    dyna_config: dict,
    global_checkpoint_path: str,
    model_checkpoint_number: str,
):
    dyna, policy, surrogate_env = _load_dyna_policy(
        dyna_config,
        model_checkpoint_number,
        global_checkpoint_path=global_checkpoint_path,
        only_real_env=True,
    )
    rewards_rollout, action_traj = policy.run_navierStokes(
        dyna.real_env,
        seed=0,
    )
    print(np.mean(rewards_rollout))
    action_traj = eval_navierStokes_policy(
        dyna=dyna,
        config=dyna_config,
        global_test_seed=global_test_seed,
        n_iter=9999999,
        baseline=baseline,
    )
    plot_navStok_controls(
        action_traj,
        dyna.real_env,
        global_plot_path="../data/paper_evaluation/navier_stokes/fom_interact/",
        model_name="control_plot",
        surrogate_model=(not baseline),
        save=True,
    )
    plot_navStok_vector_field(
        dyna.real_env,
        time_idx=199,
        global_plot_path="../data/paper_evaluation/navier_stokes/fom_interact/",
        model_name="flow_field",
        surrogate_model=(not baseline),
        file_prefix="END_HORIZON",
        save=True,
    )


def eval_NavStokes(
    dyna_config: dict,
    global_checkpoint_path: str,
    model_checkpoint_number: str,
):
    dyna, policy, surrogate_env = _load_dyna_policy(
        dyna_config,
        model_checkpoint_number,
        global_checkpoint_path=global_checkpoint_path,
        only_real_env=True,
    )

    # controlgym init
    avg_rew_over_time_eval = []
    avg_rew_over_time_extrapolate = []
    for seed in test_seeds:
        rewards_rollout, action_traj = policy.run_navierStokes(dyna.real_env, seed=seed)
        avg_rew_over_time_eval.append(np.average(rewards_rollout))
        avg_rew_over_time_extrapolate.append(np.average(rewards_rollout))
    print("Evaluation")
    print(
        "${:.2f} \\pm {:.2f}$".format(
            np.average(avg_rew_over_time_eval),
            np.std(avg_rew_over_time_eval),
        )
    )
    return (
        np.average(avg_rew_over_time_eval),
        np.average(avg_rew_over_time_extrapolate),
    )


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
    # eval_NavStokes(
    #     dyna_config,
    #     global_checkpoint_path,
    #     checkpoint,
    # )
    plot_NavStokes(
        dyna_config,
        global_checkpoint_path,
        checkpoint,
    )
