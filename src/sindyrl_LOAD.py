import warnings

from utils import (
    load_config,
    plot_action_trajectory,
    plot_training_kpis,
)
import matplotlib.pyplot as plt
import torch

warnings.filterwarnings("ignore")
from pathlib import Path
import logging
import os
import numpy as np
import logging

from sindy_rl.policy import RLlibPolicyWrapper, RandomPolicy
from sindy_rl.dyna import DynaSINDy
from sindy_rl.file_helpers import subfolder_to_full_path
from sindy_rl.utils.parse_cli_args import parse_cli_args

import random

global_test_seed = 0
np.random.seed(global_test_seed)
random.seed(global_test_seed)
torch.manual_seed(global_test_seed)


# NOTE:
# Main script used to evaluate all the models, usually called via:
# python sindyrl_LOAD.py --no-baseline --filename "<cnofig.yml>"
# the script provides several evaluation methods:
# - evaluate_model: evaluate the model in a classical sense, i.e. rewards, control
# in the deployed scenario (should be run after training any model)
# - debugging_tests: highly specific tests, often including hand-written tensors and numbers
# to manually check for specific behavior
# - evaluate_AE: script to evaluate the training of the AE, i.e. how good are the dynamics learned


def _load_dyna_policy(
    config: dict,
    model_checkpoint_number: str,
    global_checkpoint_path: str,
    only_real_env: bool = False,
):
    """Load the DynaSindy Object and the corresponding policy."""
    dyna = DynaSINDy(config, only_eval=True)
    print(global_checkpoint_path)
    print(model_checkpoint_number)
    # algo is trained once before, in the constructor, but now the correct weights are loaded
    dyna.drl_algo.restore(global_checkpoint_path + "/" + model_checkpoint_number)
    print(type(dyna.drl_algo))
    print(dyna.drl_algo.__dict__)
    policy = RLlibPolicyWrapper(dyna.drl_algo)
    if only_real_env:
        return dyna, policy, None

    dyna.dynamics_model.load(
        global_checkpoint_path + "/" + model_checkpoint_number + "/dyn_model.pkl"
    )
    surrogate_env = dyna.drl_config["environment"]["env"](
        dyna.drl_config["environment"]["env_config"]
    )
    print("Policy creation and restoring of the DRL algo worked.")
    return dyna, policy, surrogate_env


def evaluate_AE(
    config,
    model_checkpoint_number: str,
    model_prefix: str,
):
    dyna, policy, surrogate_env = _load_dyna_policy(
        config,
        model_checkpoint_number,
        global_checkpoint_path=global_checkpoint_path,
        only_real_env=False,
    )
    poly_features = dyna.dynamics_model.feature_library.get_feature_names()
    xi_matrix = dyna.dynamics_model.model.xi.weight
    latent_state_dim = xi_matrix.shape[1]
    dict_dim = xi_matrix.shape[0]
    print("\n\n")
    print("Interpretation of the dynamics")
    for idx_latent in range(latent_state_dim):
        out = "x{} =".format(idx_latent)
        for feat_idx in range(dict_dim):
            if torch.abs(xi_matrix[feat_idx, idx_latent]) > 0.1:
                out += " {:1.3f}*{} +".format(
                    xi_matrix[feat_idx, idx_latent], poly_features[feat_idx]
                )
        print(out[:-1])
    print("\n\n")

    print("Analyze the off-policy actions")

    print(dyna.dynamics_model.model.xi.weight)

    off_policy = dyna_config["off_policy_pi"]
    for _ in range(10):
        obs, _ = dyna.real_env.reset()
        # take the off policy action
        control = off_policy.compute_action(obs)
        obs_next, _, _, _, _ = dyna.real_env.step(control)

        obs_next_AE = dyna.dynamics_model.predict(obs, control)
        print("AE prediction: ", np.linalg.norm(obs_next - obs_next_AE))


def evaluate_model(
    config,
    model_checkpoint_number: str,
    model_prefix: str,
    only_real_env: bool,
):
    n_steps_extrapolation = config["drl"]["config"]["environment"]["env_config"][
        "max_episode_steps"
    ]
    dyna, policy, surrogate_env = _load_dyna_policy(
        config,
        model_checkpoint_number,
        global_checkpoint_path=global_checkpoint_path,
        only_real_env=only_real_env,
    )
    print("RUNNING THE FOM ENVIRONMENT")
    total_rew, proj_total_rew, action_traj = policy.run(
        dyna.real_env, seed=global_test_seed
    )
    total_rew_surrogate = None
    if not only_real_env:
        print("RUNNING THE SURROGATE ENVIRONMENT")
        surrogate_env.max_episode_steps = dyna.real_env.n_steps
        total_rew_surrogate, action_traj_surrogate = policy.run_surrogate(
            surrogate_env, seed=global_test_seed
        )
    # this is ugly, but otherwise RayRL is throwing an unsolvable error
    from utils import plot_state_traj, plot_reward

    # Plotting
    plot_reward(total_rew, proj_total_rew, total_rew_surrogate)

    plot_state_traj(
        dyna.real_env,
        5,
        global_plot_path,
        model_prefix + model_checkpoint_number,
        avg_reward=np.average(total_rew),
        n_steps_extrapolation=n_steps_extrapolation,
    )

    print("plotting state trajectories")
    plot_action_trajectory(
        action_traj=action_traj,
        global_plot_path=global_plot_path,
        model_name=model_prefix + model_checkpoint_number,
        n_steps_extrapolation=n_steps_extrapolation,
    )
    if not only_real_env:
        plot_state_traj(
            surrogate_env,
            5,
            global_plot_path,
            model_prefix + model_checkpoint_number,
            surrogate_model=True,
            avg_reward=np.average(total_rew_surrogate),
            n_steps_extrapolation=n_steps_extrapolation,
        )
        plot_action_trajectory(
            action_traj=action_traj_surrogate,
            global_plot_path=global_plot_path,
            model_name=model_prefix + model_checkpoint_number,
            surrogate=True,
            n_steps_extrapolation=n_steps_extrapolation,
        )
    plot_training_kpis(
        global_plot_path,
        global_log_path,
        model_name=model_prefix + model_checkpoint_number,
    )
    plt.show()


def debugging_tests(
    config,
    model_checkpoint_number: str,
    model_prefix: str,
):
    dyna, policy, surrogate_env = _load_dyna_policy(
        config, model_checkpoint_number, global_checkpoint_path=global_checkpoint_path
    )
    print("RUNNING THE REAL ENVIRONMENT")
    total_rew, proj_total_rew, action_traj = policy.run(
        dyna.real_env, seed=global_test_seed
    )
    print(dyna.drl_algo.__dict__)
    surrogate_env.max_episode_steps = dyna.real_env.n_steps
    print("RUNNING THE SURROGATE ENVIRONMENT")
    total_rew_surrogate, action_traj_surrogate = policy.run_surrogate(
        surrogate_env, seed=global_test_seed
    )
    print("testing zero states")
    surrogate_env.reset(**{"state": np.zeros(dyna.real_env.n_state)})
    obs, rew, terminated, truncated, info = surrogate_env.step(
        np.zeros(dyna.real_env.n_action)
    )
    print("result for a zero state and zero action:")
    print("obs: ", obs)
    print("rew: ", rew)

    print("test policy for zero state:")
    test_action = policy.compute_action(np.zeros(dyna.real_env.n_observation))
    print("test_action:\n", test_action)

    # surrogate_env.reset(**{"state": np.zeros(dyna.real_env.n_state)})
    surrogate_env.obs = np.zeros(dyna.real_env.n_observation)
    # surrogate_env.obs = s_reset
    test_total_rew_surrogate, test_action_traj_surrogate = policy.run_surrogate(
        surrogate_env, seed=global_test_seed, reset=False
    )
    print("first action surrogate")
    print(test_action_traj_surrogate[:, 0])
    # test the same with the real env
    dyna.real_env.state = np.zeros(dyna.real_env.n_state)
    total_rew, proj_total_rew, action_traj = policy.run(
        dyna.real_env, seed=global_test_seed, reset=False
    )
    print("first action real")
    print(action_traj[:, 0])


if __name__ == "__main__":
    from utils import get_global_paths

    logging.getLogger().setLevel(logging.INFO)
    # load the paths
    baseline, filename, _, checkpoint, debug = parse_cli_args()
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

    print("Debug flag: {}".format(debug))
    # if debug:
    # debugging_tests(
    #     config=dyna_config,
    #     model_checkpoint_number=checkpoint,
    #     model_prefix=model_prefix,
    # )
    # else:
    evaluate_model(
        config=dyna_config,
        model_checkpoint_number=checkpoint,
        model_prefix=model_prefix,
        only_real_env=baseline,
    )
    print("evaluate the AutoEncoder")
    # evaluate_AE(
    #     config=dyna_config,
    #     model_checkpoint_number=checkpoint,
    #     model_prefix=model_prefix,
    # )
