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

# main script to analyze the sample efficiency for the paper experiments
# for each checkpoint we evalute the model on 5 new evaluation seeds,
# count the number of interactions with the full order model and
# save the results to create a combined plot

global_test_seed = 0
np.random.seed(global_test_seed)
random.seed(global_test_seed)
torch.manual_seed(global_test_seed)

test_seeds = [42, 123, 789, 2024, 5678]


def eval_bell_shape_init(
    dyna_config: dict,
    global_checkpoint_path: str,
    model_checkpoint_number: str,
    num_zero_controls: int = 20,
):
    dyna, policy, surrogate_env = _load_dyna_policy(
        dyna_config,
        model_checkpoint_number,
        global_checkpoint_path=global_checkpoint_path,
        only_real_env=True,
    )
    pde_domain = dyna.real_env.domain_coordinates  # spatial domain
    pde_domain_length = dyna.real_env.domain_length  # spatial domain

    # controlgym init
    avg_rew_over_time_eval = []
    avg_rew_over_time_extrapolate = []
    for seed in test_seeds:
        rand_offset = 0.5 * np.random.random(1)  # random offset [0, 0.5)
        init_state = 5 * (
            np.cosh(10 * (pde_domain - 1 * pde_domain_length / 2 + rand_offset)) ** (-1)
        )
        # run the policy on the real environment -> use custom seed
        rew_list, _, _ = policy.run(
            dyna.real_env,
            seed=seed,
            state=init_state,
            num_zero_controls=num_zero_controls,
            reset=True,
        )
        eval = rew_list[21:40]  # t \in [1,2]
        extrapolate = rew_list[40:]  # t \in (2,6]
        avg_rew_over_time_eval.append(np.average(eval))
        avg_rew_over_time_extrapolate.append(np.average(extrapolate))
    print("Evaluation")
    print(
        "${:.2f} \\pm {:.2f}$".format(
            np.average(avg_rew_over_time_eval),
            np.std(avg_rew_over_time_eval),
        )
    )
    print("Extrapolation")
    print(
        "${:.2f} \\pm {:.2f}$".format(
            np.average(avg_rew_over_time_extrapolate),
            np.std(avg_rew_over_time_extrapolate),
        )
    )
    return (
        np.average(avg_rew_over_time_eval),
        np.average(avg_rew_over_time_extrapolate),
        np.std(avg_rew_over_time_eval),
        np.std(avg_rew_over_time_extrapolate),
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
    eval_bell_shape_init(
        dyna_config,
        global_checkpoint_path,
        checkpoint,
    )
