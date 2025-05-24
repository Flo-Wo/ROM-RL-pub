import warnings
from pathlib import Path

from analysis.helper_sindyrl_train import eval_burgers_policy, eval_navierStokes_policy

from analysis.surrogate_reward import xi_as_heatmap

warnings.filterwarnings("ignore")
import logging
import numpy as np

from sindy_rl.dyna import DynaSINDy
from sindy_rl.file_helpers import (
    read_dyna_config,
    setup_folders,
)
from sindy_rl.utils.parse_cli_args import parse_cli_args

# use different seed than the training seed
global_test_seed = 0

# NOTE:
# Main script used to train all the models, usually called via:
# python sindyrl_TRAIN.py --no-baseline --filename "<config.yml>"


def dyna_sindy(config, baseline: bool = False):
    """
    ray.Tune functional API for defining an expirement
    """
    dyna_config = config
    checkpoint_dir = config["checkpoint_dir"]
    train_iterations = config["n_train_iter"]
    dyn_fit_freq = config["dyn_fit_freq"]
    ckpt_freq = config["ray_config"]["checkpoint_freq"]

    print("creating DynaSINDy...")
    # data collected (or loaded) upon initialization
    dyna = DynaSINDy(dyna_config)
    print("DynaSINDy is created")

    n_steps_extrapolation = config["drl"]["config"]["environment"]["env_config"][
        "max_episode_steps"
    ]
    start_iter = 0

    # for PBT, session is populated with a checkpoint after evaluating the population
    # and pruning the bottom performers
    checkpoint = None
    """
    checkpoint = session.get_checkpoint()
    print("checkpoint: ", checkpoint)
    if checkpoint:
        check_dict = checkpoint.to_dict()
        dyna.load_checkpoint(check_dict)

        # grab the iteration to make sure we are checkpointing correctly
        start_iter = check_dict["epoch"] + 1
    """

    # setup the dynamics, reward, DRL algo push weights to surrogate
    # on remote workers
    print("Before the loop.")
    dyna.fit_dynamics()
    dyna.fit_rew()
    dyna.update_surrogate()

    print("Before the loop: Dynamics, reward fitted. Surrogate updated.")

    collect_dict = {"mean_rew": np.nan, "mean_len": 0}
    counter = 0

    # Main training loop
    for n_iter in range(start_iter, train_iterations):
        checkpoint = None
        print("\nTRAINING MODEL")
        train_results = dyna.train_algo()

        # periodically evaluate by collecting on-policy data
        if (n_iter % dyn_fit_freq) == dyn_fit_freq - 1:
            print("Fitting dynamics, rewrds and update the surrogate")
            (trajs_obs, trajs_acts, trajs_rew, trajs_real_rew) = dyna.collect_data(
                dyna.on_policy_buffer,
                dyna.real_env,
                dyna.on_policy_pi,
                n_rollouts=1,
                **dyna_config["on_policy_buffer"]["collect"]
            )
            train_info = dyna.fit_dynamics()
            """
            try:
                xi_as_heatmap(train_info, config, it_num=n_iter)
            except:
                print("Could not plot Xi.")
            """
            dyna.fit_rew()
            dyna.update_surrogate()

            collect_dict = {}
            collect_dict["mean_rew"] = np.mean([np.sum(rew) for rew in trajs_rew])
            collect_dict["mean_len"] = np.mean([len(obs) for obs in trajs_obs])

            counter += 1

        # Checkpoint (ideally after the latest collection)
        if (n_iter % ckpt_freq) == ckpt_freq - 1:

            # check_dict = dyna.save_checkpoint(
            dyna.save_checkpoint(
                ckpt_num=n_iter,
                save_dir=checkpoint_dir,  # session.get_trial_dir(),
            )
            # evaluate the model and the xi matrix
            if config["real_env"]["class"] == "BurgersControlEnv":
                eval_burgers_policy(
                    dyna=dyna,
                    config=config,
                    global_test_seed=global_test_seed,
                    n_iter=n_iter,
                    baseline=baseline,
                    n_steps_extrapolation=n_steps_extrapolation,
                )
            elif config["real_env"]["class"] == "NavierStokesControlEnv":
                eval_navierStokes_policy(
                    dyna=dyna,
                    config=config,
                    global_test_seed=global_test_seed,
                    n_iter=n_iter,
                    baseline=baseline,
                )
            else:
                print("No evluation method for this environment is defined.")

            """
            from utils import plot_state_traj

            policy = RLlibPolicyWrapper(dyna.drl_algo)
            _, _, _ = policy.run(dyna.real_env, seed=global_test_seed)
            surrogate_env = dyna.drl_config["environment"]["env"](
                dyna.drl_config["environment"]["env_config"]
            )

            global_plot_path = config["plot_dir"]
            model_checkpoint_number = "_{:06d}".format(n_iter)
            model_prefix = "ppo_baseline_" if baseline else "ppo_surrogate"
            if not dyna.use_real_env:
                surrogate_env.max_episode_steps = dyna.real_env.n_steps
                _, _ = policy.run_surrogate(
                    surrogate_env,
                    seed=global_test_seed,
                    n_steps=dyna.real_env.n_steps,
                )
                plot_state_traj(
                    surrogate_env,
                    5,
                    global_plot_path,
                    model_prefix + model_checkpoint_number,
                    surrogate_model=True,
                    n_steps_extrapolation=n_steps_extrapolation,
                )
            plot_state_traj(
                dyna.real_env,
                5,
                global_plot_path,
                model_prefix + model_checkpoint_number,
                n_steps_extrapolation=n_steps_extrapolation,
            )
            """
            # checkpoint = Checkpoint.from_dict(check_dict)

        # compile metrics for tune to report
        train_results["traj_buffer"] = dyna.get_buffer_metrics()
        print("Information about the trajectory buffer")
        print(train_results["traj_buffer"])
        train_results["dyn_collect"] = collect_dict


def explore(dyna_config):
    """
    Used for population based training (PBT).
    Ensures explored (continuous) parameters stay in a given range
    """
    config = dyna_config["drl"]["config"]["training"]

    config["lambda_"] = np.clip(config["lambda_"], 0, 1)
    config["gamma"] = np.clip(config["gamma"], 0, 1)

    dyna_config["drl"]["config"]["training"] = config
    return dyna_config


if __name__ == "__main__":
    import logging
    from sindy_rl.policy import RandomPolicy

    # Setup logger
    logging.getLogger().setLevel(logging.INFO)
    logging.basicConfig()
    logger = logging.getLogger("dyna-sindy")
    logger.setLevel(logging.INFO)

    _parent_dir = Path(__file__).parent

    # decide which case to run
    baseline, filename, _, _, _ = parse_cli_args()

    dyna_config = read_dyna_config(_parent_dir, filename)
    setup_folders(dyna_config)

    # Initialize default off-policy for initial collection
    n_control = dyna_config["drl"]["config"]["environment"]["env_config"]["act_dim"]
    print("creating the random policy")
    dyna_config["off_policy_pi"] = RandomPolicy(
        low=-1 * np.ones(n_control), high=np.ones(n_control), seed=0
    )

    # Run without the ray setup?
    dyna_sindy(config=dyna_config, baseline=baseline)
