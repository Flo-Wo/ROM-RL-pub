from utils import plot_state_traj
from sindy_rl.policy import RLlibPolicyWrapper
import numpy as np
import matplotlib.pyplot as plt
from sindy_rl.navierStokesData import get_desired_states

lightRed = (186 / 255, 49 / 255, 51 / 255)
darkBlue = (32 / 255, 72 / 255, 127 / 255)

plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        # "font.serif": ["Palatino"],
        "font.size": 14,
    }
)


def eval_burgers_policy(
    dyna,
    config: dict,
    global_test_seed: int,
    n_iter: int,
    baseline: bool,
    n_steps_extrapolation: int,
):
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


def eval_navierStokes_policy(
    dyna,
    config: dict,
    global_test_seed: int,
    n_iter: int,
    baseline: bool,
):
    policy = RLlibPolicyWrapper(dyna.drl_algo)
    rewards_rollout, action_traj = policy.run_navierStokes(
        dyna.real_env, seed=global_test_seed
    )
    print(np.mean(rewards_rollout))

    # path to save the plots
    global_plot_path = config["plot_dir"]
    model_checkpoint_number = "_{:06d}".format(n_iter)
    model_prefix = "ppo_baseline" if baseline else "ppo_surrogate"

    plot_navStok_controls(
        action_traj,
        dyna.real_env,
        global_plot_path=global_plot_path,
        model_name=model_prefix + model_checkpoint_number,
        surrogate_model=(not baseline),
        save=True,
    )

    plot_navStok_vector_field(
        dyna.real_env,
        time_idx=0,
        global_plot_path=global_plot_path,
        model_name=model_prefix + model_checkpoint_number,
        surrogate_model=(not baseline),
        file_prefix="INIT",
        save=True,
    )
    plot_navStok_vector_field(
        dyna.real_env,
        time_idx=199,
        global_plot_path=global_plot_path,
        model_name=model_prefix + model_checkpoint_number,
        surrogate_model=(not baseline),
        file_prefix="END_HORIZON",
        save=True,
    )
    return action_traj


def plot_navStok_controls(
    action_traj: np.ndarray,
    env,
    global_plot_path: str,
    model_name: str,
    surrogate_model: bool,
    save: bool = True,
):
    fig, ax = plt.subplots()
    time_max = 0.2 - env.dt
    print(time_max)
    x_axis = np.linspace(0, time_max, env.nt)
    plt.hlines(
        y=2,
        xmin=0,
        xmax=time_max,
        color=lightRed,
        linestyle="--",
        label=r"$u_\mathrm{ref}$",
    )
    plt.xlabel(r"Time $t$")
    plt.ylabel(r"Control $u_t$")
    plt.plot(x_axis[:-1], action_traj[:-1], label="controls", color=darkBlue)
    plt.title("Trajectory of the controls")
    plt.legend(loc="best")
    if save:
        plt.savefig(
            global_plot_path
            + "/controls{}_{}.pdf".format(
                "_surrogate" if surrogate_model else "", model_name
            )
        )
    plt.close(fig)


def plot_navStok_vector_field(
    env,
    time_idx: int,
    global_plot_path: str,
    model_name: str,
    surrogate_model: bool,
    file_prefix: str,
    save: bool = True,
):
    # laod the target
    desired_state = get_desired_states()

    X = env.X
    Y = env.Y
    U = env.U[time_idx, :, :, 0]
    V = env.U[time_idx, :, :, 1]

    desired_U = desired_state[time_idx, :, :, 0]
    desired_V = desired_state[time_idx, :, :, 1]

    magnitude_min = min(
        np.min(np.sqrt(U**2 + V**2)), np.min(np.sqrt(desired_U**2 + desired_V**2))
    )
    magnitude_max = max(
        np.max(np.sqrt(U**2 + V**2)), np.max(np.sqrt(desired_U**2 + desired_V**2))
    )
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

    c_desired = plot_vector_field(
        X,
        Y,
        desired_U,
        desired_V,
        ax[0],
        "Desired",
        magnitude_min=magnitude_min,
        magnitude_max=magnitude_max,
        colorbar=False,
    )
    c_est = plot_vector_field(
        X,
        Y,
        U,
        V,
        ax[1],
        "Estimated",
        magnitude_min=magnitude_min,
        magnitude_max=magnitude_max,
        colorbar=False,
    )
    cbar = fig.colorbar(c_est, ax=ax[1], shrink=0.95)
    plt.suptitle("Velocity fields at t = {}".format(env.dt * (time_idx + 1)))
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if save:
        plt.savefig(
            global_plot_path
            + "/flow_field_{}{}_{}.pdf".format(
                file_prefix, "_surrogate" if surrogate_model else "", model_name
            )
        )
    plt.close(fig)


def plot_vector_field(
    X,
    Y,
    U,
    V,
    ax,
    title,
    magnitude_min: float,
    magnitude_max: float,
    colorbar: bool = True,
):
    magnitude = np.sqrt(U**2 + V**2)
    c = ax.imshow(
        magnitude,
        extent=(X.min(), X.max(), Y.min(), Y.max()),
        origin="lower",
        cmap="RdBu_r",
        aspect="auto",
        alpha=0.5,
        vmin=magnitude_min,
        vmax=magnitude_max,
    )
    q = ax.quiver(X, Y, U, V, color="black")
    if colorbar:
        plt.colorbar(c, ax=ax)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    return c
