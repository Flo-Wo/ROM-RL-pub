from pathlib import Path
from sindy_rl.file_helpers import subfolder_to_full_path
from sindy_rl.policy import RandomPolicy
from sindy_rl.utils.parse_cli_args import parse_cli_args
from analysis.autoencoder import (
    _replace_variables,
    plot_AE_metrics,
    plot_xi_matrix,
    read_AE_log_file,
)
from sindyrl_LOAD import _load_dyna_policy
import matplotlib.pyplot as plt
import torch
import logging
import numpy as np

from utils import get_global_paths, load_config

# enable latex formatting
plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        # "font.serif": ["Palatino"],
        "font.size": 14,
    }
)

# main method to evaluate the AutoEncoder and the corresponding internal SINDy dynamics
# we analyze:
#   - xi matrix (plot + sparsity)
#   - number of parameters of the AutoEncoder
#   - Average error for the internal training data
# example call: NOTE: adjust the header below manually
# python sindyrl_AE_EVAL.py --baseline --checkpoint <checkpoint_number> --filename "AE/BATCH_burgers_PO_2D_surrogate"


sparsity_threshold = 0.15


def evaluate_AE(
    config,
    model_checkpoint_number: str,
    title: str = "",
    prefix: str = "",
    path: str = "",
    fully_observable: bool = True,
):
    dyna, policy, surrogate_env = _load_dyna_policy(
        config,
        model_checkpoint_number,
        global_checkpoint_path=global_checkpoint_path,
        only_real_env=False,
    )
    poly_features = dyna.dynamics_model.feature_library.get_feature_names()
    print(poly_features)
    xi_matrix = dyna.dynamics_model.model.xi.weight.data
    # plot and analyze the xi matrix
    num_entries_xi = plot_xi_matrix(
        xi_matrix,
        save=True,
        path=path,
        prefix=prefix,
        title=title,
        sparsity_threshold=sparsity_threshold,
    )

    # compute the total number of parameters of the dynamics model
    num_feat = dyna.dynamics_model.model.count_parameters()
    print("Total number of features in the AE model, including Xi")
    print(num_feat)
    print("Params for the xi matrix")
    print(num_entries_xi)

    # interpret the dynamics
    latent_state_dim = xi_matrix.shape[1]
    dict_dim = xi_matrix.shape[0]
    print("\n\n")
    print("Interpretation of the dynamics")
    for idx_latent in range(latent_state_dim):
        out = "x{}(t+1) =".format(idx_latent)
        for feat_idx in range(dict_dim):
            if torch.abs(xi_matrix[feat_idx, idx_latent]) > sparsity_threshold:
                out += " {:1.3f}*{} +".format(
                    xi_matrix[feat_idx, idx_latent], poly_features[feat_idx]
                )
        print(out[:-1])
    print("\n\n")
    latex_output = []
    for idx_latent in range(latent_state_dim):
        out = "$x_{}(t+1) =".format(idx_latent + 1)
        for feat_idx in range(dict_dim):
            coefficient = xi_matrix[feat_idx, idx_latent].item()
            if abs(coefficient) > sparsity_threshold:
                feature = _replace_variables(poly_features[feat_idx])
                if coefficient < 0:
                    out += " - {:1.3f} \\cdot {}".format(abs(coefficient), feature)
                else:
                    out += " + {:1.3f} \\cdot {}".format(coefficient, feature)
        out += "$"
        latex_output.append(out)
    for equation in latex_output:
        print(equation)

    # evalute the train and validation loss over all the epochs, as well as the training time for the encoder
    train_times, train_losses, eval_losses = read_AE_log_file(
        global_log_path + "/dynamics_AE.log", fully_observable=fully_observable
    )

    plot_AE_metrics(
        train_losses,
        eval_losses,
        train_times,
        path=path,
        prefix=prefix,
        title=title[1:-1],
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
        low=-1 * np.ones(n_control), high=np.ones(n_control), seed=1234
    )

    evaluate_AE(
        config=dyna_config,
        model_checkpoint_number=checkpoint,
        # path="../data/paper_evaluation/burgers/dynamics/",
        path="../data/paper_evaluation/navier_stokes/dynamics/",
        prefix="AE_5_small",
        title=r"(AE+SINDy-C, $k_\mathrm{dyn} = 5$)",
        fully_observable=True,
        # prefix="AE_FO",
        # title="(Autoencoder, fully observable)",
        # fully_observable=True,
    )
