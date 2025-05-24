import numpy as np
import matplotlib.pyplot as plt
import torch


def xi_as_heatmap(train_info: dict, config: dict, it_num: int, save: bool = True):
    """
    Print the Xi matrix, i.e. selection of the dynamics component, as a heatmap to analyze
    the magnitude of the entries, their distribution structure and sparsity.
    Args:
        train_info: dict
            Dict containing the information about the internal training process of the
            dynamics model.
        config: dict
            Global configuration file, used to load paths.
        it_num: int
            Number of training iteration, i.e. SINDy epoch.
        save: bool
            Flag to decide if the plot should be saved, by default True.
    """
    if not "xi_matrix" in train_info:
        return
    weight_matrix = train_info["xi_matrix"]
    max_abs_value = torch.max(torch.abs(weight_matrix))
    min_abs_value = torch.min(torch.abs(weight_matrix))
    num_entries_gt_01 = torch.sum(torch.abs(weight_matrix) > 0.1)
    total_num_entries = torch.numel(weight_matrix)
    fig = plt.figure(figsize=(8, 6))
    plt.imshow(weight_matrix, cmap="viridis", aspect="auto")
    plt.colorbar()
    plt.title(
        "Weight Matrix Heatmap\nMax Abs Val: {:.2f}, Min Abs Val: {:.2f},\n#Entries > 0.1: {}/{} = {}%".format(
            max_abs_value,
            min_abs_value,
            num_entries_gt_01,
            total_num_entries,
            num_entries_gt_01 / total_num_entries * 100,
        )
    )
    plt.xlabel("Input Dimension")
    plt.ylabel("Output Dimension")
    if save:
        path = config["plot_dir"] + "/xi_weight_matrix_{}.pdf".format(it_num)
        plt.savefig(path)
    plt.close(fig)
