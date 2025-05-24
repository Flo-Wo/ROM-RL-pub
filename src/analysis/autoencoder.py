import re
import matplotlib.pyplot as plt
import torch


def read_AE_log_file(log_file_path, fully_observable: bool = True):
    ae_train_time_pattern = re.compile(r"AE_train_time_s: ([\d\.]+)")
    ae_train_loss_pattern = re.compile(r"AE_train_loss: ([\d\.]+)")
    ae_val_loss_pattern = re.compile(r"AE_val_loss: ([\d\.]+)")

    if fully_observable:
        n_elements = 64 * 256
    else:
        n_elements = 64 * 48

    ae_train_times = []
    ae_train_losses = []
    ae_val_losses = []

    with open(log_file_path, "r") as log_file:
        for line in log_file:
            train_time_match = ae_train_time_pattern.search(line)
            train_loss_match = ae_train_loss_pattern.search(line)
            val_loss_match = ae_val_loss_pattern.search(line)

            if train_time_match:
                ae_train_times.append(float(train_time_match.group(1)))

            if train_loss_match:
                ae_train_losses.append(float(train_loss_match.group(1)) / n_elements)

            if val_loss_match:
                ae_val_losses.append(float(val_loss_match.group(1)) / n_elements)

    # Remove duplicates by converting lists to sets and back to lists
    ae_train_times = list(set(ae_train_times))
    ae_train_losses = list(set(ae_train_losses))
    ae_val_losses = list(set(ae_val_losses))

    return ae_train_times, ae_train_losses, ae_val_losses


def plot_AE_metrics(
    ae_train_losses,
    ae_val_losses,
    ae_train_times,
    path: str,
    prefix: str,
    title: str = "",
    save: bool = True,
):
    # plot the internal errors first
    # Create figure and axes
    fig, ax = plt.subplots(figsize=(6, 6))

    # Boxplot for train and evaluation loss
    ax.boxplot(
        [ae_train_losses, ae_val_losses],
        labels=["Training", "Validation"],
    )
    ax.set_title(title)
    ax.set_ylabel("Average Loss")
    ax.set_yscale("log")
    # ax.set_xlabel("Metrics")
    # Adjust layout
    plt.tight_layout()
    if save:
        plt.savefig(path + prefix + "_AE_internal_errors.pdf")
    plt.close(fig)

    # Boxplot for training time
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.boxplot(ae_train_times, labels=["Training"])
    ax.set_title(title)
    ax.set_ylabel("Time [s]")
    # ax.set_xlabel("Metrics")

    # Adjust layout
    plt.tight_layout()
    if save:
        plt.savefig(path + prefix + "_AE_training_time.pdf")
    plt.close(fig)


def plot_xi_matrix(
    weight_matrix: torch.tensor,
    save: bool = True,
    prefix: str = "",
    path: str = "",
    title: str = "",
    sparsity_threshold: float = 0.1,
):
    max_abs_value = torch.max(torch.abs(weight_matrix))
    min_abs_value = torch.min(torch.abs(weight_matrix))
    num_entries_gt_01 = torch.sum(torch.abs(weight_matrix) > sparsity_threshold)
    total_num_entries = torch.numel(weight_matrix)
    fig = plt.figure(figsize=(8, 6))
    plt.imshow(weight_matrix, cmap="RdBu_r", aspect="auto")
    num_cols = weight_matrix.shape[1]
    ticks_xi = list(range(0, num_cols))
    labels_xi = [str(i) for i in ticks_xi]
    plt.xticks(ticks=ticks_xi, labels=labels_xi)

    y_ticks_xi = range(0, weight_matrix.shape[0], 3)
    y_labels_xi = range(1, weight_matrix.shape[0] + 1, 3)

    plt.yticks(ticks=y_ticks_xi, labels=y_labels_xi)

    plt.colorbar()
    plt.title(
        "SINDy $\Xi$ matrix {}:\n".format(title)
        + "$\max |\Xi_{{i,j}}|$ = {:.2f}, $\min |\Xi_{{i,j}}|$ = {:.2f}".format(
            max_abs_value, min_abs_value
        )
        + "\n$\#\{{ |\Xi_{{i,j}}| > {:.2f} \}}$ = {}/{} = {:.2f} \%".format(
            sparsity_threshold,
            num_entries_gt_01,
            total_num_entries,
            num_entries_gt_01 / total_num_entries * 100,
        )
    )
    plt.xlabel("Surrogate Space Dimension")
    plt.ylabel("SINDy Dictionary Functions")
    plt.tight_layout()
    if save:
        plt.savefig(path + prefix + "_xi_weight_matrix.pdf")
    plt.close(fig)
    return total_num_entries


def _replace_variables(feature):
    # the last two variables are the controls and we want 1-based indices
    # for the latex formulas
    feature = feature.replace("x0", "x_1(t)")
    feature = feature.replace("x1", "x_2(t)")
    feature = feature.replace("x2", "x_3(t)")
    feature = feature.replace("x3", "x_4(t)")
    feature = feature.replace("x4", "x_5(t)")
    feature = feature.replace("x5", "x_6(t)")
    # controls
    feature = feature.replace("x6", "u_1(t)")
    feature = feature.replace("x7", "u_2(t)")
    return feature
