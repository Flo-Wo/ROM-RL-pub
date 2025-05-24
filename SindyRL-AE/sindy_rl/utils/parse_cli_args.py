import argparse


def parse_cli_args():
    """
    Parse command line arguments to simplify the training and
    evaluation of the models.

    Return two flags:
        - baseline: use the baseline (NOT surrogate) the model, default is true.
        - train: train (NOT evaluate) the model, default is true.
        - checkpoint: number of the model to load, if train is false, default is 30.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-b",
        "--baseline",
        help="Use the baseline model, default is true. If false, the surrogate model is used.",
        type=bool,
        default=True,
        action=argparse.BooleanOptionalAction,
        required=False,
    )
    parser.add_argument(
        "-t",
        "--train",
        help="Train the model, default is true. If false, the model is evaluated.",
        type=bool,
        default=True,
        action=argparse.BooleanOptionalAction,
        required=False,
    )
    parser.add_argument(
        "-c",
        "--checkpoint",
        help="Checkpoint number (int) of the model to load, if train is False.",
        type=int,
        default=30,
        required=False,
    )
    parser.add_argument(
        "-f",
        "--filename",
        help="Filename to load the model from.",
        type=str,
        default="dyn_burgers.yml",
        required=False,
    )
    parser.add_argument(
        "-d",
        "--debug",
        help="If evaluation mode, run the debugging script",
        type=bool,
        default=False,
        required=False,
    )
    args = parser.parse_args()
    baseline = args.baseline
    train = args.train
    checkpoint = f"checkpoint_{args.checkpoint:06d}"
    filename = args.filename
    debug = args.debug
    return baseline, filename, train, checkpoint, debug
