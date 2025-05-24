import logging

from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from ray.tune.logger import Logger, pretty_print
import os
from sindy_rl.file_helpers import get_subfolder_path


class TensorBoardLogger(Logger):
    def __init__(self, config: dict, logdir: str, dummy: bool = False):
        """Logger interacting with the Tensorboard. Since RayRL does not
        accept no logger, we provide a dummy-flag which creates an empty logger.

        Parameters
        ----------
        config : dict
            Global config, based on the config.yml file.
        logdir : str
            Directory for the logs
        dummy : bool, optional
            Dummy flag, to create an empty logger, by default False.
        """
        super().__init__(config, logdir)
        self.dummy = dummy
        if self.dummy:
            return
        now = "{date:%Y-%m-%d_%H-%M-%S}.log".format(date=datetime.now())
        self._log_file = os.path.join(logdir, "log_" + now)
        self._logger = logging.getLogger(__name__)
        self._logger.setLevel(logging.INFO)

        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        file_handler = logging.FileHandler(self._log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)

        self._logger.addHandler(file_handler)
        self.writer = SummaryWriter(log_dir=logdir + "/tbf" + now)

    def on_result(self, result, step=None):
        if self.dummy:
            return
        self.last_result = result
        self._logger.info("\n\nEpisode {}".format(result["training_iteration"]))
        self._logger.info(pretty_print(result))
        of_interest = [
            "episode_reward_mean",
            "episode_reward_max",
            "episode_reward_min",
            "time_this_iter_s",
        ]
        iter_num = result["training_iteration"]
        # self._logger.info("\n\n")
        for key in of_interest:
            self.writer.add_scalar(key, result[key], global_step=iter_num)
            # self._logger.info("Episode {}, {} : {}".format(iter_num, key, result[key]))
        # self._logger.info("\n\n")

    def on_end(self):
        self.writer.close()


def get_logger(global_config):
    def get_tensorboard_logger(drl_algo_config):
        """Create a logger for Ray."""
        # Define the directory for logging
        # log_dir = get_subfolder_path(global_config=global_config, key="log_dir")
        log_dir = global_config["log_dir"]
        dummy = global_config.get("dummy_logger", False)
        if log_dir is None:
            log_dir = "./no_dir_given"

        # Configure logging
        logging.basicConfig(level=logging.INFO)

        # Create and return the UnifiedLogger
        return TensorBoardLogger(global_config, log_dir, dummy=dummy)

    return get_tensorboard_logger
