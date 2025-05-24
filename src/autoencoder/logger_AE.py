import torch
import logging
from datetime import datetime


class LoggerAE:
    def __init__(self, path_to_logs: str, dummy_logger: bool = False):
        """Logger class for the AutoEncoder training.

        Parameters
        ----------
        path_to_logs : str
            Path where the logs should be saved to.
        dummy_logger : bool, optional
            Trick to create a logger but not log to file, by default False.
            Needed because of internal RlLIb configs.
        """
        self.log_file = path_to_logs

        print("Dummy Logger AE: ", dummy_logger)
        print(path_to_logs)

        self._logger = logging.getLogger(__name__)
        self._logger.setLevel(logging.INFO)

        if not dummy_logger:
            call_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            filename = path_to_logs + "/AE_training_logs_{}".format(call_time)
            print(filename)
            formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            file_handler = logging.FileHandler(path_to_logs + "/dynamics_AE.log")
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(formatter)

            self._logger.addHandler(file_handler)

    def log(self, train_info: dict):
        """Log the current AE training progress.

        Parameters
        ----------
        train_info : dict
            See SindyAutoEncoderDynamics class for more details.
        """
        # self._logger.info("Epoch {}".format(self.epoch_counter))
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._logger.info("Timestamp: {}".format(current_time))
        for key, value in train_info.items():
            self._logger.info("{}: {}".format(key, value))

    def info(self, msg: str):
        self._logger.info(msg)
