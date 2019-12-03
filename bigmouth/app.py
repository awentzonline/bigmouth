import argparse
import logging


class App:
    """Base class for runnable program."""
    def __init__(self, args):
        self.args = args
        self._setup_logger()

    def _setup_logger(self):
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)

    def run(self):
        raise NotImplementedError

    @classmethod
    def parse_args(cls):
        """
        Parse arguments

        Returns
        ---------
        argparse.Namespace
            Argparse namespace object with parsed arguments
        """
        parser = argparse.ArgumentParser()
        cls.add_arguments_to_parser(parser)
        args = parser.parse_args()
        return args

    @classmethod
    def add_arguments_to_parser(cls, parser):
        """
        Override in subclass to add specific experiment arguments to the parser

        Parameters
        ----------
        parser : argparse.ArgumentParser
            ArgumentParser object
        """
        pass

    @classmethod
    def create_from_args(cls):
        """
        Convienient method to instantiate the experiment class from arguments

        Returns
        ----------
        class
            Experiment class with args passed
        """
        args = cls.parse_args()
        return cls(args)
