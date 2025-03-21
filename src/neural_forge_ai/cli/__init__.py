from abc import ABC, abstractmethod
from argparse import ArgumentParser


class Baseneural_forge_aiCommand(ABC):
    @staticmethod
    @abstractmethod
    def register_subcommand(parser: ArgumentParser):
        raise NotImplementedError()

    @abstractmethod
    def run(self):
        raise NotImplementedError()
