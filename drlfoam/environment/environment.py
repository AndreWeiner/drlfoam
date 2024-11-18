"""Base class for all environments.

The base class provides a common interface for all derived environments
and implements shared functionality. New environments should be derived
from this class.
"""
from abc import ABC, abstractmethod
from os.path import join
from typing import Union, Tuple
from torch import Tensor

from ..utils import check_path, check_file, check_pos_int


class Environment(ABC):
    def __init__(self, path: str, initializer_script: str, run_script: str,
                 clean_script: str, mpi_ranks: int, n_states: int,
                 n_actions: int):
        """
        implements a base class for environments

        :param path: path to the current test case inside the 'openfoam' directory
        :type path: str
        :param initializer_script: name of the script which should be executed for the base case
        :type initializer_script: str
        :param run_script: name of the script which should be executed for the simulations
        :type run_script: str
        :param clean_script: name of the script which should be executed for resetting the simulations
        :type clean_script: str
        :param mpi_ranks: number of MPI ranks for executing the simulation
        :type mpi_ranks: int
        :param n_states: number of states
        :type n_states: int
        :param n_actions: number of actions
        :type n_actions: int
        """
        self.path = path
        self.initializer_script = initializer_script
        self.run_script = run_script
        self.clean_script = clean_script
        self.mpi_ranks = mpi_ranks
        self.n_states = n_states
        self.n_actions = n_actions
        self._initialized = False
        self._start_time = None
        self._end_time = None
        self._control_interval = None
        self._action_bounds = None
        self._seed = None
        self._policy = None
        self._train = None
        self._observations = None

    @abstractmethod
    def reset(self):
        pass

    @property
    def path(self) -> str:
        return self._path

    @path.setter
    def path(self, value: str):
        check_path(value)
        self._path = value

    @property
    def initializer_script(self) -> str:
        return self._initializer_script

    @initializer_script.setter
    def initializer_script(self, value: str):
        check_file(join(self.path, value))
        self._initializer_script = value

    @property
    def run_script(self) -> str:
        return self._run_script

    @run_script.setter
    def run_script(self, value: str):
        check_file(join(self.path, value))
        self._run_script = value

    @property
    def clean_script(self) -> str:
        return self._clean_script

    @clean_script.setter
    def clean_script(self, value: str):
        check_file(join(self.path, value))
        self._clean_script = value

    @property
    def mpi_ranks(self) -> int:
        return self._mpi_ranks

    @mpi_ranks.setter
    def mpi_ranks(self, value: int):
        check_pos_int(value, "mpi_ranks")
        self._mpi_ranks = value

    @property
    def n_states(self) -> int:
        return self._n_states

    @n_states.setter
    def n_states(self, value: int):
        check_pos_int(value, "n_states")
        self._n_states = value

    @property
    def n_actions(self) -> int:
        return self._n_actions

    @n_actions.setter
    def n_actions(self, value: int):
        check_pos_int(value, "n_actions")
        self._n_actions = value

    @property
    def initialized(self):
        return self._initialized

    @initialized.setter
    def initialized(self, _):
        self._initialized = True

    @property
    @abstractmethod
    def start_time(self) -> float:
        pass

    @property
    @abstractmethod
    def end_time(self) -> float:
        pass

    @property
    @abstractmethod
    def control_interval(self) -> int:
        pass

    @property
    @abstractmethod
    def action_bounds(self) -> Union[Tensor, float]:
        pass

    @property
    @abstractmethod
    def seed(self) -> int:
        pass

    @property
    @abstractmethod
    def policy(self) -> str:
        pass

    @property
    @abstractmethod
    def train(self) -> bool:
        pass

    @property
    @abstractmethod
    def observations(self) -> Tuple[Tensor]:
        pass

    @start_time.setter
    def start_time(self, value):
        self._start_time = value

    @end_time.setter
    def end_time(self, value):
        self._end_time = value

    @seed.setter
    def seed(self, value):
        self._seed = value