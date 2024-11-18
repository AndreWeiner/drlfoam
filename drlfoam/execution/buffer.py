"""
implements a base class for storing the buffer
"""
import logging
import torch as pt

from copy import deepcopy
from shutil import copytree
from subprocess import Popen
from typing import Tuple, List
from os.path import join, exists
from abc import ABC, abstractmethod

from ..agent import FCPolicy
from .manager import TaskManager
from ..environment import Environment

logger = logging.getLogger(__name__)


class Buffer(ABC):
    def __init__(
        self,
        path: str,
        base_env: Environment,
        buffer_size: int,
        n_runners_max: int,
        keep_trajectories: bool,
        timeout: int,
    ):
        self._path = path
        self._base_env = base_env
        self._buffer_size = buffer_size
        self._n_runners_max = n_runners_max
        self._keep_trajectories = keep_trajectories
        self._timeout = timeout
        self._manager = TaskManager(self._n_runners_max)
        self._envs = None
        self._n_fills = 0

    @abstractmethod
    def prepare(self):
        pass

    @abstractmethod
    def fill(self):
        pass

    def create_copies(self):
        envs = []
        for i in range(self._buffer_size):
            dest = join(self._path, f"copy_{i}")
            if not exists(dest):
                copytree(self._base_env.path, dest, dirs_exist_ok=True)
            envs.append(deepcopy(self._base_env))
            envs[-1].path = dest
            envs[-1].end_time = envs[-1].end_time
            envs[-1].start_time = envs[-1].start_time
            envs[-1].seed = i
        self._envs = envs

    def update_policy(self, policy: FCPolicy):
        for env in self.envs:
            policy.save(join(env.path, env.policy))
        policy.save(join(self.base_env.path, self.base_env.policy))

    def reset(self):
        for env in self.envs:
            env.reset()

    def clean(self):
        for env in self.envs:
            proc = Popen([f"./{env.clean_script}"], cwd=env.path)
            proc.wait()

    def save_trajectories(self):
        pt.save(
            [env.observations for env in self.envs],
            join(self._path, f"observations_{self._n_fills}.pt")
        )

    @property
    def base_env(self) -> Environment:
        return self._base_env

    @property
    def envs(self):
        if self._envs is None:
            self.create_copies()
        return self._envs

    @property
    def observations(self) -> Tuple[List[pt.Tensor], List[pt.Tensor], List[pt.Tensor]]:
        states, actions, rewards = [], [], []
        for env in self.envs:
            obs = env.observations
            if all([key in obs for key in ("states", "actions", "rewards")]):
                states.append(obs["states"])
                actions.append(obs["actions"])
                rewards.append(obs["rewards"])
            else:
                logger.warning(f"Warning: environment {env.path} returned empty observations")
        return states, actions, rewards
