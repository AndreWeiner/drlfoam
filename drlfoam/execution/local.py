"""
implements a local buffer
"""
from typing import Union
from subprocess import Popen

from .buffer import Buffer
from ..environment import Environment


def submit_and_wait(cmd: str, cwd: str, timeout: int = 1e15):
    proc = Popen([cmd], cwd=cwd)
    proc.wait(timeout)


class LocalBuffer(Buffer):
    def __init__(
        self,
        path: str,
        base_env: Environment,
        buffer_size: int,
        n_runners_max: int,
        keep_trajectories: bool = True,
        timeout: Union[int, float] = 1e15,
    ):
        super(LocalBuffer, self).__init__(
            path, base_env, buffer_size, n_runners_max, keep_trajectories, timeout
        )

    def prepare(self) -> None:
        cmd = f"./{self._base_env.initializer_script}"
        cwd = self._base_env.path
        self._manager.add(submit_and_wait, cmd, cwd, self._timeout)
        self._manager.run()
        self._base_env.initialized = True

    def fill(self) -> None:
        for env in self.envs:
            self._manager.add(submit_and_wait, f"./{env.run_script}", env.path, self._timeout)
        self._manager.run()
        if self._keep_trajectories:
            self.save_trajectories()
        self._n_fills += 1