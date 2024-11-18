"""Tools for running jobs with the SLURM workload manager.

See: https://slurm.schedmd.com/documentation.html
"""
import logging

from typing import List, Union
from os.path import join
from subprocess import Popen, PIPE
from time import sleep

from .buffer import Buffer
from ..environment import Environment

logger = logging.getLogger(__name__)

DEFAULT_SHELL = "#!/bin/bash -l"
SLURM_PREFIX = "#SBATCH"
SLURM_JOB_NAME = "--job-name"
SLURM_NTASKS = "--ntasks"
SLURM_NODES = "--nodes"
SLURM_ERROR = "--error"
SLURM_OUTPUT = "--output"
SLURM_PARTITION = "--partition"
SLURM_CONSTRAINT = "--constraint"
SLURM_MAIL_TYPE = "--mail-type"
SLURM_MAIL_USER = "--mail-user"
SLURM_TIME = "--time"
SLURM_NTASKS_PER_NODE = "--ntasks-per-node"
SLURM_MEM_PER_CPU = "--mem-per-cpu"


def submit_job(jobscript: Union[str, list]) -> int:
    if type(jobscript) is str:
        proc = Popen(["sbatch", jobscript], stdout=PIPE)
    else:
        proc = Popen(["sbatch"] + jobscript, stdout=PIPE)
    response = str(proc.stdout.read(), "utf-8")
    return int(response.split()[-1])


def get_job_status(job_id: int) -> str:
    proc = Popen(["squeue", "-j", f"{job_id}"], stdout=PIPE)
    response = str(proc.stdout.read(), "utf-8").split()
    response = [s for s in response if s]
    return response[12]


def submit_and_wait(jobscript: str, wait: int = 5, timeout: Union[int, float] = 1e15) -> None:
    job_id = submit_job(jobscript)
    running, time_passed = True, 0
    while running:
        try:
            status = get_job_status(job_id)
            if status in ["PD", "R", "CF"]:
                sleep(wait)
                time_passed += wait
                if time_passed > timeout:
                    Popen(["scancel", f"{job_id}"]).wait()
                    raise Exception(
                        f"Slurm job {job_id} exceeded time limited of {timeout}s and got canceled")
            else:
                running = False
        except Exception:
            running = False


class SlurmConfig(object):
    def __init__(
        self,
        commands_pre: List[str] = None,
        commands: List[str] = None,
        modules: List[str] = None,
        job_name: str = None,
        n_tasks: int = None,
        n_nodes: int = None,
        std_out: str = None,
        err_out: str = None,
        partition: str = None,
        constraint: str = None,
        mail_type: str = None,
        mail_user: str = None,
        time: str = None,
        n_tasks_per_node: int = None,
        mem_per_cpu: int = None,
    ):

        if commands_pre is None:
            commands_pre = []
        if commands is None:
            commands = []
        if modules is None:
            modules = []
        self._commands_pre = commands_pre
        self._commands = commands
        self._modules = modules
        self._options = {
            SLURM_JOB_NAME: job_name,
            SLURM_NTASKS: n_tasks,
            SLURM_NODES: n_nodes,
            SLURM_OUTPUT: std_out,
            SLURM_ERROR: err_out,
            SLURM_PARTITION: partition,
            SLURM_CONSTRAINT: constraint,
            SLURM_MAIL_TYPE: mail_type,
            SLURM_MAIL_USER: mail_user,
            SLURM_TIME: time,
            SLURM_NTASKS_PER_NODE: n_tasks_per_node,
            SLURM_MEM_PER_CPU: mem_per_cpu,
        }

    def write(self, path: str) -> None:
        entries = [DEFAULT_SHELL, ""]
        for key, val in self._options.items():
            if val is not None:
                entries.append(f"{SLURM_PREFIX} {key}={val}")

        if len(self._modules) > 0:
            entries.append("")
            for m in self._modules:
                entries.append(f"module load {m}")

        all_commands = self._commands_pre + self._commands
        if len(all_commands) > 0:
            entries.append("")
            entries += all_commands
        else:
            logger.warning(f"Warning: no commands specified in jobscript {path}")

        with open(path, "w+") as jobscript:
            jobscript.write("\n".join(entries))

    @property
    def commands_pre(self) -> List[str]:
        return self._commands_pre

    @commands_pre.setter
    def commands_pre(self, value: List[str]):
        self._commands_pre = value

    @property
    def commands(self) -> List[str]:
        return self._commands

    @commands.setter
    def commands(self, value: List[str]):
        self._commands = value

    @property
    def modules(self) -> List[str]:
        return self._modules

    @modules.setter
    def modules(self, value: List[str]):
        self._modules = value

    @property
    def job_name(self) -> str:
        return self._options[SLURM_JOB_NAME]

    @job_name.setter
    def job_name(self, value: str):
        self._options[SLURM_JOB_NAME] = value

    @property
    def n_tasks(self) -> int:
        return self._options[SLURM_NTASKS]

    @n_tasks.setter
    def n_tasks(self, value: int) -> None:
        self._options[SLURM_NTASKS] = value

    @property
    def n_nodes(self) -> int:
        return self._options[SLURM_NODES]

    @n_nodes.setter
    def n_nodes(self, value: int):
        self._options[SLURM_NODES] = value

    @property
    def std_out(self) -> str:
        return self._options[SLURM_OUTPUT]

    @std_out.setter
    def std_out(self, value: str) -> None:
        self._options[SLURM_OUTPUT] = value

    @property
    def err_out(self) -> str:
        return self._options[SLURM_ERROR]

    @err_out.setter
    def err_out(self, value: str) -> None:
        self._options[SLURM_ERROR] = value

    @property
    def partition(self) -> str:
        return self._options[SLURM_PARTITION]

    @partition.setter
    def partition(self, value: str) -> None:
        self._options[SLURM_PARTITION] = value

    @property
    def constraint(self) -> str:
        return self._options[SLURM_CONSTRAINT]

    @constraint.setter
    def constraint(self, value: str) -> None:
        self._options[SLURM_CONSTRAINT] = value

    @property
    def mail_type(self) -> str:
        return self._options[SLURM_MAIL_TYPE]

    @mail_type.setter
    def mail_type(self, value: str) -> None:
        self._options[SLURM_MAIL_TYPE] = value

    @property
    def mail_user(self) -> str:
        return self._options[SLURM_MAIL_USER]

    @mail_user.setter
    def mail_user(self, value: str) -> None:
        self._options[SLURM_MAIL_USER] = value

    @property
    def time(self) -> str:
        return self._options[SLURM_TIME]

    @time.setter
    def time(self, value: str) -> None:
        self._options[SLURM_TIME] = value

    @property
    def n_tasks_per_node(self) -> int:
        return self._options[SLURM_NTASKS_PER_NODE]

    @n_tasks_per_node.setter
    def n_tasks_per_node(self, value: int) -> None:
        self._options[SLURM_NTASKS_PER_NODE] = value

    @property
    def mem_per_cpu(self) -> int:
        return self._options[SLURM_MEM_PER_CPU]

    @mem_per_cpu.setter
    def mem_per_cpu(self, value: int) -> None:
        self._options[SLURM_MEM_PER_CPU] = value

    @property
    def options(self):
        return self._options


class SlurmBuffer(Buffer):
    def __init__(
        self,
        path: str,
        base_env: Environment,
        buffer_size: int,
        n_runners_max: int,
        slurm_config: SlurmConfig,
        keep_trajectories: bool = True,
        timeout: Union[int, float] = 1e15,
        wait: int = 5
    ):
        super(SlurmBuffer, self).__init__(
            path, base_env, buffer_size, n_runners_max, keep_trajectories, timeout
        )
        self._config = slurm_config
        self._wait = wait

    def prepare(self) -> None:
        self._config.commands = [
            f"cd {self._base_env.path}",
            f"./{self._base_env.initializer_script}",
        ]
        self._config.job_name = "prepare"
        self._config.err_out = join(self._base_env.path, "prepare.err")
        self._config.err_out = join(self._base_env.path, "prepare.out")
        jobscript = join(self._base_env.path, "jobscript.sh")
        self._config.write(jobscript)
        self._manager.add(submit_and_wait, jobscript,
                          wait=self._wait, timeout=self._timeout)
        self._manager.run()
        self._base_env.initialized = True

    def fill(self) -> None:
        for i, env in enumerate(self.envs):
            self._config.commands = [f"cd {env.path}", f"./{env.run_script}"]
            self._config.job_name = f"copy_{i}"
            self._config.err_out = join(env.path, f"copy_{i}.err")
            self._config.err_out = join(env.path, f"copy_{i}.out")
            jobscript = join(env.path, "jobscript.sh")
            self._config.write(jobscript)
            self._manager.add(submit_and_wait, jobscript,
                              wait=self._wait, timeout=self._timeout)
        self._manager.run()
        if self._keep_trajectories:
            self.save_trajectories()
        self._n_fills += 1
