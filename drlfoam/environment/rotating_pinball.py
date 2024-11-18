"""
implements the environment for the RotatingPinball2D
"""
import logging
import torch as pt

from re import sub
from os import remove
from glob import glob
from io import StringIO
from typing import Union
from shutil import rmtree
from pandas import read_csv, DataFrame
from os.path import join, isfile, isdir, sep

from .environment import Environment
from ..constants import TESTCASE_PATH, DEFAULT_TENSOR_TYPE
from ..utils import (check_pos_int, check_pos_float, replace_line_in_file,
                     get_time_folders, replace_line_latest)

pt.set_default_tensor_type(DEFAULT_TENSOR_TYPE)
logger = logging.getLogger(__name__)


def _parse_surface_field_sum(path: str) -> DataFrame:
    """Read and parse output of surfaceFieldValue sum.

    Extracts timestamp and force coefficient components in x/y/z
    from the output of the surfaceFieldValue function object.

    :param path: path to *surfaceFieldValue* output file
    :type path: str
    :return: time and force coefficients for a single cylinder
    :rtype: DataFrame
    """
    with open(path, "r") as ffile:
        data = sub("[()]", "", ffile.read())
        names = ("t", "cx", "cy", "cz")
        return read_csv(StringIO(data), header=None, names=names,
                        comment="#", delim_whitespace=True)


def _parse_forces(path: str) -> DataFrame:
    """Read and parse force coefficients for all three cylinders.

    The force coefficient in *i*-direction of cylinder *j* is labeled
    *ci_j* in the returned `DataFrame`. The corresponding time is labeled
    *t_j*. Note that the timestamp should be equal for all three cylinders.

    :param path: path to the pinball simulation (top-level)
    :type path: str
    :return: time and force coefficients for all three cylinders
    :rtype: DataFrame
    """
    data = DataFrame()
    for c in ("a", "b", "c"):
        folders = glob(
            join(path, "postProcessing", f"field_cylinder_{c}", "*")
        )
        time = sorted(folders, key=lambda x: x.split(sep)[-1])[0]
        data_i = _parse_surface_field_sum(join(time, "surfaceFieldValue.dat"))
        data[f"t_{c}"] = data_i.t
        data[f"cx_{c}"] = data_i.cx
        data[f"cy_{c}"] = data_i.cy
        data[f"cz_{c}"] = data_i.cz
    return data


def _parse_probes(path: str, n_probes: int) -> DataFrame:
    """Read and parse pressure probes.

    :param path: path to probes file
    :type path: str
    :param n_probes: number of pressure probes to parse
    :type n_probes: int
    :return: 
    :rtype: DataFrame
    """
    with open(path, "r") as pfile:
        pdata = sub("[()]", "", pfile.read())
    names = ["t"] + [f"p{i}" for i in range(n_probes)]
    return read_csv(
        StringIO(pdata), header=None, names=names, comment="#",
        sep=r"\s+"
    )


def _parse_trajectory(path: str) -> DataFrame:
    """Read and parse trajectory file.

    The trajectory file contains a timestamp, the parameters of
    the beta-distribution for every action, and the sampled actions
    (3 for the pinball).

    :param path: path to the trajectory file
    :type path: str
    :return: time, actions, and distribution parameters
    :rtype: DataFrame
    """
    names = ("t", "omega_a", "alpha_a", "beta_a", "omega_b", "alpha_b",
             "beta_b", "omega_c", "alpha_c", "beta_c")
    return read_csv(path, sep=",", header=0, names=names)


class RotatingPinball2D(Environment):
    def __init__(self, r1: Union[int, float] = 1.5, r2: Union[int, float] = 1.0, r3: Union[int, float] = 0.4):
        """
        implements the RotatingCylinder2D environment

        :param r1: offset in reward function
        :type r1: Union[int, float]
        :param r2: weighing factor for cd in reward function
        :type r2: Union[int, float]
        :param r3: weighing factor for cl in reward function
        :type r3: Union[int, float]
        """
        super(RotatingPinball2D, self).__init__(
            join(TESTCASE_PATH, "rotatingPinball2D"), "Allrun.pre",
            "Allrun", "Allclean", 8, 14, 3
        )
        self._r1 = r1
        self._r2 = r2
        self._r3 = r3
        self._initialized = False
        self._start_time = 0
        self._end_time = 200
        self._control_interval = 0.5
        self._train = True
        self._seed = 0
        self._action_bounds = 5.0
        self._policy = "policy.pt"

    def reward(self, data: dict) -> pt.Tensor:
        _cx = data[f"cx_a"] + data[f"cx_b"] + data[f"cx_c"]
        _cy = data[f"cy_a"] + data[f"cy_b"] + data[f"cy_c"]
        return self._r1 - (self._r2 * _cx + self._r3 * _cy.abs())

    @property
    def start_time(self) -> float:
        return self._start_time

    @start_time.setter
    def start_time(self, value: float) -> None:
        check_pos_float(value, "start_time", with_zero=True)
        replace_line_in_file(
            join(self.path, "system", "controlDict"),
            "timeStart",
            f"        timeStart       {value};"
        )
        self._start_time = value

    @property
    def end_time(self) -> float:
        return self._end_time

    @end_time.setter
    def end_time(self, value: float) -> None:
        check_pos_float(value, "end_time", with_zero=True)
        replace_line_in_file(
            join(self.path, "system", "controlDict"),
            "endTime ",
            f"endTime         {value};"
        )
        self._end_time = value

    @property
    def control_interval(self) -> Union[float, int]:
        return self._control_interval

    @control_interval.setter
    def control_interval(self, value: int) -> None:
        check_pos_float(value, "control_interval")
        replace_line_in_file(
            join(self.path, "system", "controlDict"),
            "executeInterval",
            f"        executeInterval {value};",
        )
        replace_line_in_file(
            join(self.path, "system", "controlDict"),
            "writeInterval",
            f"        writeInterval   {value};",
        )
        self._control_interval = value

    @property
    def action_bounds(self) -> float:
        return self._action_bounds

    @action_bounds.setter
    def action_bounds(self, value: float) -> None:
        proc = True if self.initialized else False
        new = f"        absOmegaMax     {value:2.4f};"
        replace_line_latest(self.path, "U", "absOmegaMax", new, proc)
        self._action_bounds = value

    @property
    def seed(self) -> int:
        return self._seed

    @seed.setter
    def seed(self, value: int) -> None:
        check_pos_int(value, "seed", with_zero=True)
        proc = True if self.initialized else False
        new = f"        seed     {value};"
        replace_line_latest(self.path, "U", "seed", new, proc)
        self._seed = value

    @property
    def policy(self) -> str:
        return self._policy

    @policy.setter
    def policy(self, value: str) -> None:
        proc = True if self.initialized else False
        new = f"        policy     {value};"
        replace_line_latest(self.path, "U", "policy", new, proc)
        self._policy = value

    @property
    def train(self) -> bool:
        return self._train

    @train.setter
    def train(self, value: bool) -> None:
        proc = True if self.initialized else False
        value_cpp = "true" if value else "false"
        new = f"        train           {value_cpp};"
        replace_line_latest(self.path, "U", "train", new, proc)
        self._train = value

    @property
    def observations(self) -> dict:
        obs = {}
        try:
            forces = _parse_forces(self.path)
            tr_path = join(self.path, "trajectory.csv")
            tr = _parse_trajectory(tr_path)
            times_folder_probes = glob(
                join(self.path, "postProcessing", "probes", "*"))
            probes_path = join(times_folder_probes[0], "p")
            probes = _parse_probes(probes_path, self._n_states)
            p_names = ["p{:d}".format(i) for i in range(self._n_states)]

            for c in ("a", "b", "c"):
                obs[f"cx_{c}"] = pt.from_numpy(forces[f"cx_{c}"].values)
                obs[f"cy_{c}"] = pt.from_numpy(forces[f"cy_{c}"].values)
                obs[f"alpha_{c}"] = pt.from_numpy(tr[f"alpha_{c}"].values)
                obs[f"beta_{c}"] = pt.from_numpy(tr[f"beta_{c}"].values)

            obs["states"] = pt.from_numpy(probes[p_names].values)
            obs["actions"] = pt.stack((
                pt.from_numpy(tr[f"omega_a"].values),
                pt.from_numpy(tr[f"omega_b"].values),
                pt.from_numpy(tr[f"omega_c"].values)
            )).T
            obs[f"rewards"] = self.reward(obs)

        except Exception as e:
            logger.warning("Could not parse observations: ", e)
        finally:
            return obs

    def reset(self) -> None:
        # if we are not in base case, then there should be a log-file from the solver used (e.g. interFoam / pimpleFoam)
        solver_log = glob(join(self.path, "log.*Foam"))
        if solver_log:
            files = [f"log.{solver_log[0].split('.')[-1]}", "finished.txt", "trajectory.csv"]
        else:
            # otherwise we are in the base case and have only a log.*Foam.pre, which we don't want to remove
            files = ["finished.txt", "trajectory.csv"]
        for f in files:
            f_path = join(self.path, f)
            if isfile(f_path):
                remove(f_path)
        post = join(self.path, "postProcessing")
        if isdir(post):
            rmtree(post)
        times = get_time_folders(join(self.path, "processor0"))
        times = [t for t in times if float(t) > self.start_time]
        for p in glob(join(self.path, "processor*")):
            for t in times:
                rmtree(join(p, t))
