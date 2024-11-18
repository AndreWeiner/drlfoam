"""
    parse the settings defined in 'config_orig.yml' located in the examples directory and set up the simulation
"""
import yaml
import logging

from os import makedirs
from os.path import join, exists
from shutil import copy, copytree
from torch import manual_seed, cuda
from torch.nn.modules import activation

from ..environment import RotatingCylinder2D, RotatingPinball2D

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ParseSetup:
    def __init__(self, base_path: str):
        """
        implements class to parse the settings defined in 'config.yml' located in the examples directory

        :param base_path: BASE_PATH to drlfoam
        """
        self._base_path = base_path
        self._file_name_orig = r"config_orig.yml"
        self._file_name = r"config.yml"
        self._keys_train = ["executer", "simulation", "seed", "training_path", "episodes", "end_time", "checkpoint"]
        self._keys_buffer = ["training_path", "n_runner", "buffer_size", "timeout"]
        logger.info(f"Loading settings from '{self._file_name_orig}'.")

        # define the correct data types within the config file and available simulation environments
        self._data_types = self.data_types()
        self._simulations = self._simulation_envs()

        # load the config file
        self._config = self._load_config_file()

        # recursively check the config file and cast to the defined data types; a required parameter is not specified
        # then replace it with a default value
        self._cast_input(self._config, self._data_types)

        # add the prefix 'examples' to the training path to ensure that we run all training sin the examples directory
        self._config["training"]["training_path"] = join("examples", self._config["training"].get("training_path"))

        # map the activation function based on the defined str in the config file
        self._map_activation_function()

        # check if the desired simulation environment exists
        self._check_if_simulation_exist()
        self.env = self._simulations[self._config["training"]["simulation"]]()

        # check if the defined finish time is larger than the end time of the base simulation
        self._check_finish_time()

        # re-organize for simpler access
        self.training = {k: self._config["training"][k] for k in self._keys_train}
        self.buffer = {k: self._config["training"][k] for k in self._keys_buffer}
        self.agent = {"value_model": self._config["value_network"], "policy_model": self._config["policy_network"],
                      "value_train": self._config["value_training"], "policy_train": self._config["policy_training"],
                      "ppo_dict": self._config["ppo_settings"]}
        del self._config

        # create a run directory, copy of the simulation environments and config file
        self._create_copy()

        # ensure reproducibility
        manual_seed(self.training.get("seed"))
        if cuda.is_available():
            cuda.manual_seed_all(self.training.get("seed"))

    def _load_config_file(self) -> dict:
        """
        load the settings from the config .yml file

        :return: loaded settings as dict
        """
        # assuming base_path is drlfoam, but the config file is located in the examples directory
        try:
            with open(join(self._base_path, "examples", self._file_name_orig), "r") as file:
                config = yaml.safe_load(file)
            return config
        except FileNotFoundError:
            logger.error(f"Unable to find '{self._file_name_orig}'. Make sure '{self._file_name_orig}' is located in "
                         f"the 'examples' directory. Exiting.")
            exit()

    def _create_copy(self) -> None:
        """
        Create a direcotry for executing the training, copy the config file and the simulation environment into it

        In case we want to run multiple trainings in parallel, each training needs to have its own config file
        (allow runtime modification, will be implemented later)

        :return: None
        """
        # create run directory and copy the config file
        makedirs(join(self._base_path, self.training.get("training_path")), exist_ok=True)
        copy(join(self._base_path, "examples", self._file_name_orig),
             join(self._base_path, self.training.get("training_path"), self._file_name))

        # copy the simulation environment
        if not exists(join(self.training.get("training_path"), "base")):
            copytree(join(self._base_path, "openfoam", "test_cases", self.training.get("simulation")),
                     join(self._base_path, self.training.get("training_path"), "base"), dirs_exist_ok=True)

    def _cast_input(self, data: dict, reference_data: dict = None, parent_dict: str = None) -> None:
        """
        Recursively cast the data types within the setup dict to the data types defined in the data_types() method,
        because sometimes the type is not determined correctly when loading the file.
        Further, replace any mandatory parameters, which are not provided with the default values.

        :param data: Setup dict (or sub-dict)
        :param reference_data: dict containing the corresponding data types
        :param parent_dict: key of the parent dict in case we are calling this function with a sub dict
        :return: None
        """
        # recursively check and cast the settings to the correct types
        for key, value in reference_data.items():
            # in case we have a sub-dict, we need to check all keys inside this dict as well
            if isinstance(value, dict):
                self._cast_input(data[key], reference_data[key], key)
            else:
                # otherwise cast to correct typ
                try:
                    # if the key is not present but mandatory, replace it with default
                    if key not in data.keys():
                        logger.warning(f"Could not find a value for parameter '{key}' in the sub-dict '{parent_dict}'. "
                                       f"Using a default value of {key} = {reference_data[key][1]}.")
                        data[key] = reference_data[key][1]

                    # otherwise cast
                    else:
                        # we need to make an exception for the checkpoint argument
                        if key == "checkpoint":
                            data[key] = str(data[key]) if data[key] is not None else None
                        else:
                            data[key] = reference_data[key][0](data[key])
                except KeyError:
                    logger.warning(f"Could not find default data type for entry '{key}'in the sub-dict '{parent_dict}'."
                                   f" Omit checking for the correct data type.")

    def _map_activation_function(self) -> None:
        """
        Map the string defining the activation function in the settings to the pyTorch module

        Taken from: https://discuss.pytorch.org/t/call-activation-function-from-string/30857/4#

        :return: None
        """
        # get all available activation functions
        all_functions = [str(a).lower() for a in activation.__all__]

        for n in ["policy_network", "value_network"]:
            if self._config[n]["activation"] in all_functions:
                idx = all_functions.index(self._config[n]["activation"].lower())
                act_name = activation.__all__[idx]
                self._config[n]["activation"] = getattr(activation, act_name)()
            else:
                _opts = "".join(["Available options are:\n"] + [f'\t{a}\n' for a in all_functions])
                raise ValueError(f"Cannot find activation function {self._config[n]['activation'].lower()}. {_opts}")

    def _check_finish_time(self) -> None:
        """
        Checks if the user-specified finish time is greater than the end time of the base case, if not then exit with
        an error message

        :return: None
        """
        pwd = join(self._base_path, "openfoam", "test_cases", self._config["training"]["simulation"], "system",
                   "controlDict")
        with open(pwd, "r") as f:
            lines = f.readlines()

        # get the end time of the base case, normally endTime is specified in l. 28, but in case of modifications, check
        # lines 20-35
        t_base = [float(i.strip(";\n").split(" ")[-1]) for i in lines[20:35] if i.startswith("endTime")][0]

        if t_base >= self._config["training"]["end_time"]:
            logger.critical(f"specified finish time is smaller than end time of base case! The finish time needs to be "
                            f"greater than {t_base}. Exiting.")
            exit(0)

    def _check_if_simulation_exist(self) -> None:
        """
        Check if the simulation exists within drlfoam or is the specified environment is not implemented

        :return: None
        """
        if self._config["training"]["simulation"] not in self._simulations.keys():
            msg = (f"Unknown simulation environment {self._config['training']['simulation']}" +
                   "Available options are:\n\n" +
                   "\n".join(self._simulations.keys()) + "\n")
            raise ValueError(msg)

    @staticmethod
    def data_types() -> dict:
        """
        dict defining all the data types and corresponding default values for the settings to ensure correct casting

        :return: dict containing the data types
        """
        return {
            "training": {
                "executer": (str, "local"),
                "simulation": (str, "rotatingCylinder2D"),
                "training_path": (str, join("examples", "test_training")),
                "n_runner": (int, 2),
                "buffer_size": (int, 4),
                "end_time": (int, 8),
                "seed": (int, 0),
                "episodes": (int, 20),
                "checkpoint": (str, None),
                "timeout": (float, float(1e15)),
            },
            "policy_network": {
                "n_layers": (int, 2),
                "n_neurons": (int, 64),
                "activation": (str, "relu")
            },
            "policy_training": {
                "lr": (float, float(4e-4)),
                "epochs": (int, 100),
                "clip": (float, 0.1),
                "grad_norm": (float, "inf"),
                "kl_stop": (float, 0.2)
            },
            "value_network": {
                "n_layers": (int, 2),
                "n_neurons": (int, 64),
                "activation": (str, "relu")
            },
            "value_training": {
                "lr": (float, float(5e-4)),
                "epochs": (int, 100),
                "clip": (float, 0.1),
                "grad_norm": (float, "inf"),
                "mse_stop": (float, 25.0)
            },
            "ppo_settings": {
                "gamma": (float, 0.99),
                "lambda": (float, 0.97),
                "entropy_weight": (float, 0.01)
            }
        }

    @staticmethod
    def _simulation_envs() -> dict:
        """
        stores all simulation environments implemented in drlfoam so far

        :return: dict with the available environments
        """
        return {
            "rotatingCylinder2D": RotatingCylinder2D,
            "rotatingPinball2D": RotatingPinball2D
        }


if __name__ == "__main__":
    pass
