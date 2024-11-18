"""
    Create a randomly initialized policy network.
"""
import sys
import torch as pt
from os import environ
from os.path import join
from typing import Union

BASE_PATH = environ.get("DRL_BASE", "")
sys.path.insert(0, BASE_PATH)

from drlfoam.agent import FCPolicy


def create_dummy_policy(n_probes: int, n_actions: int, target_dir: str,
                        abs_action: Union[int, float, pt.Tensor]) -> None:
    """
    initializes new policy

    :param n_probes: number of probes placed in the flow field
    :param n_actions: number of actions
    :param target_dir: path to the training directory
    :param abs_action: absolute value of the action boundaries
    :return: None
    """
    policy = FCPolicy(n_probes, n_actions, -abs_action, abs_action)
    script = pt.jit.script(policy)
    script.save(join(target_dir, "policy.pt"))


if __name__ == "__main__":
    # rotatingCylinder2D
    create_dummy_policy(12, 1, join("..", "openfoam", "test_cases", "rotatingCylinder2D"), 5)
    # rotatingPinball2D
    create_dummy_policy(14, 3, join("..", "openfoam", "test_cases", "rotatingCylinder2D"), 5)
