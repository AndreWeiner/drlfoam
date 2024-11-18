"""
    Example training script.
"""
import sys
import logging

from time import time
from os import environ
from os.path import join

BASE_PATH = environ.get("DRL_BASE", "")
sys.path.insert(0, BASE_PATH)

from drlfoam.agent import PPOAgent
# from examples.debug import DebugTraining
from drlfoam.execution.setup import ParseSetup
from examples.create_dummy_policy import create_dummy_policy
from drlfoam.execution import LocalBuffer, SlurmBuffer, SlurmConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def print_statistics(actions, rewards):
    rt = [r.mean().item() for r in rewards]
    at_mean = [a.mean().item() for a in actions]
    at_std = [a.std().item() for a in actions]
    reward_msg = f"Reward mean/min/max: {sum(rt) / len(rt):2.4f}/{min(rt):2.4f}/{max(rt):2.4f}"
    action_mean_msg = f"Mean action mean/min/max: {sum(at_mean) / len(at_mean):2.4f}/{min(at_mean):2.4f}/{max(at_mean):2.4f}"
    action_std_msg = f"Std. action mean/min/max: {sum(at_std) / len(at_std):2.4f}/{min(at_std):2.4f}/{max(at_std):2.4f}"
    logger.info("\n".join((reward_msg, action_mean_msg, action_std_msg)))


def main():
    # if we want to debug from an IDE, we need to set all required paths first
    # if DEBUG:
    #     debug = DebugTraining()

    # load the setup
    setup = ParseSetup(BASE_PATH)
    training_path = str(join(BASE_PATH, setup.buffer["training_path"]))
    setup.env.path = join(training_path, "base")

    # add the path to openfoam to the Allrun scripts
    # if DEBUG:
    #     debug.set_openfoam_bashrc(training_path=setup.env.path)

    # create buffer
    if setup.training["executer"] == "local":
        buffer = LocalBuffer(training_path, setup.env, setup.buffer["buffer_size"], setup.buffer["n_runner"])
    elif setup.training["executer"] == "slurm":
        # Typical Slurm configs for TU Dresden cluster
        config = SlurmConfig(
            n_tasks_per_node=setup.env.mpi_ranks, n_nodes=1, time="03:00:00", job_name="drl_train",
            modules=["development/24.04 GCC/12.3.0", "OpenMPI/4.1.5", "OpenFOAM/v2312"],
            commands_pre=["source $FOAM_BASH", f"source {BASE_PATH}/setup-env"]
        )
        buffer = SlurmBuffer(training_path, setup.env, setup.buffer["buffer_size"],
                             setup.buffer["n_runner"], config, timeout=setup.buffer["timeout"])
    else:
        raise ValueError(
            f"Unknown executer {setup.training['executer']}; available options are 'local' and 'slurm'.")

    # create PPO agent
    agent = PPOAgent(setup.env.n_states, setup.env.n_actions, -setup.env.action_bounds, setup.env.action_bounds,
                     **setup.agent)

    # load checkpoint if provided
    if setup.training["checkpoint"] is not None:
        logging.info(f"Loading checkpoint from file {setup.training['checkpoint']}")
        agent.load_state(join(training_path, setup.training["checkpoint"]))
        starting_episode = agent.history["episode"][-1] + 1
        buffer._n_fills = starting_episode
    else:
        starting_episode = 0

        # create fresh random policy and execute the base case
        create_dummy_policy(setup.env.n_states, setup.env.n_actions, setup.env.path, setup.env.action_bounds)

        # execute the base simulation
        buffer.prepare()

    buffer.base_env.start_time = buffer.base_env.end_time
    buffer.base_env.end_time = setup.training["end_time"]
    buffer.reset()

    # begin training
    start_time = time()
    for e in range(starting_episode, setup.training["episodes"]):
        logger.info(f"Start of episode {e}")
        buffer.fill()
        states, actions, rewards = buffer.observations
        print_statistics(actions, rewards)
        agent.update(states, actions, rewards)
        agent.save_state(join(training_path, f"checkpoint_{e}.pt"))
        current_policy = agent.trace_policy()
        buffer.update_policy(current_policy)
        current_policy.save(join(training_path, f"policy_trace_{e}.pt"))
        if not e == setup.training["episodes"] - 1:
            buffer.reset()
    logger.info(f"Training time (s): {time() - start_time}")


if __name__ == "__main__":
    # option for running the training in IDE, e.g. in debugger
    # DEBUG = False
    main()
