"""
Implements functions for computing the returns and GAE as well as classes for policy - and value network.
Provides further a base class for all agents.
"""
from typing import Callable, Tuple, Union
from abc import ABC, abstractmethod
import torch as pt
from ..constants import DEFAULT_TENSOR_TYPE


pt.set_default_tensor_type(DEFAULT_TENSOR_TYPE)


def compute_returns(rewards: pt.Tensor, gamma: Union[int, float] = 0.99) -> pt.Tensor:
    """
    compute the returns based on given rewards and discount factor

    :param rewards: rewards
    :type rewards: pt.Tensor
    :param gamma: discount factor
    :type gamma: Union[int, float]
    :return: returns
    :rtype: pt.Tensor
    """
    n_steps = len(rewards)
    discounts = pt.logspace(0, n_steps-1, n_steps, gamma)
    returns = [(discounts[:n_steps-t] * rewards[t:]).sum()
               for t in range(n_steps)]
    return pt.tensor(returns)


def compute_gae(rewards: pt.Tensor, values: pt.Tensor, gamma: Union[int, float] = 0.99,
                lam: Union[int, float] = 0.97) -> pt.Tensor:
    """
    Compute the generalized advantage estimate (GAE) based on

    'High-Dimensional Continuous Control Using Generalized Advantage Estimation', https://arxiv.org/abs/1506.02438

    :param rewards: rewards
    :type rewards: pt.Tensor
    :param values: values of the states (output of value network)
    :type values: pt.Tensor
    :param gamma: discount factor
    :type gamma: Union[int, float]
    :param lam: hyperparameter lambda
    :type lam: Union[int, float]
    :return: GAE
    :rtype: pt.Tensor
    """
    n_steps = len(rewards)
    factor = pt.logspace(0, n_steps-1, n_steps, gamma*lam)
    delta = rewards[:-1] + gamma * values[1:] - values[:-1]
    gae = [(factor[:n_steps-t-1] * delta[t:]).sum()
           for t in range(n_steps - 1)]
    return pt.tensor(gae)


class FCPolicy(pt.nn.Module):
    def __init__(self, n_states: int, n_actions: int, action_min: Union[int, float, pt.Tensor],
                 action_max: Union[int, float, pt.Tensor], n_layers: int = 2, n_neurons: int = 64,
                 activation: Callable = pt.nn.functional.relu):
        """
        implements policy network

        :param n_states: number of states
        :type n_states: int
        :param n_actions: number of actions
        :type n_actions: int
        :param action_min: lower bound of the actions
        :type action_min: Union[int, float, pt.Tensor]
        :param action_max: upper bound of the actions
        :type action_max: Union[int, float, pt.Tensor]
        :param n_layers: number of hidden layers
        :type n_layers: int
        :param n_neurons: number of neurons per layer
        :type n_neurons: int
        :param activation: activation function
        :type activation: pt.Callable
        """
        super(FCPolicy, self).__init__()
        self._n_states = n_states
        self._n_actions = n_actions
        self._action_min = action_min
        self._action_max = action_max
        self._n_layers = n_layers
        self._n_neurons = n_neurons
        self._activation = activation

        # set up policy network
        self._layers = pt.nn.ModuleList()
        self._layers.append(pt.nn.Linear(self._n_states, self._n_neurons))
        if self._n_layers > 1:
            for hidden in range(self._n_layers - 1):
                self._layers.append(pt.nn.Linear(self._n_neurons, self._n_neurons))
                self._layers.append(pt.nn.LayerNorm(self._n_neurons))
        self._last_layer = pt.nn.Linear(self._n_neurons, 2*self._n_actions)

    @pt.jit.ignore
    def _scale(self, actions: pt.Tensor) -> pt.Tensor:
        """
        perform min-max-scaling of the actions

        :param actions: unscaled actions
        :type actions: pt.Tensor
        :return: actions scaled to an interval of [0, 1]
        :rtype pt.Tensor
        """
        return (actions - self._action_min) / (self._action_max - self._action_min)

    def forward(self, x: pt.Tensor) -> pt.Tensor:
        for layer in self._layers:
            x = self._activation(layer(x))
        return 1.0 + pt.nn.functional.softplus(self._last_layer(x))

    @pt.jit.ignore
    def predict(self, states: pt.Tensor, actions: pt.Tensor) -> Tuple[pt.Tensor, pt.Tensor]:
        """
        predict log-probability and associated entropy based on given states and actions based on a beta distribution
        for each action

        :param states: unscaled states
        :type states: pt.Tensor
        :param actions: unscaled actions
        :type actions: pt.Tensor
        :return: log-probability and entropy of the beta distribution(s); in case of multiple distributions, the sum
                 is taken over the second axis
        :rtype Tuple[pt.Tensor, pt.Tensor]
        """
        out = self.forward(states)
        c0 = out[:, :self._n_actions]
        c1 = out[:, self._n_actions:]
        beta = pt.distributions.Beta(c0, c1)
        if len(actions.shape) == 1:
            scaled_actions = self._scale(actions.unsqueeze(-1))
        else:
            scaled_actions = self._scale(actions)
        log_p = beta.log_prob(scaled_actions)
        if len(actions.shape) == 1:
            return log_p.squeeze(), beta.entropy().squeeze()
        else:
            return log_p.sum(dim=1), beta.entropy().sum(dim=1)


class FCValue(pt.nn.Module):
    def __init__(self, n_states: int, n_layers: int = 2, n_neurons: int = 64,
                 activation: Callable = pt.nn.functional.relu):
        """
        implements value network

        :param n_states: number of states
        :type n_states: int
        :param n_layers: number of hidden layers
        :type n_layers: int
        :param n_neurons: number of neurons per layer
        :type n_neurons: int
        :param activation: activation function
        :type activation: pt.Callable
        """
        super(FCValue, self).__init__()
        self._n_states = n_states
        self._n_layers = n_layers
        self._n_neurons = n_neurons
        self._activation = activation

        # set up value network
        self._layers = pt.nn.ModuleList()
        self._layers.append(pt.nn.Linear(self._n_states, self._n_neurons))
        if self._n_layers > 1:
            for hidden in range(self._n_layers - 1):
                self._layers.append(pt.nn.Linear(self._n_neurons, self._n_neurons))
                self._layers.append(pt.nn.LayerNorm(self._n_neurons))
        self._layers.append(pt.nn.Linear(self._n_neurons, 1))

    def forward(self, x: pt.Tensor) -> pt.Tensor:
        for i_layer in range(len(self._layers) - 1):
            x = self._activation(self._layers[i_layer](x))
        return self._layers[-1](x).squeeze()


class Agent(ABC):
    """Common interface for all agents.
    """

    @abstractmethod
    def update(self, states, actions, rewards):
        pass

    @abstractmethod
    def save_state(self, path: str):
        pass

    @abstractmethod
    def load_state(self, state: Union[str, dict]):
        pass

    @abstractmethod
    def trace_policy(self):
        pass

    @property
    @abstractmethod
    def history(self):
        pass

    @property
    @abstractmethod
    def state(self):
        pass