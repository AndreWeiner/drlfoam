"""
implements PPO-agent
"""
import logging
import torch as pt

from typing import List, Union
from collections import defaultdict

from .agent import Agent, FCPolicy, FCValue, compute_gae, compute_returns
from ..constants import EPS_SP, DEFAULT_TENSOR_TYPE

logger = logging.getLogger(__name__)
pt.set_default_tensor_type(DEFAULT_TENSOR_TYPE)


PPO_STATE_KEYS = ("policy_state", "value_state", "policy_optimizer_state",
                  "value_optimizer_state", "history")


class PPOAgent(Agent):
    def __init__(self, n_states, n_actions, action_min, action_max, ppo_dict: dict, value_train: dict,
                 policy_train: dict, policy_model: dict, value_model: dict):
        """
        implements PPO-agent class

        :param n_states: number of states
        :param n_actions: number of actions
        :param action_min: lower action bound
        :param action_max: upper action bound
        :param ppo_dict: contains hyperparameter {"lambda", "gamma", "entropy_weight"} for PPO
        :param value_train: contains parameters {"lr", "epochs", "clip", "grad_norm", "mse_stop"} for training the value
                            network
        :param policy_train: contains parameters {"lr", "epochs", "clip", "grad_norm", "kl_stop"} for training the
                             policy network
        :param policy_model: contains parameters {"n_layers", "n_neurons", "activation"} for the policy network
        :param value_model: contains parameters {"n_layers", "n_neurons", "activation"} for the value network
        """
        self._n_states = n_states
        self._n_actions = n_actions
        self._action_min = action_min
        self._action_max = action_max
        self._settings_value = value_train
        self._settings_policy = policy_train
        self._gamma = ppo_dict.get("gamma")
        self._lam = ppo_dict.get("lambda")
        self._entropy_weight = ppo_dict.get("entropy_weight")

        # networks and optimizers
        self._policy = FCPolicy(self._n_states, self._n_actions, self._action_min, self._action_max, **policy_model)
        self._value = FCValue(self._n_states, **value_model)
        self._policy_optimizer = pt.optim.Adam(self._policy.parameters(), lr=self._settings_policy.get("lr"))
        self._value_optimizer = pt.optim.Adam(self._value.parameters(), lr=self._settings_value.get("lr"))

        # history
        self._history = defaultdict(list)
        self._update_counter = 0

    def update(self, states: List[pt.Tensor], actions: List[pt.Tensor],
               rewards: List[pt.Tensor]) -> None:
        """
        update the policy and value network

        :param states: states
        :param actions: actions
        :param rewards: rewards
        :return: None
        """
        values = [self._value(s).detach() for s in states]

        # compute log_p for all but the final experience tuple
        log_p_old = pt.cat([self._policy.predict(s[:-1], a[:-1])[0].detach() for s, a in zip(states, actions)])
        gaes = pt.cat([compute_gae(r, v, self._gamma, self._lam) for r, v in zip(rewards, values)])
        gaes = (gaes - gaes.mean()) / (gaes.std(0) + EPS_SP)
        # create tensors with all but the final state/action of each trajectory for convenience
        states_wf = pt.cat([s[:-1] for s in states])
        actions_wf = pt.cat([a[:-1] for a in actions])

        # policy update
        p_loss_, e_loss_, kl_ = self._train_policy(states_wf, actions_wf, log_p_old, gaes)

        # value update & new computation of values
        v_loss_, mse_ = self._train_value(states, pt.cat(values), rewards)

        # save history
        self._history["policy_loss"].append(p_loss_)
        self._history["entropy_loss"].append(e_loss_)
        self._history["policy_div"].append(kl_)
        self._history["value_loss"].append(v_loss_)
        self._history["value_mse"].append(mse_)
        self._history["episode"].append(self._update_counter)
        self._update_counter += 1

    def _train_policy(self, states_wf, actions_wf, log_p_old, gaes):
        logger.info("Updating policy network.")
        p_loss, e_loss, kl = [], [], []
        for e in range(self._settings_policy.get("epochs")):
            # compute loss and update weights
            log_p_new, entropy = self._policy.predict(states_wf, actions_wf)
            p_ratio = (log_p_new - log_p_old).exp()
            policy_objective = gaes * p_ratio
            policy_objective_clipped = gaes * p_ratio.clamp(1.0 - self._settings_policy.get("clip"),
                                                            1.0 + self._settings_policy.get("clip"))
            policy_loss = -pt.min(policy_objective, policy_objective_clipped).mean()
            entropy_loss = -entropy.mean() * self._entropy_weight
            self._policy_optimizer.zero_grad()
            (policy_loss + entropy_loss).backward()
            pt.nn.utils.clip_grad_norm_(self._policy.parameters(), self._settings_policy.get("grad_norm"))
            self._policy_optimizer.step()
            p_loss.append(policy_loss.item())
            e_loss.append(entropy_loss.item())

            # check KL-divergence
            with pt.no_grad():
                log_p, _ = self._policy.predict(states_wf, actions_wf)
                kl_ = (log_p_old - log_p).mean()
                kl.append(kl_.item())
                if kl_.item() > self._settings_policy.get("kl_stop"):
                    logger.info(f"Stopping policy training after {e} epochs due to KL-criterion.")
                    break
        return p_loss, e_loss, kl

    def _train_value(self, states, values, rewards):
        logger.info("Updating value network.")

        # compute returns
        returns = pt.cat([compute_returns(r, self._gamma) for r in rewards])

        v_loss_, mse_ = [], []
        for e in range(self._settings_value.get("epochs")):
            # compute loss and update weights
            values_new = self._value(pt.cat(states))
            values_new_clipped = values + (values_new - values).clamp(-self._settings_value.get("clip"),
                                                                      self._settings_value.get("clip"))
            v_loss = (returns - values_new).pow(2)
            v_loss_clipped = (returns - values_new_clipped).pow(2)
            value_loss = pt.max(v_loss, v_loss_clipped).mul(0.5).mean()
            self._value_optimizer.zero_grad()
            value_loss.backward()
            pt.nn.utils.clip_grad_norm_(self._value.parameters(), self._settings_value.get("grad_norm"))
            self._value_optimizer.step()
            v_loss_.append(value_loss.item())

            # check difference to old values
            with pt.no_grad():
                values_check = self._value(pt.cat(states))
                mse = (values - values_check).pow(2).mul(0.5).mean()
                mse_.append(mse.item())
                if mse.item() > self._settings_value.get("mse_stop"):
                    logger.info(f"Stopping value training after {e} epochs due to MSE-criterion.")
                    break
        return v_loss_, mse_

    def save_state(self, path: str) -> None:
        pt.save(self.state, path)

    def load_state(self, state: Union[str, dict]) -> None:
        if isinstance(state, str):
            state = pt.load(state)
        if not isinstance(state, dict):
            raise ValueError(
                "Unknown state format; state should be a state dictionary or the path to a state dictionary.")
        if not all([key in state.keys() for key in PPO_STATE_KEYS]):
            ValueError(
                "One or more keys missing in state dictionary;\n" +
                "provided keys: {:s}\n".format(", ".join(state.keys())) +
                "expected keys: {:s}".format(", ".join(PPO_STATE_KEYS))
            )
        self._policy.load_state_dict(state["policy_state"])
        self._value.load_state_dict(state["value_state"])
        self._policy_optimizer.load_state_dict(state["policy_optimizer_state"])
        self._value_optimizer.load_state_dict(state["value_optimizer_state"])
        self._history = state["history"]
        if self._history["episode"]:
            self._update_counter = self._history["episode"][-1]

    def trace_policy(self) -> pt.jit.script:
        return pt.jit.script(self._policy)

    @property
    def history(self) -> dict:
        return self._history

    @property
    def state(self) -> dict:
        return {
            "policy_state": self._policy.state_dict(),
            "value_state": self._value.state_dict(),
            "policy_optimizer_state": self._policy_optimizer.state_dict(),
            "value_optimizer_state": self._value_optimizer.state_dict(),
            "history": self._history
        }

    @property
    def value(self):
        return self._value

    @property
    def policy(self):
        return self._policy

    @property
    def gamma(self):
        return self._gamma

    @property
    def lam(self):
        return self._lam

    @property
    def policy_clip(self):
        return self._settings_policy.get("clip")
