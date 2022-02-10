from enum import Enum, auto

import gym
import numpy as np


class AgentType(Enum):
    RANDOM = auto()
    MODEL = auto()


class Agent:
    def __init__(self, agent_type: AgentType, model=None):
        self.agent_type = agent_type
        self.model = model

    def choose_action(self, env: gym.Env, choose_best_action: bool):
        action_probs = None
        if self.agent_type == AgentType.RANDOM:
            action_probs = np.array(env.get_random_action())

        action = np.argmax(action_probs)

        assert action_probs is not None
        if not choose_best_action:
            action = self.sample_action(action_probs)

        return action

    @staticmethod
    def sample_action(action_probs):
        action = np.random.choice(len(action_probs), p=action_probs)
        return action
