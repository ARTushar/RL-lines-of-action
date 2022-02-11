from enum import Enum, auto

import gym
import numpy as np

from environments.lines_of_action.lac import LACEnv


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

        # TODO: Add model support

        action = np.argmax(action_probs)

        assert action_probs is not None
        if not choose_best_action:
            action = self.sample_action(action_probs)

        return action

    @staticmethod
    def sample_action(action_probs):
        action = np.random.choice(len(action_probs), p=action_probs)
        return action


def test_agent():
    agents = [Agent(AgentType.RANDOM), Agent(AgentType.RANDOM)]
    env = LACEnv()

    env.render()
    game_done = False
    while not game_done:
        for agent in agents:
            action = agent.choose_action(env, True)
            print('move: ', env.action_to_move(action))
            _, reward, done, _ = env.step(action)
            env.render()
            if done:
                game_done = True
                break


if __name__ == '__main__':
    np.random.seed(1)
    test_agent()

