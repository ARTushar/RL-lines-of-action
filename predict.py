import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.utils import obs_as_tensor

import config
from utils.agent import test_model_agent
from utils.selfplay import SelfPlayEnv, OpponentType


def load_model():
    env = SelfPlayEnv(opponent_type=OpponentType.RANDOM)
    best_mode_name = 'best_model.zip'
    path = os.path.join(config.TMPMODELDIR, best_mode_name)
    model = PPO.load(path, env=env)
    return model


def predict(observation: np.ndarray, model: PPO):
    model.policy.set_training_mode(False)
    observation = observation.reshape((-1,) + observation.shape)
    obs_t = obs_as_tensor(observation, model.device)
    action_dist = model.policy.get_distribution(obs_t)
    print(action_dist)


if __name__ == "__main__":
    # model = load_model()
    # env = SelfPlayEnv(opponent_type=OpponentType.RANDOM)
    # obs = env.reset()
    # predict(obs, model)
    test_model_agent()
