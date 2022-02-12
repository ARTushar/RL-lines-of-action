import numpy as np
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

import config
from utils.callback import SelfPlayCallback
from utils.helpers import load_all_models, load_model, load_best_model, load_random_model, get_model_generation_stats


def train_stable_baseline3(env):
    print("Training RL agent")
    model = PPO('MlpPolicy', env, verbose=1)

    callback_args = {
        'eval_env': env,
        'best_model_save_path': config.TMPMODELDIR,
        'log_path': config.LOGDIR,
        'eval_freq': config.eval_freq,
        'n_eval_episodes': config.n_eval_episodes,
        'deterministic': False,
        'render': True,
        'verbose': 0
    }

    # Evaluate against a 'rules' agent as well
    eval_actual_callback = EvalCallback(
      eval_env = env,
      eval_freq=1,
      n_eval_episodes=config.n_eval_episodes,
      deterministic = True,
      render = True,
      verbose = 0
    )
    callback_args['callback_on_new_best'] = eval_actual_callback

    eval_callback = SelfPlayCallback(config.oponent_type, config.threshold, **callback_args)

    model.learn(total_timesteps=10000, callback=[eval_callback], reset_num_timesteps=False, tb_log_name="tb")

    print('saving the model....')
    model.save('ppo_cartpole')


def test_stable_baseline3(env):
    model = PPO.load('ppo_cartpole')

    obs = env.reset()
    print("Testing RL Agent")
    for i in range(1000):
        action, _state = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        env.render()
        if done:
            obs = env.reset()


if __name__ == '__main__':
    env = gym.make('CartPole-v1')
    env = Monitor(env)
    train_stable_baseline3(env)
    # test_stable_baseline3(env)
    # print(get_model_generation_stats())