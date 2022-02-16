import numpy as np
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

import config
from utils.callback import SelfPlayCallback
from models.model import CustomCNN, CustomActorCriticPolicy
from utils.selfplay import SelfPlayEnv, OpponentType
from utils.helpers import load_all_models, load_model, load_best_model, load_random_model, get_model_generation_stats


def train_stable_baseline3(env):
    print("Training RL agent")
    policy_kwargs = dict(
        features_extractor_class=CustomCNN,
        features_extractor_kwargs=dict(features_dim=64*64*2),

    )
    env = SelfPlayEnv(opponent_type=OpponentType.PREV_BEST, verbose=0)
    model = PPO(CustomActorCriticPolicy, env, batch_size=64, policy_kwargs=policy_kwargs, verbose=0)
    

    callback_args = {
        'eval_env': Monitor(env),
        'best_model_save_path': config.TMPMODELDIR,
        'log_path': config.LOGDIR,
        'eval_freq': config.eval_freq,
        'n_eval_episodes': config.n_eval_episodes,
        'deterministic': True,
        'render': False,
        'verbose': 0
    }

    # Evaluate against a 'random' agent as well
    eval_actual_callback = EvalCallback(
        eval_env=Monitor(SelfPlayEnv(opponent_type=OpponentType.RANDOM)),
        eval_freq=100,
        best_model_save_path=config.TMPMODELDIR,
        log_path=config.LOGDIR,
        deterministic=True,
        render=False
    )
    callback_args['callback_on_new_best'] = eval_actual_callback

    eval_callback = SelfPlayCallback(OpponentType.PREV_BEST, config.threshold, **callback_args)

    model.learn(total_timesteps=100000, callback=[eval_callback], reset_num_timesteps=False, tb_log_name="tb")

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