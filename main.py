import numpy as np
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
import os

import config
from utils.callback import SelfPlayCallback
from models.model import CustomCNN, CustomActorCriticPolicy, ResnetFeatureExtractor
from utils.selfplay import SelfPlayEnv, OpponentType
from utils.helpers import load_all_models, load_model, load_best_model, load_random_model, get_model_generation_stats


def train_model(env, continue_from_last_checkpoint=False):
    print("Training RL agent")
    policy_kwargs = dict(
        # features_extractor_class=CustomCNN,
        features_extractor_class=ResnetFeatureExtractor,
        features_extractor_kwargs=dict(features_dim=config.model_output_dim),
    )
    if continue_from_last_checkpoint:
        # model = load_best_model(env)
        model_file = os.path.join(config.MODELPOOLDIR, 'best_model-copy.zip')
        model = PPO.load(model_file, env=env)
    else:
        model = PPO(CustomActorCriticPolicy, env, policy_kwargs=policy_kwargs, verbose=0)

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
        eval_freq=config.eval_freq,
        n_eval_episodes=config.n_eval_episodes,
        best_model_save_path=config.TMPMODELDIR,
        log_path=config.LOGDIR,
        deterministic=True,
        render=False,
    )
    callback_args['callback_on_new_best'] = eval_actual_callback

    eval_callback = SelfPlayCallback(OpponentType.RANDOM, config.threshold, **callback_args)

    model.learn(total_timesteps=1000000, callback=[eval_callback], reset_num_timesteps=False, tb_log_name="tb")

    print('saving the model....')
    model.save(os.path.join(config.MODELPOOLDIR, 'model.zip'))


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
    custom_env = SelfPlayEnv(opponent_type=OpponentType.PREV_BEST, verbose=0)
    train_model(custom_env, continue_from_last_checkpoint=False)
    custom_env.close()



