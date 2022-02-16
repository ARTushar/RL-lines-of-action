import os

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

import config
from models.model import CustomCNN, CustomActorCriticPolicy, ResnetFeatureExtractor
from utils.selfplay import SelfPlayEnv, OpponentType


def train_with_random():
    policy_kwargs = dict(
        # features_extractor_class=CustomCNN,
        features_extractor_class=ResnetFeatureExtractor,
        features_extractor_kwargs=dict(features_dim=64*64*2),

    )
    env = SelfPlayEnv(opponent_type=OpponentType.RANDOM, verbose=0)
    model = PPO(CustomActorCriticPolicy, env, policy_kwargs=policy_kwargs, verbose=0)
    eval_callback = EvalCallback(
        eval_env=Monitor(SelfPlayEnv(opponent_type=OpponentType.RANDOM)),
        eval_freq=1000,
        best_model_save_path=config.TMPMODELDIR,
        log_path=config.LOGDIR,
        deterministic=True,
        render=False
    )

    model.learn(total_timesteps=int(1e9), callback=[eval_callback], tb_log_name='tb')

    env.close()


def evaluate_trained_model():
    # policy_kwargs = dict(
    #     features_extractor_class=CustomCNN,
    # )
    env = SelfPlayEnv(opponent_type=OpponentType.RANDOM)
    best_mode_name = 'best_model.zip'
    path = os.path.join(config.TMPMODELDIR, best_mode_name)
    model = PPO.load(path, env=env)
    mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=2)
    print("mean reward : ", mean_reward)


def continue_train():
    env = SelfPlayEnv(opponent_type=OpponentType.RANDOM, verbose=0)
    best_mode_name = 'best_model.zip'
    path = os.path.join(config.TMPMODELDIR, best_mode_name)
    model = PPO.load(path, env=env)
    eval_callback = EvalCallback(
        eval_env=Monitor(SelfPlayEnv(opponent_type=OpponentType.RANDOM)),
        eval_freq=1000,
        best_model_save_path=config.TMPMODELDIR,
        log_path=config.LOGDIR,
        deterministic=True,
        render=False
    )
    model.learn(total_timesteps=int(1e9), callback=[eval_callback], tb_log_name='tb')


if __name__ == '__main__':
    train_with_random()
    # evaluate_trained_model()
    # continue_train()



