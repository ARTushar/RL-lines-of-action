import os
from stable_baselines3 import PPO

from models.model import CustomCNN, CustomActorCriticPolicy, ResnetFeatureExtractor
import numpy as np

import config


def create_custom_policy_ppo_model(env):
    policy_kwargs = dict(
        # features_extractor_class=CustomCNN,
        features_extractor_class=ResnetFeatureExtractor,
        features_extractor_kwargs=dict(features_dim=config.model_output_dim),
    )
    model = PPO(CustomActorCriticPolicy, env, policy_kwargs=policy_kwargs, verbose=0)
    return model


def load_model(model_file, env):
    if os.path.isfile(model_file):
        ppo_model = PPO.load(model_file, env=env)
    else:
        print('creating new model in load model function')
        ppo_model = create_custom_policy_ppo_model(env)
        ppo_model.save(model_file)
    return ppo_model


def load_all_models(env):
    model_name_list = [f for f in os.listdir(config.MODELPOOLDIR) if f.startswith("_model")]
    model_name_list.sort()
    model_path_list = [os.path.join(config.MODELPOOLDIR, f) for f in model_name_list]
    models = [load_model(os.path.join(config.MODELPOOLDIR, 'base.zip'), env)]
    for model_path in model_path_list:
        models.append(load_model(model_path, env))
    return models


def get_best_model_name(dir):
    model_name_list = [f for f in os.listdir(dir) if f.startswith("_model")]
    model_name_list.sort()
    if len(model_name_list) == 0:
        return None
    model_name = model_name_list[-1]
    return model_name


def load_best_model(env, directory=config.MODELPOOLDIR):
    model_name = get_best_model_name(directory)
    # print('best model name:', model_name)
    if model_name is None:
        model_path = os.path.join(directory, 'best_model.zip')
    else:
        model_path = os.path.join(directory, model_name)
    model = load_model(model_path, env)
    return model


def load_random_model(env):
    model_name_list = [f for f in os.listdir(config.MODELPOOLDIR) if f.startswith("_model")]
    if len(model_name_list) == 0:
        model_name = 'base.zip'
    else:
        model_name = np.random.choice(model_name_list)
    print('random model name:', model_name)
    model_path = os.path.join(config.MODELPOOLDIR, model_name)
    model = load_model(model_path, env)
    return model


def get_model_generation_stats():
    best_model_name = get_best_model_name(config.MODELPOOLDIR)
    if best_model_name is None:
        generation = 0
        timesteps = 0
        best_rules_based = -np.inf
        best_reward = -np.inf
    else:
        best_model_name = best_model_name[:-4] # excluding .zip
        stats = best_model_name.split('_')
        generation = int(stats[2])
        best_rules_based = float(stats[3])
        best_reward = float(stats[4])
        timesteps = int(stats[5])
    return generation, best_rules_based, best_reward, timesteps
