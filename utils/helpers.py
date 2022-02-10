import os
from stable_baselines3 import PPO

import config

def load_model(model_file, env):
    if os.path.isfile(model_file):
        ppo_model = PPO.load(model_file)
    else:
        ppo_model = PPO('MlpPolicy', env, verbose=1)
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

