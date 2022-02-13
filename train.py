from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

import config
from models.model import CustomCNN
from utils.selfplay import SelfPlayEnv, OpponentType


def train_with_random():
    policy_kwargs = dict(
        features_extractor_class=CustomCNN,
    )
    env = SelfPlayEnv(opponent_type=OpponentType.RANDOM)
    model = PPO('CnnPolicy', env, policy_kwargs=policy_kwargs, verbose=0)
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


if __name__ == '__main__':
    train_with_random()

