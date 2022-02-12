from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

import config
from utils.selfplay import SelfPlayEnv, OpponentType


def train_with_bot():
    env = SelfPlayEnv(opponent_type=OpponentType.BOT)
    model = PPO('MlpPolicy', env)
    eval_callback = EvalCallback(
        eval_env=Monitor(SelfPlayEnv(opponent_type=OpponentType.RANDOM)),
        eval_freq=100,
        best_model_save_path=config.TMPMODELDIR,
        log_path=config.LOGDIR,
        deterministic=True,
        render=False
    )

    model.learn(total_timesteps=int(1e9), callback=[eval_callback], tb_log_name='tb')

    env.close()


if __name__ == '__main__':
    train_with_bot()

