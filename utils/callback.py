from stable_baselines3.common.callbacks import EvalCallback
import torch
from utils.helpers import get_model_generation_stats
from utils.selfplay import SelfPlayEnv, OpponentType
import numpy as np
import os
from shutil import copyfile

import config


class SelfPlayCallback(EvalCallback):
  def __init__(self, opponent_type, threshold, *args, **kwargs):
    super(SelfPlayCallback, self).__init__(*args, **kwargs)
    self.opponent_type = opponent_type
    # self.model_dir = os.path.join(config.MODELDIR, env_name)
    self.generation, best_rule_based_reward, best_reward, self.base_timesteps = get_model_generation_stats()

    self.threshold = best_reward
    # print('threshold initial: ', best_reward)

    #reset best_mean_reward because this is what we use to extract the rewards from the latest evaluation by each agent
    self.best_mean_reward = -np.inf
    # if self.callback is not None: #if evaling against rules-based agent as well, reset this too
    #   self.callback.best_mean_reward = -np.inf

    # if self.opponent_type == 'rules':
    #   self.threshold = bmr # the threshold is the overall best evaluation by the agent against a rules-based agent
    # else:
    #   self.threshold = threshold # the threshold is a constant


  def _on_step(self) -> bool:
    if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
      print('\nevaluating ...')
      print('episode: ', self.n_calls/self.eval_freq)
      print('generation: ', self.generation)
      print('best mean reward: ', self.best_mean_reward)
      result = super(SelfPlayCallback, self)._on_step() #this will set self.best_mean_reward to the reward from the evaluation as it's previously -np.inf
      # list_of_rewards = MPI.COMM_WORLD.allgather(self.best_mean_reward)
      # av_reward = np.mean(list_of_rewards)
      # std_reward = np.std(list_of_rewards)
      # av_timesteps = np.mean(MPI.COMM_WORLD.allgather(self.num_timesteps))
      # total_episodes = np.sum(MPI.COMM_WORLD.allgather(self.n_eval_episodes))

      # if self.callback is not None:
      #   rules_based_rewards = MPI.COMM_WORLD.allgather(self.callback.best_mean_reward)
      #   av_rules_based_reward = np.mean(rules_based_rewards)

      # rank = MPI.COMM_WORLD.Get_rank()
      # if rank == 0:
      #   logger.info("Eval num_timesteps={}, episode_reward={:.2f} +/- {:.2f}".format(self.num_timesteps, av_reward, std_reward))
      #   logger.info("Total episodes ran={}".format(total_episodes))

      print('new threshold: ', self.threshold)
      # #compare the latest reward against the threshold
      if result and self.best_mean_reward > self.threshold:
        self.generation += 1
        self.threshold = self.best_mean_reward

        generation_str = str(self.generation).zfill(5)
        rewards_str = str(round(self.best_mean_reward,3))

        # if self.callback is not None:
        #   av_rules_based_reward_str = str(round(rewards_str,3))
        # else:
        #   av_rules_based_reward_str = str(0)
        
        source_file = os.path.join(config.TMPMODELDIR, f"best_model.zip") # this is constantly being written to - not actually the best model
        target_file = os.path.join(config.MODELPOOLDIR,  f"_model_{generation_str}_{rewards_str}_{rewards_str}_{str(self.base_timesteps + self.num_timesteps)}_.zip")
        copyfile(source_file, target_file)
        target_file = os.path.join(config.MODELPOOLDIR,  f"best_model.zip")
        copyfile(source_file, target_file)

        # if playing against a rules based agent, update the global best reward to the improved metric
        # if self.opponent_type == 'rules':
        #   self.threshold  = av_reward
        
      #reset best_mean_reward because this is what we use to extract the rewards from the latest evaluation by each agent
      self.best_mean_reward = -np.inf

      # if self.callback is not None: #if evaling against rules-based agent as well, reset this too
      #   self.callback.best_mean_reward = -np.inf

    return True