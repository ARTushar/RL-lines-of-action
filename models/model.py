from typing import Tuple, Callable, List, Dict, Union, Optional, Type

import gym
import torch as th
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch.nn as nn
from torch.nn.functional import relu


class CustomCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.Space, features_dim: int = 128*8*8):
        super().__init__(observation_space, features_dim)
        # assume channel first
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=(4, 4), stride=(2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1)),
            nn.ReLU(),
            nn.Flatten(),
        )

        # compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(th.as_tensor(observation_space.sample()[None]).float()).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))


class CustomNetwork(nn.Module):
    def __init__(
            self,
            feature_dim: int = 128*8*8,
            last_layer_dim_pi: int = 64*8,
            last_layer_dim_vf: int = 64*8
    ):
        super(CustomNetwork, self).__init__()
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf

        self.policy_net = nn.Sequential(
            nn.Linear(feature_dim, last_layer_dim_pi), nn.ReLU(),
            nn.Linear(last_layer_dim_pi, last_layer_dim_pi), nn.ReLU()
        )
        self.value_net = nn.Sequential(
            nn.Linear(feature_dim, last_layer_dim_vf), nn.ReLU(),
            nn.Linear(last_layer_dim_vf, last_layer_dim_vf), nn.ReLU()
        )

    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        return self.policy_net(features), self.value_net(features)

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        return self.policy_net(features)

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        return self.value_net(features)


class CustomActorCriticPolicy(ActorCriticPolicy):
    def __init__(
            self,
            observation_space: gym.spaces.Space,
            action_space: gym.spaces.Space,
            lr_schedule: Callable[[float], float],
            net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
            activation_fn: Type[nn.Module] = nn.Tanh,
            *args,
            **kwargs,
    ):
        super(CustomActorCriticPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            *args,
            **kwargs
        )
        self.ortho_init = False

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = CustomNetwork(self.features_dim)


class ResnetFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.Space, features_dim: int = 128*8*8):
        super().__init__(observation_space, features_dim)
        self.input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            self.conv2d(in_channels=self.input_channels, out_channels=64),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
        )
        self.cnn_2 = nn.Sequential(
            self.conv2d(in_channels=128, out_channels=128),
            nn.BatchNorm2d(num_features=128),
        )
        # self.cnn_3 = nn.Sequential(
        #     self.conv2d(in_channels=128, out_channels=128),
        #     nn.BatchNorm2d(momentum=0.9, num_features=128),
        # )
        self.residual_1 = self.create_residual_1(in_channels=128, out_channels=128)
        # self.residual_2 = self.create_residual_1(in_channels=128, out_channels=128)
        self.linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, features_dim),
            nn.ReLU()
        )

        self.cnn_64 = nn.Sequential(
            self.conv2d(in_channels=64, out_channels=64),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            self.conv2d(in_channels=64, out_channels=64),
            nn.BatchNorm2d(num_features=64),
        )

        self.cnn_64_128 = nn.Sequential(
            self.conv2d(in_channels=64, out_channels=128),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
        )

        self.cnn_128 = nn.Sequential(
            self.conv2d(in_channels=128, out_channels=128),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            self.conv2d(in_channels=128, out_channels=128),
            nn.BatchNorm2d(num_features=128),
        )

        self.cnn_128_256 = nn.Sequential(
            self.conv2d(in_channels=128, out_channels=256),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
        )

        self.cnn_256 = nn.Sequential(
            self.conv2d(in_channels=256, out_channels=256),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            self.conv2d(in_channels=256, out_channels=256),
            nn.BatchNorm2d(num_features=256),
        )
        
        

    @staticmethod
    def conv2d(in_channels: int, out_channels: int, kernel_size: Tuple[int, int] = (3, 3), padding: Tuple[int, int] = (1, 1)):
        return nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding)

    def create_residual_1(self, in_channels: int, out_channels: int, kernel_size: Tuple[int, int] = (3, 3)):
        return nn.Sequential(
            self.conv2d(in_channels, out_channels, kernel_size),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU()
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        y = self.cnn(observations)
        shortcut = y
        y = self.cnn_64(y)
        y += shortcut
        y = relu(y)
        shortcut = y
        y = self.cnn_64(y)
        y += shortcut
        y = relu(y)
        y = self.cnn_64_128(y)
        shortcut = y
        y = self.cnn_128(y)
        y += shortcut
        y = relu(y)
        shortcut = y
        y = self.cnn_128(y)
        y += shortcut
        y = relu(y)
        # y = self.cnn_128_256(y)
        # shortcut = y
        # y = self.cnn_256(y)
        # y += shortcut
        # y = relu(y)
        y = self.linear(y)
        # y = self.residual_1(y)
        # y = self.cnn_2(y)
        # y += shortcut
        # y = relu(y)
        # shortcut = y
        # y = self.residual_1(y)
        # y = self.cnn_2(y)
        # y += shortcut
        # y = relu(y)
        # y = self.linear(y)
        # shortcut = y
        # y = self.residual_1(y)
        # y = self.cnn_2(y)
        # y += shortcut
        # y = relu(y)
        # y = self.linear(y)
        return y
