import gym
from stable_baselines3 import PPO


def test_stable_baseline3():
    print("Testing stable baseline 3: PPO Algorithm")
    env = gym.make('CartPole-v1')

    model = PPO('MlpPolicy', env, verbose=1)
    model.learn(total_timesteps=10000)

    obs = env.reset()
    print("Using RL Agent")
    for i in range(1000):
        action, _state = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        env.render()
        if done:
            obs = env.reset()


if __name__ == '__main__':
    test_stable_baseline3()
