from adversarial_gym.distributions import *
import gym
from gym import Wrapper, Env


class NoiseWrapper(Wrapper):
    def __init__(self, env: Env, noise_type: Distribution, noise_kwargs: dict) -> None:
        super().__init__(env)
        noise_types = {
            "normal": NormalDistribution,
            "uniform": UniformDistribution,
            "discrete_uniform": DiscreteUniformDistribution,
            "beta": BetaDistribution,
            "poisson": PoissonDistribution
        }
        assert noise_type in list(noise_types.keys()), f"Noise type {noise_type} is not valid! Choose from:\n {noise_types.keys()}"
        self.distribution = noise_types[noise_type](**noise_kwargs)
        self.obs_dim = self.observation_space.shape
        if type(self.action_space) == gym.spaces.Discrete:
            self.action_dim = (1,)
        elif type(self.action_space) == gym.spaces.Box:
            self.action_dim = self.action_space.shape

    def reset(self):
        self.distribution.reset()
        obs = super().reset()
        return obs


class ObsNoiseWrapper(NoiseWrapper):
    def __init__(self, env: Env, noise_type: Distribution, noise_kwargs: dict) -> None:
        super().__init__(env, noise_type, noise_kwargs)

    def reset(self):
        obs = super().reset()
        noise = self.distribution.sample(self.obs_dim)
        noisy_obs = obs + noise
        return noisy_obs

    def step(self, a):
        obs, rew, done, infos = super().step(a)
        noise = self.distribution.sample(self.obs_dim)
        noisy_obs = obs + noise
        step_info = {
            "obs": obs,
            "noisy_obs": noisy_obs,
            "noise": noise
        }
        infos.update(step_info)
        return noisy_obs, rew, done, infos


class ActionNoiseWrapper(NoiseWrapper):
    def __init__(self, env: Env, noise_type: Distribution, noise_kwargs: dict) -> None:
        super().__init__(env, noise_type, noise_kwargs)

    def reset(self):
        obs = super().reset()
        return obs

    def step(self, a):
        noise = self.distribution.sample(self.action_dim)
        noisy_action = a + noise
        obs, rew, done, infos = super().step(noisy_action)
        step_info = {
            "action": a,
            "noisy_action": noisy_action,
            "noise": noise
        }
        infos.update(step_info)
        return obs, rew, done, infos


class RewardNoiseWrapper(NoiseWrapper):
    def __init__(self, env: Env, noise_type: Distribution, noise_kwargs: dict) -> None:
        super().__init__(env, noise_type, noise_kwargs)

    def reset(self):
        obs = super().reset()
        return obs

    def step(self, a):
        obs, rew, done, infos = super().step(a)
        noise = self.distribution.sample((1,))
        noisy_rew = rew + noise
        step_info = {
            "reward": rew,
            "noisy_reward": noisy_rew,
            "noise": noise
        }
        infos.update(step_info)
        return obs, rew, done, infos
