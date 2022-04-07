from gym import register

register(
    id="ContinuousCartPole-v0",
    entry_point="adversarial_gym.envs.cartpole:ContinuousCartPoleEnv"
)