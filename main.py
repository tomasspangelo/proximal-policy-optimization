from ppo.learner import Learner
from ppo.environment import MountainEnv
import os

"""
Main file showcasing how to use the PPO implementation.
"""

if __name__ == "__main__":
    env = MountainEnv()
    learner = Learner(env.state_dim, env.action_dim, 0.0003, 0.001, 0.2, size=1000)

    print(os.getcwd())
    log_path = os.path.join(os.getcwd(), "./LOG3")
    net_path = os.path.join(os.getcwd(), "./testing")
    learner.learn(env, 80, 0.99, 0.97, 200, 1000000, 3, log_path, net_path, 100, update_timestep=1000)
    learner.test(MountainEnv(), 200)
    learner.test(MountainEnv(), 200)
    learner.test(MountainEnv(), 200)
    learner.test(MountainEnv(), 200)
    learner.test(MountainEnv(), 200)
    learner.test(MountainEnv(), 200)
    learner.test(MountainEnv(), 200)
    learner.test(MountainEnv(), 200)
    learner.test(MountainEnv(), 200)
    learner.test(MountainEnv(), 200)
    learner.test(MountainEnv(), 200)
