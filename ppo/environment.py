from typing import Tuple, Callable, Any

import numpy as np
import gym


class Environment:
    def __init__(self, state_dim: int, action_dim: int):
        self.state_dim = state_dim
        self.action_dim = action_dim

    def get_state_dim(self) -> int:
        return self.state_dim

    def get_action_dim(self) -> int:
        return self.action_dim

    def reset(self) -> np.ndarray:
        raise NotImplementedError("Not implemented")

    def step(self, action) -> Tuple[np.ndarray, float, bool]:
        raise NotImplementedError("Not implemented")

    def render(self):
        raise NotImplementedError("Not implemented")


class MountainEnv(Environment):
    def __init__(self):
        env = gym.make('MountainCar-v0')
        super(MountainEnv, self).__init__(sum(env.observation_space.shape), 3)
        self.env = env

    def reset(self) -> np.ndarray:
        state, _ = self.env.reset()
        return state

    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        a = self.env.step(action)
        state, reward, done, _, _ = self.env.step(action)
        return state, reward, done

    def render(self):
        self.env.render()


class CartPoleEnv(Environment):
    def __init__(self):
        env = gym.make('CartPole-v1').env
        super(CartPoleEnv, self).__init__(sum(env.observation_space.shape), 2)
        print(env.observation_space)
        print(env.action_space)
        self.env = env

    def reset(self) -> np.ndarray:
        state, _ = self.env.reset()
        return state

    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        state, reward, done, _, _ = self.env.step(action)
        return state, reward, done

    def render(self):
        self.env.render()


if __name__ == "__main__":
    pass