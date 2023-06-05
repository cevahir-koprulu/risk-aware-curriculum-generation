import numpy as np
from gym import Env
import gym


class ContextualBipedalWalker(Env):
    def __init__(self, context=np.array([3., 6.])):
        self.env = gym.make('bipedal-walker-continuous-v0').unwrapped
        # self.env.my_init({'leg_size': 'default'})
        self.env.my_init({'leg_size': 'long'})
        self.context = context
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.env.set_environment(stump_height=self.context[0], obstacle_spacing=self.context[1])

    def reset(self):
        self.env.set_environment(stump_height=self.context[0], obstacle_spacing=self.context[1])
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)

    def render(self, mode='human'):
        self.env.render(mode)
