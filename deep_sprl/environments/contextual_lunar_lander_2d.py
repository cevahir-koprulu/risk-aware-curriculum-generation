import numpy as np
from gym import Env
from .contextual_lunar_lander import ContextualLunarLander


class ContextualLunarLander2D(Env):

    def __init__(self, context=np.array([ContextualLunarLander.GRAVITY_Y, ContextualLunarLander.WIND_POWER])):
        self.env = ContextualLunarLander(context=context)
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.set_context(context)

    def set_context(self, context):
        self.env.set_context(context)

    def get_context(self):
        return self.env.get_context()

    context = property(get_context, set_context)

    def reset(self):
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)

    def render(self, mode='human'):
        self.env.render(mode)
