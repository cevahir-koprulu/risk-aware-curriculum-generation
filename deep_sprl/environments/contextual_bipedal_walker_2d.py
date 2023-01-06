import numpy as np
from gym import Env
from .contextual_bipedal_walker import ContextualBipedalWalker


class ContextualBipedalWalker2D(Env):

    def __init__(self, context=np.array([3., 6.])):
        self.env = ContextualBipedalWalker(context)
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

    def set_context(self, context):
        self.env.context = context

    def get_context(self):
        return self.env.context.copy()

    context = property(get_context, set_context)

    def reset(self):
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)

    def render(self, mode='human'):
        self.env.render(mode)
