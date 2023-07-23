import numpy as np
from gym import Env
from .contextual_point_mass import ContextualPointMass


class ContextualPointMass3D(Env):

    def __init__(self, context=np.array([0., 2., 0.])):
        self.env = ContextualPointMass(context)
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

    def set_context(self, context):
        self.env.context = np.copy(context)

    def get_context(self):
        return self.env.context.copy()

    context = property(get_context, set_context)

    def reset(self):
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)

    def render(self, mode='human'):
        self.env.render(mode)
