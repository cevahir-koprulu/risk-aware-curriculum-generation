import numpy as np
from gym import Env
import gym
from deep_sprl.environments.lunar_lander import LunarLander

class ContextualLunarLander(Env):

    GRAVITY_X = 0.
    GRAVITY_Y = -10.
    WIND_POWER = 0.

    def __init__(self, context=np.array([GRAVITY_Y, WIND_POWER])):
        self.env = LunarLander()
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.set_context(context)
        self.step_no = 0
        self.total_reward = 0.

    def set_context(self, context):
        # self.env = LunarLander()
        self.env.gravity = context[0]
        self.env.wind_power = context[1]

    def get_context(self):
        return np.array([self.env.world.gravity[1], self.env.wind_power])

    context = property(get_context, set_context)

    def reset(self):
        self.step_no = 0
        self.total_reward = 0.
        return self.env.reset()

    def step(self, action):
        self.step_no += 1
        state, reward, done, info = self.env.step(action)
        self.total_reward += reward
        info = {"success": False}
        if not self.env.lander.awake:
            info["success"] = True
        return state, reward, done, info

    def render(self, mode='human'):
        self.env.render(mode)
