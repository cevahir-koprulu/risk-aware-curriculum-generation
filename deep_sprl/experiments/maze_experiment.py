import os
import gym
import torch
import numpy as np
from deep_sprl.experiments.abstract_experiment import AbstractExperiment, Learner
from deep_sprl.teachers.goal_gan import GoalGAN, GoalGANWrapper
from deep_sprl.teachers.alp_gmm import ALPGMM, ALPGMMWrapper
from deep_sprl.teachers.spl import SelfPacedTeacherV2, SelfPacedWrapper, CurrOT
from deep_sprl.teachers.dummy_teachers import UniformSampler
from deep_sprl.teachers.abstract_teacher import BaseWrapper
from deep_sprl.teachers.acl import ACL, ACLWrapper
from deep_sprl.teachers.plr import PLR, PLRWrapper, ValueFunction
from deep_sprl.teachers.vds import VDS, VDSWrapper
from deep_sprl.environments.maze import MazeEnv
from deep_sprl.teachers.util import Subsampler


class MazeSampler:
    def __init__(self):
        self.LOWER_CONTEXT_BOUNDS = np.array([-9., -9., 0.05])
        self.UPPER_CONTEXT_BOUNDS = np.array([9., 9., 0.05])

    def sample(self):
        sample = np.random.uniform(self.LOWER_CONTEXT_BOUNDS, self.UPPER_CONTEXT_BOUNDS)
        while not MazeEnv._is_feasible(sample):
            sample = np.random.uniform(self.LOWER_CONTEXT_BOUNDS, self.UPPER_CONTEXT_BOUNDS)
        return sample

    def save(self, path):
        pass

    def load(self, path):
        pass


class MazeExperiment(AbstractExperiment):
    INITIAL_MEAN = np.array([0., 0., 10])
    INITIAL_VARIANCE = np.diag(np.square([4, 4, 5]))

    TARGET_LOWER_CONTEXT_BOUNDS = np.array([-9., -9., 0.05])
    TARGET_UPPER_CONTEXT_BOUNDS = np.array([9., 9., 0.05])

    LOWER_CONTEXT_BOUNDS = np.array([-9., -9., 0.05])
    UPPER_CONTEXT_BOUNDS = np.array([9., 9., 18.])

    DISCOUNT_FACTOR = 0.995

    DELTA = 0.6
    METRIC_EPS = 1.5
    KL_EPS = 0.25
    EP_PER_UPDATE = 50

    # ACL Parameters [found after search over [0.05, 0.1, 0.2] x [0.01, 0.025, 0.05]]
    ACL_EPS = 0.2
    ACL_ETA = 0.05

    PLR_REPLAY_RATE = 0.55
    PLR_BUFFER_SIZE = 100
    PLR_BETA = 0.15
    PLR_RHO = 0.45

    VDS_NQ = 5
    VDS_LR = 5e-4
    VDS_EPOCHS = 10
    VDS_BATCHES = 80

    STEPS_PER_ITER = 10000
    LAM = 0.995
    ACHI_NET = dict(net_arch=[128, 128, 128], activation_fn=torch.nn.ReLU)

    AG_P_RAND = {Learner.PPO: None, Learner.SAC: 0.2}
    AG_FIT_RATE = {Learner.PPO: None, Learner.SAC: 200}
    AG_MAX_SIZE = {Learner.PPO: None, Learner.SAC: 500}

    GG_NOISE_LEVEL = {Learner.PPO: None, Learner.SAC: 0.1}
    GG_FIT_RATE = {Learner.PPO: None, Learner.SAC: 200}
    GG_P_OLD = {Learner.PPO: None, Learner.SAC: 0.2}

    def target_log_likelihood(self, cs):
        norm = np.prod(self.UPPER_CONTEXT_BOUNDS[:2] - self.LOWER_CONTEXT_BOUNDS[:2]) * (0.01 * 1 + 17.94 * 1e-4)
        return np.where(cs[:, -1] < 0.06, np.log(1 / norm) * np.ones(cs.shape[0]),
                        np.log(1e-4 / norm) * np.ones(cs.shape[0]))

    def target_sampler(self, n, rng=None):
        if rng is None:
            rng = np.random
        return rng.uniform(self.TARGET_LOWER_CONTEXT_BOUNDS, self.TARGET_UPPER_CONTEXT_BOUNDS, size=(n, 3))

    def __init__(self, base_log_dir, curriculum_name, learner_name, parameters, seed, device):
        super().__init__(base_log_dir, curriculum_name, learner_name, parameters, seed, device)
        self.eval_env, self.vec_eval_env = self.create_environment(evaluation=True)

    def create_environment(self, evaluation=False):
        env = gym.make("Maze-v1")
        if evaluation or self.curriculum.default():
            teacher = MazeSampler()
            env = BaseWrapper(env, teacher, self.DISCOUNT_FACTOR, context_visible=True, reward_from_info=True)
        elif self.curriculum.alp_gmm():
            teacher = ALPGMM(self.LOWER_CONTEXT_BOUNDS.copy(), self.UPPER_CONTEXT_BOUNDS.copy(), seed=self.seed,
                             fit_rate=self.AG_FIT_RATE[self.learner], random_task_ratio=self.AG_P_RAND[self.learner],
                             max_size=self.AG_MAX_SIZE[self.learner])
            env = ALPGMMWrapper(env, teacher, self.DISCOUNT_FACTOR, context_visible=True)
        elif self.curriculum.goal_gan():
            init_samples = np.random.uniform(self.LOWER_CONTEXT_BOUNDS, self.UPPER_CONTEXT_BOUNDS,
                                             size=(1000, 3))
            teacher = GoalGAN(self.LOWER_CONTEXT_BOUNDS.copy(), self.UPPER_CONTEXT_BOUNDS.copy(),
                              state_noise_level=self.GG_NOISE_LEVEL[self.learner], success_distance_threshold=0.01,
                              update_size=self.GG_FIT_RATE[self.learner], n_rollouts=4, goid_lb=0.25, goid_ub=0.75,
                              p_old=self.GG_P_OLD[self.learner], pretrain_samples=init_samples)
            env = GoalGANWrapper(env, teacher, self.DISCOUNT_FACTOR, context_visible=True)
        elif self.curriculum.self_paced() or self.curriculum.wasserstein():
            teacher = self.create_self_paced_teacher(with_callback=False)
            env = SelfPacedWrapper(env, teacher, self.DISCOUNT_FACTOR, context_visible=True, reward_from_info=True,
                                   use_undiscounted_reward=True, episodes_per_update=self.EP_PER_UPDATE)
        elif self.curriculum.acl():
            bins = 20
            teacher = ACL(bins * bins * bins, self.ACL_ETA, eps=self.ACL_EPS, norm_hist_len=2000)
            env = ACLWrapper(env, teacher, self.DISCOUNT_FACTOR, context_visible=True,
                             context_post_processing=Subsampler(self.LOWER_CONTEXT_BOUNDS.copy(),
                                                                self.UPPER_CONTEXT_BOUNDS.copy(),
                                                                [bins, bins, bins]))
        elif self.curriculum.plr():
            teacher = PLR(self.LOWER_CONTEXT_BOUNDS, self.UPPER_CONTEXT_BOUNDS, self.PLR_REPLAY_RATE,
                          self.PLR_BUFFER_SIZE, self.PLR_BETA, self.PLR_RHO)
            env = PLRWrapper(env, teacher, self.DISCOUNT_FACTOR, context_visible=True,
                             value_fn=ValueFunction(env.observation_space.shape[0] + self.LOWER_CONTEXT_BOUNDS.shape[0],
                                                    [128, 128, 128], torch.nn.ReLU(),
                                                    {"steps_per_iter": 2048, "noptepochs": 10,
                                                     "minibatches": 32, "lr": 3e-4}), lam=self.LAM)
        elif self.curriculum.vds():
            teacher = VDS(self.LOWER_CONTEXT_BOUNDS, self.UPPER_CONTEXT_BOUNDS, self.DISCOUNT_FACTOR, self.VDS_NQ,
                          q_train_config={"replay_size": 5 * self.STEPS_PER_ITER, "lr": self.VDS_LR,
                                          "n_epochs": self.VDS_EPOCHS, "batches_per_epoch": self.VDS_BATCHES,
                                          "steps_per_update": self.STEPS_PER_ITER})
            env = VDSWrapper(env, teacher, self.DISCOUNT_FACTOR, context_visible=True)
        elif self.curriculum.random():
            teacher = UniformSampler(self.LOWER_CONTEXT_BOUNDS.copy(), self.UPPER_CONTEXT_BOUNDS.copy())
            env = BaseWrapper(env, teacher, self.DISCOUNT_FACTOR, context_visible=True)
        else:
            raise RuntimeError("Invalid learning type")

        return env, env

    def create_learner_params(self):
        return dict(common=dict(gamma=self.DISCOUNT_FACTOR, seed=self.seed, verbose=0, policy_kwargs=self.ACHI_NET,
                                device=self.device),
                    ppo=dict(n_steps=int(self.STEPS_PER_ITER), batch_size=320, gae_lambda=self.LAM),
                    sac=dict(learning_rate=3e-4, buffer_size=200000, learning_starts=1000, batch_size=512,
                             train_freq=5, target_entropy="auto"))

    def create_experiment(self):
        timesteps = 400 * self.STEPS_PER_ITER

        env, vec_env = self.create_environment(evaluation=False)
        model, interface = self.learner.create_learner(vec_env, self.create_learner_params())

        if isinstance(env, PLRWrapper):
            env.learner = interface

        if isinstance(env, VDSWrapper):
            env.teacher.initialize_teacher(env, interface,
                                           lambda contexts: np.concatenate(
                                               (MazeEnv.sample_initial_state(contexts.shape[0]), contexts), axis=-1))

        callback_params = {"learner": interface, "env_wrapper": env, "save_interval": 10,
                           "step_divider": self.STEPS_PER_ITER}
        return model, timesteps, callback_params

    def create_self_paced_teacher(self, with_callback=False):
        bounds = (self.LOWER_CONTEXT_BOUNDS.copy(), self.UPPER_CONTEXT_BOUNDS.copy())
        if self.curriculum.self_paced():
            return SelfPacedTeacherV2(self.target_log_likelihood, self.target_sampler, self.INITIAL_MEAN.copy(),
                                      self.INITIAL_VARIANCE.copy(), bounds, self.DELTA, max_kl=self.KL_EPS,
                                      std_lower_bound=None, kl_threshold=None)
        elif self.curriculum.wasserstein():
            init_samples = np.random.uniform(np.array([-9., -9., 18.]), np.array([9., 9., 18.]), size=(500, 3))
            return CurrOT(bounds, init_samples, self.target_sampler, self.DELTA, self.METRIC_EPS, self.EP_PER_UPDATE,
                          wb_max_reuse=1)
        else:
            raise RuntimeError('Invalid self-paced curriculum type')

    def get_env_name(self):
        return "maze"

    def evaluate_learner(self, path, render=False):
        model_load_path = os.path.join(path, "model.zip")
        model = self.learner.load_for_evaluation(model_load_path, self.vec_eval_env, self.device)
        for i in range(0, 200):
            obs = self.vec_eval_env.reset()
            done = False
            while not done:
                action = model.step(obs, state=None, deterministic=False)
                obs, rewards, done, infos = self.vec_eval_env.step(action)
                if render:
                    self.vec_eval_env.render(mode="human")

        stats = self.eval_env.get_statistics()
        return stats[0]
