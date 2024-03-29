import os
import gym
import torch.nn
import numpy as np
from deep_sprl.experiments.abstract_experiment import AbstractExperiment, Learner
from deep_sprl.teachers.alp_gmm import ALPGMM, ALPGMMWrapper
from deep_sprl.teachers.goal_gan import GoalGAN, GoalGANWrapper
from deep_sprl.teachers.spl import SelfPacedTeacherV2, SelfPacedWrapper, CurrOT
from deep_sprl.teachers.dummy_teachers import UniformSampler, DistributionSampler
from deep_sprl.teachers.dummy_wrapper import DummyWrapper
from deep_sprl.teachers.abstract_teacher import BaseWrapper
from deep_sprl.aux_teachers.cem import CEMGaussian, CEMCauchy
from stable_baselines3.common.vec_env import DummyVecEnv
from deep_sprl.teachers.acl import ACL, ACLWrapper
from deep_sprl.teachers.plr import PLR, PLRWrapper
from deep_sprl.teachers.vds import VDS, VDSWrapper
from deep_sprl.teachers.util import Subsampler
from scipy.stats import multivariate_normal

from deep_sprl.environments.contextual_point_mass import ContextualPointMass


def logsumexp(x):
    xmax = np.max(x)
    return np.log(np.sum(np.exp(x - xmax))) + xmax


CEM_AUX_TEACHERS = {
    "gaussian": CEMGaussian,
    "cauchy": CEMCauchy,
}


class PointMass2DExperiment(AbstractExperiment):
    TARGET_TYPE = "narrow"
    TARGET_MEAN = np.array([ContextualPointMass.ROOM_WIDTH*0.3125, 0.5])  # *0.375 for pos 3. with width 8.
    TARGET_VARIANCES = {
        "narrow": np.square(np.diag([1e-4, 1e-4])),
        "wide": np.square(np.diag([1., 1.])),  # .5 for width 8.
    }

    LOWER_CONTEXT_BOUNDS = np.array([-ContextualPointMass.ROOM_WIDTH/2, 0.5])
    UPPER_CONTEXT_BOUNDS = np.array([ContextualPointMass.ROOM_WIDTH/2, ContextualPointMass.ROOM_WIDTH])
    EXT_CONTEXT_BOUNDS = np.array([5., 5.])

    def target_log_likelihood(self, cs):
        return multivariate_normal.logpdf(cs, self.TARGET_MEAN, self.TARGET_VARIANCES[self.TARGET_TYPE])

    def target_sampler(self, n, rng=None):
        if rng is None:
            rng = np.random

        return rng.multivariate_normal(self.TARGET_MEAN, self.TARGET_VARIANCES[self.TARGET_TYPE], size=n)

    INITIAL_MEAN = np.array([0., 4.25])
    INITIAL_VARIANCE = np.diag(np.square([2, 1.875]))

    SP_DIST_TYPE = "gaussian"  # "cauchy"

    STD_LOWER_BOUND = np.array([0.2, 0.1875])
    KL_THRESHOLD = 8000.
    KL_EPS = 0.25  # 0.5
    DELTA = 4.0
    METRIC_EPS = 0.5
    EP_PER_UPDATE = 40  # 10

    # CEMGaussian
    EP_PER_AUX_UPDATE = 20
    RALPH_IN = 0.2  # initial reference alpha
    RALPH = 0.2  # final reference alpha
    RALPH_SCH = 20  # num steps for linear schedule to reach final reference alpha
    INT_ALPHA = 0.5  # internal alpha

    def risk_level_scheduler(self, update_no):
        # Set both to 1 for fixed ref_alpha
        risk_level_schedule_factor = 1

        alpha_cand = self.RALPH_IN - (self.RALPH_IN - self.RALPH) * update_no / (
                self.RALPH_SCH * risk_level_schedule_factor)
        return max(self.RALPH, alpha_cand)

    NUM_ITER = 400  # 200
    STEPS_PER_ITER = 8192  # 4096  # 1024  # 2048  # 4096
    DISCOUNT_FACTOR = 0.95
    LAM = 0.99

    # ACL Parameters [found after search over [0.05, 0.1, 0.2] x [0.01, 0.025, 0.05]]
    ACL_EPS = 0.2
    ACL_ETA = 0.025

    PLR_REPLAY_RATE = 0.85
    PLR_BUFFER_SIZE = 100
    PLR_BETA = 0.45
    PLR_RHO = 0.15

    VDS_NQ = 5
    VDS_LR = 1e-3
    VDS_EPOCHS = 3
    VDS_BATCHES = 20

    AG_P_RAND = {Learner.PPO: 0.1, Learner.SAC: None}
    AG_FIT_RATE = {Learner.PPO: 100, Learner.SAC: None}
    AG_MAX_SIZE = {Learner.PPO: 500, Learner.SAC: None}

    GG_NOISE_LEVEL = {Learner.PPO: 0.1, Learner.SAC: None}
    GG_FIT_RATE = {Learner.PPO: 200, Learner.SAC: None}
    GG_P_OLD = {Learner.PPO: 0.2, Learner.SAC: None}

    def __init__(self, base_log_dir, curriculum_name, learner_name, parameters, seed, device):
        super().__init__(base_log_dir, curriculum_name, learner_name, parameters, seed, device)
        self.eval_env, self.vec_eval_env = self.create_environment(evaluation=True)

    def create_environment(self, evaluation=False):
        env = gym.make("ContextualPointMass2D-v1")
        if evaluation or self.curriculum.default():
            teacher = DistributionSampler(self.target_sampler, self.LOWER_CONTEXT_BOUNDS, self.UPPER_CONTEXT_BOUNDS)
            env = BaseWrapper(env, teacher, self.DISCOUNT_FACTOR, context_visible=True)
        elif self.curriculum.default_with_cem():
            teacher = DistributionSampler(self.target_sampler, self.LOWER_CONTEXT_BOUNDS, self.UPPER_CONTEXT_BOUNDS)
            aux_teacher = self.create_cem_teacher(cem_type=self.DIST_TYPE,
                                                  dist_params=(self.TARGET_MEAN.copy(),
                                                               self.TARGET_VARIANCES[self.TARGET_TYPE].copy()))
            env = DummyWrapper(env, teacher, self.DISCOUNT_FACTOR,
                               episodes_per_update=self.EP_PER_UPDATE - self.EP_PER_AUX_UPDATE,
                               context_visible=True, episodes_per_aux_update=self.EP_PER_AUX_UPDATE,
                               aux_teacher=aux_teacher)
        elif self.curriculum.alp_gmm():
            teacher = ALPGMM(self.LOWER_CONTEXT_BOUNDS.copy(), self.UPPER_CONTEXT_BOUNDS.copy(), seed=self.seed,
                             fit_rate=self.AG_FIT_RATE[self.learner], random_task_ratio=self.AG_P_RAND[self.learner],
                             max_size=self.AG_MAX_SIZE[self.learner])
            env = ALPGMMWrapper(env, teacher, self.DISCOUNT_FACTOR, context_visible=True)
        elif self.curriculum.goal_gan():
            samples = np.random.uniform(self.LOWER_CONTEXT_BOUNDS, self.UPPER_CONTEXT_BOUNDS, size=(1000, 2))
            teacher = GoalGAN(self.LOWER_CONTEXT_BOUNDS.copy(), self.UPPER_CONTEXT_BOUNDS.copy(),
                              state_noise_level=self.GG_NOISE_LEVEL[self.learner], success_distance_threshold=0.01,
                              update_size=self.GG_FIT_RATE[self.learner], n_rollouts=2, goid_lb=0.25, goid_ub=0.75,
                              p_old=self.GG_P_OLD[self.learner], pretrain_samples=samples)
            env = GoalGANWrapper(env, teacher, self.DISCOUNT_FACTOR, context_visible=True)
        elif self.curriculum.self_paced() or self.curriculum.wasserstein():
            teacher = self.create_self_paced_teacher(with_callback=False)
            env = SelfPacedWrapper(env, teacher, self.DISCOUNT_FACTOR, episodes_per_update=self.EP_PER_UPDATE,
                                   context_visible=True)
        elif self.curriculum.self_paced_with_cem():
            teacher = self.create_self_paced_teacher(with_callback=False)
            aux_teacher = self.create_cem_teacher(cem_type=self.DIST_TYPE,
                                                  dist_params=(self.INITIAL_MEAN.copy(), self.INITIAL_VARIANCE.copy()))
            env = SelfPacedWrapper(env, teacher, self.DISCOUNT_FACTOR,
                                   episodes_per_update=self.EP_PER_UPDATE-self.EP_PER_AUX_UPDATE,
                                   context_visible=True, episodes_per_aux_update=self.EP_PER_AUX_UPDATE,
                                   aux_teacher=aux_teacher)
        elif self.curriculum.acl():
            bins = 50
            teacher = ACL(bins * bins, self.ACL_ETA, eps=self.ACL_EPS, norm_hist_len=2000)
            env = ACLWrapper(env, teacher, self.DISCOUNT_FACTOR, context_visible=True,
                             context_post_processing=Subsampler(self.LOWER_CONTEXT_BOUNDS.copy(),
                                                                self.UPPER_CONTEXT_BOUNDS.copy(),
                                                                [bins, bins]))
        elif self.curriculum.plr():
            teacher = PLR(self.LOWER_CONTEXT_BOUNDS, self.UPPER_CONTEXT_BOUNDS, self.PLR_REPLAY_RATE,
                          self.PLR_BUFFER_SIZE, self.PLR_BETA, self.PLR_RHO)
            env = PLRWrapper(env, teacher, self.DISCOUNT_FACTOR, context_visible=True)
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

        return env, DummyVecEnv([lambda: env])

    def create_learner_params(self):
        return dict(common=dict(gamma=self.DISCOUNT_FACTOR, seed=self.seed, verbose=0, device=self.device,
                                policy_kwargs=dict(net_arch=[128, 128, 128], activation_fn=torch.nn.Tanh)),
                    ppo=dict(n_steps=self.STEPS_PER_ITER, gae_lambda=self.LAM, batch_size=128),
                    sac=dict(learning_rate=3e-4, buffer_size=10000, learning_starts=500, batch_size=64,
                             train_freq=5, target_entropy="auto"))

    def create_experiment(self):
        timesteps = self.NUM_ITER * self.STEPS_PER_ITER

        env, vec_env = self.create_environment(evaluation=False)
        model, interface = self.learner.create_learner(vec_env, self.create_learner_params())

        if isinstance(env, PLRWrapper):
            env.learner = interface

        if isinstance(env, VDSWrapper):
            state_provider = lambda contexts: np.concatenate(
                [np.repeat(np.array([0., 0., -3., 0.])[None, :], contexts.shape[0], axis=0),
                 contexts], axis=-1)
            env.teacher.initialize_teacher(env, interface, state_provider)

        callback_params = {"learner": interface, "env_wrapper": env, "save_interval": 5,
                           "step_divider": self.STEPS_PER_ITER}
        return model, timesteps, callback_params

    def create_self_paced_teacher(self, with_callback=False):
        bounds = (self.LOWER_CONTEXT_BOUNDS.copy(), self.UPPER_CONTEXT_BOUNDS.copy())
        if self.curriculum.self_paced() or self.curriculum.self_paced_with_cem():
            return SelfPacedTeacherV2(self.target_log_likelihood, self.target_sampler, self.INITIAL_MEAN.copy(),
                                      self.INITIAL_VARIANCE.copy(), bounds, self.DELTA, max_kl=self.KL_EPS,
                                      std_lower_bound=self.STD_LOWER_BOUND.copy(), kl_threshold=self.KL_THRESHOLD,
                                      dist_type=self.SP_DIST_TYPE)
        else:
            init_samples = np.random.uniform(self.LOWER_CONTEXT_BOUNDS, self.UPPER_CONTEXT_BOUNDS, size=(200, 2))
            return CurrOT(bounds, init_samples, self.target_sampler, self.DELTA, self.METRIC_EPS, self.EP_PER_UPDATE,
                          wb_max_reuse=1)

    def create_cem_teacher(self, dist_params=None, cem_type="gaussian"):
        if dist_params is None:
            raise ValueError("dist_params should not be None!")
        if cem_type in CEM_AUX_TEACHERS:
            return CEM_AUX_TEACHERS[cem_type](dist_params=dist_params,
                                              target_log_likelihood=self.target_log_likelihood,
                                              risk_level_scheduler=self.risk_level_scheduler,
                                              data_bounds=(self.LOWER_CONTEXT_BOUNDS.copy(),
                                                           self.UPPER_CONTEXT_BOUNDS.copy()),
                                              batch_size=self.EP_PER_UPDATE,
                                              n_orig_per_batch=self.EP_PER_AUX_UPDATE,
                                              ref_alpha=self.RALPH_IN, internal_alpha=self.INT_ALPHA,
                                              ref_mode='train', force_min_samples=True, w_clip=5)
        else:
            raise ValueError(f"Given CEM type, {cem_type}, is not in {list(CEM_AUX_TEACHERS.keys())}.")

    def get_env_name(self):
        return "point_mass_2d"

    def evaluate_learner(self, path, eval_type=""):
        num_context = None
        num_run = 1

        model_load_path = os.path.join(path, "model.zip")
        model = self.learner.load_for_evaluation(model_load_path, self.vec_eval_env, self.device)
        eval_path = f"{os.getcwd()}/eval_contexts/{self.get_env_name()}_eval{eval_type}_contexts.npy"
        if os.path.exists(eval_path):
            eval_contexts = np.load(eval_path)
            num_context = eval_contexts.shape[0]
        else:
            raise ValueError(f"Invalid evaluation type: {eval_type}")

        num_succ_eps_per_c = np.zeros((num_context, 1))
        for i in range(num_context):
            context = eval_contexts[i, :]
            for j in range(num_run):
                self.eval_env.set_context(context)
                obs = self.vec_eval_env.reset()
                done = False
                success = []
                while not done:
                    action = model.step(obs, state=None, deterministic=False)
                    obs, rewards, done, infos = self.vec_eval_env.step(action)
                    success.append(infos[0]["success"]*1)
                if any(success):
                    num_succ_eps_per_c[i, 0] += 1. / num_run
        print(f"Successful Eps: {100 * np.mean(num_succ_eps_per_c)}%")

        disc_rewards = self.eval_env.get_reward_buffer()
        ave_disc_rewards = []
        for j in range(num_context):
            ave_disc_rewards.append(np.average(disc_rewards[j * num_run:(j + 1) * num_run]))
        return ave_disc_rewards, eval_contexts[:num_context, :], \
               np.exp(self.target_log_likelihood(eval_contexts[:num_context, :])), num_succ_eps_per_c
