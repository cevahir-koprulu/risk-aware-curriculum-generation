import os
import gym
import torch.nn
import numpy as np
import scipy
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

from torchquad import MonteCarlo, set_up_backend
from pyro.distributions import MultivariateStudentT
from torch.distributions.kl import register_kl
from scipy.special import gamma

CEM_AUX_TEACHERS = {
    "gaussian": CEMGaussian,
    "cauchy": CEMCauchy,
}

# set_up_backend("torch", data_type="float32", torch_enable_cuda=False)


@register_kl(MultivariateStudentT, MultivariateStudentT)
def _kl_multivariatecauchy_multivariatecauchy(p, q):
    lb = BipedalWalker2DHeavyTailedExperiment.LOWER_CONTEXT_BOUNDS - BipedalWalker2DHeavyTailedExperiment.EXT_CONTEXT_BOUNDS
    ub = BipedalWalker2DHeavyTailedExperiment.UPPER_CONTEXT_BOUNDS + BipedalWalker2DHeavyTailedExperiment.EXT_CONTEXT_BOUNDS
    def _kl_inner(x):
        return torch.exp(p.log_prob(x))*(p.log_prob(x) - q.log_prob(x))

    mc = MonteCarlo()
    kl = mc.integrate(fn=_kl_inner,
                      dim=p.loc.shape[0],
                      integration_domain=[[lb[0], ub[0]], [lb[1], ub[1]]],
                      N=5000
                      )
    return kl

class BipedalWalker2DHeavyTailedExperiment(AbstractExperiment):
    TARGET_TYPE = "wide"
    TARGET_MEAN = np.array([1.5, 3.0])
    # TARGET_MEAN = np.array([1.5, 4.5])
    # TARGET_MEAN = np.array([1.2, 4.5])
    TARGET_VARIANCES = {
        "narrow": np.square(np.diag([1e-4, 1e-4])),
        "wide": np.square(np.diag([.3, .5])),
        # "wide": np.square(np.diag([.75, .75])),
        # "wide": np.square(np.diag([.6, .75])),
    }

    # Also used in kl div fcn, but not from self
    LOWER_CONTEXT_BOUNDS = np.array([0., 0.])
    UPPER_CONTEXT_BOUNDS = np.array([3., 6.])
    EXT_CONTEXT_BOUNDS = np.array([0., 0.])

    def target_log_likelihood(self, cs):
        # Student's t distribution with DoF 1 is equivalent to a Cauchy distribution
        p = scipy.stats.multivariate_t.logpdf(cs, loc=self.TARGET_MEAN, shape=self.TARGET_VARIANCES[self.TARGET_TYPE],
                                              df=1)
        return p

    def target_sampler(self, n, rng=None):
        if rng is None:
            rng = scipy.stats

        # Student's t distribution with DoF 1 is equivalent to a Cauchy distribution
        s = rng.multivariate_t.rvs(loc=self.TARGET_MEAN, shape=self.TARGET_VARIANCES[self.TARGET_TYPE], df=1, size=n)
        if n == 1:
            s = np.array([s])
        return s

    INITIAL_MEAN = np.array([0., 6.])
    INITIAL_VARIANCE = np.diag(np.square([.1, .1]))

    DIST_TYPE = "cauchy"  # "gaussian"

    STD_LOWER_BOUND = np.array([0.1, 0.1])
    KL_THRESHOLD = 8000.
    KL_EPS = 1.5 # 1.0  # 0.5  # 0.25
    DELTA = 120. # 180.  # 5.  # 15.  # 4.0  ### Mastered means return >= 230 (page 5 of teachDeepRL CoRL 2019)
    METRIC_EPS = 0.5
    EP_PER_UPDATE = 50

    # CEM
    EP_PER_AUX_UPDATE = 10 # 15  # 25  # 35
    RALPH_IN = 1.0  # initial reference alpha
    RALPH = 0.4 # 0.2  # final reference alpha
    RALPH_SCH = 50  # num steps for linear schedule to reach final reference alpha
    INT_ALPHA = 0.5  # internal alpha

    def risk_level_scheduler(self, update_no):
        # Set both to 1 for fixed RALPH
        risk_level_schedule_factor = 1

        alpha_cand = self.RALPH_IN - (self.RALPH_IN - self.RALPH) * update_no / (
                self.RALPH_SCH * risk_level_schedule_factor)
        return max(self.RALPH, alpha_cand)

    NUM_ITER = 300  # 200
    STEPS_PER_ITER = 100000
    DISCOUNT_FACTOR = 0.99  # 0.95
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

    AG_P_RAND = {Learner.PPO: 0.1, Learner.SAC: 0.05}
    AG_FIT_RATE = {Learner.PPO: 100, Learner.SAC: 150}
    AG_MAX_SIZE = {Learner.PPO: 500, Learner.SAC: 500}

    GG_NOISE_LEVEL = {Learner.PPO: 0.1, Learner.SAC: 0.1}
    GG_FIT_RATE = {Learner.PPO: 200, Learner.SAC: 100}
    GG_P_OLD = {Learner.PPO: 0.2, Learner.SAC: 0.1}

    def __init__(self, base_log_dir, curriculum_name, learner_name, parameters, seed, device):
        super().__init__(base_log_dir, curriculum_name, learner_name, parameters, seed, device)
        self.eval_env, self.vec_eval_env = self.create_environment(evaluation=True)

    def create_environment(self, evaluation=False):
        env = gym.make("ContextualBipedalWalker2D-v1")
        print(f"EP_PER_AUX_UPDATE: {self.EP_PER_AUX_UPDATE}")
        if evaluation:
            teacher = DistributionSampler(self.target_sampler, self.LOWER_CONTEXT_BOUNDS, self.UPPER_CONTEXT_BOUNDS)
            env = BaseWrapper(env, teacher, discount_factor=1.0, context_visible=True)
        elif self.curriculum.default():
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
            env = SelfPacedWrapper(env, teacher, discount_factor=1.0, episodes_per_update=self.EP_PER_UPDATE,
                                   context_visible=True)
        elif self.curriculum.self_paced_with_cem():
            teacher = self.create_self_paced_teacher(with_callback=False)
            aux_teacher = self.create_cem_teacher(cem_type=self.DIST_TYPE,
                                                  dist_params=(self.INITIAL_MEAN.copy(), self.INITIAL_VARIANCE.copy()))
            env = SelfPacedWrapper(env, teacher, discount_factor=1.0,
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
                                          "steps_per_update": self.STEPS_PER_ITER},
                          device=self.device)
            env = VDSWrapper(env, teacher, self.DISCOUNT_FACTOR, context_visible=True)
        elif self.curriculum.random():
            teacher = UniformSampler(self.LOWER_CONTEXT_BOUNDS.copy(), self.UPPER_CONTEXT_BOUNDS.copy())
            env = BaseWrapper(env, teacher, self.DISCOUNT_FACTOR, context_visible=True)
        else:
            raise RuntimeError("Invalid learning type")

        return env, DummyVecEnv([lambda: env])

    def create_learner_params(self):
        return dict(common=dict(gamma=self.DISCOUNT_FACTOR, seed=self.seed, verbose=0, device=self.device,
                                policy_kwargs=dict(net_arch=[400, 300], activation_fn=torch.nn.ReLU)),
                    ppo=dict(n_steps=self.STEPS_PER_ITER, gae_lambda=self.LAM, batch_size=128),
                    # sac=dict(learning_rate=1e-3, buffer_size=2000000, learning_starts=10000, batch_size=100,
                    #          train_freq=10, target_entropy="auto", ent_coef=0.2)
                    sac=dict(learning_rate=1e-3, buffer_size=2000000, learning_starts=1000, batch_size=1000,
                               train_freq=10, target_entropy="auto", ent_coef=0.005)
                    )



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

        callback_params = {"learner": interface, "env_wrapper": env, "save_interval": 10,  # 5,
                           "step_divider": self.STEPS_PER_ITER}
        return model, timesteps, callback_params

    def create_self_paced_teacher(self, with_callback=False):
        bounds = (self.LOWER_CONTEXT_BOUNDS.copy(), self.UPPER_CONTEXT_BOUNDS.copy())
        if self.curriculum.self_paced() or self.curriculum.self_paced_with_cem():
            return SelfPacedTeacherV2(self.target_log_likelihood, self.target_sampler, self.INITIAL_MEAN.copy(),
                                      self.INITIAL_VARIANCE.copy(), bounds, self.DELTA, max_kl=self.KL_EPS,
                                      std_lower_bound=self.STD_LOWER_BOUND.copy(), kl_threshold=self.KL_THRESHOLD,
                                      dist_type=self.DIST_TYPE, ext_bounds=self.EXT_CONTEXT_BOUNDS)
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
        return f"bipedal_walker_2d_heavytailed_{self.TARGET_TYPE}"

    def evaluate_learner(self, path, eval_type=""):
        num_context = None
        num_run = 1

        model_load_path = os.path.join(path, "model.zip")
        model = self.learner.load_for_evaluation(model_load_path, self.vec_eval_env, self.device)
        eval_path = f"{os.getcwd()}/eval_contexts/{self.get_env_name()}_eval{eval_type}_contexts.npy"
        if os.path.exists(eval_path):
            eval_contexts = np.load(eval_path)
            if num_context is None:
                num_context = eval_contexts.shape[0]
        else:
            raise ValueError(f"Invalid evaluation type: {eval_type}")

        num_succ_eps_per_c = np.zeros((num_context, 1))
        for i in range(num_context):
            context = eval_contexts[i, :]
            # context = np.array([0., 6.])
            for j in range(num_run):
                self.eval_env.set_context(context)
                obs = self.vec_eval_env.reset()
                done = False
                success = []
                r = 0.
                while not done:
                    action = model.step(obs, state=None, deterministic=False)
                    obs, rewards, done, infos = self.vec_eval_env.step(action)
                    # self.eval_env.render()
                    # success.append(infos[0]["success"]*1)
                    success.append(False)
                    r += rewards[0]
                if any(success):
                    num_succ_eps_per_c[i, 0] += 1. / num_run
                # print(f"return: {r}")
        print(f"Successful Eps: {100 * np.mean(num_succ_eps_per_c)}%")

        disc_rewards = self.eval_env.get_reward_buffer()
        ave_disc_rewards = []
        for j in range(num_context):
            ave_disc_rewards.append(np.average(disc_rewards[j * num_run:(j + 1) * num_run]))
        return ave_disc_rewards, eval_contexts[:num_context, :], \
               np.exp(self.target_log_likelihood(eval_contexts[:num_context, :])), num_succ_eps_per_c
