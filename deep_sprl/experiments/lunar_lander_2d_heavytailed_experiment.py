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
    lb = LunarLander2DHeavyTailedExperiment.LOWER_CONTEXT_BOUNDS - LunarLander2DHeavyTailedExperiment.EXT_CONTEXT_BOUNDS
    ub = LunarLander2DHeavyTailedExperiment.UPPER_CONTEXT_BOUNDS + LunarLander2DHeavyTailedExperiment.EXT_CONTEXT_BOUNDS
    def _kl_inner(x):
        return torch.exp(p.log_prob(x))*(p.log_prob(x) - q.log_prob(x))

    mc = MonteCarlo()
    kl = mc.integrate(fn=_kl_inner,
                      dim=p.loc.shape[0],
                      integration_domain=[[lb[0], ub[0]], [lb[1], ub[1]]],
                      N=5000
                      )
    return kl

class LunarLander2DHeavyTailedExperiment(AbstractExperiment):
    TARGET_TYPE = "wide"
    TARGET_MEAN = np.array([-7.0, 5.])
    TARGET_VARIANCES = {
        "narrow": np.square(np.diag([1e-4, 1e-4])),
        "wide": np.square(np.diag([1., 1.])),
    }

    LOWER_CONTEXT_BOUNDS = np.array([-12., 0.])
    UPPER_CONTEXT_BOUNDS = np.array([-0.01, 10.])

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

    INIT_VAR = 0.5
    INITIAL_MEAN = np.array([-3.7, 0.])
    # INITIAL_VARIANCE = np.diag(np.square([.5, .5]))

    DIST_TYPE = "cauchy"  # "gaussian"

    STD_LOWER_BOUND = np.array([0.1, 0.1])
    KL_THRESHOLD = 8000.
    KL_EPS = 0.025
    DELTA = -100.
    METRIC_EPS = 0.5
    EP_PER_UPDATE = 40 

    # CEM
    EP_PER_AUX_UPDATE = 10 
    RALPH_IN = 1.0  # initial reference alpha
    RALPH = 0.2  # final reference alpha
    RALPH_SCH = 80 # 40 # 20  # num steps for linear schedule to reach final reference alpha
    INT_ALPHA = 0.5  # internal alpha

    def risk_level_scheduler(self, update_no):
        # Set both to 1 for fixed RALPH
        risk_level_schedule_factor = 1

        alpha_cand = self.RALPH_IN - (self.RALPH_IN - self.RALPH) * update_no / (
                self.RALPH_SCH * risk_level_schedule_factor)
        return max(self.RALPH, alpha_cand)

    NUM_ITER = 250 
    STEPS_PER_ITER = 10240 
    DISCOUNT_FACTOR = 0.99 
    LAM = 0.99 

    # ACL Parameters [found after search over [0.05, 0.1, 0.2] x [0.01, 0.025, 0.05]]
    ACL_EPS = 0.2
    ACL_ETA = 0.025

    PLR_REPLAY_RATE = 0.85
    PLR_BUFFER_SIZE = 100
    PLR_BETA = 0.15
    PLR_RHO = 0.45

    VDS_NQ = 5
    VDS_LR = 1e-3
    VDS_EPOCHS = 3
    VDS_BATCHES = 40

    AG_P_RAND = {Learner.PPO: 0.1, Learner.SAC: 0.1}
    AG_FIT_RATE = {Learner.PPO: 100, Learner.SAC: 50}
    AG_MAX_SIZE = {Learner.PPO: 500, Learner.SAC: 500}

    GG_NOISE_LEVEL = {Learner.PPO: 0.1, Learner.SAC: 0.05}
    GG_FIT_RATE = {Learner.PPO: 200, Learner.SAC: 200}
    GG_P_OLD = {Learner.PPO: 0.2, Learner.SAC: 0.1}

    def __init__(self, base_log_dir, curriculum_name, learner_name, parameters, seed, device):
        super().__init__(base_log_dir, curriculum_name, learner_name, parameters, seed, device)
        self.eval_env, self.vec_eval_env = self.create_environment(evaluation=True)

    def create_environment(self, evaluation=False):
        env = gym.make("ContextualLunarLander2D-v1")
        print(f"EP_PER_AUX_UPDATE: {self.EP_PER_AUX_UPDATE}")
        if evaluation:
            teacher = DistributionSampler(self.target_sampler, self.LOWER_CONTEXT_BOUNDS, self.UPPER_CONTEXT_BOUNDS)
            env = BaseWrapper(env, teacher, discount_factor=self.DISCOUNT_FACTOR, context_visible=True)
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
            samples = np.random.uniform(self.LOWER_CONTEXT_BOUNDS, self.UPPER_CONTEXT_BOUNDS, size=(1000, self.LOWER_CONTEXT_BOUNDS.shape[0]))
            teacher = GoalGAN(self.LOWER_CONTEXT_BOUNDS.copy(), self.UPPER_CONTEXT_BOUNDS.copy(),
                              state_noise_level=self.GG_NOISE_LEVEL[self.learner], success_distance_threshold=0.01,
                              update_size=self.GG_FIT_RATE[self.learner], n_rollouts=2, goid_lb=0.25, goid_ub=0.75,
                              p_old=self.GG_P_OLD[self.learner], pretrain_samples=samples)
            env = GoalGANWrapper(env, teacher, self.DISCOUNT_FACTOR, context_visible=True)
        elif self.curriculum.self_paced() or self.curriculum.wasserstein():
            teacher = self.create_self_paced_teacher(with_callback=False)
            env = SelfPacedWrapper(env, teacher, discount_factor=self.DISCOUNT_FACTOR, episodes_per_update=self.EP_PER_UPDATE,
                                   context_visible=True)
        elif self.curriculum.self_paced_with_cem():
            teacher = self.create_self_paced_teacher(with_callback=False)
            aux_teacher = self.create_cem_teacher(cem_type=self.DIST_TYPE,
                                                  dist_params=(self.INITIAL_MEAN.copy(), np.diag(np.square([self.INIT_VAR, self.INIT_VAR]))))
            env = SelfPacedWrapper(env, teacher, discount_factor=self.DISCOUNT_FACTOR,
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
                                policy_kwargs=dict(net_arch=[64, 64], 
                                activation_fn=torch.nn.Tanh,
                                )),
                    ppo=dict(learning_rate=3e-4,
                                n_steps=10240, 
                                n_epochs=4, 
                                gae_lambda=self.LAM, 
                                batch_size=64,
                                ent_coef=0.0,
                    ),
                    sac=dict(learning_rate=3e-4, buffer_size=200000, learning_starts=2000, batch_size=64,
                               train_freq=5, target_entropy="auto")
                    )

    def create_experiment(self):
        timesteps = self.NUM_ITER * self.STEPS_PER_ITER

        env, vec_env = self.create_environment(evaluation=False)
        model, interface = self.learner.create_learner(vec_env, self.create_learner_params())

        if isinstance(env, PLRWrapper):
            env.learner = interface

        if isinstance(env, VDSWrapper):
            print(env.reset()[None, :])
            state_provider = lambda contexts: np.concatenate(
                [np.repeat(env.reset()[None, :-self.LOWER_CONTEXT_BOUNDS.shape[0]], contexts.shape[0], axis=0),
                 contexts], axis=-1)
            env.teacher.initialize_teacher(env, interface, state_provider)

        callback_params = {"learner": interface, "env_wrapper": env, "save_interval": 10,  # 5,
                           "step_divider": self.STEPS_PER_ITER}
        return model, timesteps, callback_params

    def create_self_paced_teacher(self, with_callback=False):
        bounds = (self.LOWER_CONTEXT_BOUNDS.copy(), self.UPPER_CONTEXT_BOUNDS.copy())
        if self.curriculum.self_paced() or self.curriculum.self_paced_with_cem():
            return SelfPacedTeacherV2(self.target_log_likelihood, self.target_sampler, self.INITIAL_MEAN.copy(),
                                      np.diag(np.square([self.INIT_VAR, self.INIT_VAR])), bounds, self.DELTA, max_kl=self.KL_EPS,
                                      std_lower_bound=self.STD_LOWER_BOUND.copy(), kl_threshold=self.KL_THRESHOLD,
                                      dist_type=self.DIST_TYPE, ext_bounds=self.EXT_CONTEXT_BOUNDS)
        else:
            init_samples = np.random.uniform(self.LOWER_CONTEXT_BOUNDS, self.UPPER_CONTEXT_BOUNDS, size=(200, self.LOWER_CONTEXT_BOUNDS.shape[0]))
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
        return f"lunar_lander_2d_heavytailed_{self.TARGET_TYPE}"

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
            for j in range(num_run):
                self.eval_env.set_context(context)
                obs = self.vec_eval_env.reset()
                done = False
                success = []
                r = 0.
                while not done:
                    action = model.step(obs, state=None, deterministic=False)
                    obs, rewards, done, infos = self.vec_eval_env.step(action)
                    success.append(infos[0]["success"]*1)
                    r += rewards[0]
                if any(success):
                    num_succ_eps_per_c[i, 0] += 1. / num_run
        print(f"Successful Eps: {100 * np.mean(num_succ_eps_per_c)}%")

        disc_rewards = self.eval_env.get_reward_buffer()
        ave_disc_rewards = []
        for j in range(num_context):
            ave_disc_rewards.append(np.average(disc_rewards[j * num_run:(j + 1) * num_run]))
        return ave_disc_rewards, eval_contexts[:num_context, :], \
               np.exp(self.target_log_likelihood(eval_contexts[:num_context, :])), num_succ_eps_per_c

