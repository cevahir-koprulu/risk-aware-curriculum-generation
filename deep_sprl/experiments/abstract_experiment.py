import os
import time
import torch
import pickle
import numpy as np
from abc import ABC, abstractmethod
from enum import Enum
from deep_sprl.util.parameter_parser import create_override_appendix
from deep_sprl.teachers.spl import SelfPacedTeacherV2
from deep_sprl.aux_teachers.cem import CEM

from stable_baselines3.sac import SAC
from stable_baselines3.ppo import PPO
from stable_baselines3.common.vec_env import VecEnv, DummyVecEnv
from stable_baselines3.ppo.policies import MlpPolicy as PPOMlpPolicy
from stable_baselines3.sac.policies import MlpPolicy as SACMlpPolicy


class CurriculumType(Enum):
    GoalGAN = 1
    ALPGMM = 2
    SelfPaced = 3
    Default = 4
    Random = 5
    Wasserstein = 7
    ACL = 8
    PLR = 9
    VDS = 10
    SelfPacedWithCEM = 11
    DefaultWithCEM = 12

    def __str__(self):
        if self.goal_gan():
            return "goal_gan"
        elif self.alp_gmm():
            return "alp_gmm"
        elif self.self_paced():
            return "self_paced"
        elif self.wasserstein():
            return "wasserstein"
        elif self.default():
            return "default"
        elif self.acl():
            return "acl"
        elif self.plr():
            return "plr"
        elif self.vds():
            return "vds"
        elif self.self_paced_with_cem():
            return "self_paced_with_cem"
        elif self.default_with_cem():
            return "default_with_cem"
        else:
            return "random"

    def self_paced(self):
        return self.value == CurriculumType.SelfPaced.value

    def self_paced_with_cem(self):
        return self.value == CurriculumType.SelfPacedWithCEM.value

    def goal_gan(self):
        return self.value == CurriculumType.GoalGAN.value

    def alp_gmm(self):
        return self.value == CurriculumType.ALPGMM.value

    def default(self):
        return self.value == CurriculumType.Default.value

    def default_with_cem(self):
        return self.value == CurriculumType.DefaultWithCEM.value

    def wasserstein(self):
        return self.value == CurriculumType.Wasserstein.value

    def random(self):
        return self.value == CurriculumType.Random.value

    def acl(self):
        return self.value == CurriculumType.ACL.value

    def plr(self):
        return self.value == CurriculumType.PLR.value

    def vds(self):
        return self.value == CurriculumType.VDS.value

    @staticmethod
    def from_string(string):
        if string == str(CurriculumType.GoalGAN):
            return CurriculumType.GoalGAN
        elif string == str(CurriculumType.ALPGMM):
            return CurriculumType.ALPGMM
        elif string == str(CurriculumType.SelfPaced):
            return CurriculumType.SelfPaced
        elif string == str(CurriculumType.SelfPacedWithCEM):
            return CurriculumType.SelfPacedWithCEM
        elif string == str(CurriculumType.Default):
            return CurriculumType.Default
        elif string == str(CurriculumType.DefaultWithCEM):
            return CurriculumType.DefaultWithCEM
        elif string == str(CurriculumType.Random):
            return CurriculumType.Random
        elif string == str(CurriculumType.Wasserstein):
            return CurriculumType.Wasserstein
        elif string == str(CurriculumType.ACL):
            return CurriculumType.ACL
        elif string == str(CurriculumType.PLR):
            return CurriculumType.PLR
        elif string == str(CurriculumType.VDS):
            return CurriculumType.VDS
        else:
            raise RuntimeError("Invalid string: '" + string + "'")


class AgentInterface(ABC):

    def __init__(self, learner, obs_dim, device):
        self.learner = learner
        self.obs_dim = obs_dim
        self.device = device

    def estimate_value(self, inputs):
        return self.estimate_value_internal(inputs)

    @abstractmethod
    def estimate_value_internal(self, inputs):
        pass

    @abstractmethod
    def mean_policy_std(self, cb_args, cb_kwargs):
        pass

    @abstractmethod
    def get_action(self, observations):
        pass

    def save(self, log_dir):
        self.learner.save(os.path.join(log_dir, "model"))


class SACInterface(AgentInterface):

    def __init__(self, learner, obs_dim, device):
        super().__init__(learner, obs_dim, device)

    def estimate_value_internal(self, inputs):
        return np.squeeze(self.learner.sess.run([self.learner.step_ops[6]], {self.learner.observations_ph: inputs}))

    def get_action(self, observations):
        flat_obs = np.reshape(observations, (-1, observations.shape[-1]))
        flat_acts = self.learner.predict(flat_obs, deterministic=False)[0]
        acts = np.reshape(flat_acts, (observations.shape[0:-1]) + (-1,))
        return acts

    def mean_policy_std(self, cb_args, cb_kwargs):
        if "infos_values" in cb_args[0] and len(cb_args[0]["infos_values"]) > 0:
            return cb_args[0]["infos_values"][4]
        else:
            return np.nan


class PPOInterface(AgentInterface):

    def __init__(self, learner, obs_dim, device):
        super().__init__(learner, obs_dim, device)
        self.grad_fn = []

    def estimate_value_internal(self, inputs):
        return np.squeeze(self.learner.policy.predict_values(torch.from_numpy(inputs)).detach().numpy())

    def get_action(self, observations):
        flat_obs = np.reshape(observations, (-1, observations.shape[-1]))
        flat_acts = self.learner.predict(flat_obs, deterministic=False)[0]
        return np.reshape(flat_acts, (observations.shape[0:-1]) + (-1,))

    def mean_policy_std(self, cb_args, cb_kwargs):
        std_th = self.learner.policy.get_distribution(torch.zeros((1, self.obs_dim)).to(
            self.device)).distribution.stddev[0, :]
        return np.mean(std_th.detach().cpu().numpy())


class SACEvalWrapper:

    def __init__(self, model):
        self.model = model

    def step(self, observation, state=None, deterministic=False):
        return self.model.predict(observation, state=state, deterministic=deterministic)[0]


class PPOEvalWrapper:

    def __init__(self, model):
        self.model = model

    def step(self, observation, state=None, deterministic=False):
        if len(observation.shape) == 1:
            observation = observation[None, :]
            return self.model.predict(observation, state=state, deterministic=deterministic)[0][0, :]
        else:
            return self.model.predict(observation, state=state, deterministic=deterministic)[0]


class Learner(Enum):
    PPO = 1
    SAC = 2

    def __str__(self):
        if self.ppo():
            return "ppo"
        else:
            return "sac"

    def ppo(self):
        return self.value == Learner.PPO.value

    def sac(self):
        return self.value == Learner.SAC.value

    def create_learner(self, env, parameters):
        if self.ppo() and not issubclass(type(env), VecEnv):
            env = DummyVecEnv([lambda: env])

        if self.ppo():
            model = PPO(PPOMlpPolicy, env, **parameters["common"], **parameters[str(self)])
            interface = PPOInterface(model, env.observation_space.shape[0], parameters["common"]["device"])
        else:
            model = SAC(SACMlpPolicy, env, **parameters["common"], **parameters[str(self)])
            interface = SACInterface(model, env.observation_space.shape[0], parameters["common"]["device"])

        return model, interface

    def load(self, path, env, device):
        if self.ppo():
            return PPO.load(path, env=env, device=device)
        else:
            return SAC.load(path, env=env, device=device)

    def load_for_evaluation(self, path, env, device):
        if self.ppo() and not issubclass(type(env), VecEnv):
            env = DummyVecEnv([lambda: env])
        model = self.load(path, env, device)
        if self.sac():
            return SACEvalWrapper(model)
        else:
            return PPOEvalWrapper(model)

    @staticmethod
    def from_string(string):
        if string == str(Learner.PPO):
            return Learner.PPO
        elif string == str(Learner.SAC):
            return Learner.SAC
        else:
            raise RuntimeError("Invalid string: '" + string + "'")


class ExperimentCallback:

    def __init__(self, log_directory, learner, env_wrapper, save_interval=5, step_divider=1):
        self.log_dir = os.path.realpath(log_directory)
        self.learner = learner
        self.env_wrapper = env_wrapper
        self.save_interval = save_interval
        self.algorithm_iteration = 0
        self.step_divider = step_divider
        self.iteration = 0
        self.last_time = None
        self.format = "   %4d    | %.1E |   %3d    |  %.2E  |  %.2E  |  %.2E   "
        if self.env_wrapper.teacher is not None:
            if isinstance(self.env_wrapper.teacher, SelfPacedTeacherV2):
                context_dim = self.env_wrapper.teacher.context_dim
                text = "| [%.2E"
                for i in range(0, context_dim - 1):
                    text += ", %.2E"
                text += "] "
                self.format += text + text
        if env_wrapper.aux_teacher is not None:
            if isinstance(self.env_wrapper.aux_teacher, CEM):
                context_dim = self.env_wrapper.aux_teacher.context_dim
                # aux mean, aux std
                text = "| [%.2E"
                for i in range(0, context_dim - 1):
                    text += ", %.2E"
                text += "] "
                self.format += text + text
                # q_ref, q_internal, ref_alpha
                for i in range(3):
                    self.format += "|  %.2E  "

        header = " Iteration |  Time   | Ep. Len. | Mean Reward | Mean Disc. Reward | Mean Policy STD "
        if self.env_wrapper.teacher is not None:
            if isinstance(self.env_wrapper.teacher, SelfPacedTeacherV2):
                header += "|     Context mean     |      Context std     "
        if env_wrapper.aux_teacher is not None:
            if isinstance(self.env_wrapper.aux_teacher, CEM):
                # q_ref, q_internal, ref_alpha
                header += "|     AUX mean     |      AUX std     "\
                          "|     Reference q     |      Internal Q     |" \
                          "     Reference alpha     "

        print(header)

    def __call__(self, *args, **kwargs):
        if self.algorithm_iteration % self.step_divider == 0:
            data_tpl = (self.iteration,)

            t_new = time.time()
            dt = np.nan
            if self.last_time is not None:
                dt = t_new - self.last_time
            data_tpl += (dt,)

            mean_rew, mean_disc_rew, mean_length = self.env_wrapper.get_statistics()
            data_tpl += (int(mean_length), mean_rew, mean_disc_rew)

            data_tpl += (self.learner.mean_policy_std(args, kwargs),)

            # Extra logging info
            if isinstance(self.env_wrapper.teacher, SelfPacedTeacherV2):
                context_mean = self.env_wrapper.teacher.context_dist.mean()
                context_std = np.sqrt(np.diag(self.env_wrapper.teacher.context_dist.covariance_matrix()))
                data_tpl += tuple(context_mean.tolist())
                data_tpl += tuple(context_std.tolist())
            if isinstance(self.env_wrapper.aux_teacher, CEM):
                aux_mean, aux_std = self.env_wrapper.aux_teacher.get_sample_dist()
                data_tpl += tuple(aux_mean.tolist())
                data_tpl += tuple(aux_std.tolist())
                data_tpl += (self.env_wrapper.aux_teacher.get_reference_q(),
                             self.env_wrapper.aux_teacher.get_internal_q(),
                             self.env_wrapper.aux_teacher.get_reference_alpha())

            print(self.format % data_tpl)

            if self.iteration % self.save_interval == 0:
                iter_log_dir = os.path.join(self.log_dir, "iteration-" + str(self.iteration))
                os.makedirs(iter_log_dir, exist_ok=True)

                with open(os.path.join(iter_log_dir, "context_trace.pkl"), "wb") as f:
                    pickle.dump(self.env_wrapper.get_encountered_contexts(), f)

                self.learner.save(iter_log_dir)
                if self.env_wrapper.teacher is not None:
                    self.env_wrapper.teacher.save(iter_log_dir)

                # SAVE AUXILIARY DISTRIBUTION!
                if self.env_wrapper.aux_teacher is not None:
                    self.env_wrapper.aux_teacher.save(iter_log_dir)

            self.last_time = time.time()
            self.iteration += 1

        self.algorithm_iteration += 1


class AbstractExperiment(ABC):
    APPENDIX_KEYS = {"default": ["DISCOUNT_FACTOR", "STEPS_PER_ITER", "LAM"],
                     CurriculumType.SelfPaced: ["DELTA", "KL_EPS", "DIST_TYPE"],
                     CurriculumType.SelfPacedWithCEM: ["DELTA", "KL_EPS", "REF_ALPHA_IN", "REF_ALPHA",
                                                       "REF_ALPHA_SCH", "DIST_TYPE"],
                     CurriculumType.Wasserstein: ["DELTA", "METRIC_EPS"],
                     CurriculumType.GoalGAN: ["GG_NOISE_LEVEL", "GG_FIT_RATE", "GG_P_OLD"],
                     CurriculumType.ALPGMM: ["AG_P_RAND", "AG_FIT_RATE", "AG_MAX_SIZE"],
                     CurriculumType.Random: [],
                     CurriculumType.Default: [],
                     CurriculumType.DefaultWithCEM: ["REF_ALPHA_IN", "REF_ALPHA", "REF_ALPHA_SCH", "DIST_TYPE"],
                     CurriculumType.ACL: ["ACL_EPS", "ACL_ETA"],
                     CurriculumType.PLR: ["PLR_REPLAY_RATE", "PLR_BETA", "PLR_RHO"],
                     CurriculumType.VDS: ["VDS_NQ", "VDS_LR", "VDS_EPOCHS", "VDS_BATCHES"]}

    def __init__(self, base_log_dir, curriculum_name, learner_name, parameters, seed, device, view=False):
        self.device = device
        self.base_log_dir = base_log_dir
        self.parameters = parameters
        self.curriculum = CurriculumType.from_string(curriculum_name)
        self.learner = Learner.from_string(learner_name)
        self.seed = seed
        self.view = view
        self.process_parameters()

    @abstractmethod
    def create_experiment(self):
        pass

    @abstractmethod
    def get_env_name(self):
        pass

    @abstractmethod
    def create_self_paced_teacher(self):
        pass

    @abstractmethod
    def create_cem_teacher(self):
        pass

    @abstractmethod
    def evaluate_learner(self, path):
        pass

    def get_other_appendix(self):
        return ""

    @staticmethod
    def parse_max_size(val):
        if val == "None":
            return None
        else:
            return int(val)

    def process_parameters(self):
        allowed_overrides = {"DISCOUNT_FACTOR": float, "MAX_KL": float, "STEPS_PER_ITER": int,
                             "LAM": float, "AG_P_RAND": float, "AG_FIT_RATE": int,
                             "AG_MAX_SIZE": self.parse_max_size, "GG_NOISE_LEVEL": float, "GG_FIT_RATE": int,
                             "GG_P_OLD": float, "DELTA": float, "EPS": float, "MAZE_TYPE": str, "ACL_EPS": float,
                             "ACL_ETA": float, "PLR_REPLAY_RATE": float, "PLR_BETA": float, "PLR_RHO": float,
                             "VDS_NQ": int, "VDS_LR": float, "VDS_EPOCHS": int, "VDS_BATCHES": int,
                             "DIST_TYPE": str, "TARGET_TYPE": str, "KL_EPS": float,
                             "REF_ALPHA_IN": float, "REF_ALPHA": float, "REF_ALPHA_SCH": int,
        }
        for key in sorted(self.parameters.keys()):
            if key not in allowed_overrides:
                raise RuntimeError("Parameter '" + str(key) + "'not allowed'")

            value = self.parameters[key]
            tmp = getattr(self, key)
            if isinstance(tmp, dict):
                tmp[self.learner] = allowed_overrides[key](value)
            else:
                setattr(self, key, allowed_overrides[key](value))

    def get_log_dir(self):
        override_appendix = create_override_appendix(self.APPENDIX_KEYS["default"], self.parameters)
        leaner_string = str(self.learner)
        key_list = self.APPENDIX_KEYS[self.curriculum]
        for key in sorted(key_list):
            tmp = getattr(self, key)
            if isinstance(tmp, dict):
                tmp = tmp[self.learner]
            leaner_string += "_" + key + "=" + str(tmp).replace(" ", "")

        return os.path.join(self.base_log_dir, self.get_env_name(), str(self.curriculum),
                            leaner_string + override_appendix + self.get_other_appendix(), "seed-" + str(self.seed))

    def train(self):
        model, timesteps, callback_params = self.create_experiment()
        log_directory = self.get_log_dir()

        if os.path.exists(log_directory):
            print("Log directory already exists! Going directly to evaluation")
        else:
            callback = ExperimentCallback(log_directory=log_directory, **callback_params)
            model.learn(total_timesteps=timesteps, reset_num_timesteps=False, callback=callback)

        # callback = ExperimentCallback(log_directory=log_directory, **callback_params)
        # model.learn(total_timesteps=timesteps, reset_num_timesteps=False, callback=callback)

    def evaluate(self, eval_type=0):
        log_dir = self.get_log_dir()

        iteration_dirs = [d for d in os.listdir(log_dir) if d.startswith("iteration-")]
        unsorted_iterations = np.array([int(d[len("iteration-"):]) for d in iteration_dirs])
        idxs = np.argsort(unsorted_iterations)
        sorted_iteration_dirs = np.array(iteration_dirs)[idxs].tolist()

        # First evaluate the KL-Divergences if Self-Paced learning was used
        if (self.curriculum.self_paced() or self.curriculum.self_paced_with_cem()) and \
                not os.path.exists(os.path.join(log_dir, "kl_divergences.pkl")):
            kl_divergences = []
            teacher = self.create_self_paced_teacher()
            for iteration_dir in sorted_iteration_dirs:
                iteration_log_dir = os.path.join(log_dir, iteration_dir)
                teacher.load(iteration_log_dir)
                kl_divergences.append(teacher.target_context_kl())

            kl_divergences = np.array(kl_divergences)
            with open(os.path.join(log_dir, "kl_divergences.pkl"), "wb") as f:
                pickle.dump(kl_divergences, f)

        performance_files = {
            0: "performance",
            1: "performance_rare",
            2: "performance_rare_uniform",
            3: "performance_hom",
        }
        for iteration_dir in sorted_iteration_dirs:
            iteration_log_dir = os.path.join(log_dir, iteration_dir)
            performance_log_dir = os.path.join(iteration_log_dir, f"{performance_files[eval_type]}.npy")
            eval_type_str = performance_files[eval_type][len("performance"):]
            # if not os.path.exists(performance_log_dir):
            if True:
                disc_rewards, eval_contexts, context_p, successful_eps = self.evaluate_learner(
                    path=iteration_log_dir,
                    eval_type=eval_type_str,
                )
                print(f"Evaluated {iteration_dir} (eval_type={eval_type}): {np.mean(disc_rewards)}")
                disc_rewards = np.array(disc_rewards)
                eval_contexts = np.array(eval_contexts)
                num_context = eval_contexts.shape[0]
                stats = np.ones((num_context, 1))*int(iteration_dir[len("iteration")+1:])
                stats = np.concatenate((stats, disc_rewards.reshape(-1, 1)), axis=1)
                stats = np.concatenate((stats, eval_contexts), axis=1)
                stats = np.concatenate((stats, context_p.reshape(-1, 1)), axis=1)
                stats = np.concatenate((stats, successful_eps), axis=1)
                np.save(performance_log_dir, stats)

