import numpy as np
from deep_sprl.teachers.util import Buffer
from deep_sprl.teachers.abstract_teacher import BaseWrapper


class SelfPacedWrapper(BaseWrapper):

    def __init__(self, env, sp_teacher, discount_factor, context_visible, reward_from_info=False,
                 use_undiscounted_reward=False, episodes_per_update=50, episodes_per_aux_update=0, aux_teacher=None):
        self.use_undiscounted_reward = use_undiscounted_reward

        # CEM extension
        # BaseWrapper.__init__(self, env, sp_teacher, discount_factor, context_visible,
        #                      reward_from_info=reward_from_info)
        BaseWrapper.__init__(self, env, sp_teacher, discount_factor, context_visible,
                             reward_from_info=reward_from_info,
                             episodes_per_update=episodes_per_update,
                             episodes_per_aux_update=episodes_per_aux_update,
                             aux_teacher=aux_teacher)
        self.aux_context_buffer = Buffer(3, episodes_per_aux_update + 1, True)

        self.context_buffer = Buffer(3, episodes_per_update + 1, True)
        # self.episodes_per_update = episodes_per_update

    def done_callback(self, step, cur_initial_state, cur_context, discounted_reward, undiscounted_reward,
                      use_teacher=True):
        ret = undiscounted_reward if self.use_undiscounted_reward else discounted_reward
        # CEM extension
        # self.context_buffer.update_buffer((cur_initial_state, cur_context, ret))
        if use_teacher:
            self.context_buffer.update_buffer((cur_initial_state, cur_context, ret))
            if hasattr(self.teacher, "on_rollout_end"):
                self.teacher.on_rollout_end(cur_context, ret)
        else:
            self.aux_context_buffer.update_buffer((cur_initial_state, cur_context, ret))

        # CEM extension
        if len(self.context_buffer) >= self.episodes_per_update and \
                (len(self.aux_context_buffer) >= self.episodes_per_aux_update or self.aux_teacher is None):
            __, contexts, returns = self.context_buffer.read_buffer()
            self.teacher.update_distribution(np.array(contexts), np.array(returns))

            if self.aux_teacher is not None:
                __, aux_contexts, aux_returns = self.aux_context_buffer.read_buffer()
                self.aux_teacher.update_distribution(np.array(contexts), np.array(returns),
                                                     np.array(aux_contexts), np.array(aux_returns))
                context_dist_mean = self.teacher.context_dist.mean()
                context_dist_var = self.teacher.context_dist.covariance_matrix()
                self.aux_teacher.update_original_distribution(dist_params=(context_dist_mean, context_dist_var))
                self.aux_teacher.update_ref_alpha()

        # if len(self.context_buffer) >= self.episodes_per_update:
        #     __, contexts, returns = self.context_buffer.read_buffer()
        #     self.teacher.update_distribution(np.array(contexts), np.array(returns))
        #     if len(self.aux_context_buffer) >= self.episodes_per_aux_update and self.aux_teacher is not None:
        #         __, aux_contexts, aux_returns = self.aux_context_buffer.read_buffer()
        #         # How should the auxiliary teacher be updated?
        #         self.aux_teacher.update_distribution(np.array(contexts), np.array(returns),
        #                                              np.array(aux_contexts), np.array(aux_returns))
        #         context_dist_mean = self.teacher.context_dist.mean()
        #         context_dist_var = self.teacher.context_dist.covariance_matrix()
        #         self.aux_teacher.update_original_distribution(dist_params=(context_dist_mean, context_dist_var))
        #         self.aux_teacher.update_ref_alpha()

    def get_context_buffer(self):
        ins, cons, disc_rews = self.context_buffer.read_buffer()
        return np.array(ins), np.array(cons), np.array(disc_rews)

    def get_aux_context_buffer(self):
        aux_ins, aux_cons, aux_disc_rews = self.aux_context_buffer.read_buffer()
        return np.array(aux_ins), np.array(aux_cons), np.array(aux_disc_rews)
