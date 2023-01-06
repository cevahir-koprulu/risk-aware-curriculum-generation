import os
import sys
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from pathlib import Path
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from deep_sprl.util.gaussian_torch_distribution import GaussianTorchDistribution

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def get_results(base_dir, iterations, get_success=False, alpha=0.2):
    expected = []
    success = []
    cvar = []
    success_cvar = []

    for iteration in iterations:
        perf_file = os.path.join(base_dir, f"iteration-{iteration}", "performance.npy")
        if os.path.exists(perf_file):
            results = np.load(perf_file)
            disc_rewards = results[:, 1]
            expected.append(np.mean(disc_rewards))
            cvar.append(np.quantile(disc_rewards, q=alpha))

            if get_success:
                successful_eps = results[:, -1]
                success.append(np.mean(successful_eps))
                success_cvar.append(np.quantile(successful_eps, q=alpha))

        else:
            print(f"No evaluation data found: {os.path.join(base_dir, 'iteration-0', 'performance.npy')}")
            expected = []
            success = []
            break
    return expected, success, cvar, success_cvar


def get_dist_stats(base_dir, iterations, context_dim=2):
    dist_stats = []
    for iteration in iterations:
        dist_path = os.path.join(base_dir, f"iteration-{iteration}", "teacher.npy")
        dist = GaussianTorchDistribution.from_weights(context_dim, np.load(dist_path))
        stats = []
        for c_dim in range(context_dim):
            stats.append(dist.mean()[c_dim])
        for c_dim in range(context_dim):
            stats.append(dist.covariance_matrix()[c_dim, c_dim])

        dist_stats.append(stats)
    dist_stats = np.array(dist_stats)
    return dist_stats


def get_aux_dist_stats(base_dir, iterations, context_dim=2):
    dist_stats = []
    for iteration in iterations:
        filename = os.path.join(base_dir, f"iteration-{iteration}", "cem.cem")
        with open(filename, 'rb') as h:
            obj = pkl.load(h)

        title, original_dist, batch_size, w_clip, \
        ref_mode, ref_alpha, n_orig_per_batch, internal_alpha, \
        ref_scores, sample_dist, sampled_data, weights, scores, \
        ref_quantile, internal_quantile, selected_samples = obj
        dist = sample_dist[-1]
        stats = []
        for c_dim in range(context_dim):
            stats.append(dist.mean()[c_dim])
        for c_dim in range(context_dim):
            stats.append(dist.covariance_matrix()[c_dim, c_dim])

        dist_stats.append(stats)
    dist_stats = np.array(dist_stats)
    return dist_stats


def plot_results(base_log_dir, num_updates_per_iteration, seeds, env, setting, algorithms, plot_success, alpha,
                 figname_extra):
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

    num_iters = setting["num_iters"]
    steps_per_iter = setting["steps_per_iter"]
    context_dim = setting["context_dim"]
    fontsize = setting["fontsize"]
    figsize = setting["figsize"]
    grid_shape = setting["grid_shape"]
    bbox_to_anchor = setting["bbox_to_anchor"]
    axes_info = setting["axes_info"]

    iterations = np.arange(0, num_iters, num_updates_per_iteration, dtype=int)
    iterations_step = iterations*steps_per_iter

    fig = plt.figure(constrained_layout=True, figsize=figsize)
    gs = GridSpec(grid_shape[0], grid_shape[1], figure=fig)
    axes = []
    for row_no in range(grid_shape[0]):
        for col_no in range(grid_shape[1]):
            axes.append(fig.add_subplot(gs[row_no, col_no]))

    for cur_algo in algorithms:
        algorithm = algorithms[cur_algo]["algorithm"]
        label = algorithms[cur_algo]["label"]
        model = algorithms[cur_algo]["model"]
        color = algorithms[cur_algo]["color"]
        aux_color = algorithms[cur_algo]["aux_color"]
        print(algorithm)

        expected = []
        success = []
        cvar = []
        success_cvar = []
        dist_stats = []
        aux_dist_stats = []
        for seed in seeds:
            base_dir = os.path.join(base_log_dir, env, algorithm, model, f"seed-{seed}")
            print(base_dir)
            expected_seed, success_seed, cvar_seed, success_cvar_seed = get_results(
                base_dir=base_dir,
                iterations=iterations,
                get_success=plot_success,
                alpha=alpha,
            )
            if len(expected_seed) == 0:
                continue

            expected.append(expected_seed)
            cvar.append(cvar_seed)
            # if "self_paced" in algorithm:
            if False:
                dist_stats_seed = get_dist_stats(
                    base_dir=base_dir,
                    iterations=iterations,
                    context_dim=context_dim)
                dist_stats.append(dist_stats_seed)
            # if "cem" in algorithm:
            if False:
                aux_dist_stats_seed = get_aux_dist_stats(
                    base_dir=base_dir,
                    iterations=iterations,
                    context_dim=context_dim)
                aux_dist_stats.append(aux_dist_stats_seed)
            if plot_success:
                success.append(success_seed)
                success_cvar.append(success_cvar_seed)

        expected = np.array(expected)
        expected_mid = np.median(expected, axis=0)
        expected_qlow = np.quantile(expected, 0.25, axis=0)
        expected_qhigh = np.quantile(expected, 0.75, axis=0)
        # expected_low = np.min(expected, axis=0)
        # expected_high = np.max(expected, axis=0)
        # expected_mid = np.mean(expected, axis=0)
        # expected_std = np.std(expected, axis=0)
        # expected_qlow = expected_mid-expected_std
        # expected_qhigh = expected_mid+expected_std

        cvar = np.array(cvar)
        cvar_mid = np.median(cvar, axis=0)
        cvar_qlow = np.quantile(cvar, 0.25, axis=0)
        cvar_qhigh = np.quantile(cvar, 0.75, axis=0)
        # cvar_low = np.min(cvar, axis=0)
        # cvar_high = np.max(cvar, axis=0)
        # cvar_mid = np.mean(cvar, axis=0)
        # cvar_std = np.std(cvar, axis=0)
        # cvar_qlow = cvar_mid-cvar_std
        # cvar_qhigh = cvar_mid+cvar_std

        # if "self_paced" in algorithm:
        if False:
            dist_stats = np.array(dist_stats)
            dist_stats = np.swapaxes(dist_stats, 1, 2)
            dist_stats = np.swapaxes(dist_stats, 0, 1)
            dist_stats_mid = np.median(dist_stats, axis=1)
            # dist_stats_low = np.min(dist_stats, axis=1)
            # dist_stats_high = np.max(dist_stats, axis=1)
            dist_stats_low = np.quantile(dist_stats, 0.25, axis=1)
            dist_stats_high = np.quantile(dist_stats, 0.75, axis=1)

            # print(dist_stats[:context_dim, :])

            for ax_i in range(context_dim):
                axes[ax_i].plot(iterations_step, dist_stats_mid[ax_i, :], color=color, label=f"{label}-mean", marker=".")
                axes[ax_i].fill_between(iterations_step, dist_stats_low[ax_i, :], dist_stats_high[ax_i, :], color=color,
                                        alpha=0.5)
                axes[ax_i].plot(iterations_step, dist_stats_mid[context_dim+ax_i, :], color=color, label=f"{label}-var",
                                ls="--", marker="x")
                axes[ax_i].fill_between(iterations_step, dist_stats_low[context_dim+ax_i, :],
                                        dist_stats_high[context_dim+ax_i, :], color=color, alpha=0.5, ls="--")
        # if "cem" in algorithm:
        if False:
            aux_dist_stats = np.array(aux_dist_stats)
            aux_dist_stats = np.swapaxes(aux_dist_stats, 1, 2)
            aux_dist_stats = np.swapaxes(aux_dist_stats, 0, 1)
            aux_dist_stats_mid = np.median(aux_dist_stats, axis=1)
            # aux_dist_stats_low = np.min(aux_dist_stats, axis=1)
            # aux_dist_stats_high = np.max(aux_dist_stats, axis=1)
            aux_dist_stats_low = np.quantile(aux_dist_stats, 0.25, axis=1)
            aux_dist_stats_high = np.quantile(aux_dist_stats, 0.75, axis=1)


            for ax_i in range(context_dim):
                axes[ax_i].plot(iterations_step, aux_dist_stats_mid[ax_i, :], color=aux_color, label=f"aux-{label}-mean", marker=".")
                axes[ax_i].fill_between(iterations_step, aux_dist_stats_low[ax_i, :], aux_dist_stats_high[ax_i, :], color=aux_color,
                                        alpha=0.5)
                axes[ax_i].plot(iterations_step, aux_dist_stats_mid[context_dim+ax_i, :], color=aux_color, label=f"aux-{label}-var",
                                ls="--", marker="x")
                axes[ax_i].fill_between(iterations_step, aux_dist_stats_low[context_dim+ax_i, :],
                                        aux_dist_stats_high[context_dim+ax_i, :], color=aux_color, alpha=0.5, ls="--")

        axes[context_dim].plot(iterations_step, expected_mid, color=color, linewidth=2.0, label=f"{label}",
                      # marker="^",
                      )
        axes[context_dim].fill_between(iterations_step, expected_qlow, expected_qhigh, color=color, alpha=0.4)
        # axes[context_dim].fill_between(iterations_step, expected_low, expected_high, color=color, alpha=0.2)

        axes[context_dim+1].plot(iterations_step, cvar_mid, color=color, linewidth=2.0, label=f"{label}")
        axes[context_dim+1].fill_between(iterations_step, cvar_qlow, cvar_qhigh, color=color, alpha=0.4)
        # axes[context_dim+1].fill_between(iterations_step, cvar_low, cvar_high, color=color, alpha=0.2)

        if plot_success:
            success = np.array(success)
            success_mid = np.median(success, axis=0)
            # success_low = np.min(success, axis=0)
            # success_high = np.max(success, axis=0)
            success_qlow = np.quantile(success, 0.25, axis=0)
            success_qhigh = np.quantile(success, 0.75, axis=0)

            axes[-2].plot(iterations_step, success_mid, color=color, linewidth=2.0, label=f"{label}",
                          # marker="^",
                          )
            axes[-2].fill_between(iterations_step, success_qlow, success_qhigh, color=color, alpha=0.4)
            # axes[-2].fill_between(iterations_step, success_low, success_high, color=color, alpha=0.2)

            success_cvar = np.array(success_cvar)
            success_cvar_mid = np.median(success_cvar, axis=0)
            # success_cvar_low = np.min(success_cvar, axis=0)
            # success_cvar_high = np.max(success_cvar, axis=0)
            success_cvar_qlow = np.quantile(success_cvar, 0.25, axis=0)
            success_cvar_qhigh = np.quantile(success_cvar, 0.75, axis=0)

            axes[-1].plot(iterations_step, success_cvar_mid, color=color, linewidth=2.0, label=f"{label}",
                          # marker="^",
                          )
            axes[-1].fill_between(iterations_step, success_cvar_qlow, success_cvar_qhigh, color=color, alpha=0.4)
            # axes[-1].fill_between(iterations_step, success_low, success_high, color=color, alpha=0.2)

    markers = [".", "x"]
    linestyles = ["-", "--"]
    labels = ["Mean", "Variance"]
    lines = [Line2D([0], [0], color="black", linestyle=linestyles[i], marker=markers[i]) for i in range(2)]
    for ax_i in range(len(axes)):
        axes[ax_i].set_ylabel(axes_info["ylabel"][ax_i], fontsize=fontsize)
        axes[ax_i].set_ylim(axes_info["ylim"][ax_i])
        axes[ax_i].grid(True)
        if ax_i < context_dim:
            axes[ax_i].legend(lines, labels, fontsize=fontsize*0.8, loc="best", framealpha=1.)
            if grid_shape[0] == 1:
                # axes[ax_i].set_xlabel('Context Distribution Update', fontsize=fontsize)
                axes[ax_i].set_xlabel('Number of environment interactions', fontsize=fontsize)
    # axes[-1].set_xlabel('Context Distribution Update', fontsize=fontsize)
    axes[-1].set_xlabel('Number of environment interactions', fontsize=fontsize)

    colors = []
    labels = []
    num_alg = 0
    for cur_algo in algorithms:
        num_alg += 1
        colors.append(algorithms[cur_algo]["color"])
        labels.append(algorithms[cur_algo]["label"])
        if "cem" in cur_algo:
            colors.append(algorithms[cur_algo]["aux_color"])
            labels.append(algorithms[cur_algo]["label"]+"_aux")
            num_alg += 1

    markers = ["" for i in range(num_alg)]
    linestyles = ["-" for i in range(num_alg)]
    labels.reverse()
    lines = [Line2D([0], [0], color=colors[-i-1], linestyle=linestyles[i], marker=markers[i], linewidth=2.0)
             for i in range(num_alg)]
    lgd = fig.legend(lines, labels, ncol=num_alg, loc="upper center", bbox_to_anchor=bbox_to_anchor,
                     fontsize=fontsize, handlelength=1.0, labelspacing=0., handletextpad=0.5, columnspacing=1.0)

    figname = ""
    for cur_algo_i, cur_algo in enumerate(algorithms):
        figname += cur_algo
        if cur_algo_i < len(algorithms)-1:
            figname += "_vs_"

    print(f"{Path(os.getcwd()).parent}\\figures\\{env}_{figname}{figname_extra}.pdf")
    plt.savefig(f"{Path(os.getcwd()).parent}\\figures\\{env}_{figname}{figname_extra}.pdf", dpi=500,
                bbox_inches='tight', bbox_extra_artists=(lgd,))


def main():
    base_log_dir = f"{Path(os.getcwd()).parent}\\logs"
    num_updates_per_iteration = 5
    seeds = [str(i) for i in range(1, 11)]
    target_type = "wide"
    env = f"point_mass_2d_heavytailed_{target_type}"
    figname_extra = "_alpha=0.2"
    plot_success = False
    alpha = 0.2

    algorithms = {
        "point_mass_2d_heavytailed_wide": {
            "SPDL": {
                "algorithm": "self_paced",
                "label": "SPDL",
                "model": "ppo_DELTA=4.0_DIST_TYPE=cauchy_KL_EPS=0.25",
                "color": "blue",
                "aux_color": "",
            },
            "RACGEN": {
                "algorithm": "self_paced_with_cem",
                "label": "RACGEN",
                "model": "ppo_DELTA=4.0_DIST_TYPE=cauchy_KL_EPS=0.25_REF_ALPHA=0.2_REF_ALPHA_IN=0.2_REF_ALPHA_SCH=20",
                "color": "red",
                "aux_color": "green",
            },
            "DEF": {
                "algorithm": "default",
                "label": "Default",
                "model": "ppo",
                "color": "magenta",
                "aux_color": "",
            },
            "SPDL-G": {
                "algorithm": "self_paced",
                "label": "SPDL-G",
                "model": "ppo_DELTA=4.0_DIST_TYPE=gaussian_KL_EPS=0.25",
                "color": "gold",
                "aux_color": "",
            },
            "RACGEN-G": {
                "algorithm": "self_paced_with_cem",
                "label": "RACGEN-G",
                "model": "ppo_DELTA=4.0_DIST_TYPE=gaussian_KL_EPS=0.25_REF_ALPHA=0.2_REF_ALPHA_IN=0.2_REF_ALPHA_SCH=20",
                "color": "cyan",
                "aux_color": "orange",
            },
            "DEF-CEM": {
                "algorithm": "default_with_cem",
                "label": "Default-CEM",
                "model": "ppo_DIST_TYPE=cauchy_REF_ALPHA=0.2_REF_ALPHA_IN=0.2_REF_ALPHA_SCH=20",
                "color": "maroon",
                "aux_color": "violet",
            },
            "CURROT": {
                "algorithm": "wasserstein",
                "label": "CURROT",
                "model": "ppo_DELTA=4.0_METRIC_EPS=0.5",
                "color": "green",
                "aux_color": "",
            },
            "GoalGAN": {
                "algorithm": "goal_gan",
                "label": "GoalGAN",
                "model": "ppo_GG_FIT_RATE=200_GG_NOISE_LEVEL=0.1_GG_P_OLD=0.2",
                "color": "orange",
                "aux_color": "",
            },
            "ALP-GMM": {
                "algorithm": "alp_gmm",
                "label": "ALP-GMM",
                "model": "ppo_AG_FIT_RATE=100_AG_MAX_SIZE=500_AG_P_RAND=0.1",
                "color": "lawngreen",
                "aux_color": "",
            },
            "ACL": {
                "algorithm": "acl",
                "label": "ACL",
                "model": "ppo_ACL_EPS=0.2_ACL_ETA=0.025",
                "color": "lightblue",
                "aux_color": "",
            },
            "PLR": {
                "algorithm": "plr",
                "label": "PLR",
                "model": "ppo_PLR_BETA=0.45_PLR_REPLAY_RATE=0.85_PLR_RHO=0.15",
                "color": "plum",
                "aux_color": "",
            },
            "VDS": {
                "algorithm": "vds",
                "label": "VDS",
                "model": "ppo_VDS_BATCHES=20_VDS_EPOCHS=3_VDS_LR=0.001_VDS_NQ=5",
                "color": "violet",
                "aux_color": "",
            },

        },
        "point_mass_2d_heavytailed_narrow": {
            "self_paced": {
                "algorithm": "self_paced",
                "label": "self_paced",
                "model": "ppo_DELTA=4.0_DIST_TYPE=cauchy_KL_EPS=0.25",
                "color": "blue",
                "aux_color": "",
            },

            "self_paced_with_cem": {
                "algorithm": "self_paced_with_cem",
                "label": "self_paced_with_cem",
                "model": "ppo_DELTA=4.0_DIST_TYPE=cauchy_INTERNAL_ALPHA=0.5_KL_EPS=0.25_REF_ALPHA=0.2",
                "color": "red",
                "aux_color": "green",
            },

            "default": {
                "algorithm": "default",
                "label": "default",
                "model": "ppo",
                "color": "magenta",
                "aux_color": "orange",
            },
        },
        "point_mass_2d_wide": {
            "self_paced": {
                "algorithm": "self_paced",
                "label": "self_paced",
                "model": "ppo_DELTA=4.0_KL_EPS=0.25",
                "color": "blue",
                "aux_color": "",
            },

            "self_paced_with_cem": {
                "algorithm": "self_paced_with_cem",
                "label": "self_paced_with_cem",
                "model": "ppo_DELTA=4.0_INTERNAL_ALPHA=0.5_KL_EPS=0.25_REF_ALPHA=0.2",
                "color": "red",
                "aux_color": "green",
            },
            "default": {
                "algorithm": "default",
                "label": "default",
                "model": "ppo",
                "color": "magenta",
                "aux_color": "orange",
            },

            "default_with_cem": {
                "algorithm": "default_with_cem",
                "label": "default_with_cem",
                "model": "ppo_INTERNAL_ALPHA=0.5_REF_ALPHA=0.2",
                "color": "cyan",
                "aux_color": "brown",
            },
        }
    }

    settings = {
        "point_mass_2d_heavytailed":
            {
                "context_dim": 2,
                "num_iters": 300,
                "steps_per_iter": 4096,
                "fontsize": 12,
                "figsize": (5 * 2, 2.5 * (2 + plot_success)),
                "grid_shape": (2 + plot_success, 2),
                "bbox_to_anchor": (.5, 1.25),
                "axes_info": {
                    "ylabel": ['Param-1: Door position',
                               'Param-2: Door width',
                               'Expected discounted return',
                               'CVaR of discounted return',
                               'Expected rate of success',
                               'CVaR of rate of succes',
                               ],
                    "ylim": [[-4, 4.5],
                             [0, 8],
                             [-.1, 8.],
                             [-.1, 8.],
                             [-0.05, 1.05],
                             [-0.05, 1.05],
                             ],
                    },
            },
        "point_mass_2d":
            {
                "context_dim": 2,
                "num_iters": 400,
                "steps_per_iter": 1024,
                "fontsize": 12,
                "figsize": (5 * 2, 2.5 * (2 + plot_success)),
                "grid_shape": (2 + plot_success, 2),
                "bbox_to_anchor": (.5, 1.25),
                "axes_info": {
                    "ylabel": ['Param-1: Door position',
                               'Param-2: Door width',
                               'Expected discounted return',
                               'CVaR of discounted return',
                               'Expected rate of success',
                               'CVaR of rate of succes',
                               ],
                    "ylim": [[-4, 4.5],
                             [0, 8],
                             [-.1, 8.],
                             [-.1, 8.],
                             [-0.05, 1.05],
                             [-0.05, 1.05],
                             ],
                    },
            },
    }

    plot_results(
        base_log_dir=base_log_dir,
        num_updates_per_iteration=num_updates_per_iteration,
        seeds=seeds,
        env=env,
        setting=settings[env[:-len(target_type)-1]],
        algorithms=algorithms[env],
        plot_success=plot_success,
        alpha=alpha,
        figname_extra=figname_extra,
        )


if __name__ == "__main__":
    main()
