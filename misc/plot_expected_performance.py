import os
import sys
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import ticker
from pathlib import Path
sys.path.insert(1, os.path.join(sys.path[0], '..'))

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def get_results(base_dir, iterations):
    expected = []
    for iteration in iterations:
        perf_file = os.path.join(base_dir, f"iteration-{iteration}", "performance.npy")
        if os.path.exists(perf_file):
            results = np.load(perf_file)
            disc_rewards = results[:, 1]
            expected.append(np.mean(disc_rewards))
        else:
            print(f"No evaluation data found: {perf_file}")
            expected = []
            break
    return expected

def plot_results(base_log_dir, num_updates_per_iteration, seeds, env, setting, algorithms, figname_extra):
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
    plt.rcParams['font.size'] = setting["fontsize"]

    num_iters = setting["num_iters"]
    steps_per_iter = setting["steps_per_iter"]
    fontsize = setting["fontsize"]
    figsize = setting["figsize"]
    bbox_to_anchor = setting["bbox_to_anchor"]
    ylabel = setting["ylabel"]
    ylim = setting["ylim"]
    iterations = np.arange(0, num_iters, num_updates_per_iteration, dtype=int)
    iterations_step = iterations*steps_per_iter

    fig = plt.figure(constrained_layout=True, figsize=figsize)
    alg_exp_mid = {}

    for cur_algo in algorithms:
        algorithm = algorithms[cur_algo]["algorithm"]
        label = algorithms[cur_algo]["label"]
        model = algorithms[cur_algo]["model"]
        color = algorithms[cur_algo]["color"]
        print(algorithm)

        expected = []
        for seed in seeds:
            base_dir = os.path.join(base_log_dir, env, algorithm, model, f"seed-{seed}")
            print(base_dir)
            expected_seed = get_results(
                base_dir=base_dir,
                iterations=iterations,
            )
            if len(expected_seed) == 0:
                continue

            expected.append(expected_seed)

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

        alg_exp_mid[cur_algo] = expected_mid[-1]

        plt.plot(iterations_step, expected_mid, color=color, linewidth=2.0, label=f"{label}",
                      marker=".",
                      )
        plt.fill_between(iterations_step, expected_qlow, expected_qhigh, color=color, alpha=0.4)
        # axes[context_dim].fill_between(iterations_step, expected_low, expected_high, color=color, alpha=0.2)

    plt.ticklabel_format(axis='x', style='sci', scilimits=(5, 6), useMathText=True)
    plt.xticks([])
    plt.xlim([iterations_step[0], iterations_step[-1]])
    plt.ylim(ylim)
    plt.ylabel(ylabel)
    plt.xlabel("Number of environment interactions")
    plt.grid(True)

    sorted_alg_exp_mid = [b[0] for b in sorted(enumerate(list(alg_exp_mid.values()), ), key=lambda i: i[1])]
    colors = []
    labels = []
    num_alg = len(algorithms)
    for alg_i in sorted_alg_exp_mid:
        cur_algo = list(alg_exp_mid.keys())[alg_i]
        colors.append(algorithms[cur_algo]["color"])
        labels.append(algorithms[cur_algo]["label"])
    # for cur_algo in algorithms:
    #     colors.append(algorithms[cur_algo]["color"])
    #     labels.append(algorithms[cur_algo]["label"])

    markers = ["" for i in range(num_alg)]
    linestyles = ["-" for i in range(num_alg)]
    labels.reverse()
    lines = [Line2D([0], [0], color=colors[-i-1], linestyle=linestyles[i], marker=markers[i], linewidth=2.0)
             for i in range(num_alg)]
    lgd = fig.legend(lines, labels, ncol=math.ceil(num_alg/3), loc="upper center", bbox_to_anchor=bbox_to_anchor,
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
    # num_updates_per_iteration = 5
    # seeds = [str(i) for i in range(1, 11)]
    # env = "point_mass_2d_heavytailed_wide"

    num_updates_per_iteration = 10
    seeds = [1, 2, 3, 5, 6]
    env = "lunar_lander_2d_heavytailed_wide"
    figname_extra = "_expected_new"

    algorithms = {
        "lunar_lander_2d_heavytailed_wide": {
            "SPDL": {
                "algorithm": "self_paced",
                "label": "SPDL",
                "model": "ppo_DELTA=-100.0_DIST_TYPE=cauchy_KL_EPS=0.025",
                "color": "blue",
            },
            "RACGEN": {
                "algorithm": "self_paced_with_cem",
                "label": "RACGEN",
                "model": "ppo_DELTA=-100.0_DIST_TYPE=cauchy_KL_EPS=0.025_RALPH=0.2_RALPH_IN=1.0_RALPH_SCH=80",
                "color": "red",
            },
            "DEF": {
                "algorithm": "default",
                "label": "Default",
                "model": "ppo",
                "color": "magenta",
            },
            "DEF-CEM": {
                "algorithm": "default_with_cem",
                "label": "Default-CEM",
                "model": "ppo_DIST_TYPE=cauchy_RALPH=0.2_RALPH_IN=1.0_RALPH_SCH=80",
                "color": "maroon",
            },
            "CURROT": {
                "algorithm": "wasserstein",
                "label": "CURROT",
                "model": "ppo_DELTA=-50.0_METRIC_EPS=0.5",
                "color": "green",
            },
            "GoalGAN": {
                "algorithm": "goal_gan",
                "label": "GoalGAN",
                "model": "ppo_GG_FIT_RATE=200_GG_NOISE_LEVEL=0.1_GG_P_OLD=0.2",
                "color": "orange",
            },
            "ALP-GMM": {
                "algorithm": "alp_gmm",
                "label": "ALP-GMM",
                "model": "ppo_AG_FIT_RATE=100_AG_MAX_SIZE=500_AG_P_RAND=0.1",
                "color": "lawngreen",
            },
            "PLR": {
                "algorithm": "plr",
                "label": "PLR",
                "model": "ppo_PLR_BETA=0.15_PLR_REPLAY_RATE=0.85_PLR_RHO=0.45",
                "color": "purple",
            },
            "VDS": {
                "algorithm": "vds",
                "label": "VDS",
                "model": "ppo_VDS_BATCHES=40_VDS_EPOCHS=3_VDS_LR=0.001_VDS_NQ=5",
                "color": "darkcyan",
            },
        },
        "point_mass_2d_heavytailed_wide": {
            "SPDL": {
                "algorithm": "self_paced",
                "label": "SPDL",
                "model": "ppo_DELTA=4.0_DIST_TYPE=cauchy_KL_EPS=0.25",
                "color": "blue",
            },
            "RACGEN": {
                "algorithm": "self_paced_with_cem",
                "label": "RACGEN",
                "model": "ppo_DELTA=4.0_DIST_TYPE=cauchy_KL_EPS=0.25_RALPH=0.2_RALPH_IN=1.0_RALPH_SCH=20",
                "color": "red",
            },
            "DEF": {
                "algorithm": "default",
                "label": "Default",
                "model": "ppo",
                "color": "magenta",
            },
            "SPDL-N": {
                "algorithm": "self_paced",
                "label": "SPDL-N",
                "model": "ppo_DELTA=4.0_DIST_TYPE=gaussian_KL_EPS=0.25",
                "color": "gold",
            },
            "RACGEN-N": {
                "algorithm": "self_paced_with_cem",
                "label": "RACGEN-N",
                "model": "ppo_DELTA=4.0_DIST_TYPE=gaussian_KL_EPS=0.25_RALPH=0.2_RALPH_IN=1.0_RALPH_SCH=20",
                "color": "cyan",
            },
            "DEF-CEM": {
                "algorithm": "default_with_cem",
                "label": "Default-CEM",
                "model": "ppo_DIST_TYPE=cauchy_RALPH=0.2_RALPH_IN=1.0_RALPH_SCH=20",
                "color": "maroon",
            },
            "CURROT": {
                "algorithm": "wasserstein",
                "label": "CURROT",
                "model": "ppo_DELTA=4.0_METRIC_EPS=0.5",
                "color": "green",
            },
            "GoalGAN": {
                "algorithm": "goal_gan",
                "label": "GoalGAN",
                "model": "ppo_GG_FIT_RATE=200_GG_NOISE_LEVEL=0.05_GG_P_OLD=0.1",
                "color": "orange",
            },
            "ALP-GMM": {
                "algorithm": "alp_gmm",
                "label": "ALP-GMM",
                "model": "ppo_AG_FIT_RATE=50_AG_MAX_SIZE=500_AG_P_RAND=0.1",
                "color": "lawngreen",
            },
            "PLR": {
                "algorithm": "plr",
                "label": "PLR",
                "model": "ppo_PLR_BETA=0.15_PLR_REPLAY_RATE=0.85_PLR_RHO=0.45",
                "color": "purple",
            },
            "VDS": {
                "algorithm": "vds",
                "label": "VDS",
                "model": "ppo_VDS_BATCHES=40_VDS_EPOCHS=5_VDS_LR=0.001_VDS_NQ=5",
                "color": "darkcyan",
            },
        },
    }

    settings = {
        "lunar_lander_2d_heavytailed_wide":
            {
                "num_iters": 250,
                "steps_per_iter": 10240,
                "fontsize": 24,
                "figsize": (10, 5),
                "bbox_to_anchor": (.5, 1.28),
                "ylabel": 'Expected discounted return',
                "ylim": [-125., 50.],
            },
        "point_mass_2d_heavytailed_wide":
            {
                "num_iters": 300,
                "steps_per_iter": 6144,
                "fontsize": 24,
                "figsize": (10, 5),
                "bbox_to_anchor": (.5, 1.28),
                "ylabel": 'Expected discounted return',
                "ylim": [0., 6.],
            },
    }

    plot_results(
        base_log_dir=base_log_dir,
        num_updates_per_iteration=num_updates_per_iteration,
        seeds=seeds,
        env=env,
        setting=settings[env],
        algorithms=algorithms[env],
        figname_extra=figname_extra,
        )


if __name__ == "__main__":
    main()
