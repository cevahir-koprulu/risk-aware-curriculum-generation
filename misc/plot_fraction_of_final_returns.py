import os
import sys
import math
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path
from itertools import combinations
sys.path.insert(1, os.path.join(sys.path[0], '..'))

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def get_results(base_dir, iteration, min_return, max_return, num_intervals):
    perf_file = os.path.join(base_dir, f"iteration-{iteration}", "performance.npy")
    fractions = []
    if os.path.exists(perf_file):
        results = np.load(perf_file)
        returns = results[:, 1]
        interval = (max_return-min_return)/num_intervals
        for th in np.arange(min_return, max_return+interval, interval):
            # print(f"th={th} || fraction={np.sum(returns>th)/returns.shape[0]}")
            fractions.append(np.sum(returns>th)/returns.shape[0])
    else:
        print(f"No evaluation data found: {perf_file}")
        returns = []
    return np.array(fractions), np.array(returns)

def plot_results(base_log_dir, num_updates_per_iteration, seeds, env, setting, algorithms, figname_extra):
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
    plt.rcParams['font.size'] = setting["fontsize"]

    num_iters = setting["num_iters"]
    fontsize = setting["fontsize"]
    figsize = setting["figsize"]
    bbox_to_anchor = setting["bbox_to_anchor"]
    min_return = setting["min_return"]
    max_return = setting["max_return"]
    num_intervals = setting["num_intervals"]
    iteration = num_iters - num_updates_per_iteration
    xticks = setting["xticks"]
    yticks = setting["yticks"]

    fig = plt.figure(constrained_layout=True, figsize=figsize)
    alg_exp_mid = {}
    for cur_algo in algorithms:
        algorithm = algorithms[cur_algo]["algorithm"]
        label = algorithms[cur_algo]["label"]
        model = algorithms[cur_algo]["model"]
        color = algorithms[cur_algo]["color"]
        print(algorithm)

        fractions = []
        returns = []
        for seed in seeds:
            base_dir = os.path.join(base_log_dir, env, algorithm, model, f"seed-{seed}")
            print(base_dir)
            fractions_seed, returns_seed = get_results(
                base_dir=base_dir,
                iteration=iteration,
                min_return=min_return,
                max_return=max_return,
                num_intervals=num_intervals,
            )
            if len(fractions_seed) == 0:
                continue

            fractions.append(fractions_seed)
            returns.append(returns_seed)

        fractions = np.array(fractions)
        fractions_mid = np.median(fractions, axis=0)
        fractions_qlow = np.quantile(fractions, 0.25, axis=0)
        fractions_qhigh = np.quantile(fractions, 0.75, axis=0)
        
        returns = np.array(returns)
        alg_exp_mid[cur_algo] = np.median(returns.flatten())

        interval = (max_return-min_return)/num_intervals
        intervals = np.arange(min_return, max_return+interval, interval)
        
        plt.plot(intervals, fractions_mid, color=color, linewidth=2.0, label=f"{label}",
                      marker=".",
                      )
        # plt.fill_between(intervals, fractions_qlow, fractions_qhigh, color=color, alpha=0.4)
        # axes[context_dim].fill_between(intervals, fractions_low, fractions_high, color=color, alpha=0.2)

    # To DO: Set x ticks!
    # plt.ticklabel_format(axis='x', style='sci', scilimits=(5, 6), useMathText=True)
    if xticks is not None:
        plt.xticks(xticks)
    if yticks is not None:
        plt.yticks(yticks)
    plt.xlim([intervals[0], intervals[-1]])
    plt.ylim([-0.05,1.05])
    plt.ylabel(r"Fraction of runs with return $>\mathbf{r}$")
    plt.xlabel(r"Return $(\mathbf{r})$")
    plt.grid(True)

    sorted_alg_exp_mid = [b[0] for b in sorted(enumerate(list(alg_exp_mid.values()), ), key=lambda i: i[1])]
    colors = []
    labels = []
    num_alg = len(algorithms)
    for alg_i in sorted_alg_exp_mid:
        cur_algo = list(alg_exp_mid.keys())[alg_i]
        colors.append(algorithms[cur_algo]["color"])
        labels.append(algorithms[cur_algo]["label"])

    markers = ["" for i in range(num_alg)]
    linestyles = ["-" for i in range(num_alg)]
    labels.reverse()
    lines = [Line2D([0], [0], color=colors[-i-1], linestyle=linestyles[i], marker=markers[i], linewidth=2.0)
             for i in range(num_alg)]
    lgd = fig.legend(lines, labels, ncol=math.ceil(num_alg/2), loc="upper center", bbox_to_anchor=bbox_to_anchor,
                     fontsize=fontsize, handlelength=1.0, labelspacing=0., handletextpad=0.5, columnspacing=1.0)

    figname = ""
    for cur_algo_i, cur_algo in enumerate(algorithms):
        figname += cur_algo
        if cur_algo_i < len(algorithms)-1:
            figname += "_vs_"

    figpath = os.path.join(Path(os.getcwd()).parent, "figures", f"{env}_{figname}{figname_extra}.pdf")
    print(figpath)
    plt.savefig(figpath, dpi=500,
                bbox_inches='tight', bbox_extra_artists=(lgd,))

def main():
    base_log_dir = os.path.join(Path(os.getcwd()).parent, "logs")
    num_updates_per_iteration = 10
    seeds = [1, 2, 3, 5, 6]
    env = "lunar_lander_2d_heavytailed_wide"
    # num_updates_per_iteration = 5
    # seeds = [str(i) for i in range(1, 11)]
    # env = "point_mass_2d_heavytailed_wide"
    figname_extra = "_fractions_noshades"

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
            "RAC": {
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
            "RAC-N": {
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
            "CUR": {
                "algorithm": "wasserstein",
                "label": "CURROT",
                "model": "ppo_DELTA=4.0_METRIC_EPS=0.5",
                "color": "green",
            },
            "Goal": {
                "algorithm": "goal_gan",
                "label": "GoalGAN",
                "model": "ppo_GG_FIT_RATE=200_GG_NOISE_LEVEL=0.05_GG_P_OLD=0.1",
                "color": "orange",
            },
            "ALP": {
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
        "lunar_lander_2d_heavytailed_wide":
            {
                "num_iters": 250,
                "steps_per_iter": 10240,
                "fontsize": 18,
                "figsize": (10, 5),
                "bbox_to_anchor": (.5, 1.17),
                "min_return": -100.,
                "max_return": 100.,
                "num_intervals": 100,
                "xticks": [-100, -46, -30, 62, 74, 82,100],
                "yticks": None, #[0.0, 1.0],
            },
        "point_mass_2d_heavytailed_wide":
            {
                "num_iters": 300,
                "steps_per_iter": 6144,
                "fontsize": 16,
                "figsize": (10, 5),
                "bbox_to_anchor": (.5, 1.4),
                "min_return": 0.,
                "max_return": 8.,
                "num_intervals": 100,
                "xticks": None,
                "yticks": None,
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
