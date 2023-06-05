import os
import sys
import math
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from itertools import combinations
from statannotations.Annotator import Annotator
sys.path.insert(1, os.path.join(sys.path[0], '..'))

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def get_results(base_dir, iteration):
    perf_file = os.path.join(base_dir, f"iteration-{iteration}", "performance.npy")
    if os.path.exists(perf_file):
        results = np.load(perf_file)
        returns = results[:, 1]
    else:
        print(f"No evaluation data found: {perf_file}")
        returns = []
    return np.array(returns)

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
    iteration = num_iters - num_updates_per_iteration

    df_med = pd.DataFrame()
    df_all = pd.DataFrame()
    color_palette = {}
    for cur_algo in algorithms:
        algorithm = algorithms[cur_algo]["algorithm"]
        label = algorithms[cur_algo]["label"]
        model = algorithms[cur_algo]["model"]
        color = algorithms[cur_algo]["color"]
        print(algorithm)

        returns = []
        color_palette[label] = color
        for seed in seeds:
            base_dir = os.path.join(base_log_dir, env, algorithm, model, f"seed-{seed}")
            print(base_dir)
            returns_seed = get_results(
                base_dir=base_dir,
                iteration=iteration,
            )
            if len(returns_seed) == 0:
                continue

            returns.append(returns_seed)

        returns = np.array(returns)
        returns_med = np.median(returns, axis=0)
        returns_all = returns.flatten()
        df_med_alg = pd.DataFrame()
        df_med_alg["algorithm"] = list(label for i in range(returns_med.shape[0]))
        df_med_alg["return"] = returns_med.tolist()
        df_med = pd.concat([df_med, df_med_alg])
        df_all_alg = pd.DataFrame()
        df_all_alg["algorithm"] = list(label for i in range(returns_all.shape[0]))
        df_all_alg["return"] = returns_all.tolist()
        df_all = pd.concat([df_all, df_all_alg])
        # df_med[label] = returns_med.tolist()
        # df_all[label] = returns_all.tolist()

    ord_df = df_all.groupby(["algorithm"])["return"].median().sort_values(ascending=False)
    ord_labels = [x.strip() for x in ord_df.index]

    figname = ""
    for cur_algo_i, cur_algo in enumerate(algorithms):
        figname += cur_algo
        if cur_algo_i < len(algorithms)-1:
            figname += "_vs_"

    f, ax = plt.subplots(figsize=figsize)
    sns.boxplot(data=df_med, x="algorithm", y="return", order=ord_labels, palette=color_palette)
    ax.xaxis.grid(True)
    ax.set(ylabel="Discounted Return", xlabel="Algorithm")
    sns.despine(trim=True, left=True)
    # pairs = list(comb for comb in combinations(ord_labels, r=2) if "RACGEN" in comb)
    # # pairs = list(comb for comb in combinations(ord_labels, r=2))
    # annotator = Annotator(ax, pairs, data=df_all, x="algorithm", y="return", order=ord_labels)
    # annotator.configure(test="t-test_welch", loc='outside')
    # annotator.configure(
    #     # test="Mann-Whitney",
    #     test="t-test_welch",
    #     loc='outside',
    #     text_format="star",
    #     line_height=0.01,
    #     fontsize="small",
    #     # pvalue_thresholds=[[0.0001, '****'], [0.001, '***'], [0.01, '**'], [0.05, '*'], [1, 'ns']],
    # )
    # print(annotator.get_configuration())
    # annotator.apply_and_annotate()
    plt.savefig(f"{Path(os.getcwd()).parent}\\figures\\{env}_{figname}{figname_extra}_stat_med.pdf", dpi=500,
                bbox_inches='tight')

    f, ax = plt.subplots(figsize=figsize)
    sns.boxplot(data=df_all, x="algorithm", y="return", order=ord_labels, palette=color_palette)
    ax.xaxis.grid(True)
    ax.set(ylabel="Discounted Return", xlabel="Algorithm")
    sns.despine(trim=True, left=True)
    # pairs = list(comb for comb in combinations(ord_labels, r=2) if "RACGEN" in comb)
    # # pairs = list(comb for comb in combinations(ord_labels, r=2))
    # annotator = Annotator(ax, pairs, data=df_all, x="algorithm", y="return", order=ord_labels)
    # annotator.configure(test="t-test_welch", loc='outside')
    # annotator.configure(
    #     # test="Mann-Whitney",
    #     test="t-test_welch",
    #     loc='outside',
    #     text_format="star",
    #     line_height=0.01,
    #     fontsize="small",
    #     # pvalue_thresholds=[[0.0001, '****'], [0.001, '***'], [0.01, '**'], [0.05, '*'], [1, 'ns']],
    # )
    # print(annotator.get_configuration())
    # annotator.apply_and_annotate()
    plt.savefig(f"{Path(os.getcwd()).parent}\\figures\\{env}_{figname}{figname_extra}_stat_all.pdf", dpi=500,
                bbox_inches='tight')

def main():
    base_log_dir = f"{Path(os.getcwd()).parent}\\logs"
    num_updates_per_iteration = 10
    seeds = [1, 2, 3, 5, 6]
    env = "lunar_lander_2d_heavytailed_wide"
    # num_updates_per_iteration = 5
    # seeds = [str(i) for i in range(1, 11)]
    # env = "point_mass_2d_heavytailed_wide"
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
                "label": "RAC\nGEN",
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
                "label": "Default\n-CEM",
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
                "label": "Goal\nGAN",
                "model": "ppo_GG_FIT_RATE=200_GG_NOISE_LEVEL=0.1_GG_P_OLD=0.2",
                "color": "orange",
            },
            "ALP-GMM": {
                "algorithm": "alp_gmm",
                "label": "ALP\n-GMM",
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
                "label": "RAC\nGEN",
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
                "label": "RAC\nGEN-N",
                "model": "ppo_DELTA=4.0_DIST_TYPE=gaussian_KL_EPS=0.25_RALPH=0.2_RALPH_IN=1.0_RALPH_SCH=20",
                "color": "cyan",
            },
            "DEF-CEM": {
                "algorithm": "default_with_cem",
                "label": "Default\n-CEM",
                "model": "ppo_DIST_TYPE=cauchy_RALPH=0.2_RALPH_IN=1.0_RALPH_SCH=20",
                "color": "maroon",
            },
            "CUR": {
                "algorithm": "wasserstein",
                "label": "CUR\nROT",
                "model": "ppo_DELTA=4.0_METRIC_EPS=0.5",
                "color": "green",
            },
            "Goal": {
                "algorithm": "goal_gan",
                "label": "Goal\nGAN",
                "model": "ppo_GG_FIT_RATE=200_GG_NOISE_LEVEL=0.05_GG_P_OLD=0.1",
                "color": "orange",
            },
            "ALP": {
                "algorithm": "alp_gmm",
                "label": "ALP\n-GMM",
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
                "bbox_to_anchor": (.5, 1.1),
                "ylabel": 'Expected discounted return',
                "ylim": [-150., 60.],
            },
        "point_mass_2d_heavytailed_wide":
            {
                "num_iters": 300,
                "steps_per_iter": 6144,
                "fontsize": 16,
                "figsize": (10, 5),
                "bbox_to_anchor": (.5, 1.1),
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
