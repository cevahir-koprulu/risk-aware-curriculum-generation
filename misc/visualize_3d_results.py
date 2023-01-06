import os
import math
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib import cm
from pathlib import Path

PERFORMANCE_FILES = {
    0: "performance",
    1: "performance_rare",
    2: "performance_rare_uniform",
    3: "performance_hom",
}


def get_results(base_dir, seeds, iteration, context_dim, eval_type=0, result_type=0):
    all_res = []
    perf_filename = f"{PERFORMANCE_FILES[eval_type]}"

    res = []
    for seed in seeds:

        model_path = os.path.join(base_dir, f"seed-{seed}", f"iteration-{iteration}", f"{perf_filename}.npy")
        if os.path.exists(model_path):
            results = np.load(model_path)
            success = results[:, -1]
            disc_rewards = results[:, 1]
            eval_contexts = results[:, 2:2+context_dim]
            contexts_p = results[:, -2]

            if result_type == 0:
                res.append(disc_rewards)
            elif result_type == 1:
                res.append(success)
            else:
                print(f"Result type {result_type} is invalid: Should be 0 or 1")

        else:
            raise Exception(f"No evaluation data found: {model_path}")

    res = np.mean(np.array(res), axis=0)

    return eval_contexts, res, contexts_p



def plot_results(base_log_dir, figures_dir, seeds, env, setting, algorithms, eval_type,
                 result_type, figname_extra, view_angle):
    rc('font', **{'family': 'serif', 'serif': ['Times New Roman'], 'size': setting["fontsize"]})

    num_iters = setting["num_iters"]
    context_dim = setting["context_dim"]
    num_updates_per_iteration = setting["num_updates_per_iteration"]
    figure_setting = setting["figure"]

    iterations = np.arange(0, num_iters, num_updates_per_iteration)

    for cur_algo in algorithms:
        algorithm = algorithms[cur_algo]["algorithm"]
        label = algorithms[cur_algo]["label"]
        model = algorithms[cur_algo]["model"]
        color = algorithms[cur_algo]["color"]
        base_dir = os.path.join(base_log_dir, env, algorithm, model)

        for iteration in iterations:

            fig = plt.figure(figsize=(12, 10))
            ax = plt.axes(projection='3d')
            ax.view_init(view_angle[0], view_angle[1])

            contexts, results, contexts_p = get_results(
                base_dir=base_dir,
                seeds=seeds,
                iteration=iteration,
                context_dim=context_dim,
                eval_type=eval_type,
                result_type=result_type,
            )
            if eval_type != 3:
                surf = ax.scatter(xs=contexts[:, 0], ys=contexts[:, 1], zs=results,
                                  cmap=cm.coolwarm, c=results,
                                  )
            else:
                num_per_axis = int(math.sqrt(contexts[:, 0].shape[0]))
                xs = contexts[:, 0:1].reshape((num_per_axis, num_per_axis))
                ys = contexts[:, 1:].reshape((num_per_axis, num_per_axis))
                zs = results.reshape((num_per_axis, num_per_axis))
                surf = ax.plot_surface(X=xs, Y=ys, Z=zs,
                                       cmap=cm.coolwarm,
                                       )

            ax.set_title(f"Iteration {iteration}")
            ax.set_xlabel(figure_setting["x_label"])
            ax.set_ylabel(figure_setting["y_label"])
            if result_type == 0:
                ax.set_zlabel('Expected Return')
                ax.set_zlim([-.1, 10.])
                surf.set_clim(-.1, 10.)
            elif result_type == 1:
                ax.set_zlabel('Success Ratio')
                ax.set_zlim([-.05, 1.05])
                surf.set_clim(-.05, 1.05)
            fig.colorbar(surf, shrink=0.5, aspect=8)

            # if eval_type != 0:
            #     figname += f"_{PERFORMANCE_FILES[eval_type][len('performance')+1:]}"

            fig_dir = f"{Path(os.getcwd())}\\{figures_dir}\\eval_type={eval_type}\\result_type={result_type}\\view_angle={view_angle}"
            if not os.path.exists(fig_dir):
                os.makedirs(fig_dir)
            plt.savefig(f"{fig_dir}\\{env}_{algorithm}_iter={iteration}_{figname_extra}"
                        f"_eval={eval_type}_res={result_type}_view={view_angle}.png", dpi=500,
                        # bbox_inches='tight',
                        # bbox_extra_artists=(lgd,),
                        )
            plt.close()


def main():
    figures_dir = "figures\\3D"
    algorithms_for_env = {
        "point_mass_2d_heavytailed": {
            "self_paced": {
                "algorithm": "self_paced",
                "label": "self_paced",
                "model": "ppo_DELTA=4.0_KL_EPS=0.25",
                "color": "Blues",
                "marker": "x",
            },

            "self_paced_with_cem": {
                "algorithm": "self_paced_with_cem",
                "label": "self_paced_with_cem",
                "model": "ppo_DELTA=4.0_INTERNAL_ALPHA=0.5_KL_EPS=0.25_REF_ALPHA=0.2",
                "color": "Blues",
                "marker": "x",
            },
        },
        "point_mass_2d": {
            "self_paced": {
                "algorithm": "self_paced",
                "label": "self_paced",
                "model": "ppo_DELTA=4.0_KL_EPS=0.25",
                "color": "Blues",
                "marker": "x",
            },

            "self_paced_with_cem": {
                "algorithm": "self_paced_with_cem",
                "label": "self_paced_with_cem",
                "model": "ppo_DELTA=4.0_INTERNAL_ALPHA=0.5_KL_EPS=0.25_REF_ALPHA=0.2",
                "color": "Blues",
                "marker": "x",
            },
            "default": {
                "algorithm": "default",
                "label": "default",
                "model": "ppo",
                "color": "Blues",
                "marker": "x",
            },

            "default_with_cem": {
                "algorithm": "default_with_cem",
                "label": "default_with_cem",
                "model": "ppo_INTERNAL_ALPHA=0.5_REF_ALPHA=0.2",
                "color": "Blues",
                "marker": "x",
            },
        }
    }

    settings = {
        "point_mass_2d_heavytailed":
            {
                "context_dim": 2,
                "num_updates_per_iteration": 5,
                "num_iters": 200,
                "fontsize": 12,
                "figure": {
                    "x_label": "Door Position",
                    "y_label": "Door Width",
                }
            },

        "point_mass_2d":
            {
                "context_dim": 2,
                "num_updates_per_iteration": 5,
                "num_iters": 400,
                "fontsize": 12,
                "figure": {
                    "x_label": "Door Position",
                    "y_label": "Door Width",
                }
            },
    }


    parser = argparse.ArgumentParser("Visualize 3D Results")
    parser.add_argument("--base_log_dir", type=str, default="logs")
    parser.add_argument("--figname_extra", type=str, default="")
    parser.add_argument("--algorithms", nargs='+', default=["self_paced"],
                        choices=["default", "random", "self_paced", "wasserstein", "alp_gmm",
                                 "goal_gan", "acl", "plr", "vds", "self_paced_with_cem", "default_with_cem"])
    parser.add_argument("--env", type=str, default="point_mass_2d_heavytailed",
                        choices=["point_mass_2d", "maze", "point_mass_2d_heavytailed"])
    parser.add_argument('--seeds', nargs='+2', default=[1, 11], type=int)
    parser.add_argument("--eval_type", type=int, default=0)
    parser.add_argument("--result_type", type=int, default=0)
    parser.add_argument('--view_angle', default=[-90, 90], nargs='+', type=int)

    args, remainder = parser.parse_known_args()
    print(args)

    base_log_dir = f"{Path(os.getcwd())}\\{args.base_log_dir}"
    seeds = args.seeds
    env = args.env
    figname_extra = args.figname_extra
    eval_type = args.eval_type  # 0: eval, 1: rare, 2: rare uniform, 3: hom
    result_type = args.result_type  # 0: reward, 1: success
    view_angle = tuple(args.view_angle)

    algorithms = dict()
    for alg in args.algorithms:
        algorithms[alg] = algorithms_for_env[env][alg]

    plot_results(
        base_log_dir=base_log_dir,
        figures_dir=figures_dir,
        seeds=seeds,
        env=env,
        setting=settings[env],
        algorithms=algorithms,
        eval_type=eval_type,
        result_type=result_type,
        figname_extra=figname_extra,
        view_angle=view_angle,
        )


if __name__ == "__main__":
    main()