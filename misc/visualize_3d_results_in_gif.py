from PIL import Image
from pathlib import Path
import numpy as np
import os
import argparse

def create_gif(figures_dir, env, algorithms, eval_type, result_type, figname_extra, view_angle):

    for algorithm in algorithms:
        dir = f"{Path(os.getcwd())}\\{figures_dir}\\eval_type={eval_type}\\result_type={result_type}\\view_angle={view_angle}"
        files = [file for file in os.listdir(dir) if (f"{env}_{algorithm}_iter=" in file and
                 figname_extra in file and "gif" not in file)]
        files.sort(key=lambda x: int(x[len(f"{env}_{algorithm}_iter="):-len(f"_{figname_extra}_eval={eval_type}_res={result_type}_view={view_angle}.png")]))
        print(files)

        frames = []
        for filename in files:
            new_frame = Image.open(f"{dir}\\{filename}")
            frames.append(new_frame)

        # Save into a GIF file that loops forever
        frames[0].save(
            f"{dir}\\{env}_{algorithm}_eval={eval_type}_res={result_type}_view={view_angle}.gif",
            format='GIF',
            append_images=frames[1:],
            save_all=True,
            duration=500, loop=0)


def main():
    figures_dir = "figures\\3D"

    parser = argparse.ArgumentParser("Visualize 3D Results in GIF")
    parser.add_argument("--figname_extra", type=str, default="")
    parser.add_argument("--algorithms", nargs='+', default=["self_paced"],
                        choices=["default", "random", "self_paced", "wasserstein", "alp_gmm",
                                 "goal_gan", "acl", "plr", "vds", "self_paced_with_cem", "default_with_cem"])
    parser.add_argument("--env", type=str, default="point_mass_2d_heavytailed",
                        choices=["point_mass_2d", "maze", "point_mass_2d_heavytailed"])
    parser.add_argument("--eval_type", type=int, default=0)
    parser.add_argument("--result_type", type=int, default=0)
    parser.add_argument('--view_angle', default=[-90, 90], nargs='+', type=int)

    args, remainder = parser.parse_known_args()
    print(args)

    env = args.env
    figname_extra = args.figname_extra
    eval_type = args.eval_type  # 0: eval, 1: rare, 2: rare uniform, 3: hom
    result_type = args.result_type  # 0: reward, 1: success
    view_angle = tuple(args.view_angle)
    algorithms = args.algorithms

    create_gif(
        figures_dir=figures_dir,
        env=env,
        algorithms=algorithms,
        eval_type=eval_type,
        result_type=result_type,
        figname_extra=figname_extra,
        view_angle=view_angle,
    )


if __name__ == "__main__":
    main()