# Risk-aware Curriculum Generation for Heavy-tailed Task Distributions

Cevahir Koprulu (1), Thiago D. Simão (2), Nils Jansen (2) and Ufuk Topcu (1)

(1) The University of Texas at Austin

(2) Radbound University Nijmegen

Accepted for the 39th Conference on Uncertainty in Artificial Intelligence (UAI 2023).

Our codebase is built on the repository of _Curriculum Reinforcement Learning via Constrained Optimal Transport_ (CURROT) by Klink et al. (2022).

Web sources for CURROT:

Source code: https://github.com/psclklnk/currot/tree/icml (ICML branch)

Paper: https://proceedings.mlr.press/v162/klink22a.html

Cross-entropy methods are originally implemented in the repository of _Efficient Risk-Averse Reinforcement Learning_ (CeSoR) by Greenberg et al. (2022).

Web sources for CeSoR:

Source code: https://github.com/ido90/CeSoR

Paper: https://proceedings.neurips.cc/paper_files/paper/2022/hash/d2511dfb731fa336739782ba825cd98c-Abstract-Conference.html

We run our codebase on Ubuntu 20.04.5 LTS with Python 3.9.16

## Installation

The required packages are provided in a requirements.txt file which can be installed via the following command;
```bash
pip install -r requirements.txt
```

## How to run
To run a single experiment (training + evaluation), *run.py* can be called as follows (you can put additional parameters):
```bash
python run.py --train --eval 0 --env point_mass_2d_heavytailed --type self_paced_with_cem --target_type wide --DIST_TYPE cauchy --seed 1 # RACGEN
python run.py --train --eval 0 --env point_mass_2d_heavytailed --type self_paced_with_cem --target_type wide --DIST_TYPE gaussian --seed 1 # RACGEN-N
python run.py --train --eval 0 --env point_mass_2d_heavytailed --type self_paced --target_type wide --DIST_TYPE cauchy --seed 1 # SPDL
python run.py --train --eval 0 --env point_mass_2d_heavytailed --type self_paced --target_type wide --DIST_TYPE gaussian --seed 1 # SPDL-N
python run.py --train --eval 0 --env point_mass_2d_heavytailed --type default_with_cem --target_type wide --seed 1 # Default-CEM
python run.py --train --eval 0 --env point_mass_2d_heavytailed --type default --target_type wide --seed 1 # Default
python run.py --train --eval 0 --env point_mass_2d_heavytailed --type wasserstein --target_type wide --seed 1 # CURROT
python run.py --train --eval 0 --env point_mass_2d_heavytailed --type alp_gmm --target_type wide --seed 1 # ALP-GMM
python run.py --train --eval 0 --env point_mass_2d_heavytailed --type goal_gan --target_type wide --seed 1 # GoalGAN
python run.py --train --eval 0 --env point_mass_2d_heavytailed --type plr --target_type wide --seed 1 # PLR
python run.py --train --eval 0 --env point_mass_2d_heavytailed --type vds --target_type wide --seed 1 # VDS
```
The results demonstrated in our submitted paper can be run via *run_{environment_name}_experiments.py* by changing environment_name to one of the following:
- point_mass_2d_heavytailed_wide
- lunar_lander_2d_heavytailed_wide


## Evaluation
Under *misc* directory, there are three scripts:
1) *plot_expected_performance.py*: We use this script to plot the progression of expected return during training.
3) *plot_stat_return.py*: We run this script to obtain a box plot for the distribution of the discounted return obtained by final policies in contexts drawn from the target context distribution. One can use this script to run statistical significance tests, as we did for the paper.
4) *sample_eval_contexts.py*: We run this script to draw contexts from the target context distributions and record them to be used for evaluation of trained policies.