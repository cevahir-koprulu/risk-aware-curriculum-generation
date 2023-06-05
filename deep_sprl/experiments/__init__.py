# Do this import to ensure that the Gym environments get registered properly
import deep_sprl.environments
from .point_mass_2d_experiment import PointMass2DExperiment
from .point_mass_2d_heavytailed_experiment import PointMass2DHeavyTailedExperiment
from .bipedal_walker_2d_heavytailed_experiment import BipedalWalker2DHeavyTailedExperiment
from .lunar_lander_2d_heavytailed_experiment import LunarLander2DHeavyTailedExperiment
from .abstract_experiment import CurriculumType, Learner

__all__ = ['CurriculumType', 'PointMass2DExperiment', 'Learner',
           'PointMass2DHeavyTailedExperiment', 'BipedalWalker2DHeavyTailedExperiment',
           'LunarLander2DHeavyTailedExperiment',
           ] 
