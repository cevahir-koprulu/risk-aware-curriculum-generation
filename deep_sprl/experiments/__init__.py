# Do this import to ensure that the Gym environments get registered properly
import deep_sprl.environments
from .point_mass_2d_experiment import PointMass2DExperiment
from .point_mass_2d_heavytailed_experiment import PointMass2DHeavyTailedExperiment
from .abstract_experiment import CurriculumType, Learner

__all__ = ['CurriculumType', 'PointMass2DExperiment', 'Learner',
           'PointMass2DHeavyTailedExperiment',
           ] 
