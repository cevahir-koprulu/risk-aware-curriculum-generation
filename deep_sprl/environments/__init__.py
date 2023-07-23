from gym.envs.registration import register

register(
    id='ContextualPointMass2D-v1',
    max_episode_steps=200,
    entry_point='deep_sprl.environments.contextual_point_mass_2d:ContextualPointMass2D'
)

register(
    id='ContextualPointMass3D-v1',
    max_episode_steps=200,
    entry_point='deep_sprl.environments.contextual_point_mass_3d:ContextualPointMass3D'
)