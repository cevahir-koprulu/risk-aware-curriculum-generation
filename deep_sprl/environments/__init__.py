from gym.envs.registration import register

register(
    id='ContextualPointMass2D-v1',
    max_episode_steps=200,
    entry_point='deep_sprl.environments.contextual_point_mass_2d:ContextualPointMass2D'
)

register(
    id='bipedal-walker-continuous-v0',
    entry_point='deep_sprl.environments.teachDeepRLenvs.bipedal_walker_continuous:BipedalWalkerContinuous',
    max_episode_steps=2000,
    reward_threshold=300,
)

register(
    id='ContextualBipedalWalker-v1',
    entry_point='deep_sprl.environments.contextual_bipedal_walker:ContextualBipedalWalker',
    max_episode_steps=2000,
    reward_threshold=300,
)

register(
    id='ContextualBipedalWalker2D-v1',
    entry_point='deep_sprl.environments.contextual_bipedal_walker_2d:ContextualBipedalWalker2D',
    max_episode_steps=2000,
    reward_threshold=300,
)

register(
    id='ContextualLunarLander-v1',
    max_episode_steps=1000,
    reward_threshold=200,
    entry_point='deep_sprl.environments.contextual_lunar_lander:ContextualLunarLander'
)

register(
    id='ContextualLunarLander2D-v1',
    max_episode_steps=1000,
    reward_threshold=200,
    entry_point='deep_sprl.environments.contextual_lunar_lander_2d:ContextualLunarLander2D'
)