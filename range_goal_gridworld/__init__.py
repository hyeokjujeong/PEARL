"""Convenience import for the RangeGoalGridWorld benchmark."""

from rlkit.envs.range_goal_gridworld import (  # noqa: F401
    RangeGoalGridWorldEnv,
    RangeGoalGridWorldGymEnv,
    bfs_distance_field,
    compute_likelihood_support,
    compute_oracle_posterior,
    count_posterior_modes,
    posterior_entropy,
    quantize_distance,
)
