# RangeGoalGridWorld-LevelA

`RangeGoalGridWorld` is a 2D hidden-goal benchmark for online belief
refinement. It is intended to make one transition ambiguous: each observed
range bin leaves many possible goal hypotheses, and the agent has to combine
evidence over time.

## Task

- Hidden context: fixed goal cell `g`
- Observed state: agent position, full obstacle map, coarse range bin
- Range signal: `y_t = Q(d_M(s_t, g))`
- Distance: obstacle-aware shortest-path distance, not Euclidean distance
- Action: up/down/left/right
- Reward: `+50` at goal, `-1` per step, extra `-0.2` on collision
- Episode end: goal reached or horizon reached

## PEARL Integration

The PEARL launcher expects Box observations/actions and the old Gym API, so
the registered PEARL envs are:

- `range-goal-gridworld`
- `range-goal-gridworld-ambiguous`

They expose:

- flat observation: `[agent_pos_normalized, map_flattened, range_bin_one_hot]`
- Box action: shape `(4,)`; `argmax(action)` selects the discrete move
- task API: `get_all_task_idx()` and `reset_task(idx)`

Example:

```bash
python launch_experiment.py configs/range-goal-gridworld-varibad-smoke.json
python launch_experiment.py configs/range-goal-gridworld-pearl-smoke.json
python launch_experiment.py configs/range-goal-gridworld-cfm-current-gauss-smoke.json
```

## Direct Gymnasium Inspection

For direct benchmark inspection, importing `range_goal_gridworld` registers:

- `RangeGoalGridWorld-LevelA-15x15-v0`
- `RangeGoalGridWorld-LevelA-21x21-v0`
- `RangeGoalGridWorld-LevelA-Ambiguous-15x15-v0`

Example:

```python
import gymnasium as gym
import range_goal_gridworld

env = gym.make("RangeGoalGridWorld-LevelA-15x15-v0")
obs, info = env.reset(seed=0)

done = False
while not done:
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    done = terminated or truncated

posterior = env.unwrapped.compute_oracle_posterior()
support = env.unwrapped.compute_likelihood_support()
```

## Debug Tools

The environment exposes:

- `compute_likelihood_support()`: one-step candidate support
- `compute_oracle_posterior()`: posterior over goals using all observed bins
- `render_debug(show_goal=True, show_oracle=True)`: RGB debug render

The hidden goal is never included in the observation. It is available only in
`info` for evaluation/debugging.
