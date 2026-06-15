import argparse
import os
import sys
import time

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from rlkit.envs import ENVS


ACTION_NAMES = ("up", "down", "left", "right")


def soft_action(action_idx, strength=4.0):
    action = np.zeros(4, dtype=np.float32)
    action[int(action_idx)] = float(strength)
    return action


def oracle_greedy_action(env):
    best_action = 0
    best_distance = np.inf
    for action_idx in range(4):
        next_pos, collided = env._transition(env.agent_pos, action_idx)
        if collided:
            continue
        distance = float(env.goal_distance_field[next_pos])
        if distance < best_distance:
            best_distance = distance
            best_action = action_idx
    return soft_action(best_action), best_action


def random_valid_action(env, rng):
    valid = np.flatnonzero(env._valid_action_mask(env.agent_pos) > 0.0)
    action_idx = int(rng.choice(valid)) if len(valid) else int(rng.randint(4))
    return soft_action(action_idx), action_idx


def ascii_frame(env, step, reward=0.0, done=False):
    posterior = env.compute_oracle_posterior()
    support = posterior > 0.0
    rows = []
    for i in range(env.size):
        row = []
        for j in range(env.size):
            pos = (i, j)
            if env.grid[pos] == 1:
                char = "#"
            elif pos == env.agent_pos:
                char = "A"
            elif pos == env.goal_pos:
                char = "G"
            elif pos in env.state_history[:-1]:
                char = "*"
            elif support[pos]:
                char = "?"
            else:
                char = "."
            row.append(char)
        rows.append(" ".join(row))
    print("\n".join(rows))
    print(
        "step={} reward={:.2f} done={} bin={} dist={:.1f} support={} modes={}".format(
            step,
            reward,
            done,
            env.range_bin_history[-1],
            float(env.goal_distance_field[env.agent_pos]),
            int(np.sum(support)),
            _count_modes(env, posterior),
        )
    )
    print()


def _count_modes(env, posterior):
    from rlkit.envs.range_goal_gridworld import count_posterior_modes
    return count_posterior_modes(posterior)


def run_human_view(env, args):
    import matplotlib.pyplot as plt

    plt.ion()
    fig, ax = plt.subplots(figsize=(7, 7))
    image = env.render_debug(show_goal=args.show_goal, show_oracle=True)
    artist = ax.imshow(image)
    ax.axis("off")

    def update_title(step, reward=0.0, done=False, action_name="reset"):
        posterior = env.compute_oracle_posterior()
        support = int(np.sum(posterior > 0.0))
        modes = _count_modes(env, posterior)
        ax.set_title(
            "RangeGoal Main | task={} | step={} | action={} | reward={:.2f}\n"
            "range_bin={} | dist={:.1f} | posterior_support={} | modes={} | done={}".format(
                env._task_idx,
                step,
                action_name,
                reward,
                env.range_bin_history[-1],
                float(env.goal_distance_field[env.agent_pos]),
                support,
                modes,
                done,
            )
        )

    update_title(0)
    fig.canvas.draw()
    fig.canvas.flush_events()
    time.sleep(args.delay)

    rng = np.random.RandomState(args.seed + 17)
    done = False
    for step in range(1, args.steps + 1):
        if args.policy == "greedy-oracle-demo":
            action, intended_idx = oracle_greedy_action(env)
        else:
            action, intended_idx = random_valid_action(env, rng)
        _, reward, done, _ = env.step(action)
        image = env.render_debug(show_goal=args.show_goal, show_oracle=True)
        artist.set_data(image)
        update_title(step, reward, done, ACTION_NAMES[intended_idx])
        fig.canvas.draw()
        fig.canvas.flush_events()
        time.sleep(args.delay)
        if done:
            break

    if args.hold:
        print("Close the matplotlib window to finish.")
        plt.ioff()
        plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="range-goal-gridworld-main")
    parser.add_argument("--task", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--delay", type=float, default=0.35)
    parser.add_argument("--mode", choices=("human", "ascii"), default="human")
    parser.add_argument("--policy", choices=("greedy-oracle-demo", "random"), default="greedy-oracle-demo")
    parser.add_argument("--hide-goal", action="store_true")
    parser.add_argument("--hold", action="store_true")
    args = parser.parse_args()
    args.show_goal = not args.hide_goal

    env = ENVS[args.env](
        n_tasks=40,
        randomize_tasks=False,
        seed=args.seed,
        candidate_goal_mode="random",
        num_candidate_goals=None,
        include_oracle_in_info=False,
    )
    env.reset_task(args.task)

    if args.mode == "human":
        run_human_view(env, args)
        return

    rng = np.random.RandomState(args.seed + 17)
    ascii_frame(env, 0)
    done = False
    for step in range(1, args.steps + 1):
        if args.policy == "greedy-oracle-demo":
            action, _ = oracle_greedy_action(env)
        else:
            action, _ = random_valid_action(env, rng)
        _, reward, done, _ = env.step(action)
        ascii_frame(env, step, reward, done)
        time.sleep(args.delay)
        if done:
            break


if __name__ == "__main__":
    main()
