import argparse
import pickle
import sys
import time
from pathlib import Path

from src.environment import GridEnvironment
from src.policy import epsilon_greedy_policy


def initialize_q(env):
    q = {}
    velocities = range(-env.max_speed, env.max_speed + 1)

    for x in range(env.width):
        for y in range(env.height):
            for vx in velocities:
                for vy in velocities:
                    state = (x, y, vx, vy)
                    q[state] = {action: 0.0 for action in env.actions}

    return q


def generate_episode(env, q, epsilon, max_steps):
    episode = []
    state = env.reset()

    for _ in range(max_steps):
        action = epsilon_greedy_policy(state, q, epsilon)
        next_state, reward, done = env.step(state, action)
        episode.append((state, action, reward))
        state = next_state

        if done:
            break

    return episode


def get_epsilon(episode_number, num_episodes, epsilon_start, epsilon_min):
    progress = (episode_number - 1) / max(1, num_episodes - 1)
    return max(epsilon_min, epsilon_start - (epsilon_start - epsilon_min) * progress)


def update_q_first_visit(episode, returns, q, gamma):
    returns_from_t = []
    g = 0.0

    for _, _, reward in reversed(episode):
        g = gamma * g + reward
        returns_from_t.append(g)

    returns_from_t.reverse()
    seen = set()

    for (state, action, _), g in zip(episode, returns_from_t):
        state_action = (state, action)

        if state_action in seen:
            continue

        seen.add(state_action)
        returns.setdefault(state_action, []).append(g)
        q[state][action] = sum(returns[state_action]) / len(returns[state_action])


def on_policy_first_visit_mc_control(
    gridfilename="grid_1.csv",
    gamma=0.9,
    num_episodes=1000,
    epsilon=0.1,
    epsilon_min=0.01,
):
    grid_path = Path("data") / "grids" / gridfilename
    env = GridEnvironment(grid_path)
    q = initialize_q(env)
    returns = {}
    max_steps = env.width * env.height * 100
    start_time = time.perf_counter()

    for episode_number in range(1, num_episodes + 1):
        current_epsilon = get_epsilon(episode_number, num_episodes, epsilon, epsilon_min)
        episode = generate_episode(env, q, current_epsilon, max_steps)
        update_q_first_visit(episode, returns, q, gamma)

        sys.stdout.write(f"\r[{episode_number}/{num_episodes}] epsilon={current_epsilon:.4f}")
        sys.stdout.flush()

    runtime = time.perf_counter() - start_time
    print()
    print(f"Runtime: {runtime:.2f} seconds")

    output_dir = Path("output") / grid_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)
    gamma_text = str(gamma).replace(".", "_")
    epsilon_text = str(epsilon).replace(".", "_")
    output_path = output_dir / f"q_mc_episodes_{num_episodes}_gamma_{gamma_text}_epsilon_{epsilon_text}.pkl"

    with output_path.open("wb") as file:
        pickle.dump(q, file)

    print(f"Saved Q to: {output_path}")
    return q


def build_parser():
    parser = argparse.ArgumentParser(description="Simple on-policy first-visit Monte Carlo control.")
    parser.add_argument(
        "--gridfilename",
        default="grid_1.csv",
        help="Grid file inside data/grids. Default: grid_1.csv",
    )
    parser.add_argument("--gamma", type=float, default=0.9, help="Discount factor. Default: 0.9")
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=1000,
        help="Number of episodes. Default: 1000",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.1,
        help="Initial probability of choosing a non-greedy action. Default: 0.1",
    )
    parser.add_argument(
        "--epsilon-min",
        type=float,
        default=0.01,
        help="Minimum epsilon after decay. Default: 0.01",
    )
    return parser


def main():
    args = build_parser().parse_args()
    on_policy_first_visit_mc_control(
        gridfilename=args.gridfilename,
        gamma=args.gamma,
        num_episodes=args.num_episodes,
        epsilon=args.epsilon,
        epsilon_min=args.epsilon_min,
    )


if __name__ == "__main__":
    main()
