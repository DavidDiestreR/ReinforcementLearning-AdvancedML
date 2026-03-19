import argparse
import pickle
from pathlib import Path

from src.environment import GridEnvironment
from src.policy import greedy_policy


def build_parser():
    parser = argparse.ArgumentParser(
        description="Run one greedy episode from a saved Q-table and save the resulting path data."
    )
    parser.add_argument(
        "q_path",
        help="Path to the Q-table relative to output/, e.g. grid_1/q_mc_episodes_1000_gamma_0_9_epsilon_0_1.pkl",
    )
    parser.add_argument(
        "--start-x",
        type=int,
        default=None,
        help="Starting x position. If omitted, a random start cell is used.",
    )
    return parser


def load_q_table(q_file):
    with q_file.open("rb") as file:
        return pickle.load(file)


def infer_grid_path(q_file):
    grid_name = q_file.parent.name
    grid_path = Path("data") / "grids" / f"{grid_name}.csv"

    if not grid_path.exists():
        raise FileNotFoundError(f"Grid file not found for Q-table: {grid_path}")

    return grid_path


def get_start_state(env, start_x):
    if start_x is None:
        return env.reset()

    valid_x_positions = sorted(x for x, _ in env.start_cells)

    if start_x not in valid_x_positions:
        raise ValueError(
            f"Starting x position {start_x} is not valid. Allowed x positions: {valid_x_positions}"
        )

    start_y = next(y for x, y in env.start_cells if x == start_x)
    return (start_x, start_y, 0, 0)


def run_greedy_episode(env, q, start_state):
    state = start_state
    path = [state]

    while True:
        action = greedy_policy(state, q)
        next_state, reward, done = env.step(state, action)
        path.append(next_state)
        state = next_state

        if done:
            return path, reward


def build_output_path(q_file):
    if not q_file.name.startswith("q_"):
        raise ValueError("Expected a Q-table file starting with 'q_'.")

    return q_file.with_name(q_file.name.replace("q_", "path_", 1))


def save_path(path_file, path_data):
    path_file.parent.mkdir(parents=True, exist_ok=True)
    with path_file.open("wb") as file:
        pickle.dump(path_data, file)


def main():
    args = build_parser().parse_args()
    q_file = Path("output") / args.q_path

    if not q_file.exists():
        raise FileNotFoundError(f"Q-table not found: {q_file}")

    q = load_q_table(q_file)
    grid_path = infer_grid_path(q_file)
    env = GridEnvironment(grid_path)
    start_state = get_start_state(env, args.start_x)

    path, final_reward = run_greedy_episode(env, q, start_state)
    output_path = build_output_path(q_file)
    save_path(
        output_path,
        {
            "grid_path": str(grid_path),
            "q_path": str(q_file),
            "path": path,
            "final_reward": final_reward,
            "start_state": start_state,
        },
    )

    print(f"Saved path to: {output_path}")
    print(f"States in path: {len(path)}")


if __name__ == "__main__":
    main()
