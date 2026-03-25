# ReinforcementLearning-AdvancedML

Course project for Advanced ML focused on solving a grid-based pathfinding task with on-policy first-visit Monte Carlo reinforcement learning.

## Project structure

```text
ReinforcementLearning-AdvancedML/
├─ data/
│  └─ grids/
│     ├─ grid_1.csv
│     ├─ grid_2.csv
│     └─ grid_3.csv
├─ notebooks/
│  └─ figures.ipynb
├─ output/
│  ├─ grid_1/
│  ├─ grid_2/
│  └─ grid_3/
├─ src/
│  ├─ environment.py
│  ├─ policy.py
│  └─ __init__.py
├─ on_policy_first_visit_mc_rl.py
└─ README.md
```

## Core idea of `on_policy_first_visit_mc_rl.py`

`on_policy_first_visit_mc_rl.py` trains an agent with on-policy first-visit Monte Carlo control. The agent interacts with a racing-style grid environment where each state is defined by position and velocity `(x, y, vx, vy)`, actions modify velocity, and the policy is epsilon-greedy over learned Q-values.

The script generates episodes, computes discounted returns, and updates each state-action pair only on its first visit within an episode. Exploration starts at the chosen `epsilon` value and decays linearly until `epsilon-min`.

## Environment and grid format

The environment is implemented in `src/environment.py`.

Grid CSV legend:

- `0`: empty cell
- `1`: obstacle
- `2`: target cell
- `3`: starting cell

Main environment rules:

- The agent starts from one of the `3` cells with zero velocity.
- Each action applies an acceleration from the 9 possible combinations in `{ -1, 0, 1 } x { -1, 0, 1 }`.
- Velocity is clipped to the range `[-2, 2]` on both axes.
- Hitting an obstacle or going out of bounds resets the agent to a start cell and gives reward `-1`.
- Reaching the target ends the episode with reward `0`.
- All non-terminal transitions give reward `-1`, so the agent is encouraged to reach the goal in as few steps as possible.

## Setup

Create a Conda environment:

```bash
conda create -n rl-advancedml python=3.11 -y
conda activate rl-advancedml
```

Install the libraries used in the project and notebook:

```bash
conda install -c conda-forge matplotlib numpy pandas jupyter notebook ipykernel
```

The training script itself only uses the Python standard library. The extra packages are needed for `notebooks/figures.ipynb`.

## How to run

Train on the default grid with default hyperparameters:

```bash
python on_policy_first_visit_mc_rl.py
```

Run with a specific grid and custom parameters:

```bash
python on_policy_first_visit_mc_rl.py --gridfilename grid_1.csv --gamma 0.9 --num-episodes 1000000 --epsilon 0.3 --epsilon-min 0.01
```

Example configuration used in this project:

```bash
python on_policy_first_visit_mc_rl.py --gridfilename grid_1.csv --epsilon 0.4 --num-episodes 1000000
python on_policy_first_visit_mc_rl.py --gridfilename grid_2.csv --epsilon 0.4 --num-episodes 1000000
python on_policy_first_visit_mc_rl.py --gridfilename grid_3.csv --epsilon 0.4 --num-episodes 1000000
```

For the full list of arguments:

```bash
python on_policy_first_visit_mc_rl.py -h
```

## Outputs

After training, the learned Q-table is saved as a `.pkl` file inside `output/<grid_name>/`.

Example output path:

```text
output/grid_1/q_mc_episodes_1000000_gamma_0_9_epsilon_0_4.pkl
```

These files can be loaded later for analysis and visualization in `notebooks/figures.ipynb`.

## Notebook

`notebooks/figures.ipynb` is used to visualize the grids, inspect learned policies, and plot greedy paths from the start cells using a saved Q-table.