# next_state, reward, done = env.step(state, action)
# next_state, reward, done = env.step(state, action)
# src/environment.py

import csv
import random


class GridEnvironment:
    """
    State = (x, y, vx, vy)

    x  -> column
    y  -> row
    vx -> horizontal velocity  (positive = right)
    vy -> vertical velocity    (positive = up)

    step(state, action) returns:
        next_state, reward, done
    """

    # 9 possible actions: change vx by {-1,0,1} and vy by {-1,0,1}
    ACTIONS = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),  (0, 0),  (0, 1),
        (1, -1),  (1, 0),  (1, 1),
    ]

    # You can adapt this if your CSV uses other symbols
    TOKEN_MAP = {
        "0": "empty",
        ".": "empty",
        "": "empty",

        "1": "obstacle",
        "#": "obstacle",
        "X": "obstacle",

        "2": "target",
        "T": "target",
        "G": "target",

        "3": "start",
        "S": "start",
    }

    def __init__(self, grid_path, max_speed=2, seed=None):
        """
        max_speed=2 means vx, vy are clipped to [-2, 2]
        because the statement says the velocity components must be < 3.
        """
        self.grid = self._load_grid(grid_path)
        self.height = len(self.grid)
        self.width = len(self.grid[0])
        self.max_speed = max_speed
        self.rng = random.Random(seed)

        self.actions = self.ACTIONS.copy()
        self.start_cells = self._find_cells("start")
        self.target_cells = self._find_cells("target")

        if not self.start_cells:
            raise ValueError("The grid has no starting cells.")
        if not self.target_cells:
            raise ValueError("The grid has no target cells.")

    # ---------- Public methods ----------

    def reset(self):
        """Start a new episode from a random start cell with zero velocity."""
        x, y = self.rng.choice(self.start_cells)
        return (x, y, 0, 0)

    def step(self, state, action):
        """
        Input:
            state  = (x, y, vx, vy)
            action = (ax, ay), where each component is -1, 0 or 1

        Output:
            next_state, reward, done
        """
        x, y, vx, vy = state
        ax, ay = action

        # 1) Update velocity
        new_vx = self._clip(vx + ax)
        new_vy = self._clip(vy + ay)

        # 2) Avoid zero velocity outside the starting line
        #    If an action would make velocity (0,0) outside start,
        #    we keep the previous velocity.
        if not self._is_start_cell(x, y) and new_vx == 0 and new_vy == 0:
            new_vx, new_vy = vx, vy

        # 3) Compute candidate next position
        #    y decreases when vy is positive because row 0 is at the top
        new_x = x + new_vx
        new_y = y - new_vy

        # 4) Check every cell crossed along the path
        path_cells = self._trace_path(x, y, new_x, new_y)

        for px, py in path_cells:
            # Collision with wall
            if not self._inside(px, py):
                return self.reset(), -1, False

            cell_type = self._cell(px, py)

            # Collision with obstacle
            if cell_type == "obstacle":
                return self.reset(), -1, False

            # Reached target
            if cell_type == "target":
                return (px, py, new_vx, new_vy), 0, True

        # 5) Normal move
        return (new_x, new_y, new_vx, new_vy), -1, False

    # ---------- Internal helpers ----------

    def _load_grid(self, grid_path):
        grid = []

        with open(grid_path, "r", newline="") as f:
            reader = csv.reader(f)
            for row in reader:
                clean_row = []
                for token in row:
                    token = token.strip()
                    if token not in self.TOKEN_MAP:
                        raise ValueError(
                            f"Unknown token '{token}' in grid. "
                            f"Change TOKEN_MAP in environment.py."
                        )
                    clean_row.append(self.TOKEN_MAP[token])

                if clean_row:  # ignore empty rows
                    grid.append(clean_row)

        if not grid:
            raise ValueError("The grid file is empty.")

        width = len(grid[0])
        for row in grid:
            if len(row) != width:
                raise ValueError("All rows in the grid must have the same length.")

        return grid

    def _inside(self, x, y):
        return 0 <= x < self.width and 0 <= y < self.height


    def _trace_path(self, x0, y0, x1, y1):
        """
        Return all intermediate cells visited from (x0, y0) to (x1, y1),
        excluding the starting cell.

        This is important because with velocity > 1 the robot may cross
        walls/obstacles/target before reaching the final cell.
        """
        steps = max(abs(x1 - x0), abs(y1 - y0))

        if steps == 0:
            return []

        path = []
        last_cell = None

        for i in range(1, steps + 1):
            px = round(x0 + (x1 - x0) * i / steps)
            py = round(y0 + (y1 - y0) * i / steps)

            if (px, py) != last_cell:
                path.append((px, py))
                last_cell = (px, py)

        return path