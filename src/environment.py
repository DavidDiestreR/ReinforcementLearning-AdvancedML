import csv
import random


class GridEnvironment:
    ACTIONS = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1), (0, 0), (0, 1),
        (1, -1), (1, 0), (1, 1),
    ]

    TOKEN_MAP = {
        "0": "empty",
        "1": "obstacle",
        "2": "target",
        "3": "start",
    }

    def __init__(self, grid_path, max_speed=2, seed=None):
        self.grid = self._load_grid(grid_path)
        self.height = len(self.grid)
        self.width = len(self.grid[0])
        self.max_speed = max_speed
        self.rng = random.Random(seed)
        self.actions = self.ACTIONS.copy()
        self.start_cells = [(x, y) for y, row in enumerate(self.grid) for x, cell in enumerate(row) if cell == "start"]

        if not self.start_cells:
            raise ValueError("The grid has no starting cells.")

    def reset(self):
        x, y = self.rng.choice(self.start_cells)
        return (x, y, 0, 0)

    def step(self, state, action):
        x, y, vx, vy = state
        ax, ay = action

        vx = max(-self.max_speed, min(self.max_speed, vx + ax))
        vy = max(-self.max_speed, min(self.max_speed, vy + ay))

        if self.grid[y][x] != "start" and vx == 0 and vy == 0:
            vx, vy = state[2], state[3]

        new_x = x + vx
        new_y = y - vy

        for px, py in self._trace_path(x, y, new_x, new_y):
            if px < 0 or px >= self.width or py < 0 or py >= self.height:
                return self.reset(), -1, False

            cell = self.grid[py][px]
            if cell == "obstacle":
                return self.reset(), -1, False
            if cell == "target":
                return (px, py, vx, vy), 0, True

        return (new_x, new_y, vx, vy), -1, False

    def _load_grid(self, grid_path):
        with open(grid_path, "r", newline="") as file:
            rows = [[self.TOKEN_MAP[token.strip()] for token in row] for row in csv.reader(file) if row]

        if not rows:
            raise ValueError("The grid file is empty.")

        width = len(rows[0])
        if any(len(row) != width for row in rows):
            raise ValueError("All rows in the grid must have the same length.")

        return rows

    def _trace_path(self, x0, y0, x1, y1):
        steps = max(abs(x1 - x0), abs(y1 - y0))
        if steps == 0:
            return []

        path = []
        for i in range(1, steps + 1):
            cell = (round(x0 + (x1 - x0) * i / steps), round(y0 + (y1 - y0) * i / steps))
            if not path or cell != path[-1]:
                path.append(cell)

        return path