import torch

from .engine import ROTATIONS, Tetromino


class BatchedTetris:
    def __init__(self, population_size: int, width: int, height: int, device: torch.device):
        self.population_size = population_size
        self.width = width
        self.height = height
        self.device = device
        self.boards = torch.zeros((population_size, height, width), dtype=torch.bool, device=device)
        self.active = torch.ones(population_size, dtype=torch.bool, device=device)
        self.scores = torch.zeros(population_size, dtype=torch.int64, device=device)
        self.lines = torch.zeros(population_size, dtype=torch.int64, device=device)
        self.pieces = torch.zeros(population_size, dtype=torch.int64, device=device)
        self.rotated_moves = torch.zeros((), dtype=torch.int64, device=device)
        self.total_moves = torch.zeros((), dtype=torch.int64, device=device)
        self.geometry: dict[Tetromino, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = {}

    def active_indices(self) -> torch.Tensor:
        return torch.nonzero(self.active, as_tuple=False).flatten()

    def state_vectors(self, indices: torch.Tensor, piece: Tetromino) -> torch.Tensor:
        boards = self.boards[indices]
        occupied = boards.any(dim=1)
        top = boards.to(torch.int8).argmax(dim=1)
        heights = torch.where(occupied, self.height - top, 0)
        seen = boards.to(torch.int8).cumsum(dim=1) > 0
        holes = (seen & ~boards).sum(dim=(1, 2)).float() / 40.0
        bumpiness = torch.abs(heights[:, 1:] - heights[:, :-1]).sum(dim=1).float() / 40.0
        maximum = heights.max(dim=1).values.float() / self.height
        piece_values = torch.zeros((len(indices), len(Tetromino)), dtype=torch.float32, device=self.device)
        piece_values[:, int(piece) - 1] = 1.0
        features = torch.stack((holes, bumpiness, maximum), dim=1)
        return torch.cat((boards.flatten(1).float(), piece_values, heights.float() / self.height, features), dim=1)

    def apply_logits(self, indices: torch.Tensor, logits: torch.Tensor, piece: Tetromino) -> None:
        if len(indices) == 0:
            return
        boards = self.boards[indices]
        legal, landing, cell_x, cell_y = self._placements(boards, piece)
        has_legal = legal.any(dim=1)
        if (~has_legal).any():
            self.active[indices[~has_legal]] = False
        if not has_legal.any():
            return
        live_indices = indices[has_legal]
        live_legal = legal[has_legal]
        live_logits = logits[has_legal].masked_fill(~live_legal, float("-inf"))
        actions = live_logits.argmax(dim=1)
        drop_y = landing[has_legal].gather(1, actions.unsqueeze(1)).squeeze(1)
        selected_x = cell_x[actions]
        selected_y = cell_y[actions] + drop_y.unsqueeze(1)
        linear = selected_y * self.width + selected_x
        live_boards = self.boards[live_indices].clone()
        live_boards.flatten(1).scatter_(1, linear, True)
        full_rows = live_boards.all(dim=2)
        cleared = full_rows.sum(dim=1)
        live_boards = self._compact_rows(live_boards, full_rows)
        self.boards[live_indices] = live_boards
        rewards = torch.tensor((4, 104, 304, 504, 804), dtype=torch.int64, device=self.device)
        self.scores[live_indices] += rewards[cleared]
        self.lines[live_indices] += cleared
        self.pieces[live_indices] += 1
        rotations = actions // self.width
        actual_rotations = rotations % len(ROTATIONS[piece])
        self.rotated_moves += (actual_rotations != 0).sum()
        self.total_moves += len(live_indices)

    def fitness(self) -> torch.Tensor:
        occupied = self.boards.any(dim=1)
        top = self.boards.to(torch.int8).argmax(dim=1)
        heights = torch.where(occupied, self.height - top, 0)
        seen = self.boards.to(torch.int8).cumsum(dim=1) > 0
        holes = (seen & ~self.boards).sum(dim=(1, 2))
        bumpiness = torch.abs(heights[:, 1:] - heights[:, :-1]).sum(dim=1)
        return self.scores.float() + self.lines.float() * 500.0 + self.pieces.float() * 1.5 - heights.sum(dim=1).float() * 0.4 - holes.float() * 7.5 - bumpiness.float() * 0.25

    def rotation_rate(self) -> float:
        if int(self.total_moves.item()) == 0:
            return 0.0
        return float((self.rotated_moves.float() * 100.0 / self.total_moves).item())

    def _placements(self, boards: torch.Tensor, piece: Tetromino) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        cell_x, cell_y, positions, valid_geometry = self._geometry(piece)
        count = len(boards)
        actions = 4 * self.width
        source = boards.flatten(1).view(count, 1, 1, -1).expand(-1, actions, self.height, -1)
        gathered = torch.gather(source, 3, positions.unsqueeze(0).expand(count, -1, -1, -1))
        valid = valid_geometry.unsqueeze(0) & ~gathered.any(dim=3)
        blocked, first_invalid = torch.max((~valid).to(torch.int8), dim=2)
        first_invalid = torch.where(blocked.bool(), first_invalid, torch.full_like(first_invalid, self.height))
        landing = first_invalid - 1
        legal = landing >= 0
        return legal, landing, cell_x, cell_y

    def _geometry(self, piece: Tetromino) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        cached = self.geometry.get(piece)
        if cached is not None:
            return cached
        actions = 4 * self.width
        cell_x = torch.zeros((actions, 4), dtype=torch.int64, device=self.device)
        cell_y = torch.zeros((actions, 4), dtype=torch.int64, device=self.device)
        geometry_valid = torch.ones(actions, dtype=torch.bool, device=self.device)
        for action in range(actions):
            rotation = action // self.width
            x = action % self.width
            shape = ROTATIONS[piece][rotation % len(ROTATIONS[piece])]
            xs = [x + offset_x for offset_x, _ in shape]
            ys = [offset_y for _, offset_y in shape]
            cell_x[action] = torch.tensor(xs, dtype=torch.int64, device=self.device)
            cell_y[action] = torch.tensor(ys, dtype=torch.int64, device=self.device)
            geometry_valid[action] = max(xs) < self.width
        drops = torch.arange(self.height, dtype=torch.int64, device=self.device).view(1, self.height, 1)
        absolute_y = cell_y.unsqueeze(1) + drops
        valid = geometry_valid.view(actions, 1, 1) & (absolute_y < self.height).all(dim=2, keepdim=True)
        positions = absolute_y * self.width + cell_x.unsqueeze(1)
        positions = torch.where(valid, positions, torch.full_like(positions, -1))
        safe_positions = positions.clamp_min(0)
        result = (cell_x, cell_y, safe_positions, valid.squeeze(2))
        self.geometry[piece] = result
        return result

    def _compact_rows(self, boards: torch.Tensor, full_rows: torch.Tensor) -> torch.Tensor:
        shifts = full_rows.flip(1).to(torch.int64).cumsum(dim=1).flip(1)
        rows = torch.arange(self.height, dtype=torch.int64, device=self.device).view(1, self.height)
        targets = (rows + shifts).clamp_max(self.height - 1)
        values = (boards & ~full_rows.unsqueeze(2)).to(torch.int16)
        compacted = torch.zeros_like(values)
        compacted.scatter_add_(1, targets.unsqueeze(2).expand(-1, -1, self.width), values)
        return compacted.bool()
