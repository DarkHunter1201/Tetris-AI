import torch

from .engine import FitnessWeights, ROTATIONS, RuleWeights, Tetromino


class BatchedTetris:
    def __init__(self, population_size: int, width: int, height: int, device: torch.device, rule_weights: RuleWeights = RuleWeights(), fitness_weights: FitnessWeights = FitnessWeights(), network_policy_weight: float = 1.5):
        self.population_size = population_size
        self.width = width
        self.height = height
        self.device = device
        self.rule_weights = rule_weights
        self.fitness_weights = fitness_weights
        self.network_policy_weight = network_policy_weight
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
        heights, holes, bumpiness, maximum = self._metrics(boards)
        piece_values = torch.zeros((len(indices), len(Tetromino)), dtype=torch.float32, device=self.device)
        piece_values[:, int(piece) - 1] = 1.0
        features = torch.stack((holes.float() / 40.0, bumpiness.float() / 40.0, maximum.float() / self.height), dim=1)
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
        candidates, cleared = self._candidate_boards(boards, landing, cell_x, cell_y)
        flat_candidates = candidates.flatten(0, 1)
        heights, holes, bumpiness, maximum = self._metrics(flat_candidates)
        actions = 4 * self.width
        aggregate = heights.sum(dim=1).view(-1, actions)
        holes = holes.view(-1, actions)
        bumpiness = bumpiness.view(-1, actions)
        maximum = maximum.view(-1, actions)
        rules = cleared.float() * self.rule_weights.completed_lines - aggregate.float() * self.rule_weights.aggregate_height - holes.float() * self.rule_weights.holes - bumpiness.float() * self.rule_weights.bumpiness - maximum.float() * self.rule_weights.maximum_height
        combined = rules + torch.tanh(logits) * self.network_policy_weight
        combined = combined.masked_fill(~legal, float("-inf"))
        live_indices = indices[has_legal]
        selected_actions = combined[has_legal].argmax(dim=1)
        selected_boards = candidates[has_legal, selected_actions]
        selected_cleared = cleared[has_legal, selected_actions]
        self.boards[live_indices] = selected_boards
        rewards = torch.tensor((4, 104, 304, 504, 804), dtype=torch.int64, device=self.device)
        self.scores[live_indices] += rewards[selected_cleared]
        self.lines[live_indices] += selected_cleared
        self.pieces[live_indices] += 1
        rotations = selected_actions // self.width
        actual_rotations = rotations % len(ROTATIONS[piece])
        self.rotated_moves += (actual_rotations != 0).sum()
        self.total_moves += len(live_indices)

    def fitness(self) -> torch.Tensor:
        heights, holes, bumpiness, maximum = self._metrics(self.boards)
        weights = self.fitness_weights
        penalty = heights.sum(dim=1).float() * weights.aggregate_height + holes.float() * weights.holes + bumpiness.float() * weights.bumpiness + maximum.float() * weights.maximum_height
        terminal = (~self.active).float() * weights.game_over
        return self.scores.float() + self.lines.float() * weights.completed_lines + self.pieces.float() * weights.placed_pieces - penalty - terminal

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

    def _candidate_boards(self, boards: torch.Tensor, landing: torch.Tensor, cell_x: torch.Tensor, cell_y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        count = len(boards)
        actions = 4 * self.width
        candidates = boards.unsqueeze(1).expand(-1, actions, -1, -1).clone()
        y = landing.clamp_min(0).unsqueeze(2) + cell_y.unsqueeze(0)
        x = cell_x.unsqueeze(0).expand(count, -1, -1)
        linear = (y * self.width + x).clamp(0, self.height * self.width - 1)
        candidates.flatten(2).scatter_(2, linear, True)
        full_rows = candidates.all(dim=3)
        cleared = full_rows.sum(dim=2)
        compacted = self._compact_rows(candidates.flatten(0, 1), full_rows.flatten(0, 1))
        return compacted.view(count, actions, self.height, self.width), cleared

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
        safe_positions = torch.where(valid, positions, torch.zeros_like(positions))
        result = (cell_x, cell_y, safe_positions, valid.squeeze(2))
        self.geometry[piece] = result
        return result

    def _metrics(self, boards: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        occupied = boards.any(dim=1)
        top = boards.to(torch.int8).argmax(dim=1)
        heights = torch.where(occupied, self.height - top, 0)
        seen = boards.to(torch.int8).cumsum(dim=1) > 0
        holes = (seen & ~boards).sum(dim=(1, 2))
        bumpiness = torch.abs(heights[:, 1:] - heights[:, :-1]).sum(dim=1)
        maximum = heights.max(dim=1).values
        return heights, holes, bumpiness, maximum

    def _compact_rows(self, boards: torch.Tensor, full_rows: torch.Tensor) -> torch.Tensor:
        shifts = full_rows.flip(1).to(torch.int64).cumsum(dim=1).flip(1)
        rows = torch.arange(self.height, dtype=torch.int64, device=self.device).view(1, self.height)
        targets = (rows + shifts).clamp_max(self.height - 1)
        values = (boards & ~full_rows.unsqueeze(2)).to(torch.int16)
        compacted = torch.zeros_like(values)
        compacted.scatter_add_(1, targets.unsqueeze(2).expand(-1, -1, self.width), values)
        return compacted.bool()
