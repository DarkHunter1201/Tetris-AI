import random
from dataclasses import dataclass
from enum import IntEnum


class Tetromino(IntEnum):
    I = 1
    O = 2
    T = 3
    S = 4
    Z = 5
    J = 6
    L = 7


BASE_SHAPES = {
    Tetromino.I: ((0, 0), (1, 0), (2, 0), (3, 0)),
    Tetromino.O: ((0, 0), (1, 0), (0, 1), (1, 1)),
    Tetromino.T: ((0, 0), (1, 0), (2, 0), (1, 1)),
    Tetromino.S: ((1, 0), (2, 0), (0, 1), (1, 1)),
    Tetromino.Z: ((0, 0), (1, 0), (1, 1), (2, 1)),
    Tetromino.J: ((0, 0), (0, 1), (1, 1), (2, 1)),
    Tetromino.L: ((2, 0), (0, 1), (1, 1), (2, 1)),
}


def normalize_shape(cells: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    min_x = min(x for x, _ in cells)
    min_y = min(y for _, y in cells)
    return tuple(sorted((x - min_x, y - min_y) for x, y in cells))


def rotate_shape(cells: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    return normalize_shape(tuple((-y, x) for x, y in cells))


def build_rotations(cells: tuple[tuple[int, int], ...]) -> tuple[tuple[tuple[int, int], ...], ...]:
    rotations = []
    current = normalize_shape(cells)
    for _ in range(4):
        if current not in rotations:
            rotations.append(current)
        current = rotate_shape(current)
    return tuple(rotations)


ROTATIONS = {piece: build_rotations(cells) for piece, cells in BASE_SHAPES.items()}


def piece_sequence(seed: int, count: int) -> list[Tetromino]:
    generator = random.Random(seed)
    sequence = []
    bag: list[Tetromino] = []
    while len(sequence) < count:
        if not bag:
            bag = list(Tetromino)
            generator.shuffle(bag)
        sequence.append(bag.pop())
    return sequence


@dataclass(frozen=True)
class Placement:
    piece: Tetromino
    rotation: int
    x: int
    y: int


class TetrisGame:
    def __init__(self, width: int = 10, height: int = 20, seed: int | None = None):
        self.width = width
        self.height = height
        self.random = random.Random(seed)
        self.reset(seed)

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self.random.seed(seed)
        self.board = [[0 for _ in range(self.width)] for _ in range(self.height)]
        self.bag: list[Tetromino] = []
        self.current_piece = self._next_piece()
        self.next_piece = self._next_piece()
        self.score = 0
        self.lines = 0
        self.pieces = 0
        self.game_over = False

    def _next_piece(self) -> Tetromino:
        if not self.bag:
            self.bag = list(Tetromino)
            self.random.shuffle(self.bag)
        return self.bag.pop()

    def shape(self, rotation: int) -> tuple[tuple[int, int], ...]:
        rotations = ROTATIONS[self.current_piece]
        return rotations[rotation % len(rotations)]

    def collides(self, shape: tuple[tuple[int, int], ...], x: int, y: int) -> bool:
        for cell_x, cell_y in shape:
            board_x = x + cell_x
            board_y = y + cell_y
            if board_x < 0 or board_x >= self.width or board_y >= self.height:
                return True
            if board_y >= 0 and self.board[board_y][board_x]:
                return True
        return False

    def landing_y(self, shape: tuple[tuple[int, int], ...], x: int) -> int | None:
        if self.collides(shape, x, 0):
            return None
        y = 0
        while not self.collides(shape, x, y + 1):
            y += 1
        return y

    def placement_for_action(self, action: int) -> Placement | None:
        rotation = action // self.width
        x = action % self.width
        shape = self.shape(rotation)
        y = self.landing_y(shape, x)
        if y is None:
            return None
        return Placement(self.current_piece, rotation, x, y)

    def legal_actions(self) -> list[int]:
        return [action for action in range(4 * self.width) if self.placement_for_action(action)]

    def apply_action(self, action: int) -> bool:
        placement = self.placement_for_action(action)
        if placement is None:
            return False
        self.lock_placement(placement)
        return True

    def lock_placement(self, placement: Placement) -> None:
        shape = ROTATIONS[placement.piece][placement.rotation % len(ROTATIONS[placement.piece])]
        for cell_x, cell_y in shape:
            self.board[placement.y + cell_y][placement.x + cell_x] = int(placement.piece)
        cleared = self.clear_lines()
        rewards = (0, 100, 300, 500, 800)
        self.score += rewards[cleared] + 4
        self.lines += cleared
        self.pieces += 1
        self.current_piece = self.next_piece
        self.next_piece = self._next_piece()
        self.game_over = not bool(self.legal_actions())

    def clear_lines(self) -> int:
        remaining = [row for row in self.board if not all(row)]
        cleared = self.height - len(remaining)
        self.board = [[0 for _ in range(self.width)] for _ in range(cleared)] + remaining
        return cleared

    def column_heights(self) -> list[int]:
        heights = []
        for x in range(self.width):
            top = next((y for y in range(self.height) if self.board[y][x]), self.height)
            heights.append(self.height - top)
        return heights

    def holes(self) -> int:
        total = 0
        for x in range(self.width):
            occupied = False
            for y in range(self.height):
                occupied = occupied or bool(self.board[y][x])
                if occupied and not self.board[y][x]:
                    total += 1
        return total

    def bumpiness(self) -> int:
        heights = self.column_heights()
        return sum(abs(left - right) for left, right in zip(heights, heights[1:]))

    def state_vector(self) -> list[float]:
        board = [float(bool(cell)) for row in self.board for cell in row]
        piece = [float(self.current_piece == item) for item in Tetromino]
        heights = [height / self.height for height in self.column_heights()]
        features = [self.holes() / 40.0, self.bumpiness() / 40.0, max(self.column_heights()) / self.height]
        return board + piece + heights + features

    def fitness(self) -> float:
        height_penalty = sum(self.column_heights()) * 0.4
        hole_penalty = self.holes() * 7.5
        bump_penalty = self.bumpiness() * 0.25
        return self.score + self.lines * 500.0 + self.pieces * 1.5 - height_penalty - hole_penalty - bump_penalty
