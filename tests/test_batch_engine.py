import unittest

import torch

from tetris_ai.batch_engine import BatchedTetris
from tetris_ai.engine import TetrisGame, Tetromino, piece_sequence


class BatchEngineTests(unittest.TestCase):
    def test_logits_can_select_rotated_piece(self):
        games = BatchedTetris(4, 10, 20, torch.device("cpu"))
        indices = games.active_indices()
        logits = torch.zeros((4, 40))
        logits[:, 10] = 20.0
        games.apply_logits(indices, logits, Tetromino.I)
        self.assertEqual(games.rotated_moves.item(), 4)
        self.assertEqual(games.pieces.tolist(), [1, 1, 1, 1])
        self.assertEqual(games.boards[:, :, 0].sum(dim=1).tolist(), [4, 4, 4, 4])

    def test_line_clear_compacts_board(self):
        games = BatchedTetris(1, 10, 20, torch.device("cpu"))
        games.boards[0, -1, 4:] = True
        logits = torch.full((1, 40), -10.0)
        logits[0, 0] = 10.0
        games.apply_logits(games.active_indices(), logits, Tetromino.I)
        self.assertEqual(games.lines.item(), 1)
        self.assertEqual(games.scores.item(), 104)
        self.assertFalse(games.boards[0, -1].any())

    def test_state_vector_matches_network_input_size(self):
        games = BatchedTetris(3, 10, 20, torch.device("cpu"))
        states = games.state_vectors(games.active_indices(), Tetromino.T)
        self.assertEqual(states.shape, (3, 220))

    def test_batch_game_matches_scalar_engine(self):
        batch = BatchedTetris(1, 10, 20, torch.device("cpu"))
        scalar = TetrisGame(seed=7)
        generator = torch.Generator().manual_seed(91)
        for piece in piece_sequence(22, 40):
            scalar.current_piece = piece
            legal = scalar.legal_actions()
            if not legal:
                break
            logits = torch.randn((1, 40), generator=generator)
            action = max(legal, key=lambda candidate: float(logits[0, candidate]))
            scalar.apply_action(action)
            batch.apply_logits(batch.active_indices(), logits, piece)
            self.assertEqual(batch.boards[0].tolist(), [[bool(cell) for cell in row] for row in scalar.board])
            self.assertEqual(batch.scores.item(), scalar.score)
            self.assertEqual(batch.lines.item(), scalar.lines)
            if scalar.game_over:
                break
        self.assertAlmostEqual(batch.fitness().item(), scalar.fitness(), places=4)


if __name__ == "__main__":
    unittest.main()
