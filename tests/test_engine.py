import unittest

from tetris_ai.engine import ROTATIONS, TetrisGame, Tetromino, piece_sequence


class EngineTests(unittest.TestCase):
    def test_wall_and_floor_collisions(self):
        game = TetrisGame(seed=1)
        shape = ROTATIONS[Tetromino.I][0]
        self.assertTrue(game.collides(shape, -1, 0))
        self.assertTrue(game.collides(shape, 0, game.height))
        self.assertFalse(game.collides(shape, 0, 0))

    def test_rotations_are_normalized_and_distinct(self):
        rotations = ROTATIONS[Tetromino.T]
        self.assertEqual(len(rotations), 4)
        self.assertEqual(len(set(rotations)), 4)
        for rotation in rotations:
            self.assertEqual(min(x for x, _ in rotation), 0)
            self.assertEqual(min(y for _, y in rotation), 0)

    def test_action_selects_rotated_i_piece(self):
        game = TetrisGame(seed=5)
        game.current_piece = Tetromino.I
        action = game.width
        placement = game.placement_for_action(action)
        self.assertIsNotNone(placement)
        self.assertEqual(placement.rotation, 1)
        game.apply_action(action)
        self.assertEqual(sum(game.board[y][0] for y in range(game.height)), 4)
        self.assertEqual(sum(game.board[-1]), 1)

    def test_piece_sequence_uses_fair_seven_bags(self):
        sequence = piece_sequence(33, 14)
        expected = set(Tetromino)
        self.assertEqual(set(sequence[:7]), expected)
        self.assertEqual(set(sequence[7:]), expected)

    def test_completed_lines_are_cleared(self):
        game = TetrisGame(seed=2)
        game.board[-1] = [1] * game.width
        game.board[-2][0] = 2
        self.assertEqual(game.clear_lines(), 1)
        self.assertEqual(sum(game.board[-1]), 2)

    def test_game_over_detection(self):
        game = TetrisGame(seed=3)
        game.board[0] = [1] * game.width
        self.assertEqual(game.legal_actions(), [])

    def test_reset_restores_fresh_game(self):
        game = TetrisGame(seed=4)
        game.board[-1][0] = 1
        game.score = 900
        game.lines = 4
        game.game_over = True
        game.reset(4)
        self.assertEqual(sum(sum(row) for row in game.board), 0)
        self.assertEqual(game.score, 0)
        self.assertEqual(game.lines, 0)
        self.assertFalse(game.game_over)


if __name__ == "__main__":
    unittest.main()
