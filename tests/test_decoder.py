import unittest

import numpy as np

from phase3.decoder import TemporalDecoder


class TemporalDecoderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.decoder = TemporalDecoder(
            ["A", "B", "del", "nothing", "space"],
            confidence_threshold=0.75,
            hold_seconds=2.0,
            empty_seconds=5.0,
            release_seconds=0.3,
            smoothing_frames=1,
        )

    def probabilities(self, index: int, confidence: float = 0.95) -> np.ndarray:
        values = np.full(5, (1.0 - confidence) / 4.0)
        values[index] = confidence
        return values

    def test_stable_sign_is_emitted_once(self) -> None:
        self.decoder.update(self.probabilities(0), 0.0)
        result = self.decoder.update(self.probabilities(0), 2.1)
        self.assertEqual(result.text, "A")
        self.assertEqual(result.emitted, "A")
        result = self.decoder.update(self.probabilities(0), 4.5)
        self.assertEqual(result.text, "A")
        self.assertIsNone(result.emitted)

    def test_repeated_letter_requires_release(self) -> None:
        self.decoder.update(self.probabilities(0), 0.0)
        self.decoder.update(self.probabilities(0), 2.1)
        self.decoder.update(self.probabilities(3), 2.2)
        self.decoder.update(self.probabilities(3), 2.6)
        self.decoder.update(self.probabilities(0), 3.0)
        result = self.decoder.update(self.probabilities(0), 5.1)
        self.assertEqual(result.text, "AA")

    def test_blank_finalizes_phrase(self) -> None:
        self.decoder.update(self.probabilities(1), 0.0)
        self.decoder.update(self.probabilities(1), 2.1)
        self.decoder.update(self.probabilities(3), 3.0)
        result = self.decoder.update(self.probabilities(3), 8.1)
        self.assertEqual(result.finalized, "B")
        self.assertEqual(result.text, "")

    def test_delete_and_space(self) -> None:
        self.decoder.text = "AB"
        self.decoder.update(self.probabilities(2), 0.0)
        result = self.decoder.update(self.probabilities(2), 2.1)
        self.assertEqual(result.text, "A")
        self.decoder.update(self.probabilities(3), 2.2)
        self.decoder.update(self.probabilities(3), 2.6)
        self.decoder.update(self.probabilities(4), 3.0)
        result = self.decoder.update(self.probabilities(4), 5.1)
        self.assertEqual(result.text, "A ")

    def test_low_confidence_does_not_emit(self) -> None:
        self.decoder.update(self.probabilities(0, 0.6), 0.0)
        result = self.decoder.update(self.probabilities(0, 0.6), 3.0)
        self.assertEqual(result.text, "")

    def test_probability_shape_is_validated(self) -> None:
        with self.assertRaises(ValueError):
            self.decoder.update(np.array([1.0, 0.0]), 0.0)


if __name__ == "__main__":
    unittest.main()
