import unittest

import numpy as np

from scripts.prepare_data import tone_map_hdr_linear_to_sdr_linear


class PrepareDataTests(unittest.TestCase):
    def test_tone_map_hdr_linear_to_sdr_linear_returns_bounded_frame(self) -> None:
        frame = np.full((8, 8, 3), 0.75, dtype=np.float32)
        out = tone_map_hdr_linear_to_sdr_linear(frame)
        self.assertEqual(out.shape, frame.shape)
        self.assertTrue(np.all(out >= 0.0))
        self.assertTrue(np.all(out <= 1.0))


if __name__ == "__main__":
    unittest.main()
