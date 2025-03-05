"""Unit tests for the estimate_batch_size function in the ypl.embeddings.server module.

Run with:

    python -m unittest ypl/embeddings/test_server.py
"""
import unittest

from ypl.embeddings.server import estimate_batch_size


class TestEstimateBatchSize(unittest.TestCase):
    def test_estimate_batch_size_bge_m3(self) -> None:
        # Test case 1: Basic test with BGE-M3 model
        sequence_lengths = [8096, 6, 6, 12]
        default_batch_size = 32
        max_memory = 805306368  # 768MB
        model_name = "BAAI/bge-m3"
        actual_batch_size = estimate_batch_size(
            sequence_lengths, default_batch_size=default_batch_size, max_memory=max_memory, model_name=model_name
        )
        expected_batch_size = 2
        self.assertEqual(actual_batch_size, expected_batch_size)

    def test_estimate_batch_size_no_model_name(self) -> None:
        # Test case 2: No model name provided
        sequence_lengths = [5, 10, 15]
        default_batch_size = 32
        max_memory = 805306368
        actual_batch_size = estimate_batch_size(
            sequence_lengths, default_batch_size=default_batch_size, max_memory=max_memory
        )
        self.assertEqual(actual_batch_size, default_batch_size)

    def test_estimate_batch_size_empty_sequence_lengths(self) -> None:
        # Test case 3: Empty sequence lengths
        sequence_lengths: list[int] = []
        default_batch_size = 32
        max_memory = 805306368
        model_name = "BAAI/bge-m3"
        actual_batch_size = estimate_batch_size(
            sequence_lengths, default_batch_size=default_batch_size, max_memory=max_memory, model_name=model_name
        )
        self.assertEqual(actual_batch_size, default_batch_size)

    def test_estimate_batch_size_zero_max_memory(self) -> None:
        # Test case 4: Zero max memory
        sequence_lengths = [100, 150, 200]
        default_batch_size = 32
        max_memory = 0
        model_name = "BAAI/bge-m3"
        with self.assertRaises(ValueError):
            estimate_batch_size(
                sequence_lengths, default_batch_size=default_batch_size, max_memory=max_memory, model_name=model_name
            )

    def test_estimate_batch_size_zero_batch_size(self) -> None:
        # Test case 5: Zero batch size
        sequence_lengths = [100, 150, 200]
        default_batch_size = 0
        max_memory = 805306368
        model_name = "BAAI/bge-m3"
        with self.assertRaises(ValueError):
            estimate_batch_size(
                sequence_lengths, default_batch_size=default_batch_size, max_memory=max_memory, model_name=model_name
            )


if __name__ == "__main__":
    unittest.main()
