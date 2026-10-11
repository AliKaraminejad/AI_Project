import unittest

try:
    import torch

    from phase3.model import SignLanguageCNN
except ImportError:
    torch = None
    SignLanguageCNN = None


@unittest.skipIf(torch is None, "PyTorch is not installed")
class ModelTests(unittest.TestCase):
    def test_output_shape(self) -> None:
        model = SignLanguageCNN(29).eval()
        with torch.inference_mode():
            output = model(torch.zeros(2, 3, 64, 64))
        self.assertEqual(tuple(output.shape), (2, 29))


if __name__ == "__main__":
    unittest.main()
