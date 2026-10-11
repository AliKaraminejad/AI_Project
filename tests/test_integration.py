import tempfile
import unittest
from pathlib import Path

try:
    import torch

    from main import load_checkpoint
    from phase3.model import SignLanguageCNN
except ImportError:
    torch = None
    SignLanguageCNN = None
    load_checkpoint = None


@unittest.skipIf(torch is None, "Phase 3 dependencies are not installed")
class IntegrationTests(unittest.TestCase):
    def test_notebook_checkpoint_contract(self) -> None:
        class_names = [chr(code) for code in range(ord("A"), ord("Z") + 1)] + ["del", "nothing", "space"]
        model = SignLanguageCNN(len(class_names))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "weights.pth"
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "class_names": class_names,
                    "architecture": "sign_language_cnn",
                    "image_size": 64,
                    "mean": [0.0, 0.0, 0.0],
                    "std": [1.0, 1.0, 1.0],
                },
                path,
            )
            bundle = load_checkpoint(path, "cpu")
        self.assertEqual(bundle.class_names, tuple(class_names))
        self.assertEqual(bundle.image_size, 64)
        with torch.inference_mode():
            output = bundle.model(torch.zeros(1, 3, 64, 64))
        self.assertEqual(tuple(output.shape), (1, 29))


if __name__ == "__main__":
    unittest.main()
