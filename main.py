import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch

from phase3.decoder import DecoderResult, TemporalDecoder
from phase3.model import SignLanguageCNN
from phase3.speech import SpeechWorker


@dataclass(frozen=True)
class ModelBundle:
    model: SignLanguageCNN
    class_names: tuple[str, ...]
    image_size: int
    mean: tuple[float, float, float]
    standard_deviation: tuple[float, float, float]
    device: torch.device


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Real-time sign-language-to-speech translator"
    )

    base_dir = Path(__file__).resolve().parent

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=base_dir / "weights.pth"
    )

    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--confidence", type=float, default=0.8)
    parser.add_argument("--hold-seconds", type=float, default=2.0)
    parser.add_argument("--empty-seconds", type=float, default=5.0)
    parser.add_argument("--release-seconds", type=float, default=0.35)
    parser.add_argument("--smoothing-frames", type=int, default=5)
    parser.add_argument("--roi", default="0.56,0.12,0.38,0.68")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--no-mirror", action="store_true")
    parser.add_argument("--no-tts", action="store_true")
    parser.add_argument("--tts-rate", type=int, default=165)

    return parser.parse_args()

def select_device(name: str) -> torch.device:
    if name != "auto":
        device = torch.device(name)
        if name == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        if name == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            raise RuntimeError("Apple MPS is not available")
        return device
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_checkpoint(path: Path, device_name: str) -> ModelBundle:
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    device = select_device(device_name)
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError("Checkpoint must be a dictionary created by the notebook")
    state_dict = payload.get("model_state_dict")
    class_names = payload.get("class_names")
    if not isinstance(state_dict, dict) or not isinstance(class_names, (list, tuple)) or not class_names:
        raise ValueError("Checkpoint is missing model_state_dict or class_names")
    if payload.get("architecture") not in {None, "sign_language_cnn"}:
        raise ValueError("Checkpoint architecture is not compatible with this Phase 3 application")
    image_size = int(payload.get("image_size", 64))
    if image_size != 64:
        raise ValueError("This CNN architecture requires image_size=64")
    mean = tuple(float(value) for value in payload.get("mean", (0.0, 0.0, 0.0)))
    standard_deviation = tuple(float(value) for value in payload.get("std", (1.0, 1.0, 1.0)))
    if len(mean) != 3 or len(standard_deviation) != 3 or any(value <= 0 for value in standard_deviation):
        raise ValueError("Checkpoint preprocessing metadata is invalid")
    model = SignLanguageCNN(len(class_names))
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device).eval()
    with torch.inference_mode():
        output = model(torch.zeros(1, 3, image_size, image_size, device=device))
    if output.shape != (1, len(class_names)):
        raise ValueError("Checkpoint output size does not match class_names")
    return ModelBundle(model, tuple(str(name) for name in class_names), image_size, mean, standard_deviation, device)


def parse_roi(value: str) -> tuple[float, float, float, float]:
    try:
        left, top, width, height = (float(item.strip()) for item in value.split(","))
    except ValueError as error:
        raise ValueError("ROI must be left,top,width,height") from error
    if min(left, top, width, height) < 0 or left + width > 1 or top + height > 1 or width == 0 or height == 0:
        raise ValueError("ROI values must define a non-empty rectangle inside the frame")
    return left, top, width, height


def roi_coordinates(frame: np.ndarray, roi: tuple[float, float, float, float]) -> tuple[int, int, int, int]:
    height, width = frame.shape[:2]
    left = int(roi[0] * width)
    top = int(roi[1] * height)
    right = int((roi[0] + roi[2]) * width)
    bottom = int((roi[1] + roi[3]) * height)
    return left, top, right, bottom


def preprocess(region: np.ndarray, bundle: ModelBundle) -> torch.Tensor:
    rgb = cv2.cvtColor(region, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (bundle.image_size, bundle.image_size), interpolation=cv2.INTER_AREA)
    tensor = torch.from_numpy(resized).permute(2, 0, 1).unsqueeze(0).float().to(bundle.device) / 255.0
    mean = torch.tensor(bundle.mean, device=bundle.device).view(1, 3, 1, 1)
    standard_deviation = torch.tensor(bundle.standard_deviation, device=bundle.device).view(1, 3, 1, 1)
    return (tensor - mean) / standard_deviation


def predict(region: np.ndarray, bundle: ModelBundle) -> np.ndarray:
    with torch.inference_mode():
        probabilities = torch.softmax(bundle.model(preprocess(region, bundle)), dim=1)[0]
    return probabilities.detach().cpu().numpy()


def draw_progress(frame: np.ndarray, x: int, y: int, value: float, color: tuple[int, int, int]) -> None:
    width = 280
    progress = min(1.0, max(0.0, value))
    cv2.rectangle(frame, (x, y), (x + width, y + 12), (50, 50, 50), -1)
    cv2.rectangle(frame, (x, y), (x + int(width * progress), y + 12), color, -1)


def draw_overlay(
    frame: np.ndarray,
    coordinates: tuple[int, int, int, int],
    result: DecoderResult,
    fps: float,
    confidence_threshold: float,
    speech_error: str | None,
) -> None:
    left, top, right, bottom = coordinates
    color = (80, 220, 80) if result.confidence >= confidence_threshold else (0, 180, 255)
    cv2.rectangle(frame, (left, top), (right, bottom), color, 3)
    cv2.putText(frame, "Place one hand inside the box", (left, max(30, top - 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], 170), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)
    cv2.putText(frame, f"Prediction: {result.label}  Confidence: {result.confidence:.1%}", (24, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (245, 245, 245), 2)
    cv2.putText(frame, f"Text: {result.text or '-'}", (24, 74), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (120, 240, 255), 2)
    cv2.putText(frame, f"FPS: {fps:.1f}", (frame.shape[1] - 150, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (220, 220, 220), 2)
    draw_progress(frame, 24, 92, result.stable_progress, (80, 220, 80))
    cv2.putText(frame, "Stable sign", (315, 104), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)
    draw_progress(frame, 24, 120, result.empty_progress, (255, 180, 40))
    cv2.putText(frame, "Empty frame", (315, 132), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)
    cv2.putText(frame, "Q/Esc quit  C clear  S speak  Backspace delete  Space add", (24, 158), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 200, 200), 1)
    if speech_error:
        cv2.putText(frame, f"TTS unavailable: {speech_error[:70]}", (24, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 80, 255), 1)


def run(arguments: argparse.Namespace) -> None:
    bundle = load_checkpoint(arguments.checkpoint, arguments.device)
    decoder = TemporalDecoder(
        bundle.class_names,
        arguments.confidence,
        arguments.hold_seconds,
        arguments.empty_seconds,
        arguments.release_seconds,
        arguments.smoothing_frames,
    )
    speaker = SpeechWorker(not arguments.no_tts, arguments.tts_rate)
    capture = cv2.VideoCapture(arguments.camera)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, arguments.width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, arguments.height)
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not capture.isOpened():
        speaker.close()
        raise RuntimeError(f"Unable to open camera {arguments.camera}")
    roi = parse_roi(arguments.roi)
    window_name = "AI Project 3 - Sign Language Translator"
    last_time = time.monotonic()
    fps = 0.0
    failures = 0
    try:
        while True:
            success, frame = capture.read()
            if not success:
                failures += 1
                if failures >= 20:
                    raise RuntimeError("Camera stopped returning frames")
                continue
            failures = 0
            if not arguments.no_mirror:
                frame = cv2.flip(frame, 1)
            coordinates = roi_coordinates(frame, roi)
            left, top, right, bottom = coordinates
            now = time.monotonic()
            result = decoder.update(predict(frame[top:bottom, left:right], bundle), now)
            if result.finalized:
                speaker.speak(result.finalized)
            elapsed = max(now - last_time, 1e-6)
            current_fps = 1.0 / elapsed
            fps = current_fps if fps == 0 else fps * 0.9 + current_fps * 0.1
            last_time = now
            draw_overlay(frame, coordinates, result, fps, arguments.confidence, speaker.error)
            cv2.imshow(window_name, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("c"):
                decoder.clear()
            elif key == ord("s"):
                speaker.speak(decoder.text)
            elif key in (8, 127):
                decoder.delete_last()
            elif key == 32:
                decoder.add_space()
    finally:
        capture.release()
        cv2.destroyAllWindows()
        speaker.close()


if __name__ == "__main__":
    run(parse_arguments())
