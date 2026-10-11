# AI Project 3

This package contains the merged Phase 1 and Phase 2 notebook and the connected real-time Phase 3 application.

## Files

- `SignLanguageCNN_Project (1).ipynb`: dataset analysis, preprocessing, CNN training, evaluation, and checkpoint export
- `main.py`: webcam inference, continuous recognition, visible text, and text-to-speech
- `phase3/model.py`: the exact CNN architecture used by the notebook
- `phase3/decoder.py`: temporal smoothing and phrase-building logic
- `phase3/speech.py`: non-blocking offline speech worker
- `requirements.txt`: Python dependencies

## Dataset

Download `alphabet.zip` from the provided Google Drive link and place it beside the notebook:

```text
AI_Project/
  alphabet.zip
  SignLanguageCNN_Project (1).ipynb
  main.py
```

The notebook validates every archive path, extracts the dataset into `dataset/`, detects the nested ASL Alphabet folders, and loads all 87,000 classified images. It performs a reproducible stratified split:

- 69,600 training images
- 8,700 validation images
- 8,700 test images
- 29 classes in every split
- 2,400 training, 300 validation, and 300 test images per class

The 28 separate images from `asl_alphabet_test` are evaluated as an additional reference set. They are not used as the official test set because that folder contains only one image per available class and has no `del` image.

## Installation

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

Linux or macOS:

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Phase 1 and Phase 2

Place `alphabet.zip` beside the notebook, open `SignLanguageCNN_Project (1).ipynb`, and run every cell in order. The archive is extracted automatically. The final cell evaluates the best validation model and creates:

```text
weights.pth
```

The checkpoint contains the model weights, exact class order, architecture name, image size, normalization settings, and test accuracy.

## Phase 3

After the notebook creates `weights.pth`, run:

```bash
python main.py --checkpoint weights.pth
```

Useful options:

```bash
python main.py --camera 1
python main.py --device cpu
python main.py --confidence 0.85
python main.py --no-tts
python main.py --roi 0.56,0.12,0.38,0.68
```

The prediction must remain above the confidence threshold for two seconds before it is added to the visible text. Five seconds of a confident blank-class prediction finalizes the phrase, speaks it, and clears the text.

## Controls

| Key | Action |
| --- | --- |
| `Q` or `Esc` | Quit |
| `C` | Clear text |
| `S` | Speak current text |
| `Backspace` | Delete the last character |
| `Space` | Add a space |

To enter the same letter twice, briefly show the blank class or move the hand away between letters.

## Validation

```bash
python -m compileall -q main.py phase3 tests
python -m unittest discover -s tests -v
```

Final recognition accuracy and webcam behavior must be validated after training with the real dataset and on the target computer.
