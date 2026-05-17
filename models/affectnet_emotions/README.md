# EmotiEffLib pre-trained models

In this folder you can find pre-trained models used by EmotiEffLib for facial
emotion analysis.

## What's bundled in this archive

To keep the archive small, this distribution ships only one model
(`enet_b0_8_va_mtl`) in both formats:

- `enet_b0_8_va_mtl.pt`     — PyTorch weights
- `onnx/enet_b0_8_va_mtl.onnx` — ONNX export

This model is the project's recommended general-purpose multi-task model and
is sufficient to run the Python smoke tests and the C++ unit tests.

## Getting the full model collection

The Python library auto-downloads any model on first use to
`~/.emotiefflib/`. As long as the host has internet access, no manual step is
needed — calling `EmotiEffLibRecognizer(engine=..., model_name=...)` will
fetch the requested weights from the upstream repository.

If you need the additional models offline (for example, to run all
parametrizations of the test suite without internet, or to prepare C++ model
artefacts for every model variant), download them from the upstream
repository or the project's Google Drive:

- Upstream repo: <https://github.com/sb-ai-lab/EmotiEffLib/tree/main/models/affectnet_emotions>
- Google Drive:  <https://drive.google.com/drive/folders/1eBIIooLKnHoHYfbnJmNZlEHyOkckFif1?usp=sharing>

The additional models are:

| Filename                       | Format            | Task                              |
| ------------------------------ | ----------------- | --------------------------------- |
| `enet_b0_8_best_afew.pt/onnx`  | EfficientNet B0   | 8-class emotions, AFEW-tuned      |
| `enet_b0_8_best_vgaf.pt/onnx`  | EfficientNet B0   | 8-class emotions, VGAF-tuned      |
| `enet_b2_7.pt/onnx`            | EfficientNet B2   | 7-class emotions                  |
| `enet_b2_8.pt/onnx`            | EfficientNet B2   | 8-class emotions                  |
| `mbf_va_mtl.pt/onnx`           | MobileFaceNet     | Valence-Arousal + Multi-Task      |
| `mobilevit_va_mtl.pt/onnx`     | MobileViT         | Valence-Arousal + Multi-Task      |
| `mobilenet_7.h5`               | MobileNet (Keras) | 7-class emotions                  |

Place the downloaded files in this directory (`.pt` files alongside this
README, `.onnx` files in the `onnx/` subdirectory). After that the Python
auto-download path is no longer needed, and you can run
`python models/prepare_models_for_emotieffcpplib.py` to convert all of them
for use with the C++ library.

## Behaviour when a model is missing

The Python test suite (`tests/test_facial_analysis.py`) uses
`make_recognizer()`, which calls `pytest.skip()` if a model cannot be obtained
(no local file and no internet access). With only `enet_b0_8_va_mtl` present
and no internet, only that model's parametrizations run; the rest are
reported as skipped, not failed.
