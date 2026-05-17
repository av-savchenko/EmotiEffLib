# EmotiEffLib — Build and Install Guide

This document contains step-by-step build and install instructions for the
EmotiEffLib library that is shipped in this archive. The commands below have
been executed end-to-end in a clean Anaconda environment and the test suites
referenced here pass.

For a high-level overview of the library see [`README.md`](README.md). The
Python and C++ libraries each have their own README with reference
documentation:

- Python: [`emotiefflib/README.md`](emotiefflib/README.md)
- C++:    [`emotieffcpplib/README.md`](emotieffcpplib/README.md)

Online documentation: <https://sb-ai-lab.github.io/EmotiEffLib/>
Project page:         <https://github.com/sb-ai-lab/EmotiEffLib>

---

## 1. Prerequisites

| Component       | Version used for validation                          | Notes                                          |
| --------------- | ---------------------------------------------------- | ---------------------------------------------- |
| OS              | macOS 26.3 (arm64) — also verified on Ubuntu via CI  | Linux/Windows users substitute commands below  |
| Python          | 3.10                                                 | Required by the Python package                 |
| Anaconda/Miniconda | conda 25.5                                        | Any conda distribution works                   |
| CMake           | 4.1.1 (≥3.10.2 required)                             | Only needed for the C++ build                  |
| C++ compiler    | Apple clang 17 (or gcc ≥9)                           | C++17 required                                 |
| OpenCV          | 4.12 (via Homebrew on macOS, `libopencv-dev` on Ubuntu) | Required for the C++ build                  |
| ONNX Runtime    | 1.20.1                                               | Required for the C++ build (or Libtorch — see below) |
| Internet access | Required on first run                                | Pre-trained models are auto-downloaded to `~/.emotiefflib/` |

### Pre-trained models in this archive

To keep the archive compact, only one model is bundled:
`models/affectnet_emotions/enet_b0_8_va_mtl.{pt,onnx}`. This is enough to run
the full Python smoke test suite and the C++ unit tests.

The Python library automatically downloads any other model on first use into
`~/.emotiefflib/` (requires internet on first call). If you want the
remaining six emotion models locally (for offline use, or to convert all of
them for C++ via `python models/prepare_models_for_emotieffcpplib.py`),
follow the instructions in
[`models/affectnet_emotions/README.md`](models/affectnet_emotions/README.md).

When a model file is not present locally and cannot be downloaded, the
relevant Python tests skip cleanly rather than fail.

---

## 2. Python package

The Python package is published on PyPI as `emotiefflib==1.1.1` and can also be
installed from this source archive. Both paths are supported.

### 2.1. From PyPI

Use one of the following depending on which inference backend you need:

```sh
# ONNX Runtime only (smallest install)
pip install emotiefflib

# With PyTorch backend
pip install "emotiefflib[torch]"

# With TensorFlow backend (required for engagement classification)
pip install "emotiefflib[engagement]"

# Everything (PyTorch + TensorFlow + ONNX)
pip install "emotiefflib[all]"
```

### 2.2. From this source archive

After unpacking the archive, from the repository root:

```sh
pip install .             # ONNX Runtime only
pip install ".[torch]"    # + PyTorch
pip install ".[engagement]"  # + TensorFlow
pip install ".[all]"      # everything
```

### 2.3. Recommended: clean conda environment

To reproduce a known-good setup exactly:

```sh
conda create -n emotiefflib python=3.10 -y
conda activate emotiefflib
pip install ".[all]"      # or: pip install "emotiefflib[all]"
```

### 2.4. Verifying the install (smoke test)

```sh
python -c "
import emotiefflib
print('version:', emotiefflib.__version__)
from emotiefflib.facial_analysis import EmotiEffLibRecognizer, get_model_list
print('available models:', get_model_list())
fer = EmotiEffLibRecognizer(engine='onnx', model_name='enet_b0_8_best_vgaf')
print('recognizer created OK')
"
```

The first call downloads the model from GitHub into `~/.emotiefflib/`.

---

## 3. Python test suite

```sh
# 1. Install test dependencies (this also pulls in facenet-pytorch for MTCNN face detection)
pip install -r tests/requirements.txt

# 2. Download the bundled test dataset (~85 MB) from Google Drive
cd tests
./download_test_data.sh
tar -xzf data.tar.gz
cd ..

# 3. Run the suite
pytest --disable-warnings tests/
```

Expected result on a `[all]`-style install **with internet** (other models
auto-download): 87 passed, 4 skipped, 24 xfailed.

Expected result **without internet** (only the bundled `enet_b0_8_va_mtl` is
available): tests for `enet_b0_8_va_mtl` pass; tests for the other six models
skip cleanly with a message like `Model 'enet_b0_X' for engine 'onnx' is not
available: ...`.

Two of the skips on a full run are `test_distraction_on_video[{torch,onnx}-enet_b0_8_best_afew]` —
this specific model misclassifies the bundled distracted-video sample as
`"Engaged"`. It is a known model-accuracy limitation, not an install problem;
the other six emotion models pass this test, and the same model passes every
other test. The skips are wired into the test itself (`tests/test_facial_analysis.py`).

For the ONNX-only install (`pip install emotiefflib`), the full
`test_facial_analysis.py` cannot be collected because it imports `torch` and
`facenet_pytorch`. Use the loader test instead:

```sh
WITHOUT_TORCH=1 pytest --disable-warnings tests/test_module_loading.py
```

### 3.2. Troubleshooting

- **`gdown` download fails / Google Drive rate-limits.** Open
  `tests/download_test_data.sh` and download the file with the listed Google
  Drive file id manually, then place it at `tests/data.tar.gz` and run
  `tar -xzf data.tar.gz` from `tests/`.

- **`opencv-python ... requires numpy>=2` warning during pip install.** This
  is a known pip resolver warning when `tests/requirements.txt` installs
  `facenet-pytorch` (which pins `numpy<2`). It is a warning, not a runtime
  error — tests pass with `numpy==1.26.4` and `opencv-python==4.13`.

---

## 4. C++ library

### 4.1. Dependencies

You need OpenCV and at least one inference engine (ONNX Runtime or Libtorch).
The instructions below use ONNX Runtime; for Libtorch, see
[`emotieffcpplib/README.md`](emotieffcpplib/README.md).

```sh
# macOS
brew install opencv cmake

# Ubuntu / Debian
sudo apt update
sudo apt install -y cmake build-essential libopencv-dev
```

Download ONNX Runtime 1.20.1 for your platform from
<https://github.com/microsoft/onnxruntime/releases/tag/v1.20.1>:

```sh
# macOS arm64 (Apple Silicon)
mkdir -p $HOME/onnxruntime && cd $HOME/onnxruntime
curl -L -o onnxruntime.tgz \
  https://github.com/microsoft/onnxruntime/releases/download/v1.20.1/onnxruntime-osx-arm64-1.20.1.tgz
tar -xzf onnxruntime.tgz
export ORT_DIR=$HOME/onnxruntime/onnxruntime-osx-arm64-1.20.1

# Linux x64
mkdir -p $HOME/onnxruntime && cd $HOME/onnxruntime
curl -L -o onnxruntime.tgz \
  https://github.com/microsoft/onnxruntime/releases/download/v1.20.1/onnxruntime-linux-x64-1.20.1.tgz
tar -xzf onnxruntime.tgz
export ORT_DIR=$HOME/onnxruntime/onnxruntime-linux-x64-1.20.1
```

### 4.2. Prepare ONNX model artefacts for the C++ tests

The C++ unit tests consume converted versions of the PyTorch models that ship
with the Python library. Convert them once from any environment that has
`torch`, `onnx`, and `tensorflow` installed (the `[all]` env from §2.3 works):

```sh
cd <repo root>
python models/prepare_models_for_emotieffcpplib.py
# Produces models/emotieffcpplib_prepared_models/*.onnx
```

### 4.3. Configure and build

```sh
cd <repo root>
cmake -S emotieffcpplib -B build \
    -DWITH_ONNX=$ORT_DIR \
    -DBUILD_TESTS=ON \
    -DCMAKE_GTEST_DISCOVER_TESTS_DISCOVERY_MODE=PRE_TEST
cmake --build build -j$(sysctl -n hw.ncpu 2>/dev/null || nproc)
```

The `-DCMAKE_GTEST_DISCOVER_TESTS_DISCOVERY_MODE=PRE_TEST` flag is recommended
because it defers GoogleTest's test enumeration until test run-time. The
default `POST_BUILD` mode executes the test binary at build time and can hit
the 5-second CMake timeout on slower hosts.

### 4.4. Run the unit tests

```sh
export EMOTIEFFLIB_ROOT=<repo root>
./build/bin/unit_tests
```

Expected result with an ONNX-only build: 68 tests pass, the Torch-backend
tests are skipped. To include Torch tests, also pass `-DWITH_TORCH=/path/to/libtorch`
during cmake configuration.

### 4.5. macOS troubleshooting

- **`dyld[...]: Library not loaded: /opt/homebrew/opt/protobuf/lib/libprotobuf.32.X.X.dylib`
  when running `unit_tests`.** Homebrew has rolled to a newer major version of
  protobuf but the cached `libopencv_dnn` is linked against the previous one.
  Make the older protobuf visible at run time:
  ```sh
  export DYLD_LIBRARY_PATH=/opt/homebrew/Cellar/protobuf/32.1/lib
  ./build/bin/unit_tests
  ```
  Replace `32.1` with whichever protobuf version OpenCV's `libopencv_dnn` is
  actually linked against (check with `otool -L $(brew --prefix opencv)/lib/libopencv_dnn.*.dylib`).
  A permanent fix is to `brew uninstall protobuf@33` (or whichever newer version is installed)
  or `brew link --force protobuf@32` to make 32.1 the default.

- **Linker error `symbol(s) not found for architecture arm64` while building
  `libmtcnn.dylib` with `-DBUILD_SHARED_LIBS=ON`.** This is a known cross-platform
  issue with the upstream `opencv-mtcnn` submodule's CMake on macOS, which is
  stricter than the GNU linker. Use the default static build (`-DBUILD_SHARED_LIBS=OFF`,
  i.e. omit the flag) — `unit_tests` works the same.

---

## 5. License

EmotiEffLib is released under the Apache-2.0 License (see [`LICENSE`](LICENSE)).
There is no limitation for academic or commercial use.
