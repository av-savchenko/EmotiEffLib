# EmotiEffLib Python implementation

EmotiEffLib is a Python library designed for efficient emotion and engagement
recognition in images and videos. It supports multiple inference engines,
including Torch and ONNX Runtime.

## Installing
### Install version of the library from actual code
- Install EmotiEffLib with ONNX support:
    ```sh
    pip install .
    ```
- Install EmotiEffLib with ONNX and Torch support:
   ```sh
   pip install .[torch]
   ```
- Install EmotiEffLib with ONNX and TensorFlow support (the last one is necessary for enagement prediction):
   ```sh
   pip install .[engagement]
   ```
- Install EmotiEffLib with all possible dependencies:
   ```sh
   pip install .[all]
   ```
   or
   ```sh
   pip install .[torch,engagement]
   ```

### Install version of the library from PyPI
It is also possible to install it via pip:
- Install EmotiEffLib with ONNX support:
   ```sh
   pip install emotiefflib
   ```
- Install EmotiEffLib with ONNX and Torch support:
   ```sh
   pip install emotiefflib[torch]
   ```
- Install EmotiEffLib with ONNX and TensorFlow support (the last one is necessary for enagement prediction):
   ```sh
   pip install emotiefflib[engagement]
   ```
- Install EmotiEffLib with all possible dependencies:
   ```sh
   pip install emotiefflib[all]
   ```
   or
   ```sh
   pip install emotiefflib[torch,engagement]
   ```

## Usage
Examples of usage EmotiEffLib you can find in the [python examples section](../docs/tutorials/python/README.md).

## Running tests
Pytest framework is used for testing EmotiEffLib. To run the tests, you need to
follow the following steps:
1. Install dependencies:
  ```sh
  pip install -r tests/requirements.txt
  ```
2. Download testing data:
  ```sh
  cd <emotiefflib_root>/tests
  ./download_test_data.sh
  tar -xzf data.tar.gz
  cd -
  ```
2. Run tests:
  ```sh
  pytest <emotiefflib_root>/tests
  ```

## License

The code of EmotiEffLib Python Library is released under the Apache-2.0 License. There is no limitation for both academic and commercial usage.
