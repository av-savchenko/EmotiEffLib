"""
Facial emotions recognition implementation
"""

from __future__ import absolute_import, division, print_function

import os
import pathlib
import sys
from abc import ABC, abstractmethod
from typing import List, Tuple, Union

import cv2
import numpy as np
from PIL import Image

from .engagement_classification_model import get_engagement_model
from .utils import get_model_path_onnx, get_model_path_torch

FILE_DIR = pathlib.Path(__file__).parent.resolve()

try:
    import torch
    from torchvision import transforms

    # It is required for mbf_va_mtl
    path_to_backbones = os.path.join(FILE_DIR, "backbones")
    if path_to_backbones not in sys.path:
        sys.path.append(path_to_backbones)
except ImportError:
    pass
try:
    import onnx
    import onnxruntime as ort
    from onnx import TensorProto, helper, numpy_helper
except ImportError:
    pass


def get_model_list() -> List[str]:
    """
    Returns a list of available model names.

    These models are supported by HSEmoitonRecognizer.

    Returns:
        List[str]: A list of model names.
    """
    return [
        "enet_b0_8_best_vgaf",
        "enet_b0_8_best_afew",
        "enet_b2_8",
        "enet_b0_8_va_mtl",
        "enet_b2_7",
        "mbf_va_mtl",
        "mobilevit_va_mtl",
    ]


def get_supported_engines() -> List[str]:
    """
    Returns a list of supported inference engines.

    Returns:
        List[str]: A list of inference engines.
    """
    return ["torch", "onnx"]


class EmotiEffLibRecognizerBase(ABC):
    """
    Abstract class for emotion recognizer classes
    """

    def __init__(self, model_name: str) -> None:
        self.is_mtl = "_mtl" in model_name
        self.idx_to_engagement_class = {
            0: "Distracted",
            1: "Engaged",
        }
        if "_7" in model_name:
            self.idx_to_emotion_class = {
                0: "Anger",
                1: "Disgust",
                2: "Fear",
                3: "Happiness",
                4: "Neutral",
                5: "Sadness",
                6: "Surprise",
            }
        else:
            self.idx_to_emotion_class = {
                0: "Anger",
                1: "Contempt",
                2: "Disgust",
                3: "Fear",
                4: "Happiness",
                5: "Neutral",
                6: "Sadness",
                7: "Surprise",
            }

        if "mbf_" in model_name:
            self.mean = [0.5, 0.5, 0.5]
            self.std = [0.5, 0.5, 0.5]
            self.img_size = 112
        else:
            self.mean = [0.485, 0.456, 0.406]
            self.std = [0.229, 0.224, 0.225]
            if "_b2_" in model_name:
                self.img_size = 260
            elif "ddamfnet" in model_name:
                self.img_size = 112
            else:
                self.img_size = 224

        self.classifier_weights = None
        self.classifier_bias = None

    def _get_probab(self, features: np.ndarray) -> np.ndarray:
        """
        Compute the final classification scores for the given feature representations.

        Args:
            features (np.ndarray): The extracted feature vectors.

        Returns:
            np.ndarray: The raw classification scores (logits) before applying any activation
                        function.
        """
        x = np.dot(features, np.transpose(self.classifier_weights)) + self.classifier_bias
        return x

    @abstractmethod
    def _preprocess(self, img: np.ndarray) -> np.ndarray:
        """
        Prepare an image for input to the model.

        Args:
            img (np.ndarray): The input image for preprocessing.

        Returns:
            np.ndarray: Preprocessed image.
        """
        raise NotImplementedError("It should be implemented")

    @abstractmethod
    def extract_features(self, face_img: Union[np.ndarray, List[np.ndarray]]) -> np.ndarray:
        """
        Extract visual features from a facial image or a list of facial images.

        Args:
            face_img (Union[np.ndarray, List[np.ndarray]]):
                A single face image (as a NumPy array) or a list of face images.

        Returns:
            np.ndarray: The extracted feature vectors.
        """
        raise NotImplementedError("It should be implemented")

    def classify_emotions(
        self, features: np.ndarray, logits: bool = True
    ) -> Tuple[List[str], np.ndarray]:
        """
        Classify emotions based on extracted feature representations.

        Args:
            features (np.ndarray): The extracted feature vectors.
            logits (bool, optional):
                If True, returns raw model scores (logits). If False, applies softmax normalization
                to obtain probability distributions. Defaults to True.

        Returns:
            Tuple[List[str], np.ndarray]:
                - A list of predicted emotion labels.
                - The corresponding model output scores (logits or probabilities), as a NumPy array.
        """
        scores = self._get_probab(features)
        if self.is_mtl:
            x = scores[:, :-2]
        else:
            x = scores
        preds = np.argmax(x, axis=1)

        if not logits:
            e_x = np.exp(x - np.max(x, axis=1)[:, np.newaxis])
            e_x = e_x / e_x.sum(axis=1)[:, None]
            if self.is_mtl:
                scores[:, :-2] = e_x
            else:
                scores = e_x

        return [self.idx_to_emotion_class[pred] for pred in preds], scores

    def classify_engagement(self, features: np.ndarray, sliding_window_width: int = 128):
        """
        Classify engagement levels based on extracted feature representations using a sliding
        window approach.

        Args:
            features (np.ndarray): The extracted feature vectors from a video sequence.
            sliding_window_width (int, optional):
                The width of the sliding window used for engagement classification. Defaults to 128.

        Returns:
            Tuple[List[str], np.ndarray]:
                - A list of predicted engagement levels.
                - The corresponding model output scores.

        Raises:
            ValueError: If the number of frames in the video is smaller than the sliding window
                        width.
        """
        stat_func = np.std
        if features.shape[0] < sliding_window_width:
            raise ValueError(
                f"Not enough frames to predict engagement. "
                f"Sliding window width: {sliding_window_width}, "
                f"but number of frames in video: {features.shape[0]}"
            )
        max_iters = features.shape[0] - sliding_window_width
        features_slices = []
        for i in range(max_iters):
            start = i
            end = sliding_window_width + i
            x = features[start:end]
            mean_x = np.repeat(stat_func(x, axis=0).reshape((1, -1)), len(x), axis=0)
            features_slices.append(np.concatenate((mean_x, x), axis=1))
        features_slices = np.array(features_slices)
        model = get_engagement_model(features_slices.shape[-1], sliding_window_width)
        scores = model.predict(features_slices, verbose=0)
        preds = np.argmax(scores, axis=1)
        return [self.idx_to_engagement_class[pred] for pred in preds], scores

    def predict_engagement(self, face_imgs: List[np.ndarray], sliding_window_width: int = 128):
        """
        Predict the engagement presented on a sequence of facial images.

        Args:
            face_imgs (List[np.ndarray]):
                A sequence of face images.
            sliding_window_width (int, optional):
                The width of the sliding window used for engagement classification. Defaults to 128.

        Returns:
            Tuple[List[str], np.ndarray]:
                - A list of predicted engagement levels.
                - The corresponding model output scores.

        Raises:
            ValueError: If the number of frames in the video is smaller than the sliding window
                        width.
        """
        features = self.extract_features(face_imgs)
        return self.classify_engagement(features, sliding_window_width)

    def predict_emotions(
        self, face_img: Union[np.ndarray, List[np.ndarray]], logits: bool = True
    ) -> Tuple[List[str], np.ndarray]:
        """
        Predict the emotions presented on a given facial image or a list of facial images.

        Args:
            face_img (Union[np.ndarray, List[np.ndarray]]):
                A single face image (as a NumPy array) or a list of face images.
            logits (bool, optional):
                If True, returns raw model scores (logits). If False, applies softmax normalization
                to obtain probability distributions. Defaults to True.

        Returns:
            Tuple[Union[str, List[str]], np.ndarray]:
                - The predicted emotion label(s) as a list of strings (for single image only with
                  one element).
                - The corresponding model output scores (logits or probabilities), as a NumPy array.
        """
        features = self.extract_features(face_img)
        return self.classify_emotions(features, logits)


class EmotiEffLibRecognizerTorch(EmotiEffLibRecognizerBase):
    """
    Torch implementation of EmotiEffLibRecognizer.
    """

    def __init__(self, model_name: str = "enet_b0_8_best_vgaf", device: str = "cpu") -> None:
        super().__init__(model_name)
        self.device = device

        path = get_model_path_torch(model_name)
        if device == "cpu":
            model = torch.load(path, map_location=torch.device("cpu"), weights_only=False)
        else:
            model = torch.load(path, weights_only=False)
        if model_name == "mbf_va_mtl":
            self.classifier_weights = model.fc.weight.cpu().data.numpy()
            self.classifier_bias = model.fc.bias.cpu().data.numpy()
            model.fc = torch.nn.Identity()
        elif model_name == "mobilevit_va_mtl":
            self.classifier_weights = model.head.fc.weight.cpu().data.numpy()
            self.classifier_bias = model.head.fc.bias.cpu().data.numpy()
            model.head.fc = torch.nn.Identity()
        elif isinstance(model.classifier, torch.nn.Sequential):
            self.classifier_weights = model.classifier[0].weight.cpu().data.numpy()
            self.classifier_bias = model.classifier[0].bias.cpu().data.numpy()
            model.classifier = torch.nn.Identity()
        else:
            self.classifier_weights = model.classifier.weight.cpu().data.numpy()
            self.classifier_bias = model.classifier.bias.cpu().data.numpy()
            model.classifier = torch.nn.Identity()

        model = model.to(device)
        self.model = model.eval()

    def _preprocess(self, img: np.ndarray) -> np.ndarray:
        """
        Prepare an image for input to the model.

        Args:
            img (np.ndarray): The input image for preprocessing.

        Returns:
            np.ndarray: Preprocessed image.
        """
        test_transforms = transforms.Compose(
            [
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.mean, std=self.std),
            ]
        )
        return test_transforms(Image.fromarray(img))

    def extract_features(self, face_img: Union[np.ndarray, List[np.ndarray]]) -> np.ndarray:
        """
        Extract visual features from a facial image or a list of facial images.

        Args:
            face_img (Union[np.ndarray, List[np.ndarray]]):
                A single face image (as a NumPy array) or a list of face images.

        Returns:
            np.ndarray: The extracted feature vectors.
        """
        if isinstance(face_img, np.ndarray):
            img_tensor = self._preprocess(face_img)
            img_tensor.unsqueeze_(0)
        elif isinstance(face_img, list) and all(isinstance(i, np.ndarray) for i in face_img):
            img_tensor = [self._preprocess(img) for img in face_img]
            img_tensor = torch.stack(img_tensor, dim=0)
        else:
            raise TypeError("Expected np.ndarray or List[np.ndarray]")
        features = self.model(img_tensor.to(self.device))
        features = features.data.cpu().numpy()
        return features


class EmotiEffLibRecognizerOnnx(EmotiEffLibRecognizerBase):
    """
    ONNX implementation of EmotiEffLibRecognizer.
    """

    def __init__(self, model_name: str = "enet_b0_8_best_vgaf") -> None:
        super().__init__(model_name)

        path = get_model_path_onnx(model_name)
        model = onnx.load(path)
        graph = model.graph
        nodes = graph.node
        gemm_node = nodes[-1]
        new_output_name = gemm_node.input[0]
        if gemm_node is None or len(gemm_node.input) < 3:
            raise RuntimeError("Unexpected gemm node!")
        weight_name = gemm_node.input[1]
        bias_name = gemm_node.input[2]
        weight_tensor = next((t for t in graph.initializer if t.name == weight_name), None)
        bias_tensor = next((t for t in graph.initializer if t.name == bias_name), None)
        self.classifier_weights = numpy_helper.to_array(weight_tensor) if weight_tensor else None
        self.classifier_bias = numpy_helper.to_array(bias_tensor) if bias_tensor else None

        # Remove the last node
        graph.node.remove(gemm_node)
        graph.output.remove(graph.output[0])
        new_output_shape = [None, self.classifier_weights.shape[1]]
        new_output = helper.make_tensor_value_info(
            new_output_name, TensorProto.FLOAT, new_output_shape
        )
        graph.output.append(new_output)

        model_bytes = model.SerializeToString()
        ort.set_default_logger_severity(3)
        self.ort_session = ort.InferenceSession(model_bytes, providers=["CPUExecutionProvider"])

    def _preprocess(self, img: np.ndarray) -> np.ndarray:
        """
        Prepare an image for input to the model.

        Args:
            img (np.ndarray): The input image for preprocessing.

        Returns:
            np.ndarray: Preprocessed image.
        """
        x = cv2.resize(img, (self.img_size, self.img_size)) / 255
        for i in range(3):
            x[..., i] = (x[..., i] - self.mean[i]) / self.std[i]
        return x.transpose(2, 0, 1).astype("float32")[np.newaxis, ...]

    def extract_features(self, face_img: Union[np.ndarray, List[np.ndarray]]) -> np.ndarray:
        """
        Extract visual features from a facial image or a list of facial images.

        Args:
            face_img (Union[np.ndarray, List[np.ndarray]]):
                A single face image (as a NumPy array) or a list of face images.

        Returns:
            np.ndarray: The extracted feature vectors.
        """
        if isinstance(face_img, np.ndarray):
            img_tensor = self._preprocess(face_img)
        elif isinstance(face_img, list) and all(isinstance(i, np.ndarray) for i in face_img):
            img_tensor = np.concatenate([self._preprocess(img) for img in face_img], axis=0)
        else:
            raise TypeError("Expected np.ndarray or List[np.ndarray]")
        features = self.ort_session.run(None, {"input": img_tensor})[0]
        return features


# pylint: disable=invalid-name
def EmotiEffLibRecognizer(
    engine: str = "torch", model_name: str = "enet_b0_8_best_vgaf", device: str = "cpu"
) -> Union[EmotiEffLibRecognizerOnnx, EmotiEffLibRecognizerTorch]:
    """
    Creates EmotiEffLibRecognizer instance.

    Args:
        engine (str): The engine to use for inference. Can be either "torch" or "onnx".
                      Default is "torch".
        model_name (str): The name of the model to be used for emotion prediction.
                          Default is "enet_b0_8_best_vgaf".
        device (str): The device on which to run the model, either "cpu" or "cuda".
                      Default is "cpu".

    Returns:
        EmotiEffLibRecognizerTorch or EmotiEffLibRecognizerOnnx: An instance of the corresponding
        emotion recognition class based on the selected engine.
    """
    # pylint: disable=unused-import, import-outside-toplevel, redefined-outer-name
    if engine not in get_supported_engines():
        raise ValueError("Unsupported engine specified")
    if engine == "torch":
        try:
            import torch
            from torchvision import transforms
        except ImportError as e:
            raise ImportError("Looks like torch module is not installed: ", e) from e
        return EmotiEffLibRecognizerTorch(model_name, device)
    # ONNX
    try:
        import onnx
        import onnxruntime as ort
        from onnx import helper, numpy_helper
    except ImportError as e:
        raise ImportError("Looks like onnx module is not installed: ", e) from e
    return EmotiEffLibRecognizerOnnx(model_name)
