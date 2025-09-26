"""
Trace torch models for usage in C++ Libtorch
"""

import os
import pathlib

# pylint: disable=duplicate-code
try:
    import torch
except ImportError:
    pass
try:
    import onnx
    from onnx import helper
except ImportError:
    pass

from engagement_classification.engagement_single_attention_converter import (
    engagement_to_onnx_converter,
    engagement_to_torch_converter,
)

from emotiefflib.facial_analysis import get_model_list

FILE_DIR = pathlib.Path(__file__).parent.resolve()


def trace_model(
    model_name: str,
    torch_model: str,
    model_out: str,
    features_extractor_out: str,
    classifier_out: str,
) -> None:
    """
    Convert a PyTorch model into a TorchScript traced model and save it.

    Args:
        model_name (str): Name of the model with extension.
        torch_model (str): Path to the input PyTorch model file.
        model_out (str): Path to save the traced TorchScript original model.
        features_extractor (str): Path to save the traced TorchScript feature extraction model.
        classifier_out (str): Path to save the traced TorchScript classification model.

    Returns:
        None
    """
    print(f"Processing {torch_model}...")
    if model_name in ("mbf_va_mtl.pt", "mobilevit_va_mtl.pt"):
        print(f"Skip unsupported model: {model_name}...")
        return
    if "mbf_" in model_name:
        img_size = 112
    elif "_b2_" in model_name:
        img_size = 260
    elif "ddamfnet" in model_name:
        img_size = 112
    else:
        img_size = 224
    input_shape = (1, 3, img_size, img_size)
    model_example = torch.rand(*input_shape)
    model = torch.load(torch_model, map_location=torch.device("cpu"), weights_only=False)
    # pylint: disable=no-else-return
    if model_name in ("mbf_va_mtl.pt", "mobilevit_va_mtl.pt"):
        # This is a workaround but it still is not working because of shapes issues in runtime
        traced_script_module = torch.jit.trace(model, model_example)
        traced_script_module = torch.jit.script(traced_script_module, model_example)
    else:
        traced_script_module = torch.jit.script(model, model_example)
    traced_script_module.save(model_out)
    if model_name == "mbf_va_mtl.pt":
        classifier_shape = (1, model.fc.in_features)
        classifier = model.fc
        model.fc = torch.nn.Identity()
    elif model_name == "mobilevit_va_mtl.pt":
        classifier_shape = (1, model.head.fc.in_features)
        classifier = model.head.fc
        model.head.fc = torch.nn.Identity()
    elif isinstance(model.classifier, torch.nn.Sequential):
        classifier_shape = (1, model.classifier[0].in_features)
        classifier = model.classifier
        model.classifier = torch.nn.Identity()
    else:
        classifier_shape = (1, model.classifier.in_features)
        classifier = model.classifier
        model.classifier = torch.nn.Identity()
    classifier_example = torch.rand(*classifier_shape)
    traced_script_features_extractor = torch.jit.script(model, model_example)
    traced_script_features_extractor.save(features_extractor_out)
    traced_script_classifier = torch.jit.script(classifier, classifier_example)
    traced_script_classifier.save(classifier_out)


def prepare_torch_models(output_dir: str) -> None:
    """
    Trace torch models from EmotiEffLib.
    """
    models_dir = os.path.join(FILE_DIR, "affectnet_emotions")
    # model_files = [f for f in os.listdir(models_dir) if f.endswith(".pt")]
    model_files = [f + ".pt" for f in get_model_list()]

    for mf in model_files:
        inp = os.path.join(models_dir, mf)
        model_out = os.path.join(output_dir, mf)
        features_extractor_out = os.path.join(output_dir, "features_extractor_" + mf)
        classifier_out = os.path.join(output_dir, "classifier_" + mf)
        if (
            os.path.exists(model_out)
            and os.path.exists(features_extractor_out)
            and os.path.exists(classifier_out)
        ):
            print(f"SKIP {mf}")
            continue
        trace_model(mf, inp, model_out, features_extractor_out, classifier_out)


def split_onnx_model(
    model_path: str, model_out: str, features_extractor_out: str, classifier_out: str
) -> None:
    """
    Split an ONNX model into a feature extractor and a classifier.

    Args:
        model_path (str): Path to the input ONNX model.
        model_out (str): Path to save the original model.
        features_extractor_out (str): Path to save the features extraction model.
        classifier_out (str): Path to save the extracted classifier as a separate ONNX model.

    Returns:
        None

    Raises:
        RuntimeError: If the model has no nodes or if the last classification node is unexpected.
    """
    print(f"Processing {model_path}...")
    model = onnx.load(model_path)
    onnx.save(model, model_out)
    graph = model.graph
    # Ensure the model has at least one node
    if not graph.node:
        raise RuntimeError("Model has no nodes")

    # Extract last node (assumed to be the final layer)
    gemm_node = graph.node[-1]
    new_output_name = gemm_node.input[0]
    if gemm_node is None or len(gemm_node.input) < 3:
        raise RuntimeError("Unexpected gemm node!")

    weight_name = gemm_node.input[1]
    bias_name = gemm_node.input[2]
    weight_tensor = next((t for t in graph.initializer if t.name == weight_name), None)

    # Remove the last node
    graph.node.remove(gemm_node)
    graph.output.remove(graph.output[0])

    if weight_tensor:
        new_output_shape = [None, onnx.numpy_helper.to_array(weight_tensor).shape[1]]
    else:
        new_output_shape = [None, None]

    new_output = helper.make_tensor_value_info(
        new_output_name, onnx.TensorProto.FLOAT, new_output_shape
    )
    graph.output.append(new_output)

    # Create new model for classifier
    classifier_graph = helper.make_graph(
        nodes=[gemm_node],
        name="classifier",
        inputs=[
            helper.make_tensor_value_info(inp, onnx.TensorProto.FLOAT, new_output_shape)
            for inp in gemm_node.input
        ],
        outputs=[helper.make_tensor_value_info(gemm_node.output[0], onnx.TensorProto.FLOAT, None)],
        initializer=[t for t in graph.initializer if t.name in gemm_node.input],
    )

    # Remove initializers from inputs
    inputs = classifier_graph.input
    name_to_input = {}
    for inp in inputs:
        name_to_input[inp.name] = inp

    for initializer in classifier_graph.initializer:
        if initializer.name in name_to_input:
            inputs.remove(name_to_input[initializer.name])

    classifier = helper.make_model(
        classifier_graph,
        producer_name="onnx-classifier",
        opset_imports=[helper.make_opsetid("", 21)],
    )

    # Remove unused initializers
    name_to_init = {}
    for init in graph.initializer:
        name_to_init[init.name] = init
    graph.initializer.remove(name_to_init[weight_name])
    graph.initializer.remove(name_to_init[bias_name])

    onnx.save(model, features_extractor_out)
    onnx.save(classifier, classifier_out)


def prepare_onnx_models(output_dir: str) -> None:
    """
    Prepare ONNX models from EmotiEffLib.
    """
    models_dir = os.path.join(FILE_DIR, "affectnet_emotions", "onnx")
    model_files = [f + ".onnx" for f in get_model_list()]

    for mf in model_files:
        inp = os.path.join(models_dir, mf)
        model_out = os.path.join(output_dir, mf)
        features_extractor_out = os.path.join(output_dir, "features_extractor_" + mf)
        classifier_out = os.path.join(output_dir, "classifier_" + mf)
        if (
            os.path.exists(model_out)
            and os.path.exists(features_extractor_out)
            and os.path.exists(classifier_out)
        ):
            print(f"SKIP {mf}")
            continue
        split_onnx_model(inp, model_out, features_extractor_out, classifier_out)


if __name__ == "__main__":
    # pylint: disable=unused-import, import-outside-toplevel, redefined-outer-name, ungrouped-imports
    out_models_dir = os.path.join(FILE_DIR, "emotieffcpplib_prepared_models")
    if not os.path.exists(out_models_dir):
        os.makedirs(out_models_dir)
    try:
        import onnx

        prepare_onnx_models(out_models_dir)
    except ImportError:
        pass
    try:
        import torch

        prepare_torch_models(out_models_dir)
    except ImportError:
        pass
    engagement_onnx_name = os.path.join(out_models_dir, "engagement_classifier_2560_128.onnx")
    engagement_torch_name = os.path.join(out_models_dir, "engagement_classifier_2560_128.pt")
    engagement_to_onnx_converter(engagement_onnx_name)
    engagement_to_torch_converter(engagement_torch_name)
