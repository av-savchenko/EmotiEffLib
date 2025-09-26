/**
 * @file facial_analysis.cpp
 * @brief Implementation of the ONNX backend for the EmotiEffLibRecognizer.
 */

#include "emotiefflib/backends/onnx/facial_analysis.h"

#include <xtensor/containers/xadapt.hpp>
#include <xtensor/io/xio.hpp>
#include <xtensor/views/xview.hpp>

namespace {
/**
 * @brief Converts an xt::xarray to an ONNX Runtime tensor.
 *
 * Important! Looks like Ort::Value doesn't contain the own handler in it.
 * Instead, it points to the memory allocated by other objects, e.g. std::vector or xt::xarray.
 * We have to make sure that during operation with Ort::Value the initial object is in valid state.
 *
 * @param xarray The input xt::xarray.
 * @return The converted ONNX Runtime tensor.
 */
Ort::Value xarray2tensor(xt::xarray<float>& xarray) {
    auto xtensor = xt::eval(xarray);
    // Extract shape
    std::vector<int64_t> shape(xtensor.shape().begin(), xtensor.shape().end());

    // Create ONNX Runtime memory info (CPU)
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

    // Create ONNX Runtime tensor
    Ort::Value onnx_tensor = Ort::Value::CreateTensor<float>(
        memory_info, xarray.data(), xarray.size(), shape.data(), shape.size());

    // Verify tensor is valid
    if (!onnx_tensor.IsTensor()) {
        throw std::runtime_error("Error during ONNX tensor creation!");
    }
    return onnx_tensor;
}

/**
 * @brief Converts an ONNX Runtime tensor to an xt::xarray.
 *
 * This function creates a deep copy of Ort::Value and returns xt::xarray.
 *
 * @param tensor The input ONNX Runtime tensor.
 * @return The converted xt::xarray.
 */
xt::xarray<float> tensor2xarray(const Ort::Value& tensor) {
    if (!tensor.IsTensor()) {
        throw std::runtime_error("Input ONNX Value is not a tensor!");
    }

    // Get shape information
    Ort::TensorTypeAndShapeInfo tensor_info = tensor.GetTensorTypeAndShapeInfo();
    std::vector<int64_t> shape = tensor_info.GetShape();

    // Get raw data pointer (use GetTensorData instead of GetTensorMutableData)
    const float* data_ptr = tensor.GetTensorData<float>();
    size_t element_count = tensor_info.GetElementCount();

    // Convert shape to xtensor format (size_t instead of int64_t)
    std::vector<size_t> xtensor_shape(shape.begin(), shape.end());

    // Create an xt::xarray and copy the data
    xt::xarray<float> result = xt::zeros<float>(xtensor_shape);
    std::copy(data_ptr, data_ptr + element_count, result.begin());
    return result;
}

/**
 * @brief Checks if the model has the correct number of inputs and outputs.
 *
 * @param session The ONNX Runtime session.
 */
void checkModelInputs(const Ort::Session& session) {
    if (session.GetInputCount() != 1 || session.GetOutputCount() != 1)
        throw std::runtime_error("Only models with one input and one output are supported!");
}
} // namespace

namespace EmotiEffLib {
EmotiEffLibRecognizerOnnx::EmotiEffLibRecognizerOnnx(const std::string& fullPipelineModel) {
    // Define session options
    Ort::SessionOptions session_options;

    // Load the ONNX model
    Ort::Session model(env_, fullPipelineModel.c_str(), session_options);
    models_.push_back(std::move(model));
    fullPipelineModelIdx_ = 0;
    initRecognizer(fullPipelineModel);
}

EmotiEffLibRecognizerOnnx::EmotiEffLibRecognizerOnnx(const EmotiEffLibConfig& config) {
    configParser(config);
    initRecognizer(config.featureExtractorPath);
}

xt::xarray<float> EmotiEffLibRecognizerOnnx::extractFeatures(const cv::Mat& faceImg) {
    if (featureExtractorIdx_ == -1)
        throw std::runtime_error("Model for features extraction wasn't specified in the config!");
    auto imgTensor = preprocess(faceImg);
    std::vector<Ort::Value> inputTensors;
    inputTensors.push_back(xarray2tensor(imgTensor));
    auto outputTensors = modelRunWrapper(featureExtractorIdx_, inputTensors);
    auto features = tensor2xarray(outputTensors[0]);
    return features;
}

EmotiEffLibRes EmotiEffLibRecognizerOnnx::classifyEmotions(const xt::xarray<float>& features,
                                                           bool logits) {
    if (classifierIdx_ == -1)
        throw std::runtime_error(
            "Model for emotions classification wasn't specified in the config!");
    xt::xarray<float> featuresHandler = features;
    std::vector<Ort::Value> inputTensors;
    inputTensors.push_back(xarray2tensor(featuresHandler));
    auto outputTensors = modelRunWrapper(classifierIdx_, inputTensors);
    auto scores = tensor2xarray(outputTensors[0]);
    return processEmotionScores(scores, logits);
}

EmotiEffLibRes EmotiEffLibRecognizerOnnx::classifyEngagement(const xt::xarray<float>& features) {
    if (engagementClassifierIdx_ == -1)
        throw std::runtime_error(
            "Model for engagement classification wasn't specified in the config!");
    if (features.shape(0) < engagementSlidingWindowSize_)
        throw std::runtime_error(
            "Not enough features to predict engagement. Sliding window width: " +
            std::to_string(engagementSlidingWindowSize_) +
            ", but number of features in a sequence: " + std::to_string(features.shape(0)));

    auto featuresSlices = engagementFeaturesPreprocess(features);

    std::vector<Ort::Value> inputTensors;
    inputTensors.push_back(xarray2tensor(featuresSlices));
    auto engClassifierOutput = modelRunWrapper(engagementClassifierIdx_, inputTensors);
    auto scores = tensor2xarray(engClassifierOutput[0]);
    return processEngagementScores(scores);
}

EmotiEffLibRes EmotiEffLibRecognizerOnnx::predictEmotions(const cv::Mat& faceImg, bool logits) {
    if (fullPipelineModelIdx_ == -1 && (featureExtractorIdx_ == -1 || classifierIdx_ == -1))
        throw std::runtime_error("predictEmotions method requires fillPipeline model or "
                                 "featureExtractor and classifier models");
    auto imgTensor = preprocess(faceImg);

    int extractorIdx = (fullPipelineModelIdx_ > -1) ? fullPipelineModelIdx_ : featureExtractorIdx_;
    std::vector<Ort::Value> inputTensors;
    inputTensors.push_back(xarray2tensor(imgTensor));
    auto outputTensors = modelRunWrapper(extractorIdx, inputTensors);

    xt::xarray<float> scores;
    if (fullPipelineModelIdx_ == -1 && classifierIdx_ > -1) {
        auto classifierOutputTensors = modelRunWrapper(classifierIdx_, outputTensors);
        scores = tensor2xarray(classifierOutputTensors[0]);
    } else {
        scores = tensor2xarray(outputTensors[0]);
    }

    return processEmotionScores(scores, logits);
}

EmotiEffLibRes EmotiEffLibRecognizerOnnx::predictEngagement(const std::vector<cv::Mat>& faceImgs) {
    if (featureExtractorIdx_ == -1 || engagementClassifierIdx_ == -1)
        throw std::runtime_error(
            "predictEngagement method requires featureExtractor and enagement classifier models");
    if (faceImgs.size() < engagementSlidingWindowSize_)
        throw std::runtime_error(
            "Not enough frames to predict engagement. Sliding window width: " +
            std::to_string(engagementSlidingWindowSize_) +
            ", but number of frames in video: " + std::to_string(faceImgs.size()));
    auto features = EmotiEffLibRecognizer::extractFeatures(faceImgs);

    return classifyEngagement(features);
}

void EmotiEffLibRecognizerOnnx::initRecognizer(const std::string& modelPath) {
    EmotiEffLibRecognizer::initRecognizer(modelPath);

    mean_ = {0.485, 0.456, 0.406};
    std_ = {0.229, 0.224, 0.225};
    if (modelName_.find("mbf_") != std::string::npos) {
        imgSize_ = 112;
        mean_ = {0.5, 0.5, 0.5};
        std_ = {0.5, 0.5, 0.5};
    } else if (modelName_.find("_b2_") != std::string::npos) {
        imgSize_ = 260;
    } else if (modelName_.find("ddamfnet") != std::string::npos) {
        imgSize_ = 112;
    } else {
        imgSize_ = 224;
    }
}

xt::xarray<float> EmotiEffLibRecognizerOnnx::preprocess(const cv::Mat& img) {
    cv::Mat resized_img, float_img, normalized_img;

    // Resize the image
    cv::resize(img, resized_img, cv::Size(imgSize_, imgSize_));

    // Convert to float32 and scale to [0, 1]
    resized_img.convertTo(float_img, CV_32FC3, 1.0 / 255.0);

    // Normalize each channel
    std::vector<cv::Mat> channels(3);
    cv::split(float_img, channels);
    for (int i = 0; i < 3; i++) {
        channels[i] = (channels[i] - mean_[i]) / std_[i];
    }

    // Merge back the channels
    cv::merge(channels, normalized_img);

    // Convert HWC OpenCV Mat to CHW xtensor
    std::vector<float> chwData;
    chwData.reserve(3 * imgSize_ * imgSize_);

    for (int c = 0; c < 3; ++c) {
        for (int h = 0; h < imgSize_; ++h) {
            for (int w = 0; w < imgSize_; ++w) {
                chwData.push_back(normalized_img.at<cv::Vec3f>(h, w)[c]);
            }
        }
    }

    // Adapt vector to xt::xarray<float> with NCHW shape
    return xt::adapt(chwData, {1, 3, imgSize_, imgSize_});
}

void EmotiEffLibRecognizerOnnx::configParser(const EmotiEffLibConfig& config) {
    if (!config.modelName.empty()) {
        modelName_ = config.modelName;
    }
    // Define session options
    Ort::SessionOptions session_options;

    if (config.fullPipelineEmotionModelPath.empty() && config.featureExtractorPath.empty()) {
        throw std::runtime_error("fullPipelineEmotionModelPath or featureExtractorPath MUST be "
                                 "specified in the EmotiEffLibConfig.");
    }
    if (!config.fullPipelineEmotionModelPath.empty()) {
        Ort::Session model(env_, config.fullPipelineEmotionModelPath.c_str(), session_options);
        models_.push_back(std::move(model));
        fullPipelineModelIdx_ = models_.size() - 1;
    }
    if (!config.featureExtractorPath.empty()) {
        Ort::Session model(env_, config.featureExtractorPath.c_str(), session_options);
        models_.push_back(std::move(model));
        featureExtractorIdx_ = models_.size() - 1;
    }
    if (!config.classifierPath.empty()) {
        Ort::Session model(env_, config.classifierPath.c_str(), session_options);
        models_.push_back(std::move(model));
        classifierIdx_ = models_.size() - 1;
    }
    if (!config.engagementClassifierPath.empty()) {
        Ort::Session model(env_, config.engagementClassifierPath.c_str(), session_options);
        models_.push_back(std::move(model));
        engagementClassifierIdx_ = models_.size() - 1;
    }
}

std::vector<Ort::Value>
EmotiEffLibRecognizerOnnx::modelRunWrapper(int modelIdx, const std::vector<Ort::Value>& inputs) {
    auto& session = models_[modelIdx];
    checkModelInputs(session);

    auto input_name = session.GetInputNameAllocated(0, allocator_);
    std::vector<const char*> inputNames = {input_name.get()};

    auto outputName = session.GetOutputNameAllocated(0, allocator_);
    std::vector<const char*> outputNames = {outputName.get()};

    // Run inference
    return session.Run(Ort::RunOptions{nullptr}, inputNames.data(), inputs.data(), 1,
                       outputNames.data(), 1);
}
} // namespace EmotiEffLib
