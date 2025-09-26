/**
 * @file facial_analysis.cpp
 * @brief Implementation of the EmotiEffLibRecognizer base class and its utility functions.
 */

#include "emotiefflib/facial_analysis.h"
#ifdef WITH_ONNX
#include "emotiefflib/backends/onnx/facial_analysis.h"
#endif
#ifdef WITH_TORCH
#include "emotiefflib/backends/torch/facial_analysis.h"
#endif

#include <filesystem>

#include <xtensor/core/xmath.hpp>
#include <xtensor/misc/xsort.hpp>
#include <xtensor/views/xview.hpp>

namespace fs = std::filesystem;

namespace EmotiEffLib {
std::vector<std::string> getAvailableBackends() {
    return {
#ifdef WITH_TORCH
        "torch",
#endif
#ifdef WITH_ONNX
        "onnx",
#endif
    };
}

std::vector<std::string> getSupportedModels(const std::string& backendName) {
    std::vector<std::string> modelsList = {"enet_b0_8_best_vgaf", "enet_b0_8_best_afew",
                                           "enet_b2_8", "enet_b0_8_va_mtl", "enet_b2_7"};
    if (backendName == "onnx") {
        modelsList.push_back("mbf_va_mtl");
        modelsList.push_back("mobilevit_va_mtl");
    }
    return modelsList;
}

std::unique_ptr<EmotiEffLibRecognizer>
EmotiEffLibRecognizer::createInstance(const std::string& backend,
                                      const std::string& fullPipelineModelPath) {
    checkBackend(backend);
#ifdef WITH_TORCH
    if (backend == "torch")
        return std::make_unique<EmotiEffLibRecognizerTorch>(fullPipelineModelPath);
#elif defined(WITH_ONNX)
    return std::make_unique<EmotiEffLibRecognizerOnnx>(fullPipelineModelPath);
#else
    throw std::runtime_error("EmotiEffCppLib wasn't compiled with any backend support.");
#endif
}

std::unique_ptr<EmotiEffLibRecognizer>
EmotiEffLibRecognizer::createInstance(const EmotiEffLibConfig& config) {
    checkBackend(config.backend);
#ifdef WITH_TORCH
    if (config.backend == "torch")
        return std::make_unique<EmotiEffLibRecognizerTorch>(config);
#elif defined(WITH_ONNX)
    return std::make_unique<EmotiEffLibRecognizerOnnx>(config);
#else
    throw std::runtime_error("EmotiEffCppLib wasn't compiled with any backend support.");
#endif
}

xt::xarray<float> EmotiEffLibRecognizer::extractFeatures(const std::vector<cv::Mat>& faceImgs) {
    xt::xarray<float> result;
    for (size_t i = 0; i < faceImgs.size(); ++i) {
        auto features = extractFeatures(faceImgs[i]);
        // We need such workaround for concatenating because of known issue in xtensor:
        // https://github.com/xtensor-stack/xtensor/issues/2579
        if (i == 0) {
            result = xt::zeros<float>({faceImgs.size(), features.shape(1)});
        }
        xt::view(result, i, xt::all()) = xt::view(features, 0, xt::all());
    }

    return result;
}

EmotiEffLibRes EmotiEffLibRecognizer::predictEmotions(const std::vector<cv::Mat>& faceImgs,
                                                      bool logits) {
    EmotiEffLibRes result;
    result.labels = {};
    for (size_t i = 0; i < faceImgs.size(); ++i) {
        auto res = predictEmotions(faceImgs[i], logits);
        if (i == 0) {
            result.labels = {};
            result.scores = xt::zeros<float>({faceImgs.size(), res.scores.shape(1)});
        }
        result.labels.insert(result.labels.end(), res.labels.begin(), res.labels.end());
        // We need such workaround for concatenating because of known issue in xtensor:
        // https://github.com/xtensor-stack/xtensor/issues/2579
        xt::view(result.scores, i, xt::all()) = xt::view(res.scores, 0, xt::all());
    }
    return result;
}

void EmotiEffLibRecognizer::initRecognizer(const std::string& modelPath) {
    // Do not change modelName if it was explicitly specified in the config
    if (modelName_.empty()) {
        modelName_ = fs::path(modelPath).filename().string();
    }
    isMtl_ = modelName_.find("_mtl") != std::string::npos;
    bool is7 = modelName_.find("_7") != std::string::npos;
    if (is7) {
        idxToEmotionClass_.resize(7);
        idxToEmotionClass_[0] = "Anger";
        idxToEmotionClass_[1] = "Disgust";
        idxToEmotionClass_[2] = "Fear";
        idxToEmotionClass_[3] = "Happiness";
        idxToEmotionClass_[4] = "Neutral";
        idxToEmotionClass_[5] = "Sadness";
        idxToEmotionClass_[6] = "Surprise";
    } else {
        idxToEmotionClass_.resize(8);
        idxToEmotionClass_[0] = "Anger";
        idxToEmotionClass_[1] = "Contempt";
        idxToEmotionClass_[2] = "Disgust";
        idxToEmotionClass_[3] = "Fear";
        idxToEmotionClass_[4] = "Happiness";
        idxToEmotionClass_[5] = "Neutral";
        idxToEmotionClass_[6] = "Sadness";
        idxToEmotionClass_[7] = "Surprise";
    }
}

EmotiEffLibRes EmotiEffLibRecognizer::processEmotionScores(const xt::xarray<float>& score,
                                                           bool logits) {
    xt::xarray<float> x;
    xt::xarray<float> scores = score;
    // Select relevant part of the scores based on is_mtl
    if (isMtl_) {
        x = xt::view(scores, xt::all(),
                     xt::range(xt::placeholders::_, -2)); // Equivalent to scores[:, :-2]
    } else {
        x = scores;
    }

    // Compute predictions
    auto preds = xt::argmax(x, 1);

    // Apply softmax if logits is false
    if (!logits) {
        xt::xarray<float> max_x = xt::amax(x, {1}, xt::evaluation_strategy::immediate);
        xt::xarray<float> e_x = xt::exp(x - max_x);
        e_x /= xt::sum(e_x, {1}, xt::evaluation_strategy::immediate);

        if (isMtl_) {
            xt::view(scores, xt::all(), xt::range(xt::placeholders::_, -2)) =
                e_x; // Modify in-place
        } else {
            scores = e_x; // Replace scores with softmaxed values
        }
    }

    // Convert predictions to emotion class names
    EmotiEffLibRes res;
    for (auto pred : preds) {
        res.labels.push_back(idxToEmotionClass_[pred]);
    }
    res.scores = scores;
    return res;
}

EmotiEffLibRes EmotiEffLibRecognizer::processEngagementScores(const xt::xarray<float>& score) {
    // Compute predictions
    auto preds = xt::argmax(score, 1);

    // Convert predictions to engagement class names
    EmotiEffLibRes res;
    for (auto pred : preds) {
        res.labels.push_back(idxToEngagementClass_[pred]);
    }
    res.scores = score;
    return res;
}

xt::xarray<float>
EmotiEffLibRecognizer::engagementFeaturesPreprocess(const xt::xarray<float> features) {
    int maxIters = features.shape(0) - engagementSlidingWindowSize_;
    xt::xarray<float> features_slices;
    for (int i = 0; i < maxIters; ++i) {
        if (i == 0) {
            features_slices = xt::zeros<float>({static_cast<size_t>(maxIters),
                                                static_cast<size_t>(engagementSlidingWindowSize_),
                                                features.shape(1) * 2});
        }
        auto x = xt::view(features, xt::range(i, i + engagementSlidingWindowSize_), xt::all());
        auto mean_x = xt::repeat(xt::expand_dims(xt::stddev(x, {0}), 0), x.shape(0), 0);
        xt::view(features_slices, i, xt::all()) = xt::concatenate(xt::xtuple(mean_x, x), 1);
    }
    return features_slices;
}

void EmotiEffLibRecognizer::checkBackend(const std::string& backend) {
    auto backends = getAvailableBackends();
    auto it = std::find(backends.begin(), backends.end(), backend);
    if (it == backends.end()) {
        throw std::runtime_error(
            "This backend (" + backend +
            ") is not supported. Please check your EmotiEffLib build or configuration.");
    }
}

} // namespace EmotiEffLib
