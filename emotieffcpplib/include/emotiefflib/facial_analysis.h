/**
 * @file facial_analysis.h
 * @brief Header file for the EmotiEffLib library, providing emotion and engagement recognition
 * functionality.
 */

#ifndef FACIAL_ANALYSIS_H
#define FACIAL_ANALYSIS_H

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <xtensor/containers/xarray.hpp>

namespace EmotiEffLib {

/**
 * @brief Retrieves the list of available inference backends.
 *
 * This function returns a list of supported inference backends based on the compilation options.
 * The available backends depend on whether the library was built with Torch (LibTorch) or ONNX
 * Runtime support.
 *
 * @return A vector of strings containing the names of the available backends.
 *         Possible values: "torch", "onnx".
 */
std::vector<std::string> getAvailableBackends();

/**
 * @brief Retrieves the list of supported emotion or engagement recognition models.
 *
 * This function returns a list of pre-trained models that can be used for emotion recognition or
 * visual features extraction. Each model has been trained on different datasets and may have
 * varying levels of accuracy and performance depending on the application.
 *
 * @return A vector of strings containing the names of the supported models.
 */
std::vector<std::string> getSupportedModels(const std::string& backendName);

/**
 * @brief Structure to store the results of EmotiEffLib.
 *
 * This structure holds the predicted emotion/engagement labels and their corresponding confidence
 * scores for a given input.
 */
struct EmotiEffLibRes {
    /**
     * @brief The predicted emotion/engagement labels.
     *
     * A list of strings representing the emotion/engagement categories detected by the model.
     */
    std::vector<std::string> labels;

    /**
     * @brief The confidence scores for each input frame.
     *
     * An array of floating-point values representing the model's confidence in each
     * predicted class. The scores correspond to the labels in the same order.
     */
    xt::xarray<float> scores;
};

/**
 * @brief Configuration structure for the EmotiEffLibRecognizer.
 *
 * This structure holds the configuration parameters required to initialize and run
 * the emotion and engagement recognition models. It includes paths to model files
 * and the selected inference backend.
 */
struct EmotiEffLibConfig {
    /**
     * @brief The inference backend to use.
     *
     * Specifies the deep learning framework for model inference.
     * Possible values: "torch", "onnx".
     */
    std::string backend = "";

    /**
     * @brief Path to the model with full emotion recognition pipeline.
     *
     * The model extracts visual features from the input images and classify facial emotions.
     */
    std::string fullPipelineEmotionModelPath = "";

    /**
     * @brief Path to the feature extractor model file.
     *
     * The feature extractor processes input frames to generate embeddings for classification.
     */
    std::string featureExtractorPath = "";

    /**
     * @brief Path to the emotion classifier model file.
     *
     * The classifier predicts emotion labels based on extracted features.
     */
    std::string classifierPath = "";

    /**
     * @brief Path to the engagement classifier model file.
     *
     * This model predicts the engagement level of the user.
     */
    std::string engagementClassifierPath = "";

    /**
     * @brief The name of the selected model.
     *
     * Specifies the name of feature extraction model being used.
     */
    std::string modelName = "";
};

/**
 * @brief Base class for emotion and engagement recognition.
 *
 * This class provides the interface for emotion and engagement recognition using different
 * backends. It supports feature extraction, emotion classification, and engagement prediction.
 */
class EmotiEffLibRecognizer {
public:
    virtual ~EmotiEffLibRecognizer() = default;

    /*****************************************************************************
     * Initializers
     *****************************************************************************/
    /**
     * @brief Creates an instance of the EmotiEffLibRecognizer with the specified backend and model
     * path.
     *
     * @param backend The inference backend to use ("torch" or "onnx").
     * @param fullPipelineModelPath Path to the full pipeline model file.
     * @return A unique pointer to the created EmotiEffLibRecognizer instance.
     */
    static std::unique_ptr<EmotiEffLibRecognizer>
    createInstance(const std::string& backend, const std::string& fullPipelineModelPath);

    /**
     * @brief Creates an instance of the EmotiEffLibRecognizer with the specified configuration.
     *
     * @param config The configuration structure containing backend and model paths.
     * @return A unique pointer to the created EmotiEffLibRecognizer instance.
     */
    static std::unique_ptr<EmotiEffLibRecognizer> createInstance(const EmotiEffLibConfig& config);

    /*****************************************************************************
     * Getters for class name
     *****************************************************************************/
    /**
     * @brief Gets the emotion class name by index.
     *
     * @param idx The index of the emotion class.
     * @return The name of the emotion class.
     */
    inline std::string getEmotionClassById(int idx) { return idxToEmotionClass_[idx]; }

    /**
     * @brief Gets the engagement class name by index.
     *
     * @param idx The index of the engagement class.
     * @return The name of the engagement class.
     */
    inline std::string getEngagementClassById(int idx) { return idxToEngagementClass_[idx]; }

    /*****************************************************************************
     * Public API
     *****************************************************************************/
    /**
     * @brief Extracts features from a single face image.
     *
     * @param faceImg The input face image.
     * @return An array of extracted features.
     */
    virtual xt::xarray<float> extractFeatures(const cv::Mat& faceImg) = 0;

    /**
     * @brief Extracts features from a sequence of face images.
     *
     * @param faceImgs A list of input face images.
     * @return An array of extracted features.
     */
    virtual xt::xarray<float> extractFeatures(const std::vector<cv::Mat>& faceImgs);

    /**
     * @brief Classifies emotions based on extracted features.
     *
     * @param features The extracted features.
     * @param logits Whether to return raw logits or apply softmax.
     * @return The classification results.
     */
    virtual EmotiEffLibRes classifyEmotions(const xt::xarray<float>& features,
                                            bool logits = true) = 0;

    /**
     * @brief Classifies engagement based on extracted features.
     *
     * @param features The extracted features.
     * @return The classification results.
     */
    virtual EmotiEffLibRes classifyEngagement(const xt::xarray<float>& features) = 0;

    /**
     * @brief Predicts emotions for a single face image.
     *
     * @param faceImg The input face image.
     * @param logits Whether to return raw logits or apply softmax.
     * @return The prediction results.
     */
    virtual EmotiEffLibRes predictEmotions(const cv::Mat& faceImg, bool logits = true) = 0;

    /**
     * @brief Predicts emotions for a sequence of face images.
     *
     * @param faceImgs A list of input face images.
     * @param logits Whether to return raw logits or apply softmax.
     * @return The prediction results.
     */
    virtual EmotiEffLibRes predictEmotions(const std::vector<cv::Mat>& faceImgs,
                                           bool logits = true);

    /**
     * @brief Predicts engagement for a sequence of face images.
     *
     * @param faceImgs A list of input face images.
     * @return The prediction results.
     */
    virtual EmotiEffLibRes predictEngagement(const std::vector<cv::Mat>& faceImgs) = 0;

protected:
    /**
     * @brief Initializes the internal state of recognizer based on a modelPath.
     *
     * @param modelPath Path to the model file.
     */
    virtual void initRecognizer(const std::string& modelPath);

    /**
     * @brief Common method to processes emotion scores and generate classification results.
     *
     * @param scores The raw emotion scores.
     * @param logits Whether to return raw logits or apply softmax.
     * @return The processed classification results.
     */
    EmotiEffLibRes processEmotionScores(const xt::xarray<float>& scores, bool logits);

    /**
     * @brief Common method to processes engagement scores and generate classification results.
     *
     * @param scores The raw engagement scores.
     * @return The processed classification results.
     */
    EmotiEffLibRes processEngagementScores(const xt::xarray<float>& scores);

    /**
     * @brief Parses the configuration to initialize the recognizer.
     *
     * @param config The configuration structure.
     */
    virtual void configParser(const EmotiEffLibConfig& config) = 0;

    /**
     * @brief Preprocesses an image for feature extraction.
     *
     * @param img The input image.
     * @return The preprocessed image as an array.
     */
    virtual xt::xarray<float> preprocess(const cv::Mat& img) = 0;

    /**
     * @brief Preprocesses engagement features.
     *
     * @param features The extracted features.
     * @return The preprocessed features.
     */
    xt::xarray<float> engagementFeaturesPreprocess(const xt::xarray<float> features);

private:
    /**
     * @brief Checks if the specified backend is supported.
     *
     * @param backend The backend to check.
     */
    static void checkBackend(const std::string& backend);

protected:
    /// Name of the model.
    std::string modelName_ = "";
    /// Index of the full pipeline model in the array with models.
    int fullPipelineModelIdx_ = -1;
    /// Index of the model for features extraction in the array with models.
    int featureExtractorIdx_ = -1;
    /// Index of the classifier model in the array with models.
    int classifierIdx_ = -1;
    /// Index of the engagement classifier model in the array with models.
    int engagementClassifierIdx_ = -1;
    /// Sliding window size for engagement prediction.
    const int engagementSlidingWindowSize_ = 128;
    /// Engagement class names.
    std::vector<std::string> idxToEngagementClass_ = {"Distracted", "Engaged"};
    /// Emotion class names.
    std::vector<std::string> idxToEmotionClass_;
    /// Whether the model is multi-task learning (MTL).
    bool isMtl_;
    /// Input image size.
    int imgSize_;
};

} // namespace EmotiEffLib

#endif
