#include "test_utils.h"
#include "gtest/gtest.h"
#include <emotiefflib/facial_analysis.h>
#include <filesystem>
#include <gtest/gtest.h>
#include <string>

#include <xtensor/containers/xarray.hpp>
#include <xtensor/core/xmath.hpp>
#include <xtensor/io/xio.hpp>
#include <xtensor/misc/xsort.hpp>

namespace fs = std::filesystem;

namespace {
std::vector<std::string> getOneImageExpEmotions(const std::string& backend,
                                                const std::string& modelName) {
    if (modelName == "enet_b0_8_va_mtl" || modelName == "enet_b0_8_best_afew") {
        return {"Anger", "Happiness", "Happiness"};
    }
    if (modelName == "mbf_va_mtl") {
        return {"Anger", "Contempt", "Happiness"};
    }
    if (modelName == "mobilevit_va_mtl") {
        return {"Anger", "Contempt", "Fear"};
    }
    return {"Anger", "Happiness", "Fear"};
}

std::vector<cv::Mat> getOneImageFaces() {
    std::string pyTestDir = getPathToPythonTestDir();
    fs::path imagePath(pyTestDir);
    imagePath = imagePath / "test_images" / "20180720_174416.jpg";
    cv::Mat frame = cv::imread(imagePath);
    cv::Mat frameRgb;
    cv::cvtColor(frame, frameRgb, cv::COLOR_BGR2RGB);
    return recognizeFaces(frameRgb);
}
} // namespace

using EmotiEffLibTestParams = std::tuple<std::string, std::string>;

class EmotiEffLibTests : public ::testing::TestWithParam<EmotiEffLibTestParams> {
protected:
    void SetUp() override {
        std::tie(backend_, modelName_) = GetParam();
        auto supportedBackends = EmotiEffLib::getAvailableBackends();
        if (std::find(supportedBackends.begin(), supportedBackends.end(), backend_) ==
            supportedBackends.end()) {
            GTEST_SKIP() << "Skipping test because of unsupported backend.";
        }
        ext_ = (backend_ == "torch") ? ".pt" : ".onnx";
        modelPath_ = getEmotiEffLibRootDir();
        modelPath_ = modelPath_ / "models" / "emotieffcpplib_prepared_models";
    }

protected:
    std::string backend_;
    std::string modelName_;
    std::string ext_;
    fs::path modelPath_;
};

class EmotiEffLibOnlyModelTests : public ::testing::TestWithParam<std::string> {};

TEST_P(EmotiEffLibTests, OneImagePredictionOneModel) {
    auto facialImages = getOneImageFaces();

    modelPath_ /= modelName_ + ext_;
    std::vector<std::string> emotions;
    std::vector<std::string> scorePrediction;
    auto fer = EmotiEffLib::EmotiEffLibRecognizer::createInstance(backend_, modelPath_);
    for (auto& face : facialImages) {
        auto res = fer->predictEmotions(face, true);
        emotions.push_back(res.labels[0]);
        auto pred = xt::argmax(res.scores, 1);
        scorePrediction.push_back(fer->getEmotionClassById(pred[0]));
    }

    ASSERT_TRUE(AreVectorsEqual(emotions, scorePrediction));
    ASSERT_TRUE(AreVectorsEqual(emotions, getOneImageExpEmotions(backend_, modelName_)));

    // Try to call unsuitable functions
    try {
        fer->extractFeatures(facialImages[0]);
        FAIL();
    } catch (const std::runtime_error& e) {
        EXPECT_EQ("Model for features extraction wasn't specified in the config!",
                  std::string(e.what()));
    } catch (...) {
        FAIL();
    }
    try {
        xt::xarray<float> tmp;
        fer->classifyEmotions(tmp);
        FAIL();
    } catch (const std::runtime_error& e) {
        EXPECT_EQ("Model for emotions classification wasn't specified in the config!",
                  std::string(e.what()));
    } catch (...) {
        FAIL();
    }
}

TEST_P(EmotiEffLibTests, OneImageMultiPredictionOneModel) {
    auto facialImages = getOneImageFaces();

    modelPath_ /= modelName_ + ext_;
    auto fer = EmotiEffLib::EmotiEffLibRecognizer::createInstance(backend_, modelPath_);
    auto result = fer->predictEmotions(facialImages, true);
    auto preds = xt::argmax(result.scores, 1);

    std::vector<std::string> scorePrediction;
    for (auto& pred : preds) {
        scorePrediction.push_back(fer->getEmotionClassById(pred));
    }

    ASSERT_TRUE(AreVectorsEqual(result.labels, scorePrediction));
    ASSERT_TRUE(AreVectorsEqual(result.labels, getOneImageExpEmotions(backend_, modelName_)));

    // Try to call unsuitable functions
    try {
        fer->extractFeatures(facialImages[0]);
        FAIL();
    } catch (const std::runtime_error& e) {
        EXPECT_EQ("Model for features extraction wasn't specified in the config!",
                  std::string(e.what()));
    } catch (...) {
        FAIL();
    }
    try {
        xt::xarray<float> tmp;
        fer->classifyEmotions(tmp);
        FAIL();
    } catch (const std::runtime_error& e) {
        EXPECT_EQ("Model for emotions classification wasn't specified in the config!",
                  std::string(e.what()));
    } catch (...) {
        FAIL();
    }
}

TEST_P(EmotiEffLibTests, OneImagePredictionTwoModels) {
    auto facialImages = getOneImageFaces();

    std::string featureExtractorPath = modelPath_ / ("features_extractor_" + modelName_ + ext_);
    std::string classifierPath = modelPath_ / ("classifier_" + modelName_ + ext_);
    EmotiEffLib::EmotiEffLibConfig config;
    config.backend = backend_;
    config.featureExtractorPath = featureExtractorPath;
    config.classifierPath = classifierPath;
    config.modelName = modelName_;
    std::vector<std::string> emotions;
    std::vector<std::string> scorePrediction;
    auto fer = EmotiEffLib::EmotiEffLibRecognizer::createInstance(config);
    for (auto& face : facialImages) {
        auto res = fer->predictEmotions(face, true);
        emotions.push_back(res.labels[0]);
        auto pred = xt::argmax(res.scores, 1);
        scorePrediction.push_back(fer->getEmotionClassById(pred[0]));
    }

    ASSERT_TRUE(AreVectorsEqual(emotions, scorePrediction));
    ASSERT_TRUE(AreVectorsEqual(emotions, getOneImageExpEmotions(backend_, modelName_)));
}

TEST_P(EmotiEffLibTests, OneImageMultiPredictionTwoModels) {
    auto facialImages = getOneImageFaces();

    std::string featureExtractorPath = modelPath_ / ("features_extractor_" + modelName_ + ext_);
    std::string classifierPath = modelPath_ / ("classifier_" + modelName_ + ext_);
    EmotiEffLib::EmotiEffLibConfig config;
    config.backend = backend_;
    config.featureExtractorPath = featureExtractorPath;
    config.classifierPath = classifierPath;
    config.modelName = modelName_;
    auto fer = EmotiEffLib::EmotiEffLibRecognizer::createInstance(config);
    auto result = fer->predictEmotions(facialImages, true);
    auto preds = xt::argmax(result.scores, 1);

    std::vector<std::string> scorePrediction;
    for (auto& pred : preds) {
        scorePrediction.push_back(fer->getEmotionClassById(pred));
    }

    ASSERT_TRUE(AreVectorsEqual(result.labels, scorePrediction));
    ASSERT_TRUE(AreVectorsEqual(result.labels, getOneImageExpEmotions(backend_, modelName_)));
}

TEST_P(EmotiEffLibTests, OneImageClassification) {
    auto facialImages = getOneImageFaces();

    std::string featureExtractorPath = modelPath_ / ("features_extractor_" + modelName_ + ext_);
    std::string classifierPath = modelPath_ / ("classifier_" + modelName_ + ext_);
    EmotiEffLib::EmotiEffLibConfig config;
    config.backend = backend_;
    config.featureExtractorPath = featureExtractorPath;
    config.classifierPath = classifierPath;
    config.modelName = modelName_;
    auto fer = EmotiEffLib::EmotiEffLibRecognizer::createInstance(config);
    std::vector<std::string> emotions;
    std::vector<std::string> scorePrediction;
    for (auto& face : facialImages) {
        auto features = fer->extractFeatures(face);
        auto res = fer->classifyEmotions(features);
        emotions.push_back(res.labels[0]);
        auto pred = xt::argmax(res.scores, 1);
        scorePrediction.push_back(fer->getEmotionClassById(pred[0]));
    }

    ASSERT_TRUE(AreVectorsEqual(emotions, scorePrediction));
    ASSERT_TRUE(AreVectorsEqual(emotions, getOneImageExpEmotions(backend_, modelName_)));
}

TEST_P(EmotiEffLibTests, OneImageMultiClassification) {
    auto facialImages = getOneImageFaces();

    std::string featureExtractorPath = modelPath_ / ("features_extractor_" + modelName_ + ext_);
    std::string classifierPath = modelPath_ / ("classifier_" + modelName_ + ext_);
    EmotiEffLib::EmotiEffLibConfig config;
    config.backend = backend_;
    config.featureExtractorPath = featureExtractorPath;
    config.classifierPath = classifierPath;
    config.modelName = modelName_;
    auto fer = EmotiEffLib::EmotiEffLibRecognizer::createInstance(config);
    auto features = fer->extractFeatures(facialImages);
    auto result = fer->classifyEmotions(features);
    auto preds = xt::argmax(result.scores, 1);

    std::vector<std::string> scorePrediction;
    for (auto& pred : preds) {
        scorePrediction.push_back(fer->getEmotionClassById(pred));
    }

    ASSERT_TRUE(AreVectorsEqual(result.labels, scorePrediction));
    ASSERT_TRUE(AreVectorsEqual(result.labels, getOneImageExpEmotions(backend_, modelName_)));
}

TEST_P(EmotiEffLibTests, AffectNetPredictionOneModel) {
    const int filesLimit = 100;
    std::string pyTestDir = getPathToPythonTestDir();
    fs::path inputsDir(pyTestDir);
    inputsDir = inputsDir / "data" / "AffectNet_val";
    std::vector<std::string> inputFiles, inputLabels;
    for (const auto& labelEntry : fs::directory_iterator(inputsDir)) {
        const auto& labelPath = labelEntry.path();
        auto label = labelPath.filename();

        // Skip hidden files and non-directories
        if (label.string().compare(0, 1, ".") == 0 || !fs::is_directory(labelPath)) {
            continue;
        }

        size_t count = 0;
        for (const auto& img_entry : fs::directory_iterator(labelPath)) {
            if (count >= filesLimit) {
                break;
            }

            const auto& img_path = img_entry.path();
            if (img_path.filename().string().compare(0, 1, ".") == 0) {
                continue;
            }

            inputFiles.push_back(img_path.string());
            inputLabels.push_back(label.string());
            ++count;
        }
    }

    modelPath_ /= modelName_ + ext_;
    auto fer = EmotiEffLib::EmotiEffLibRecognizer::createInstance(backend_, modelPath_);

    std::vector<std::string> emotions;
    for (auto& img : inputFiles) {
        cv::Mat frame = cv::imread(img);
        cv::Mat frameRgb;
        cv::cvtColor(frame, frameRgb, cv::COLOR_BGR2RGB);

        auto res = fer->predictEmotions(frameRgb, true);
        emotions.push_back(res.labels[0]);
    }

    ASSERT_FALSE(emotions.empty());
    ASSERT_EQ(emotions.size(), inputLabels.size());
    size_t correct = std::count_if(emotions.begin(), emotions.end(),
                                   [&inputLabels, i = 0](const std::string& pred) mutable {
                                       return pred == inputLabels[i++];
                                   });
    float acc = static_cast<float>(correct) / emotions.size();

    ASSERT_TRUE(acc > 0.55);
}

TEST_P(EmotiEffLibTests, OnVideoOneModel) {
    std::string pyTestDir = getPathToPythonTestDir();
    fs::path videoPath(pyTestDir);
    videoPath = videoPath / "data" / "video_samples" / "emotions" / "Angry" / "Angry.mp4";

    cv::VideoCapture cap(videoPath);
    ASSERT_TRUE(cap.isOpened()) << "Error: Could not open video file!";

    std::vector<cv::Mat> facialImgs;
    cv::Mat frame;

    while (cap.read(frame)) { // Read frames until end
        cv::Mat frameRgb;
        cv::cvtColor(frame, frameRgb, cv::COLOR_BGR2RGB);
        auto faces = recognizeFaces(frameRgb);
        facialImgs.insert(facialImgs.end(), faces.begin(), faces.end());
    }

    cap.release();

    modelPath_ /= modelName_ + ext_;
    auto fer = EmotiEffLib::EmotiEffLibRecognizer::createInstance(backend_, modelPath_);
    auto result = fer->predictEmotions(facialImgs, true);
    auto score = xt::mean(result.scores, {0});
    auto emotion_idx = xt::argmax(score)[0];

    EXPECT_EQ(fer->getEmotionClassById(emotion_idx), "Anger");
}

TEST_P(EmotiEffLibTests, OnVideoEngagement) {
    if (modelName_.find("enet_b0") == std::string::npos) {
        GTEST_SKIP() << "Skipping test because of unsupported model.";
    }
    std::string pyTestDir = getPathToPythonTestDir();
    fs::path videoPath(pyTestDir);
    videoPath = videoPath / "data" / "video_samples" / "engagement" / "engaged" / "1_video1.mp4";

    cv::VideoCapture cap(videoPath);
    ASSERT_TRUE(cap.isOpened()) << "Error: Could not open video file!";

    std::vector<cv::Mat> facialImgs;
    cv::Mat frame;

    while (cap.read(frame)) { // Read frames until end
        cv::Mat frameRgb;
        cv::cvtColor(frame, frameRgb, cv::COLOR_BGR2RGB);
        auto faces = recognizeFaces(frameRgb);
        facialImgs.insert(facialImgs.end(), faces.begin(), faces.end());
    }

    cap.release();

    std::string featureExtractorPath = modelPath_ / ("features_extractor_" + modelName_ + ext_);
    std::string engagementClassifierPath = modelPath_ / ("engagement_classifier_2560_128" + ext_);
    EmotiEffLib::EmotiEffLibConfig config;
    config.backend = backend_;
    config.featureExtractorPath = featureExtractorPath;
    config.engagementClassifierPath = engagementClassifierPath;
    config.modelName = modelName_;
    auto fer = EmotiEffLib::EmotiEffLibRecognizer::createInstance(config);
    auto result = fer->predictEngagement(facialImgs);
    auto score = xt::mean(result.scores, {0});
    auto engagement_idx = xt::argmax(score)[0];

    EXPECT_EQ(fer->getEngagementClassById(engagement_idx), "Engaged");
}

TEST_P(EmotiEffLibTests, OnVideoDistraction) {
    if (modelName_.find("enet_b0") == std::string::npos) {
        GTEST_SKIP() << "Skipping test because of unsupported model.";
    }
    std::string pyTestDir = getPathToPythonTestDir();
    fs::path videoPath(pyTestDir);
    videoPath = videoPath / "data" / "video_samples" / "engagement" / "distracted" / "0_video1.mp4";

    cv::VideoCapture cap(videoPath);
    ASSERT_TRUE(cap.isOpened()) << "Error: Could not open video file!";

    std::vector<cv::Mat> facialImgs;
    cv::Mat frame;

    while (cap.read(frame)) { // Read frames until end
        cv::Mat frameRgb;
        cv::cvtColor(frame, frameRgb, cv::COLOR_BGR2RGB);
        auto faces = recognizeFaces(frameRgb);
        facialImgs.insert(facialImgs.end(), faces.begin(), faces.end());
    }

    cap.release();

    std::string featureExtractorPath = modelPath_ / ("features_extractor_" + modelName_ + ext_);
    std::string engagementClassifierPath = modelPath_ / ("engagement_classifier_2560_128" + ext_);
    EmotiEffLib::EmotiEffLibConfig config;
    config.backend = backend_;
    config.featureExtractorPath = featureExtractorPath;
    config.engagementClassifierPath = engagementClassifierPath;
    config.modelName = modelName_;
    auto fer = EmotiEffLib::EmotiEffLibRecognizer::createInstance(config);
    auto features = fer->extractFeatures(facialImgs);
    auto result = fer->classifyEngagement(features);
    auto score = xt::mean(result.scores, {0});
    auto engagement_idx = xt::argmax(score)[0];

    EXPECT_EQ(fer->getEngagementClassById(engagement_idx), "Distracted");
}

TEST_P(EmotiEffLibTests, OnVideoEmotionAndEngagement) {
    if (modelName_.find("enet_b0") == std::string::npos) {
        GTEST_SKIP() << "Skipping test because of unsupported model.";
    }
    std::string pyTestDir = getPathToPythonTestDir();
    fs::path videoPath(pyTestDir);
    videoPath = videoPath / "data" / "video_samples" / "engagement" / "engaged" / "1_video1.mp4";

    cv::VideoCapture cap(videoPath);
    ASSERT_TRUE(cap.isOpened()) << "Error: Could not open video file!";

    std::vector<cv::Mat> facialImgs;
    cv::Mat frame;

    while (cap.read(frame)) { // Read frames until end
        cv::Mat frameRgb;
        cv::cvtColor(frame, frameRgb, cv::COLOR_BGR2RGB);
        auto faces = recognizeFaces(frameRgb);
        facialImgs.insert(facialImgs.end(), faces.begin(), faces.end());
    }

    cap.release();

    std::string featureExtractorPath = modelPath_ / ("features_extractor_" + modelName_ + ext_);
    std::string engagementClassifierPath = modelPath_ / ("engagement_classifier_2560_128" + ext_);
    std::string classifierPath = modelPath_ / ("classifier_" + modelName_ + ext_);
    EmotiEffLib::EmotiEffLibConfig config;
    config.backend = backend_;
    config.featureExtractorPath = featureExtractorPath;
    config.classifierPath = classifierPath;
    config.engagementClassifierPath = engagementClassifierPath;
    config.modelName = modelName_;
    auto fer = EmotiEffLib::EmotiEffLibRecognizer::createInstance(config);
    auto features = fer->extractFeatures(facialImgs);
    auto emo_result = fer->classifyEmotions(features, true);
    auto eng_result = fer->classifyEngagement(features);
    auto emo_score = xt::mean(emo_result.scores, {0});
    auto eng_score = xt::mean(eng_result.scores, {0});
    auto emotion_idx = xt::argmax(emo_score)[0];
    auto engagement_idx = xt::argmax(eng_score)[0];

    if (modelName_ == "enet_b0_8_best_vgaf") {
        EXPECT_EQ(fer->getEmotionClassById(emotion_idx), "Anger");
    } else {
        EXPECT_EQ(fer->getEmotionClassById(emotion_idx), "Sadness");
    }
    EXPECT_EQ(fer->getEngagementClassById(engagement_idx), "Engaged");
}

std::string TestNameGenerator(const ::testing::TestParamInfo<EmotiEffLibTests::ParamType>& info) {
    auto& [backend, modelName] = info.param;
    std::ostringstream name;
    name << "backend_" << backend << "_model_" << modelName;

    // Replace invalid characters for test names
    std::string name_str = name.str();
    std::replace(name_str.begin(), name_str.end(), '.', '_'); // Replace dots
    return name_str;
}

INSTANTIATE_TEST_SUITE_P(
    EmotionsOnnx, EmotiEffLibTests,
    ::testing::Combine(::testing::Values("onnx"),
                       ::testing::ValuesIn(EmotiEffLib::getSupportedModels("onnx"))),
    TestNameGenerator);

INSTANTIATE_TEST_SUITE_P(
    EmotionsTorch, EmotiEffLibTests,
    ::testing::Combine(::testing::Values("torch"),
                       ::testing::ValuesIn(EmotiEffLib::getSupportedModels("torch"))),
    TestNameGenerator);

TEST_P(EmotiEffLibOnlyModelTests, OneImageFeatures) {
    std::string modelName = GetParam();
    auto facialImages = getOneImageFaces();
    auto supportedBackends = EmotiEffLib::getAvailableBackends();
    auto findOnnx = std::find(supportedBackends.begin(), supportedBackends.end(), "onnx");
    auto findTorch = std::find(supportedBackends.begin(), supportedBackends.end(), "torch");
    if (findOnnx == supportedBackends.end() or findTorch == supportedBackends.end()) {
        GTEST_SKIP() << "Skipping test because of unsupported backend.";
    }

    fs::path modelPath(getEmotiEffLibRootDir());
    modelPath = modelPath / "models" / "emotieffcpplib_prepared_models";
    std::string featureExtractorOnnxPath =
        modelPath / ("features_extractor_" + modelName + ".onnx");
    std::string featureExtractorTorchPath = modelPath / ("features_extractor_" + modelName + ".pt");
    EmotiEffLib::EmotiEffLibConfig configOnnx;
    configOnnx.backend = "onnx";
    configOnnx.featureExtractorPath = featureExtractorOnnxPath;
    EmotiEffLib::EmotiEffLibConfig configTorch;
    configTorch.backend = "torch";
    configTorch.featureExtractorPath = featureExtractorTorchPath;
    auto ferOnnx = EmotiEffLib::EmotiEffLibRecognizer::createInstance(configOnnx);
    auto ferTorch = EmotiEffLib::EmotiEffLibRecognizer::createInstance(configTorch);
    for (auto& face : facialImages) {
        auto featuresOnnx = ferOnnx->extractFeatures(face);
        auto featuresTorch = ferTorch->extractFeatures(face);
        EXPECT_EQ(featuresOnnx.shape(), featuresTorch.shape());
        ASSERT_TRUE(xt::allclose(featuresOnnx, featuresTorch, 1e-2, 1e-2));
    }
}

TEST_P(EmotiEffLibOnlyModelTests, OneImageMultiFeatures) {
    std::string modelName = GetParam();
    auto facialImages = getOneImageFaces();
    auto supportedBackends = EmotiEffLib::getAvailableBackends();
    auto findOnnx = std::find(supportedBackends.begin(), supportedBackends.end(), "onnx");
    auto findTorch = std::find(supportedBackends.begin(), supportedBackends.end(), "torch");
    if (findOnnx == supportedBackends.end() or findTorch == supportedBackends.end()) {
        GTEST_SKIP() << "Skipping test because of unsupported backend.";
    }

    fs::path modelPath(getEmotiEffLibRootDir());
    modelPath = modelPath / "models" / "emotieffcpplib_prepared_models";
    std::string featureExtractorOnnxPath =
        modelPath / ("features_extractor_" + modelName + ".onnx");
    std::string featureExtractorTorchPath = modelPath / ("features_extractor_" + modelName + ".pt");
    EmotiEffLib::EmotiEffLibConfig configOnnx;
    configOnnx.backend = "onnx";
    configOnnx.featureExtractorPath = featureExtractorOnnxPath;
    EmotiEffLib::EmotiEffLibConfig configTorch;
    configTorch.backend = "torch";
    configTorch.featureExtractorPath = featureExtractorTorchPath;
    auto ferOnnx = EmotiEffLib::EmotiEffLibRecognizer::createInstance(configOnnx);
    auto ferTorch = EmotiEffLib::EmotiEffLibRecognizer::createInstance(configTorch);
    auto featuresOnnx = ferOnnx->extractFeatures(facialImages);
    auto featuresTorch = ferTorch->extractFeatures(facialImages);
    EXPECT_EQ(featuresOnnx.shape()[0], 3);
    EXPECT_EQ(featuresOnnx.shape(), featuresTorch.shape());
    ASSERT_TRUE(xt::allclose(featuresOnnx, featuresTorch, 1e-2, 1e-2));
}

std::string OnlyModelTestNameGenerator(
    const ::testing::TestParamInfo<EmotiEffLibOnlyModelTests::ParamType>& info) {
    auto modelName = info.param;
    std::ostringstream name;
    name << "model_" << modelName;

    // Replace invalid characters for test names
    std::string name_str = name.str();
    std::replace(name_str.begin(), name_str.end(), '.', '_'); // Replace dots
    return name_str;
}

INSTANTIATE_TEST_SUITE_P(FeaturesExtraction, EmotiEffLibOnlyModelTests,
                         ::testing::ValuesIn(EmotiEffLib::getSupportedModels("generic")),
                         OnlyModelTestNameGenerator);

TEST(EmotiEffLibTests, CheckUnsupportedBackend) {
    try {
        EmotiEffLib::EmotiEffLibRecognizer::createInstance("OpenVINO", "my_model");
        FAIL();
    } catch (const std::runtime_error& e) {
        EXPECT_EQ("This backend (OpenVINO) is not supported. Please check your EmotiEffLib build "
                  "or configuration.",
                  std::string(e.what()));
    } catch (...) {
        FAIL();
    }
}

TEST(EmotiEffLibTests, CheckIncorrectConfig) {
    EmotiEffLib::EmotiEffLibConfig config;
    try {
        EmotiEffLib::EmotiEffLibRecognizer::createInstance(config);
        FAIL();
    } catch (const std::runtime_error& e) {
        EXPECT_EQ("This backend () is not supported. Please check your EmotiEffLib build or "
                  "configuration.",
                  std::string(e.what()));
    } catch (...) {
        FAIL();
    }
    auto supportedBackends = EmotiEffLib::getAvailableBackends();
    config = {.backend = "torch", .classifierPath = "bla-bla", .modelName = "bla-bla"};
    if (std::find(supportedBackends.begin(), supportedBackends.end(), config.backend) !=
        supportedBackends.end()) {
        try {
            EmotiEffLib::EmotiEffLibRecognizer::createInstance(config);
            FAIL();
        } catch (const std::runtime_error& e) {
            EXPECT_EQ(
                "fullPipelineEmotionModelPath or featureExtractorPath MUST be specified in the "
                "EmotiEffLibConfig.",
                std::string(e.what()));
        } catch (...) {
            FAIL();
        }
    }
    config.backend = "onnx";
    if (std::find(supportedBackends.begin(), supportedBackends.end(), config.backend) !=
        supportedBackends.end()) {
        try {
            EmotiEffLib::EmotiEffLibRecognizer::createInstance(config);
            FAIL();
        } catch (const std::runtime_error& e) {
            EXPECT_EQ(
                "fullPipelineEmotionModelPath or featureExtractorPath MUST be specified in the "
                "EmotiEffLibConfig.",
                std::string(e.what()));
        } catch (...) {
            FAIL();
        }
    }
}
