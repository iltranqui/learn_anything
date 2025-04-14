#pragma once

#include <string>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>
#include <DarkHelp.hpp>
#include <yolo_v2_class.hpp>

struct Detection {
    std::string label;
    float confidence;
    cv::Rect bbox;
};

class OCRProcessor {
public:
    // Constructor with paths to model files
    OCRProcessor(
        const std::string& linesConfigPath = "",
        const std::string& linesWeightsPath = "",
        const std::string& linesNamesPath = "",
        const std::string& charsConfigPath = "",
        const std::string& charsWeightsPath = "",
        const std::string& charsNamesPath = "",
        const std::string& smallCharsConfigPath = "",
        const std::string& smallCharsWeightsPath = "",
        const std::string& smallCharsNamesPath = ""
    );

    ~OCRProcessor();

    // Initialize the networks
    bool initialize(
        const std::string& linesConfigPath,
        const std::string& linesWeightsPath,
        const std::string& linesNamesPath,
        const std::string& charsConfigPath,
        const std::string& charsWeightsPath,
        const std::string& charsNamesPath,
        const std::string& smallCharsConfigPath,
        const std::string& smallCharsWeightsPath,
        const std::string& smallCharsNamesPath
    );

    // Process an image from file path and return the recognized text
    std::string processImageFile(const std::string& imagePath);

    // Process an image from cv::Mat and return the recognized text
    std::string processImage(const cv::Mat& image);

    // Process an image and return the raw prediction results
    DarkHelp::PredictionResults getPredictions(const cv::Mat& image);

    // Set confidence thresholds for each network
    void setThresholds(float lineThreshold, float charThreshold, float smallCharThreshold);

private:
    // Neural network models using DarkHelp
    std::unique_ptr<DarkHelp::NN> lineNetwork;
    std::unique_ptr<DarkHelp::NN> charNetwork;
    std::unique_ptr<DarkHelp::NN> smallCharNetwork;

    // Alternative neural network models using Detector class
    std::unique_ptr<Detector> lineDetector;
    std::unique_ptr<Detector> charDetector;
    std::unique_ptr<Detector> smallCharDetector;

    // Confidence thresholds
    float lineThreshold;
    float charThreshold;
    float smallCharThreshold;

    // Helper functions
    std::vector<Detection> detectLines(const cv::Mat& image);
    std::vector<Detection> detectCharacters(const cv::Mat& image);
    std::vector<Detection> detectSmallCharacters(const cv::Mat& image);

    // Convert Darknet/DarkHelp results to our Detection format
    std::vector<Detection> convertPredictions(const DarkHelp::PredictionResults& predictions);
    std::vector<Detection> convertBBoxes(const std::vector<bbox_t>& bboxes, const std::vector<std::string>& names);

    // Load class names from file
    std::vector<std::string> loadNames(const std::string& namesPath);

    // Sort characters from left to right
    void sortDetectionsLeftToRight(std::vector<Detection>& detections);

    // Flag to determine which API to use
    bool useDarkHelp;

    // Class names for each network (only needed for Detector API)
    std::vector<std::string> lineNames;
    std::vector<std::string> charNames;
    std::vector<std::string> smallCharNames;
};
