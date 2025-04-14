#include "OCRProcessor.h"
#include <fstream>
#include <algorithm>
#include <iostream>

// Constructor with paths to model files
OCRProcessor::OCRProcessor(
    const std::string& linesConfigPath,
    const std::string& linesWeightsPath,
    const std::string& linesNamesPath,
    const std::string& charsConfigPath,
    const std::string& charsWeightsPath,
    const std::string& charsNamesPath,
    const std::string& smallCharsConfigPath,
    const std::string& smallCharsWeightsPath,
    const std::string& smallCharsNamesPath) :
    lineThreshold(0.5f),
    charThreshold(0.5f),
    smallCharThreshold(0.5f),
    useDarkHelp(true) {

    // If all paths are provided, initialize the networks
    if (!linesConfigPath.empty() && !linesWeightsPath.empty() && !linesNamesPath.empty() &&
        !charsConfigPath.empty() && !charsWeightsPath.empty() && !charsNamesPath.empty() &&
        !smallCharsConfigPath.empty() && !smallCharsWeightsPath.empty() && !smallCharsNamesPath.empty()) {

        initialize(linesConfigPath, linesWeightsPath, linesNamesPath,
                  charsConfigPath, charsWeightsPath, charsNamesPath,
                  smallCharsConfigPath, smallCharsWeightsPath, smallCharsNamesPath);
    }
}

OCRProcessor::~OCRProcessor() {
    // The unique_ptr will automatically clean up resources, Destructor
}

bool OCRProcessor::initialize(
    const std::string& linesConfigPath,
    const std::string& linesWeightsPath,
    const std::string& linesNamesPath,
    const std::string& charsConfigPath,
    const std::string& charsWeightsPath,
    const std::string& charsNamesPath,
    const std::string& smallCharsConfigPath,
    const std::string& smallCharsWeightsPath,
    const std::string& smallCharsNamesPath) {

    try {
        // Initialize using DarkHelp
        if (useDarkHelp) {
            // Load the line detection network
            lineNetwork = std::make_unique<DarkHelp::NN>(linesConfigPath, linesWeightsPath, linesNamesPath);
            lineNetwork->config.threshold = lineThreshold;

            // Load the character detection network
            charNetwork = std::make_unique<DarkHelp::NN>(charsConfigPath, charsWeightsPath, charsNamesPath);
            charNetwork->config.threshold = charThreshold;

            // Load the small character detection network
            smallCharNetwork = std::make_unique<DarkHelp::NN>(smallCharsConfigPath, smallCharsWeightsPath, smallCharsNamesPath);
            smallCharNetwork->config.threshold = smallCharThreshold;
        }
        // Initialize using Detector class (alternative)
        else {
            // Load the line detection network
            lineDetector = std::make_unique<Detector>(linesConfigPath, linesWeightsPath);

            // Load the character detection network
            charDetector = std::make_unique<Detector>(charsConfigPath, charsWeightsPath);

            // Load the small character detection network
            smallCharDetector = std::make_unique<Detector>(smallCharsConfigPath, smallCharsWeightsPath);
        }

        std::cout << "Networks initialized successfully" << std::endl;

        if (useDarkHelp) {
            std::cout << "Line classes: " << lineNetwork->names.size() << std::endl;
            std::cout << "Character classes: " << charNetwork->names.size() << std::endl;
            std::cout << "Small character classes: " << smallCharNetwork->names.size() << std::endl;
        }

        return true;
    }
    catch (const std::exception& e) {
        std::cerr << "Error initializing networks: " << e.what() << std::endl;
        return false;
    }
}

std::string OCRProcessor::processImage(const cv::Mat& image) {
    if (image.empty()) {
        std::cerr << "Empty image provided" << std::endl;
        return "";
    }

    std::string result;

    try {
        // Step 1: Detect lines in the image
        std::vector<Detection> lineDetections = detectLines(image);
        std::cout << "Detected " << lineDetections.size() << " lines" << std::endl;

        // Sort lines from top to bottom
        std::sort(lineDetections.begin(), lineDetections.end(),
                [](const Detection& a, const Detection& b) {
                    return a.bbox.y < b.bbox.y;
                });

        // Process each line
        for (const auto& lineDetection : lineDetections) {
            // Extract the line region
            cv::Rect lineRect = lineDetection.bbox;

            // Ensure the rectangle is within the image bounds
            lineRect = lineRect & cv::Rect(0, 0, image.cols, image.rows);

            if (lineRect.width <= 0 || lineRect.height <= 0) {
                continue;
            }

            cv::Mat lineImage = image(lineRect).clone();

            // Step 2: Detect characters in the line
            std::vector<Detection> charDetections = detectCharacters(lineImage);
            std::cout << "Detected " << charDetections.size() << " characters in line" << std::endl;

            // Sort characters from left to right
            sortDetectionsLeftToRight(charDetections);

            // Process each character
            for (const auto& charDetection : charDetections) {
                // Extract the character region
                cv::Rect charRect = charDetection.bbox;

                // Ensure the rectangle is within the line image bounds
                charRect = charRect & cv::Rect(0, 0, lineImage.cols, lineImage.rows);

                if (charRect.width <= 0 || charRect.height <= 0) {
                    continue;
                }

                cv::Mat charImage = lineImage(charRect).clone();

                // Step 3: Recognize the small character
                std::vector<Detection> smallCharDetections = detectSmallCharacters(charImage);

                if (!smallCharDetections.empty()) {
                    // Sort by confidence and take the highest confidence detection
                    std::sort(smallCharDetections.begin(), smallCharDetections.end(),
                            [](const Detection& a, const Detection& b) {
                                return a.confidence > b.confidence;
                            });

                    // Add the recognized character to the result
                    result += smallCharDetections[0].label;
                }
            }

            // Add a newline after each line
            result += "\n";
        }
    }
    catch (const std::exception& e) {
        std::cerr << "Error processing image: " << e.what() << std::endl;
    }

    return result;
}

void OCRProcessor::setThresholds(float lineThreshold, float charThreshold, float smallCharThreshold) {
    this->lineThreshold = lineThreshold;
    this->charThreshold = charThreshold;
    this->smallCharThreshold = smallCharThreshold;
}

std::string OCRProcessor::processImageFile(const std::string& imagePath) {
    // Load the image from file
    cv::Mat image = cv::imread(imagePath);
    if (image.empty()) {
        std::cerr << "Failed to load image: " << imagePath << std::endl;
        return "";
    }

    // Process the image using the existing method
    return processImage(image);
}

DarkHelp::PredictionResults OCRProcessor::getPredictions(const cv::Mat& image) {
    if (image.empty()) {
        std::cerr << "Empty image provided" << std::endl;
        return DarkHelp::PredictionResults();
    }

    try {
        // Use the appropriate neural network based on the image content
        // For this example, we'll use the line detection network
        if (useDarkHelp && lineNetwork) {
            lineNetwork->config.threshold = lineThreshold;
            return lineNetwork->predict(image);
        }
        else {
            std::cerr << "Neural network not initialized or DarkHelp not enabled" << std::endl;
            return DarkHelp::PredictionResults();
        }
    }
    catch (const std::exception& e) {
        std::cerr << "Error getting predictions: " << e.what() << std::endl;
        return DarkHelp::PredictionResults();
    }
}

std::vector<Detection> OCRProcessor::detectLines(const cv::Mat& image) {
    if (useDarkHelp) {
        // Use DarkHelp for detection
        lineNetwork->config.threshold = lineThreshold;
        DarkHelp::PredictionResults predictions = lineNetwork->predict(image);
        return convertPredictions(predictions);
    } else {
        // Use Detector class for detection
        auto detections = lineDetector->detect(image, lineThreshold);
        return convertBBoxes(detections, lineNetwork->names);
    }
}

std::vector<Detection> OCRProcessor::detectCharacters(const cv::Mat& image) {
    if (useDarkHelp) {
        // Use DarkHelp for detection
        charNetwork->config.threshold = charThreshold;
        DarkHelp::PredictionResults predictions = charNetwork->predict(image);
        return convertPredictions(predictions);
    } else {
        // Use Detector class for detection
        auto detections = charDetector->detect(image, charThreshold);
        return convertBBoxes(detections, charNetwork->names);
    }
}

std::vector<Detection> OCRProcessor::detectSmallCharacters(const cv::Mat& image) {
    if (useDarkHelp) {
        // Use DarkHelp for detection
        smallCharNetwork->config.threshold = smallCharThreshold;
        DarkHelp::PredictionResults predictions = smallCharNetwork->predict(image);
        return convertPredictions(predictions);
    } else {
        // Use Detector class for detection
        auto detections = smallCharDetector->detect(image, smallCharThreshold);
        return convertBBoxes(detections, smallCharNetwork->names);
    }
}

std::vector<Detection> OCRProcessor::convertPredictions(const DarkHelp::PredictionResults& predictions) {
    std::vector<Detection> detections;

    for (const auto& pred : predictions) {
        Detection det;
        det.label = pred.name;
        det.confidence = pred.best_probability;
        det.bbox = pred.rect;
        detections.push_back(det);
    }

    return detections;
}

std::vector<Detection> OCRProcessor::convertBBoxes(const std::vector<bbox_t>& bboxes, const std::vector<std::string>& names) {
    std::vector<Detection> detections;

    for (const auto& bbox : bboxes) {
        Detection det;
        det.label = names[bbox.obj_id];
        det.confidence = bbox.prob;
        det.bbox = cv::Rect(bbox.x, bbox.y, bbox.w, bbox.h);
        detections.push_back(det);
    }

    return detections;
}

std::vector<std::string> OCRProcessor::loadNames(const std::string& namesPath) {
    std::vector<std::string> names;
    std::ifstream file(namesPath);
    std::string line;

    if (!file.is_open()) {
        std::cerr << "Failed to open names file: " << namesPath << std::endl;
        return names;
    }

    while (std::getline(file, line)) {
        // Trim whitespace
        line.erase(0, line.find_first_not_of(" \t\n\r\f\v"));
        line.erase(line.find_last_not_of(" \t\n\r\f\v") + 1);

        if (!line.empty()) {
            names.push_back(line);
        }
    }

    return names;
}

void OCRProcessor::sortDetectionsLeftToRight(std::vector<Detection>& detections) {
    std::sort(detections.begin(), detections.end(),
              [](const Detection& a, const Detection& b) {
                  return a.bbox.x < b.bbox.x;
              });
}
