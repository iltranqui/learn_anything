/**
 * @file OCRProcessor.cpp
 * @brief Implementation of the OCRProcessor class for optical character recognition
 *
 * This file contains the implementation of the OCRProcessor class, which uses
 * three neural networks to detect and recognize text in images:
 * 1. Line detection network - Detects lines of text in an image
 * 2. Character detection network - Detects individual characters in a line
 * 3. Small character recognition network - Recognizes specific characters
 *
 * The class supports both DarkHelp and Darknet APIs for neural network inference.
 */

#include "OCRProcessor.h"
#include <fstream>    // For file operations (reading/writing files)
#include <algorithm>  // For std::sort and other algorithms
#include <iostream>   // For standard input/output operations

/**
 * @brief Constructor that initializes the OCR processor with model paths
 *
 * This constructor initializes three neural networks for OCR processing:
 * - Line detection network: Detects lines of text in an image
 * - Character detection network: Detects individual characters in a line
 * - Small character recognition network: Recognizes specific characters
 *
 * @param linesConfigPath Path to the configuration file for line detection network
 * @param linesWeightsPath Path to the weights file for line detection network
 * @param linesNamesPath Path to the class names file for line detection network
 * @param charsConfigPath Path to the configuration file for character detection network
 * @param charsWeightsPath Path to the weights file for character detection network
 * @param charsNamesPath Path to the class names file for character detection network
 * @param smallCharsConfigPath Path to the configuration file for small character recognition network
 * @param smallCharsWeightsPath Path to the weights file for small character recognition network
 * @param smallCharsNamesPath Path to the class names file for small character recognition network
 */
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
    // Initialize default detection thresholds
    lineThreshold(0.5f),        // Default threshold for line detection
    charThreshold(0.5f),        // Default threshold for character detection
    smallCharThreshold(0.5f),   // Default threshold for small character recognition
    useDarkHelp(true) {         // Use DarkHelp API by default

    // If all paths are provided, initialize the networks
    if (!linesConfigPath.empty() && !linesWeightsPath.empty() && !linesNamesPath.empty() &&
        !charsConfigPath.empty() && !charsWeightsPath.empty() && !charsNamesPath.empty() &&
        !smallCharsConfigPath.empty() && !smallCharsWeightsPath.empty() && !smallCharsNamesPath.empty()) {

        initialize(linesConfigPath, linesWeightsPath, linesNamesPath,
                  charsConfigPath, charsWeightsPath, charsNamesPath,
                  smallCharsConfigPath, smallCharsWeightsPath, smallCharsNamesPath);
    }
}

/**
 * @brief Destructor for OCRProcessor
 *
 * The destructor automatically cleans up resources through the use of unique_ptr.
 * All neural network objects and detectors are automatically destroyed.
 */
OCRProcessor::~OCRProcessor() {
    // The unique_ptr will automatically clean up resources when the object is destroyed
    // No explicit cleanup needed for lineNetwork, charNetwork, smallCharNetwork,
    // lineDetector, charDetector, or smallCharDetector
}

/**
 * @brief Initialize the OCR processor with the specified model paths
 *
 * This method initializes the neural networks or detectors based on the provided paths.
 * It supports two different backends:
 * 1. DarkHelp API (preferred) - Used when useDarkHelp is true
 * 2. Detector class - Used as a fallback when DarkHelp initialization fails
 *
 * @param linesConfigPath Path to the configuration file for line detection network
 * @param linesWeightsPath Path to the weights file for line detection network
 * @param linesNamesPath Path to the class names file for line detection network
 * @param charsConfigPath Path to the configuration file for character detection network
 * @param charsWeightsPath Path to the weights file for character detection network
 * @param charsNamesPath Path to the class names file for character detection network
 * @param smallCharsConfigPath Path to the configuration file for small character recognition network
 * @param smallCharsWeightsPath Path to the weights file for small character recognition network
 * @param smallCharsNamesPath Path to the class names file for small character recognition network
 *
 * @return true if initialization was successful, false otherwise
 */
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
        // Branch based on which API to use (DarkHelp or Detector)
        if (useDarkHelp) {
            // === DarkHelp API Initialization ===

            // Initialize the line detection network using DarkHelp
            // This loads the YOLOv4 model for detecting lines of text in images
            lineNetwork = std::make_unique<DarkHelp::NN>(linesConfigPath, linesWeightsPath, linesNamesPath);
            lineNetwork->config.threshold = lineThreshold;  // Set confidence threshold for line detection

            // Initialize the character detection network using DarkHelp
            // This loads the YOLOv4 model for detecting individual characters within a line
            charNetwork = std::make_unique<DarkHelp::NN>(charsConfigPath, charsWeightsPath, charsNamesPath);
            charNetwork->config.threshold = charThreshold;  // Set confidence threshold for character detection

            // Initialize the small character recognition network using DarkHelp
            // This loads the YOLOv4 model for recognizing specific characters
            smallCharNetwork = std::make_unique<DarkHelp::NN>(smallCharsConfigPath, smallCharsWeightsPath, smallCharsNamesPath);
            smallCharNetwork->config.threshold = smallCharThreshold;  // Set confidence threshold for small character recognition
        }
        else {
            // === Detector API Initialization (Alternative) ===

            // Initialize the line detection network using Detector class
            // This is an alternative implementation that uses the Darknet API directly
            lineDetector = std::make_unique<Detector>(linesConfigPath, linesWeightsPath);

            // Initialize the character detection network using Detector class
            // This is an alternative implementation that uses the Darknet API directly
            charDetector = std::make_unique<Detector>(charsConfigPath, charsWeightsPath);

            // Initialize the small character detection network using Detector class
            // This is an alternative implementation that uses the Darknet API directly
            smallCharDetector = std::make_unique<Detector>(smallCharsConfigPath, smallCharsWeightsPath);

            // Note: When using the Detector class, we need to load class names separately
            // These will be loaded in the loadNames method
        }

        // Log successful initialization
        std::cout << "Networks initialized successfully" << std::endl;

        // Log information about the loaded networks when using DarkHelp
        if (useDarkHelp) {
            // Display the number of classes in each network
            // This helps verify that the networks were loaded correctly
            std::cout << "Line classes: " << lineNetwork->names.size() << std::endl;
            std::cout << "Character classes: " << charNetwork->names.size() << std::endl;
            std::cout << "Small character classes: " << smallCharNetwork->names.size() << std::endl;
        }

        return true;  // Return success
    }
    catch (const std::exception& e) {
        // Log any exceptions that occur during initialization
        std::cerr << "Error initializing networks: " << e.what() << std::endl;
        return false;  // Return failure
    }
}

/**
 * @brief Process an image and extract text using OCR
 *
 * This method implements the core OCR pipeline:
 * 1. Detect lines of text in the image
 * 2. For each line, detect individual characters
 * 3. For each character, recognize the specific character
 * 4. Combine the recognized characters into lines of text
 *
 * @param image OpenCV Mat containing the image to process
 * @return std::string The recognized text with each line separated by a newline
 */
std::string OCRProcessor::processImage(const cv::Mat& image) {
    // Check if the input image is valid
    if (image.empty()) {
        std::cerr << "Empty image provided" << std::endl;
        return "";  // Return empty string for invalid input
    }

    std::string result;  // Will hold the final OCR result

    try {
        // === STEP 1: Detect lines in the image ===
        // Use the line detection network to find rectangular regions containing lines of text
        std::vector<Detection> lineDetections = detectLines(image);
        std::cout << "Detected " << lineDetections.size() << " lines" << std::endl;

        // Sort lines from top to bottom to process them in reading order
        std::sort(lineDetections.begin(), lineDetections.end(),
                [](const Detection& a, const Detection& b) {
                    return a.bbox.y < b.bbox.y;  // Compare y-coordinates
                });

        // === STEP 2: Process each detected line ===
        for (const auto& lineDetection : lineDetections) {
            // Extract the line region from the original image
            cv::Rect lineRect = lineDetection.bbox;

            // Ensure the rectangle is within the image bounds to prevent access violations
            // The & operator performs a rectangle intersection
            lineRect = lineRect & cv::Rect(0, 0, image.cols, image.rows);

            // Skip invalid rectangles (those with zero or negative width/height)
            if (lineRect.width <= 0 || lineRect.height <= 0) {
                continue;
            }

            // Create a copy of the line image for processing
            cv::Mat lineImage = image(lineRect).clone();

            // === STEP 3: Detect characters within the line ===
            // Use the character detection network to find individual characters
            std::vector<Detection> charDetections = detectCharacters(lineImage);
            std::cout << "Detected " << charDetections.size() << " characters in line" << std::endl;

            // Sort characters from left to right to process them in reading order
            sortDetectionsLeftToRight(charDetections);

            // === STEP 4: Process each detected character ===
            for (const auto& charDetection : charDetections) {
                // Extract the character region from the line image
                cv::Rect charRect = charDetection.bbox;

                // Ensure the character rectangle is within the line image bounds
                charRect = charRect & cv::Rect(0, 0, lineImage.cols, lineImage.rows);

                // Skip invalid rectangles
                if (charRect.width <= 0 || charRect.height <= 0) {
                    continue;
                }

                // Create a copy of the character image for processing
                cv::Mat charImage = lineImage(charRect).clone();

                // === STEP 5: Recognize the specific character ===
                // Use the small character recognition network for precise character recognition
                std::vector<Detection> smallCharDetections = detectSmallCharacters(charImage);

                // If small character detection found something, use it
                if (!smallCharDetections.empty()) {
                    // Sort by confidence (highest first) to get the most likely character
                    std::sort(smallCharDetections.begin(), smallCharDetections.end(),
                            [](const Detection& a, const Detection& b) {
                                return a.confidence > b.confidence;  // Compare confidence scores
                            });

                    // Add the recognized character to the result string
                    result += smallCharDetections[0].label;  // Use the highest confidence detection
                }
            }

            // Add a newline after processing each line of text
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

/**
 * @brief Process an image file and extract text using OCR
 *
 * This method loads an image from a file path and processes it using the OCR pipeline.
 * It's a convenience wrapper around the processImage method that handles image loading.
 *
 * @param imagePath Path to the image file to process
 * @return std::string The recognized text with each line separated by a newline
 */
std::string OCRProcessor::processImageFile(const std::string& imagePath) {
    // Load the image from file using OpenCV's imread function
    cv::Mat image = cv::imread(imagePath);

    // Check if the image was loaded successfully
    if (image.empty()) {
        std::cerr << "Failed to load image: " << imagePath << std::endl;
        return "";  // Return empty string if image loading failed
    }

    // Process the image using the existing processImage method
    // This delegates the actual OCR processing to the main implementation
    return processImage(image);
}

/**
 * @brief Get raw predictions from the neural network for an image
 *
 * This method provides direct access to the raw neural network predictions,
 * allowing for more advanced processing or custom visualization of the results.
 * It uses the line detection network by default.
 *
 * @param image OpenCV Mat containing the image to process
 * @return DarkHelp::PredictionResults Vector of prediction results from the neural network
 */
DarkHelp::PredictionResults OCRProcessor::getPredictions(const cv::Mat& image) {
    // Check if the input image is valid
    if (image.empty()) {
        std::cerr << "Empty image provided" << std::endl;
        return DarkHelp::PredictionResults();  // Return empty results for invalid input
    }

    try {
        // Use the line detection network to get predictions
        // This is useful for getting the raw detection results before processing
        if (useDarkHelp && lineNetwork) {
            // Set the confidence threshold for detections
            lineNetwork->config.threshold = lineThreshold;

            // Run inference on the image and return the raw predictions
            // This returns a vector of DarkHelp::PredictionResult objects
            return lineNetwork->predict(image);
        }
        else {
            // Handle the case where the neural network is not initialized
            std::cerr << "Neural network not initialized or DarkHelp not enabled" << std::endl;
            return DarkHelp::PredictionResults();  // Return empty results
        }
    }
    catch (const std::exception& e) {
        // Handle any exceptions that occur during prediction
        std::cerr << "Error getting predictions: " << e.what() << std::endl;
        return DarkHelp::PredictionResults();  // Return empty results on error
    }
}

/**
 * @brief Detect lines of text in an image
 *
 * This method uses either the DarkHelp API or the Detector class to detect
 * lines of text in the input image. It returns a vector of Detection objects
 * representing the detected lines.
 *
 * @param image OpenCV Mat containing the image to process
 * @return std::vector<Detection> Vector of detected lines
 */
std::vector<Detection> OCRProcessor::detectLines(const cv::Mat& image) {
    if (useDarkHelp) {
        // === Use DarkHelp API for line detection ===
        // Set the confidence threshold for line detection
        lineNetwork->config.threshold = lineThreshold;

        // Run inference on the image to get line predictions
        DarkHelp::PredictionResults predictions = lineNetwork->predict(image);

        // Convert DarkHelp predictions to our Detection format
        return convertPredictions(predictions);
    } else {
        // === Use Detector class for line detection (alternative) ===
        // Run inference using the Darknet Detector API
        auto detections = lineDetector->detect(image, lineThreshold);

        // Convert Darknet bounding boxes to our Detection format
        return convertBBoxes(detections, lineNetwork->names);
    }
}

/**
 * @brief Detect individual characters in an image (typically a line of text)
 *
 * This method uses either the DarkHelp API or the Detector class to detect
 * individual characters in the input image. It returns a vector of Detection objects
 * representing the detected characters.
 *
 * @param image OpenCV Mat containing the image to process (usually a cropped line)
 * @return std::vector<Detection> Vector of detected characters
 */
std::vector<Detection> OCRProcessor::detectCharacters(const cv::Mat& image) {
    if (useDarkHelp) {
        // === Use DarkHelp API for character detection ===
        // Set the confidence threshold for character detection
        charNetwork->config.threshold = charThreshold;

        // Run inference on the image to get character predictions
        DarkHelp::PredictionResults predictions = charNetwork->predict(image);

        // Convert DarkHelp predictions to our Detection format
        return convertPredictions(predictions);
    } else {
        // === Use Detector class for character detection (alternative) ===
        // Run inference using the Darknet Detector API
        auto detections = charDetector->detect(image, charThreshold);

        // Convert Darknet bounding boxes to our Detection format
        return convertBBoxes(detections, charNetwork->names);
    }
}

/**
 * @brief Recognize specific characters in an image (typically a cropped character)
 *
 * This method uses either the DarkHelp API or the Detector class to recognize
 * specific characters in the input image. It returns a vector of Detection objects
 * representing the recognized characters, typically with just one high-confidence detection.
 *
 * @param image OpenCV Mat containing the image to process (usually a cropped character)
 * @return std::vector<Detection> Vector of recognized characters
 */
std::vector<Detection> OCRProcessor::detectSmallCharacters(const cv::Mat& image) {
    if (useDarkHelp) {
        // === Use DarkHelp API for small character recognition ===
        // Set the confidence threshold for small character recognition
        smallCharNetwork->config.threshold = smallCharThreshold;

        // Run inference on the image to get small character predictions
        DarkHelp::PredictionResults predictions = smallCharNetwork->predict(image);

        // Convert DarkHelp predictions to our Detection format
        return convertPredictions(predictions);
    } else {
        // === Use Detector class for small character recognition (alternative) ===
        // Run inference using the Darknet Detector API
        auto detections = smallCharDetector->detect(image, smallCharThreshold);

        // Convert Darknet bounding boxes to our Detection format
        return convertBBoxes(detections, smallCharNetwork->names);
    }
}

/**
 * @brief Convert DarkHelp prediction results to our Detection format
 *
 * This method converts the DarkHelp::PredictionResults format to our internal
 * Detection format for consistent handling of detections regardless of the backend used.
 *
 * @param predictions DarkHelp prediction results to convert
 * @return std::vector<Detection> Converted detections
 */
std::vector<Detection> OCRProcessor::convertPredictions(const DarkHelp::PredictionResults& predictions) {
    std::vector<Detection> detections;  // Will hold the converted detections

    // Process each prediction from DarkHelp
    for (const auto& pred : predictions) {
        // Create a new Detection object
        Detection det;

        // Copy the relevant information from the DarkHelp prediction
        det.label = pred.name;                  // The class name (character label)
        det.confidence = pred.best_probability;  // The confidence score (0.0 to 1.0)
        det.bbox = pred.rect;                    // The bounding box (cv::Rect)

        // Add the detection to our result vector
        detections.push_back(det);
    }

    return detections;  // Return the converted detections
}

/**
 * @brief Convert Darknet bounding boxes to our Detection format
 *
 * This method converts the Darknet bbox_t format to our internal Detection format
 * for consistent handling of detections regardless of the backend used.
 *
 * @param bboxes Vector of Darknet bounding boxes to convert
 * @param names Vector of class names for labeling the detections
 * @return std::vector<Detection> Converted detections
 */
std::vector<Detection> OCRProcessor::convertBBoxes(const std::vector<bbox_t>& bboxes, const std::vector<std::string>& names) {
    std::vector<Detection> detections;  // Will hold the converted detections

    // Process each bounding box from Darknet
    for (const auto& bbox : bboxes) {
        // Create a new Detection object
        Detection det;

        // Copy the relevant information from the Darknet bounding box
        det.label = names[bbox.obj_id];          // Get the class name using the object ID
        det.confidence = bbox.prob;              // The confidence score (0.0 to 1.0)
        det.bbox = cv::Rect(bbox.x, bbox.y, bbox.w, bbox.h);  // Convert to OpenCV Rect

        // Add the detection to our result vector
        detections.push_back(det);
    }

    return detections;  // Return the converted detections
}

/**
 * @brief Load class names from a file
 *
 * This method loads class names from a file, where each line contains one class name.
 * These names are used to label the detected objects when using the Detector class.
 * The method also trims whitespace from each line and skips empty lines.
 *
 * @param namesPath Path to the file containing class names
 * @return std::vector<std::string> Vector of class names
 */
std::vector<std::string> OCRProcessor::loadNames(const std::string& namesPath) {
    std::vector<std::string> names;  // Will hold the loaded class names
    std::ifstream file(namesPath);   // Open the file for reading
    std::string line;                // Will hold each line read from the file

    // Check if the file was opened successfully
    if (!file.is_open()) {
        // Log an error if the file couldn't be opened
        std::cerr << "Failed to open names file: " << namesPath << std::endl;
        return names;  // Return empty vector
    }

    // Read the file line by line
    while (std::getline(file, line)) {
        // Trim whitespace from the beginning of the line
        line.erase(0, line.find_first_not_of(" \t\n\r\f\v"));
        // Trim whitespace from the end of the line
        line.erase(line.find_last_not_of(" \t\n\r\f\v") + 1);

        // Skip empty lines
        if (!line.empty()) {
            // Add the trimmed class name to the vector
            names.push_back(line);
        }
    }

    return names;
}

/**
 * @brief Sort detections from left to right
 *
 * This method sorts a vector of Detection objects from left to right based on
 * their x-coordinate. This is useful for processing characters in reading order.
 *
 * @param detections Vector of Detection objects to sort (modified in-place)
 */
void OCRProcessor::sortDetectionsLeftToRight(std::vector<Detection>& detections) {
    // Sort the detections using the x-coordinate of the bounding box
    std::sort(detections.begin(), detections.end(),
              [](const Detection& a, const Detection& b) {
                  // Compare x-coordinates to sort from left to right
                  return a.bbox.x < b.bbox.x;
              });
}
