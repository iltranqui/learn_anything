/**
 * @file main.cpp
 * @brief Main application for OCR processing using neural networks
 *
 * This application demonstrates the use of the OCRProcessor class to perform
 * optical character recognition on images. It uses three neural networks:
 * 1. Line detection network - Detects lines of text in an image
 * 2. Character detection network - Detects individual characters in a line
 * 3. Small character recognition network - Recognizes specific characters
 *
 * The application takes an image path as input, processes the image using the
 * OCRProcessor, and outputs the recognized text to the console and a text file.
 */

#include "OCRProcessor.h"     // OCR processor class definition
#include <iostream>           // For standard input/output operations
#include <string>             // For string handling
#include <filesystem>         // For file and directory operations
#include <chrono>             // For timing operations
#include <fstream>            // For file operations

namespace fs = std::filesystem;  // Alias for the filesystem namespace

/**
 * @brief Structure to hold paths to model files
 *
 * This structure stores the paths to all the model files needed by the OCRProcessor:
 * - Configuration files (.cfg)
 * - Weights files (.weights)
 * - Class names files (.txt)
 *
 * The paths are constructed relative to a base path, which is typically the
 * current working directory of the application.
 */
struct ModelPaths {
    // Paths for line detection model
    std::string linesConfigPath;      // Path to line detection configuration file
    std::string linesWeightsPath;     // Path to line detection weights file
    std::string linesNamesPath;       // Path to line detection class names file

    // Paths for character detection model
    std::string charsConfigPath;      // Path to character detection configuration file
    std::string charsWeightsPath;     // Path to character detection weights file
    std::string charsNamesPath;       // Path to character detection class names file

    // Paths for small character recognition model
    std::string smallCharsConfigPath; // Path to small character recognition configuration file
    std::string smallCharsWeightsPath;// Path to small character recognition weights file
    std::string smallCharsNamesPath;  // Path to small character recognition class names file

    /**
     * @brief Constructor that initializes all model paths
     *
     * @param basePath Base directory path to construct model paths from
     */
    ModelPaths(const std::string& basePath) {
        // Line detection model
        linesConfigPath = basePath + "/cfg/lines/yolov4-tiny-lines.cfg";
        linesWeightsPath = basePath + "/backup/lines/yolov4-tiny-lines_best.weights";
        linesNamesPath = basePath + "/cfg/lines/classes.txt";

        // Character detection model
        charsConfigPath = basePath + "/cfg/characters/yolov4-tiny-characters.cfg";
        charsWeightsPath = basePath + "/backup/characters/yolov4-tiny-characters_best.weights";
        charsNamesPath = basePath + "/cfg/characters/classes.txt";

        // Small character detection model
        smallCharsConfigPath = basePath + "/cfg/small_characters/yolov4-tiny-small-characters.cfg";
        smallCharsWeightsPath = basePath + "/backup/small_characters/yolov4-tiny-small-characters_best.weights";
        smallCharsNamesPath = basePath + "/cfg/small_characters/classes.txt";
    }

    /**
     * @brief Validate that all model files exist
     *
     * This method checks if all the model files (configuration, weights, and names)
     * exist on disk. It returns false if any file is missing, and logs an error
     * message indicating which file is missing.
     *
     * @return bool True if all files exist, false otherwise
     */
    bool validatePaths() const {
        // Create a list of all files to check, with descriptive names
        std::vector<std::pair<std::string, std::string>> files = {
            {linesConfigPath, "Line config"},             // Line detection configuration file
            {linesWeightsPath, "Line weights"},           // Line detection weights file
            {linesNamesPath, "Line names"},               // Line detection class names file
            {charsConfigPath, "Character config"},        // Character detection configuration file
            {charsWeightsPath, "Character weights"},      // Character detection weights file
            {charsNamesPath, "Character names"},          // Character detection class names file
            {smallCharsConfigPath, "Small character config"},    // Small character recognition configuration file
            {smallCharsWeightsPath, "Small character weights"},  // Small character recognition weights file
            {smallCharsNamesPath, "Small character names"}      // Small character recognition class names file
        };

        // Check each file
        for (const auto& [path, desc] : files) {
            // If the file doesn't exist, log an error and return false
            if (!fs::exists(path)) {
                std::cerr << "Error: " << desc << " file does not exist: " << path << std::endl;
                return false;
            }
        }
        return true;  // All files exist
    }
};

/**
 * @brief Create and initialize the OCR processor
 *
 * This function creates an instance of the OCRProcessor class and initializes it
 * with the paths to the model files. It also sets the detection thresholds for
 * the three neural networks.
 *
 * @return std::unique_ptr<OCRProcessor> Pointer to the initialized OCR processor,
 *         or nullptr if initialization failed
 */
std::unique_ptr<OCRProcessor> createOCRProcessor() {
    // Get the current working directory as the base path
    std::string basePath = fs::current_path().string();

    // Create a ModelPaths object with paths relative to the base path
    ModelPaths paths(basePath);

    // Validate that all model files exist
    if (!paths.validatePaths()) {
        return nullptr;  // Return nullptr if any file is missing
    }

    // Log initialization
    std::cout << "Initializing OCR processor..." << std::endl;

    // Create and initialize the OCR processor with the model paths
    auto processor = std::make_unique<OCRProcessor>(
        paths.linesConfigPath, paths.linesWeightsPath, paths.linesNamesPath,
        paths.charsConfigPath, paths.charsWeightsPath, paths.charsNamesPath,
        paths.smallCharsConfigPath, paths.smallCharsWeightsPath, paths.smallCharsNamesPath
    );

    // Set detection thresholds for the three neural networks
    // Parameters: lineThreshold, charThreshold, smallCharThreshold
    processor->setThresholds(0.5f, 0.5f, 0.5f);

    return processor;  // Return the initialized processor
}

/**
 * @brief Display usage information for the application
 *
 * This function prints the command-line usage information for the application,
 * showing the expected arguments and their descriptions.
 */
void displayUsage() {
    std::cout << "Usage: OCRProcessor.exe <image_path>" << std::endl;
    std::cout << "  <image_path>: Path to the image file to process" << std::endl;
}

/**
 * @brief Main entry point for the application
 *
 * This function is the entry point for the OCR processor application. It:
 * 1. Initializes the OCR processor
 * 2. Validates command-line arguments
 * 3. Processes the input image
 * 4. Displays and saves the recognized text
 *
 * @param argc Number of command-line arguments
 * @param argv Array of command-line argument strings
 * @return int Exit code (0 for success, non-zero for error)
 */
int main(int argc, char* argv[]) {
    // === STEP 1: Initialize the OCR processor ===
    auto processor = createOCRProcessor();
    if (!processor) {
        std::cerr << "Failed to initialize OCR processor" << std::endl;
        return 1;  // Return error code
    }

    // === STEP 2: Validate command-line arguments ===
    // Check if an image path was provided
    if (argc < 2) {
        std::cerr << "Error: Missing image path" << std::endl;
        displayUsage();  // Show usage information
        return 1;  // Return error code
    }

    // Get the image path from the command-line arguments
    std::string imagePath = argv[1];

    // Check if the image file exists
    if (!fs::exists(imagePath)) {
        std::cerr << "Error: Image file does not exist: " << imagePath << std::endl;
        return 1;  // Return error code
    }

    // === STEP 3: Process the image ===
    std::cout << "Processing image: " << imagePath << std::endl;

    // Start timing the processing
    auto startTime = std::chrono::high_resolution_clock::now();

    // Load the image using OpenCV
    cv::Mat image = cv::imread(imagePath);  // Convert image path to cv::Mat object
    if (image.empty()) {
        std::cerr << "Failed to load image: " << imagePath << std::endl;
        return 1;  // Return error code
    }

    // === STEP 3.1: Get raw predictions for demonstration ===
    std::cout << "Getting raw predictions..." << std::endl;
    // Get raw predictions from the line detection network
    DarkHelp::PredictionResults predictions = processor->getPredictions(image);
    std::cout << "Found " << predictions.size() << " predictions" << std::endl;

    // Print detailed information about each prediction
    for (size_t i = 0; i < predictions.size(); ++i) {
        const auto& pred = predictions[i];  // Get reference to current prediction

        // Print prediction details
        std::cout << "Prediction " << i + 1 << ": "
                  << "Class: " << pred.best_class            // Class ID
                  << ", Name: " << pred.name                 // Class name
                  << ", Confidence: " << pred.best_probability  // Confidence score
                  << ", Position: (" << pred.rect.x << ", " << pred.rect.y << ")"  // Position
                  << ", Size: " << pred.rect.width << "x" << pred.rect.height      // Size
                  << std::endl;
    }

    // === STEP 3.2: Process the image to get the recognized text ===
    // This performs the full OCR pipeline: line detection -> character detection -> character recognition
    std::string result = processor->processImageFile(imagePath);

    // Calculate processing time
    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();

    // === STEP 4: Display the result ===
    std::cout << "\nRecognized Text:" << std::endl;
    std::cout << "----------------" << std::endl;
    std::cout << result << std::endl;  // Print the recognized text
    std::cout << "----------------" << std::endl;
    std::cout << "Processing time: " << duration << " ms" << std::endl;  // Print processing time

    // === STEP 5: Save the result to a text file ===
    // Create output file path by replacing the image extension with .txt
    std::string outputPath = fs::path(imagePath).replace_extension(".txt").string();

    // Open the output file for writing
    std::ofstream outputFile(outputPath);
    if (outputFile.is_open()) {
        // Write the recognized text to the file
        outputFile << result;
        outputFile.close();  // Close the file
        std::cout << "Result saved to: " << outputPath << std::endl;  // Log success
    } else {
        // Log error if the file couldn't be opened
        std::cerr << "Failed to save result to: " << outputPath << std::endl;
    }

    return 0;  // Return success code
}
