#include "OCRProcessor.h"
#include <iostream>
#include <string>
#include <filesystem>
#include <chrono>
#include <fstream>

namespace fs = std::filesystem;

void displayUsage() {
    std::cout << "Usage: OCRProcessor.exe <image_path>" << std::endl;
    std::cout << "  <image_path>: Path to the image file to process" << std::endl;
}

int main(int argc, char* argv[]) {
    // Check command line arguments
    if (argc < 2) {
        std::cerr << "Error: Missing image path" << std::endl;
        displayUsage();
        return 1;
    }

    std::string imagePath = argv[1];

    // Check if the image file exists
    if (!fs::exists(imagePath)) {
        std::cerr << "Error: Image file does not exist: " << imagePath << std::endl;
        return 1;
    }

    // Define paths to model files
    std::string basePath = fs::current_path().string();

    // Line detection model
    std::string linesConfigPath = basePath + "/cfg/lines/yolov4-tiny-lines.cfg";
    std::string linesWeightsPath = basePath + "/backup/lines/yolov4-tiny-lines_best.weights";
    std::string linesNamesPath = basePath + "/cfg/lines/classes.txt";

    // Character detection model
    std::string charsConfigPath = basePath + "/cfg/characters/yolov4-tiny-characters.cfg";
    std::string charsWeightsPath = basePath + "/backup/characters/yolov4-tiny-characters_best.weights";
    std::string charsNamesPath = basePath + "/cfg/characters/classes.txt";

    // Small character detection model
    std::string smallCharsConfigPath = basePath + "/cfg/small_characters/yolov4-tiny-small-characters.cfg";
    std::string smallCharsWeightsPath = basePath + "/backup/small_characters/yolov4-tiny-small-characters_best.weights";
    std::string smallCharsNamesPath = basePath + "/cfg/small_characters/classes.txt";

    // Check if all model files exist
    std::vector<std::pair<std::string, std::string>> files = {
        {linesConfigPath, "Line config"},
        {linesWeightsPath, "Line weights"},
        {linesNamesPath, "Line names"},
        {charsConfigPath, "Character config"},
        {charsWeightsPath, "Character weights"},
        {charsNamesPath, "Character names"},
        {smallCharsConfigPath, "Small character config"},
        {smallCharsWeightsPath, "Small character weights"},
        {smallCharsNamesPath, "Small character names"}
    };

    for (const auto& [path, desc] : files) {
        if (!fs::exists(path)) {
            std::cerr << "Error: " << desc << " file does not exist: " << path << std::endl;
            return 1;
        }
    }

    // Initialize the OCR processor with model paths
    std::cout << "Initializing OCR processor..." << std::endl;
    OCRProcessor processor(
        linesConfigPath, linesWeightsPath, linesNamesPath,
        charsConfigPath, charsWeightsPath, charsNamesPath,
        smallCharsConfigPath, smallCharsWeightsPath, smallCharsNamesPath
    );

    // Set detection thresholds
    processor.setThresholds(0.5f, 0.5f, 0.5f);

    // Process the image directly from file
    std::cout << "Processing image: " << imagePath << std::endl;
    auto startTime = std::chrono::high_resolution_clock::now();

    // Load the image for demonstration purposes
    cv::Mat image = cv::imread(imagePath);
    if (image.empty()) {
        std::cerr << "Failed to load image: " << imagePath << std::endl;
        return 1;
    }

    // Get raw predictions from the line detection network
    std::cout << "Getting raw predictions..." << std::endl;
    DarkHelp::PredictionResults predictions = processor.getPredictions(image);
    std::cout << "Found " << predictions.size() << " predictions" << std::endl;

    // Print some information about the predictions
    for (size_t i = 0; i < predictions.size(); ++i) {
        const auto& pred = predictions[i];
        std::cout << "Prediction " << i + 1 << ": "
                  << "Class: " << pred.best_class
                  << ", Name: " << pred.name
                  << ", Confidence: " << pred.best_probability
                  << ", Position: (" << pred.rect.x << ", " << pred.rect.y << ")"
                  << ", Size: " << pred.rect.width << "x" << pred.rect.height
                  << std::endl;
    }

    // Process the image to get the recognized text
    std::string result = processor.processImageFile(imagePath);

    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();

    // Display the result
    std::cout << "\nRecognized Text:" << std::endl;
    std::cout << "----------------" << std::endl;
    std::cout << result << std::endl;
    std::cout << "----------------" << std::endl;
    std::cout << "Processing time: " << duration << " ms" << std::endl;

    // Save the result to a text file
    std::string outputPath = fs::path(imagePath).replace_extension(".txt").string();
    std::ofstream outputFile(outputPath);
    if (outputFile.is_open()) {
        outputFile << result;
        outputFile.close();
        std::cout << "Result saved to: " << outputPath << std::endl;
    } else {
        std::cerr << "Failed to save result to: " << outputPath << std::endl;
    }

    return 0;
}
