#include <gtest/gtest.h>
#include "OCRProcessor.h"
#include <filesystem>
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>

// Simple JSON parsing for test cases
class SimpleJSON {
public:
    struct Value {
        std::vector<std::string> strings;
        std::vector<Value> children;
        int size() const { return children.size(); }
        Value operator[](const std::string& key) const {
            for (size_t i = 0; i < strings.size() - 1; i += 2) {
                if (strings[i] == key) return children[i/2];
            }
            return Value();
        }
        Value operator[](size_t index) const {
            return index < children.size() ? children[index] : Value();
        }
        std::string get() const {
            return strings.empty() ? "" : strings[0];
        }
        std::vector<std::string> get_array() const {
            std::vector<std::string> result;
            for (const auto& child : children) {
                result.push_back(child.get());
            }
            return result;
        }
    };

    static Value parse(const std::string& json_str) {
        std::istringstream stream(json_str);
        return parse_value(stream);
    }

    static Value parse_file(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            return Value();
        }
        std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
        return parse(content);
    }

private:
    static void skip_whitespace(std::istringstream& stream) {
        while (stream.good() && std::isspace(stream.peek())) {
            stream.get();
        }
    }

    static std::string parse_string(std::istringstream& stream) {
        std::string result;
        // Skip opening quote
        stream.get();
        while (stream.good() && stream.peek() != '"') {
            char c = stream.get();
            if (c == '\\') {
                if (stream.good()) {
                    c = stream.get();
                }
            }
            result += c;
        }
        // Skip closing quote
        if (stream.good()) {
            stream.get();
        }
        return result;
    }

    static Value parse_value(std::istringstream& stream) {
        Value value;
        skip_whitespace(stream);
        if (stream.peek() == '{') {
            stream.get(); // Skip {
            skip_whitespace(stream);
            while (stream.good() && stream.peek() != '}') {
                skip_whitespace(stream);
                if (stream.peek() == '"') {
                    std::string key = parse_string(stream);
                    value.strings.push_back(key);
                    skip_whitespace(stream);
                    if (stream.peek() == ':') {
                        stream.get(); // Skip :
                        value.children.push_back(parse_value(stream));
                    }
                }
                skip_whitespace(stream);
                if (stream.peek() == ',') {
                    stream.get(); // Skip ,
                }
            }
            if (stream.good()) {
                stream.get(); // Skip }
            }
        } else if (stream.peek() == '[') {
            stream.get(); // Skip [
            skip_whitespace(stream);
            while (stream.good() && stream.peek() != ']') {
                value.children.push_back(parse_value(stream));
                skip_whitespace(stream);
                if (stream.peek() == ',') {
                    stream.get(); // Skip ,
                }
                skip_whitespace(stream);
            }
            if (stream.good()) {
                stream.get(); // Skip ]
            }
        } else if (stream.peek() == '"') {
            value.strings.push_back(parse_string(stream));
        }
        return value;
    }
};

// For convenience
using json = SimpleJSON::Value;

namespace fs = std::filesystem;

// Test fixture for OCR tests
class OCRProcessorTest : public ::testing::Test {
protected:
    std::unique_ptr<OCRProcessor> processor;
    std::string testDataDir;

    void SetUp() override {
        // Initialize the OCR processor
        processor = createOCRProcessor();
        ASSERT_TRUE(processor != nullptr) << "Failed to initialize OCR processor";

        // Set the test data directory
        testDataDir = fs::current_path().string() + "/test_data";
        ASSERT_TRUE(fs::exists(testDataDir)) << "Test data directory does not exist: " << testDataDir;
    }

    // Helper function to create OCR processor
    std::unique_ptr<OCRProcessor> createOCRProcessor() {
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

        // Validate paths
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
                return nullptr;
            }
        }

        // Create and initialize the OCR processor
        auto processor = std::make_unique<OCRProcessor>(
            linesConfigPath, linesWeightsPath, linesNamesPath,
            charsConfigPath, charsWeightsPath, charsNamesPath,
            smallCharsConfigPath, smallCharsWeightsPath, smallCharsNamesPath
        );

        // Set detection thresholds
        processor->setThresholds(0.5f, 0.5f, 0.5f);

        return processor;
    }

    // Helper function to split a string into lines
    std::vector<std::string> splitLines(const std::string& text) {
        std::vector<std::string> lines;
        std::string line;
        std::istringstream stream(text);

        while (std::getline(stream, line)) {
            // Remove any trailing whitespace
            line.erase(line.find_last_not_of(" \t\n\r\f\v") + 1);
            // Skip empty lines
            if (!line.empty()) {
                lines.push_back(line);
            }
        }

        return lines;
    }
};

// Test case for verifying OCR on a single image with expected lines
TEST_F(OCRProcessorTest, TestOCRWithExpectedLines) {
    // Test data: image path and expected lines
    struct TestCase {
        std::string imagePath;
        std::vector<std::string> expectedLines;
    };

    // Load test cases from JSON file
    std::vector<TestCase> testCases;
    std::string testCasesPath = testDataDir + "/test_cases.json";

    if (fs::exists(testCasesPath)) {
        try {
            json testData = SimpleJSON::parse_file(testCasesPath);

            for (size_t i = 0; i < testData["test_cases"].size(); ++i) {
                auto testCase = testData["test_cases"][i];
                std::string imagePath = testDataDir + "/" + testCase["image_path"].get();
                std::vector<std::string> expectedLines = testCase["expected_lines"].get_array();
                testCases.push_back({imagePath, expectedLines});
            }
        } catch (const std::exception& e) {
            std::cerr << "Error loading test cases: " << e.what() << std::endl;
        }
    } else {
        std::cerr << "Test cases file not found: " << testCasesPath << std::endl;
        // Add some default test cases
        testCases = {
            {testDataDir + "/test_image1.jpg", {"123456", "hellothere", "eonegi"}},
            {testDataDir + "/test_image2.jpg", {"ABCDEF", "test123", "sample"}}
        };
    }

    // Run tests for each test case
    for (const auto& testCase : testCases) {
        // Skip if image doesn't exist
        if (!fs::exists(testCase.imagePath)) {
            std::cerr << "Warning: Test image does not exist: " << testCase.imagePath << std::endl;
            continue;
        }

        // Process the image
        std::string result = processor->processImageFile(testCase.imagePath);

        // Split the result into lines
        std::vector<std::string> actualLines = splitLines(result);

        // Verify the number of lines
        EXPECT_EQ(actualLines.size(), testCase.expectedLines.size())
            << "Number of lines mismatch for image: " << testCase.imagePath;

        // Verify each line
        for (size_t i = 0; i < std::min(actualLines.size(), testCase.expectedLines.size()); ++i) {
            EXPECT_EQ(actualLines[i], testCase.expectedLines[i])
                << "Line " << i + 1 << " mismatch for image: " << testCase.imagePath;
        }
    }
}

// Test case for verifying line detection
TEST_F(OCRProcessorTest, TestLineDetection) {
    // Test data: image path and expected number of lines
    struct TestCase {
        std::string imagePath;
        int expectedLineCount;
    };

    // Load test cases from JSON file
    std::vector<TestCase> testCases;
    std::string testCasesPath = testDataDir + "/test_cases.json";

    if (fs::exists(testCasesPath)) {
        try {
            json testData = SimpleJSON::parse_file(testCasesPath);

            for (size_t i = 0; i < testData["test_cases"].size(); ++i) {
                auto testCase = testData["test_cases"][i];
                std::string imagePath = testDataDir + "/" + testCase["image_path"].get();
                // Count the number of expected lines
                int expectedLineCount = testCase["expected_lines"].size();
                testCases.push_back({imagePath, expectedLineCount});
            }
        } catch (const std::exception& e) {
            std::cerr << "Error loading test cases: " << e.what() << std::endl;
        }
    } else {
        std::cerr << "Test cases file not found: " << testCasesPath << std::endl;
        // Add some default test cases
        testCases = {
            {testDataDir + "/test_image1.jpg", 3},
            {testDataDir + "/test_image2.jpg", 3}
        };
    }

    // Run tests for each test case
    for (const auto& testCase : testCases) {
        // Skip if image doesn't exist
        if (!fs::exists(testCase.imagePath)) {
            std::cerr << "Warning: Test image does not exist: " << testCase.imagePath << std::endl;
            continue;
        }

        // Load the image
        cv::Mat image = cv::imread(testCase.imagePath);
        ASSERT_FALSE(image.empty()) << "Failed to load image: " << testCase.imagePath;

        // Get predictions from the line detection network
        DarkHelp::PredictionResults predictions = processor->getPredictions(image);

        // Verify the number of lines detected
        EXPECT_EQ(predictions.size(), testCase.expectedLineCount)
            << "Number of lines detected mismatch for image: " << testCase.imagePath;
    }
}

// Test case for verifying character recognition
TEST_F(OCRProcessorTest, TestCharacterRecognition) {
    // Test data: image path, line index, and expected characters
    struct TestCase {
        std::string imagePath;
        int lineIndex;  // 0-based index of the line to test
        std::string expectedCharacters;
    };

    // Load test cases from JSON file
    std::vector<TestCase> testCases;
    std::string testCasesPath = testDataDir + "/test_cases.json";

    if (fs::exists(testCasesPath)) {
        try {
            json testData = SimpleJSON::parse_file(testCasesPath);

            for (size_t i = 0; i < testData["test_cases"].size(); ++i) {
                auto testCase = testData["test_cases"][i];
                std::string imagePath = testDataDir + "/" + testCase["image_path"].get();
                std::vector<std::string> expectedLines = testCase["expected_lines"].get_array();

                // Create a test case for each line in the image
                for (size_t i = 0; i < expectedLines.size(); ++i) {
                    testCases.push_back({imagePath, static_cast<int>(i), expectedLines[i]});
                }
            }
        } catch (const std::exception& e) {
            std::cerr << "Error loading test cases: " << e.what() << std::endl;
        }
    } else {
        std::cerr << "Test cases file not found: " << testCasesPath << std::endl;
        // Add some default test cases
        testCases = {
            {testDataDir + "/test_image1.jpg", 0, "123456"},
            {testDataDir + "/test_image1.jpg", 1, "hellothere"},
            {testDataDir + "/test_image1.jpg", 2, "eonegi"},
            {testDataDir + "/test_image2.jpg", 0, "ABCDEF"},
            {testDataDir + "/test_image2.jpg", 1, "test123"},
            {testDataDir + "/test_image2.jpg", 2, "sample"}
        };
    }

    // Run tests for each test case
    for (const auto& testCase : testCases) {
        // Skip if image doesn't exist
        if (!fs::exists(testCase.imagePath)) {
            std::cerr << "Warning: Test image does not exist: " << testCase.imagePath << std::endl;
            continue;
        }

        // Process the image
        std::string result = processor->processImageFile(testCase.imagePath);

        // Split the result into lines
        std::vector<std::string> lines = splitLines(result);

        // Skip if the line index is out of range
        if (testCase.lineIndex >= lines.size()) {
            std::cerr << "Warning: Line index out of range for image: " << testCase.imagePath << std::endl;
            continue;
        }

        // Verify the characters in the specified line
        EXPECT_EQ(lines[testCase.lineIndex], testCase.expectedCharacters)
            << "Character recognition mismatch for line " << testCase.lineIndex + 1
            << " in image: " << testCase.imagePath;
    }
}

// Main function to run all tests
int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
