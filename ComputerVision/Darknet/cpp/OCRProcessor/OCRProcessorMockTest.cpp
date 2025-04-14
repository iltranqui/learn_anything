#include <gtest/gtest.h>
#include <string>
#include <vector>
#include <filesystem>
#include <fstream>
#include <sstream>

namespace fs = std::filesystem;

// Mock OCR processor for testing
class MockOCRProcessor {
public:
    MockOCRProcessor() {}
    
    // Mock method to process an image file
    std::string processImageFile(const std::string& imagePath) {
        // For testing, just return a predefined result based on the image name
        std::string filename = fs::path(imagePath).filename().string();
        
        if (filename == "test_image1.jpg") {
            return "123456\nhellothere\neonegi\n";
        } else if (filename == "test_image2.jpg") {
            return "ABCDEF\ntest123\nsample\n";
        } else {
            return "Unknown image\n";
        }
    }
};

// Test fixture for OCR tests
class OCRProcessorMockTest : public ::testing::Test {
protected:
    MockOCRProcessor processor;
    std::string testDataDir;

    void SetUp() override {
        // Set the test data directory
        testDataDir = fs::current_path().string() + "/test_data";
        
        // Create test data directory if it doesn't exist
        if (!fs::exists(testDataDir)) {
            fs::create_directory(testDataDir);
        }
        
        // Create test image files (empty files for testing)
        std::ofstream test_image1(testDataDir + "/test_image1.jpg");
        test_image1.close();
        
        std::ofstream test_image2(testDataDir + "/test_image2.jpg");
        test_image2.close();
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
TEST_F(OCRProcessorMockTest, TestOCRWithExpectedLines) {
    // Test data: image path and expected lines
    struct TestCase {
        std::string imagePath;
        std::vector<std::string> expectedLines;
    };

    // Define test cases
    std::vector<TestCase> testCases = {
        {testDataDir + "/test_image1.jpg", {"123456", "hellothere", "eonegi"}},
        {testDataDir + "/test_image2.jpg", {"ABCDEF", "test123", "sample"}}
    };

    // Run tests for each test case
    for (const auto& testCase : testCases) {
        // Skip if image doesn't exist
        if (!fs::exists(testCase.imagePath)) {
            std::cerr << "Warning: Test image does not exist: " << testCase.imagePath << std::endl;
            continue;
        }

        // Process the image
        std::string result = processor.processImageFile(testCase.imagePath);
        
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

// Main function to run all tests
int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
