#include <gtest/gtest.h>
#include <string>
#include <vector>

// Simple test case
TEST(SimpleTest, BasicAssertions) {
    // Expect equality
    EXPECT_EQ(2 + 2, 4);
    
    // Expect inequality
    EXPECT_NE(2 + 2, 5);
    
    // Expect true
    EXPECT_TRUE(true);
    
    // Expect false
    EXPECT_FALSE(false);
}

// Test with strings
TEST(SimpleTest, StringOperations) {
    std::string str1 = "Hello";
    std::string str2 = "World";
    
    // Concatenate strings
    std::string result = str1 + " " + str2;
    
    // Expect equality
    EXPECT_EQ(result, "Hello World");
    
    // Expect contains
    EXPECT_TRUE(result.find(str1) != std::string::npos);
    EXPECT_TRUE(result.find(str2) != std::string::npos);
}

// Test with vectors
TEST(SimpleTest, VectorOperations) {
    std::vector<int> vec = {1, 2, 3, 4, 5};
    
    // Expect size
    EXPECT_EQ(vec.size(), 5);
    
    // Expect elements
    EXPECT_EQ(vec[0], 1);
    EXPECT_EQ(vec[4], 5);
    
    // Modify vector
    vec.push_back(6);
    
    // Expect new size
    EXPECT_EQ(vec.size(), 6);
    
    // Expect new element
    EXPECT_EQ(vec[5], 6);
}

// Main function to run all tests
int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
