// data2train.cpp : This file contains the 'main' function. Program execution begins and ends there.
// c++ -std=c++17 data2train.cpp -o data2train
// data2train -data_folder_path:data -train_name:train.txt -validation_name:validation.txt -test_name:test.txt -test_set:True -split_ratio:0.8 -split_ratio_validation_test:0.2,0.1

// This program reads all .txt files in a folder and splits them into train, validation, and test sets. (default: 80%, 20%, 10%)
// test_set: True if test set is required, False otherwise ( only train and validation sets)
// split_ratio: ( only train and validation sets) 0 < split_ratio < 1
// split_ratio_validation_test: (test_set: True) 0 < split_ratio_validation_test[0] < 1, 0 < split_ratio_validation_test[1] < 1

/*
BUG: 
Attempting to move file from: "train.txt" to: "data\\train.txt"
Rename failed for "train.txt" -> "data\\train.txt".
  Attempting copy-and-delete. Error: rename: unknown error: "train.txt", "data\train.txt"
Error copying "train.txt" -> "data\\train.txt". Final Error: remove: unknown error: "train.txt"
Attempting to move file from: "validation.txt" to: "data\\validation.txt"
Rename failed for "validation.txt" -> "data\\validation.txt".
  Attempting copy-and-delete. Error: rename: unknown error: "validation.txt", "data\validation.txt"
Error copying "validation.txt" -> "data\\validation.txt". Final Error: remove: unknown error: "validation.txt"
Attempting to move file from: "test.txt" to: "data\\test.txt"
test.txt moved to data folder.
*/

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <filesystem>
#include <random>
#include <algorithm>
#include <cassert>
#include <sstream>

namespace fs = std::filesystem;

void split_files(const std::string& folder_path,
    const std::string& train_file,
    const std::string& validation_file,
    const std::string& test_file,
    bool test_set = false,
    double split_ratio = 0.8,
    const std::vector<double>& split_ratio_validation_test = { 0.2, 0.1 }) {

    // Check if folder exists
    if (!fs::exists(folder_path)) {
        throw std::runtime_error("Folder path " + folder_path + " does not exist");
    }

    // Ensure correct ratios
    if (test_set) {
        assert(split_ratio_validation_test.size() == 2 && "split_ratio_validation_test must have exactly two elements");
    }

    // Get list of .txt files
    std::vector<std::string> txt_files;
    for (const auto& entry : fs::directory_iterator(folder_path)) {
        if (entry.path().extension() == ".txt") {
            txt_files.push_back(entry.path().filename().string());
        }
    }

    // Shuffle the files
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(txt_files.begin(), txt_files.end(), g);

    // Split the files
    std::vector<std::string> train_files, validation_files, test_files;

    if (!test_set) {
        size_t split_index = static_cast<size_t>(txt_files.size() * split_ratio);
        train_files.assign(txt_files.begin(), txt_files.begin() + split_index);
        validation_files.assign(txt_files.begin() + split_index, txt_files.end());

        // Print lengths of train_files and validation_files
        std::cout << "Train files count: " << train_files.size() << "\n";
        std::cout << "Validation files count: " << validation_files.size() << "\n\n";
    }
    else {
        size_t train_split_index = static_cast<size_t>(txt_files.size() * (1 - split_ratio_validation_test[0] - split_ratio_validation_test[1]));
        size_t validation_split_index = static_cast<size_t>(txt_files.size() * split_ratio_validation_test[0]);

        train_files.assign(txt_files.begin(), txt_files.begin() + train_split_index);
        validation_files.assign(txt_files.begin() + train_split_index, txt_files.begin() + train_split_index + validation_split_index);
        test_files.assign(txt_files.begin() + train_split_index + validation_split_index, txt_files.end());

        // Print lengths of train_files, validation_files, and test_files
        std::cout << "Train files count: " << train_files.size() << "\n";
        std::cout << "Validation files count: " << validation_files.size() << "\n";
        std::cout << "Test files count: " << test_files.size() << "\n\n";
    }

    // Write to train file
    std::ofstream train_out(train_file);
    if (!train_out) {
        throw std::runtime_error("Error opening " + train_file);
    }
    for (const auto& file : train_files) {
        train_out << "data/" << file.substr(0, file.find_last_of('.')) << ".jpg" << "\n";
    }

    // Write to validation file
    std::ofstream validation_out(validation_file);
    if (!validation_out) {
        throw std::runtime_error("Error opening " + validation_file);
    }
    for (const auto& file : validation_files) {
        validation_out << "data/" << file.substr(0, file.find_last_of('.')) << ".jpg" << "\n";
    }

    if (test_set) {
        // Write to test file
        std::ofstream test_out(test_file);
        if (!test_out) {
            throw std::runtime_error("Error opening " + test_file);
        }
        for (const auto& file : test_files) {
            test_out << "data/" << file.substr(0, file.find_last_of('.')) << ".jpg" << "\n";
        }
    }

    // Move files to "data" folder
    std::string data_folder = "data";
    if (!fs::exists(data_folder)) {
        fs::create_directory(data_folder);
    }

    try {
        for (const auto& file_name : { train_file, validation_file, test_set ? test_file : "" }) {
            if (!file_name.empty()) {
                auto src_path = fs::path(file_name);
                auto dest_path = fs::path(data_folder) / file_name;
                // Debugging paths
                std::cout << "Attempting to move file from: " << src_path << " to: " << dest_path << "\n";

                if (!fs::exists(src_path)) {
                    std::cerr << "Error: Source file " << src_path << " does not exist.\n";
                    continue;
                }
                try {
                    // Attempt to rename (move)
                    fs::rename(src_path, dest_path);
                    std::cout << file_name << " moved to " << data_folder << " folder.\n";
                }
                catch (const fs::filesystem_error& e) {
                    std::cerr << "Rename failed for " << src_path << " -> " << dest_path
                        << ".\n  Attempting copy-and-delete. Error: " << e.what() << "\n";

                    try {
                        // Copy and delete fallback
                        //fs::copy(src_path, dest_path, fs::copy_options::overwrite_existing);
                        fs::remove(src_path);
                        std::cout << file_name << " successfully copied and deleted to " << data_folder << " folder.\n";
                    }
                    catch (const fs::filesystem_error& e) {
                        std::cerr << "Error copying " << src_path << " -> " << dest_path
                            << ". Final Error: " << e.what() << "\n";
                    }
                }
            }
        }
    } catch (const fs::filesystem_error& e) {
        std::cerr << "Filesystem error: " << e.what() << "\n";
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
    }
}

std::vector<std::string> split_string(const std::string& str, char delimiter) {
    std::vector<std::string> tokens;
    std::stringstream ss(str);
    std::string token;
    while (std::getline(ss, token, delimiter)) {
        tokens.push_back(token);
    }
    return tokens;
}


int main(int argc, char* argv[]) {
	// default values
    std::string folder_path = "data";
    std::string train_file = "train.txt";
    std::string validation_file = "validation.txt";
    std::string test_file = "test.txt";
    bool test_set = false;
    double split_ratio = 0.8;
    std::vector<double> split_ratio_validation_test = { 0.2, 0.1 };

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg.find("-data_folder_path:") == 0) {
            folder_path = arg.substr(arg.find(':') + 1);
        }
        else if (arg.find("-train_name:") == 0) {
            train_file = arg.substr(arg.find(':') + 1);
        }
        else if (arg.find("-validation_name:") == 0) {
            validation_file = arg.substr(arg.find(':') + 1);
        }
        else if (arg.find("-test_name:") == 0) {
            test_file = arg.substr(arg.find(':') + 1);
        }
        else if (arg.find("-test_set:") == 0) {
            test_set = arg.substr(arg.find(':') + 1) == "True";
        }
        else if (arg.find("-split_ratio:") == 0) {
            split_ratio = std::stod(arg.substr(arg.find(':') + 1));
        }
        else if (arg.find("-split_ratio_validation_test:") == 0) {
            auto ratios = split_string(arg.substr(arg.find(':') + 1), ',');
            split_ratio_validation_test = { std::stod(ratios[0]), std::stod(ratios[1]) };
        }
    }

    if (folder_path.empty()) {
        std::cerr << "Error: -data_folder_path is required.\n";
        return 1;
    }

    try {
        split_files(folder_path, train_file, validation_file, test_file, test_set, split_ratio, split_ratio_validation_test);
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}


