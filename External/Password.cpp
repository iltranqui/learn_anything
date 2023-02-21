// In this implementation, the generatePassword function takes the desired length of the password and boolean variables indicating
//  whether to include letters, numbers, and special characters. It uses the srand function to seed the random number generator 
// with the current time, and then defines sets of characters to use based on the input options. 
// It then adds characters to the password based on the options, making sure to include at least 2 letters, 
// 1 number, and/or 1 special character if those options are selected. Finally, it returns the generated password.

// The main function prompts the user for input, generates a password using the generatePassword function,
// and outputs the generated password.

#include <iostream>
#include <string>
#include <cstdlib>
#include <ctime>

using namespace std;

string generatePassword(int length, bool useLetters, bool useNumbers, bool useSpecial)
{
    string password;
    srand(time(NULL)); // Seed random number generator with current time

    // Define the sets of characters to use
    string letters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";
    string numbers = "0123456789";
    string specials = "!@#$%^&*()_-+=";

    // Add characters to the password based on the options
    for (int i = 0; i < length; i++)
    {
        if (useLetters && i < length - 2) // Make sure there's room for at least 2 digits/special characters
        {
            password += letters[rand() % letters.length()];
        }
        else if (useNumbers && i < length - 1) // Make sure there's room for at least 1 special character
        {
            password += numbers[rand() % numbers.length()];
        }
        else if (useSpecial)
        {
            password += specials[rand() % specials.length()];
        }
    }

    return password;
}

int main()
{
    int length;
    bool useLetters, useNumbers, useSpecial;

    // Get user input
    cout << "Enter the desired length of the password: ";
    cin >> length;
    cout << "Include letters? (0/1): ";
    cin >> useLetters;
    cout << "Include numbers? (0/1): ";
    cin >> useNumbers;
    cout << "Include special characters? (0/1): ";
    cin >> useSpecial;

    // Generate and output the password
    string password = generatePassword(length, useLetters, useNumbers, useSpecial);
    cout << "Your generated password is: " << password << endl;

    return 0;
}
