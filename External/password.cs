using System;

namespace PasswordGenerator
{
    class Program
    {
        static void Main(string[] args)
        {
            // Get user input
            Console.Write("Enter the desired length of the password: ");
            int length = int.Parse(Console.ReadLine());
            Console.Write("Include letters? (Y/N): ");
            bool useLetters = Console.ReadLine().ToUpper() == "Y";
            Console.Write("Include numbers? (Y/N): ");
            bool useNumbers = Console.ReadLine().ToUpper() == "Y";
            Console.Write("Include special characters? (Y/N): ");
            bool useSpecial = Console.ReadLine().ToUpper() == "Y";

            // Generate and output the password
            string password = GeneratePassword(length, useLetters, useNumbers, useSpecial);
            Console.WriteLine("Your generated password is: " + password);
        }

        static string GeneratePassword(int length, bool useLetters, bool useNumbers, bool useSpecial)
        {
            string password = "";
            Random rand = new Random();

            // Define the sets of characters to use
            string letters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";
            string numbers = "0123456789";
            string specials = "!@#$%^&*()_-+=";

            // Add characters to the password based on the options
            for (int i = 0; i < length; i++)
            {
                if (useLetters && i < length - 2) // Make sure there's room for at least 2 digits/special characters
                {
                    password += letters[rand.Next(letters.Length)];
                }
                else if (useNumbers && i < length - 1) // Make sure there's room for at least 1 special character
                {
                    password += numbers[rand.Next(numbers.Length)];
                }
                else if (useSpecial)
                {
                    password += specials[rand.Next(specials.Length)];
                }
            }

            return password;
        }
    }
}
