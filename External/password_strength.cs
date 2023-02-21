using System;
using System.Text;

namespace PasswordGenerator
{
    class Program
    {
        static void Main(string[] args)
        {
            int passwordLength = 12;
            bool useLetters = true;
            bool useNumbers = true;
            bool useSpecialCharacters = true;

            string password = GeneratePassword(passwordLength, useLetters, useNumbers, useSpecialCharacters);
            Console.WriteLine("Generated password: " + password);

            int passwordStrengthScore = CalculatePasswordStrength(password);
            string passwordStrength = GetPasswordStrength(passwordStrengthScore);
            Console.WriteLine("Password strength: " + passwordStrength);
        }

        static string GeneratePassword(int length, bool useLetters, bool useNumbers, bool useSpecialCharacters)
        {
            const string letters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";
            const string numbers = "0123456789";
            const string specialCharacters = "!@#$%^&*()-_=+[]{}\\|;:'\",.<>/?";

            StringBuilder password = new StringBuilder();

            Random random = new Random();

            for (int i = 0; i < length; i++)
            {
                if (useLetters)
                {
                    password.Append(letters[random.Next(letters.Length)]);
                }

                if (useNumbers)
                {
                    password.Append(numbers[random.Next(numbers.Length)]);
                }

                if (useSpecialCharacters)
                {
                    password.Append(specialCharacters[random.Next(specialCharacters.Length)]);
                }
            }

            return password.ToString();
        }

        static int CalculatePasswordStrength(string password)
        {
            int score = 0;

            if (password.Length >= 8)
            {
                score++;
            }

            if (password.Length >= 12)
            {
                score++;
            }

            if (System.Text.RegularExpressions.Regex.IsMatch(password, @"[A-Z]"))
            {
                score++;
            }

            if (System.Text.RegularExpressions.Regex.IsMatch(password, @"[a-z]"))
            {
                score++;
            }

            if (System.Text.RegularExpressions.Regex.IsMatch(password, @"[0-9]"))
            {
                score++;
            }

            if (System.Text.RegularExpressions.Regex.IsMatch(password, @"[!@#$%^&*()_+=[{\]};:<>|./?,-]"))
            {
                score++;
            }

            return score;
        }

        static string GetPasswordStrength(int score)
        {
            switch (score)
            {
                case 0:
                    return "Very Weak";
                case 1:
                    return "Weak";
                case 2:
                    return "Fair";
                case 3:
                    return "Strong";
                case 4:
                    return "Very Strong";
                case 5:
                    return "Extremely Strong";
                default:
                    return "Unknown";
            }
        }
    }
}
