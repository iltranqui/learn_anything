from faker import Faker

# Initialize the Faker generator
fake = Faker()

# Generate a list of 1000 random names
random_names = [fake.name() for _ in range(1000*1000)]

# Print the names (optional)
#for name in random_names:
    #print(name)

# Optionally, save to a file
with open("random_names.txt", "w") as file:
    for name in random_names:
        file.write(f"{name}\n")