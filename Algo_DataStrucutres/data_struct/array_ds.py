import random
import time
import numpy as np  # Import numpy for array handling

# Step 1: Load random_names.txt file
def load_names(file_path):
    with open(file_path, "r") as file:
        names = file.read().splitlines()  # Read each line as an element in a list
    return np.array(names)  # Convert the list to a numpy array

def array_access(names, index):
    assert isinstance(names, np.ndarray), "names should be a numpy array"
    total_time = 0
    best_time = float('inf')
    worst_time = float('-inf')
    for _ in range(100):
        start_time = time.time()
        result = names[index]
        end_time = time.time()
        elapsed_time = (end_time - start_time) * 1000  # Convert to milliseconds
        total_time += elapsed_time
        if elapsed_time < best_time:
            best_time = elapsed_time
        if elapsed_time > worst_time:
            worst_time = elapsed_time
    average_time = total_time / 100
    print(f"Average time taken to ACCESS a name: {average_time:.5f} ms | Best time: {best_time:.5f} ms | Worst time: {worst_time:.5f} ms | Length: {len(names)}")
    return result

def array_search(names, name):
    assert isinstance(names, np.ndarray), "names should be a numpy array"
    total_time = 0
    best_time = float('inf')
    worst_time = float('-inf')
    for _ in range(100):
        start_time = time.time()
        result = np.isin(name, names)  # Use numpy's isin function
        end_time = time.time()
        elapsed_time = (end_time - start_time) * 1000  # Convert to milliseconds
        total_time += elapsed_time
        if elapsed_time < best_time:
            best_time = elapsed_time
        if elapsed_time > worst_time:
            worst_time = elapsed_time
    average_time = total_time / 100
    print(f"Average time taken to SEARCH a name: {average_time:.5f} ms | Best time: {best_time:.5f} ms | Worst time: {worst_time:.5f} ms | Length: {len(names)}")
    return result, average_time

def array_insert(names, name):
    assert isinstance(names, np.ndarray), "names should be a numpy array"
    total_time = 0
    best_time = float('inf')
    worst_time = float('-inf')
    for _ in range(100):
        start_time = time.time()
        names = np.append(names, name)  # Use numpy's append
        end_time = time.time()
        elapsed_time = (end_time - start_time) * 1000  # Convert to milliseconds
        total_time += elapsed_time
        if elapsed_time < best_time:
            best_time = elapsed_time
        if elapsed_time > worst_time:
            worst_time = elapsed_time
    average_time = total_time / 100
    print(f"Average time taken to INSERT a name: {average_time:.5f} ms | Best time: {best_time:.5f} ms | Worst time: {worst_time:.5f} ms | Length: {len(names)}")
    return names, average_time

def array_delete(names, name):
    assert isinstance(names, np.ndarray), "names should be a numpy array"
    total_time = 0
    best_time = float('inf')
    worst_time = float('-inf')
    for _ in range(100):
        start_time = time.time()
        names = np.delete(names, np.where(names == name))  # Use numpy's delete and where
        end_time = time.time()
        elapsed_time = (end_time - start_time) * 1000  # Convert to milliseconds
        total_time += elapsed_time
        if elapsed_time < best_time:
            best_time = elapsed_time
        if elapsed_time > worst_time:
            worst_time = elapsed_time
    average_time = total_time / 100
    print(f"Average time taken to DELETE a name: {average_time:.5f} ms | Best time: {best_time:.5f} ms | Worst time: {worst_time:.5f} ms | Length: {len(names)}")
    return names, average_time

if __name__ == "__main__":
    # Load names in a numpy array data structure
    file_path = "../random_names.txt"
    names = load_names(file_path)
    print(type(names))
    print(len(names))

    # Access Test
    index = random.randint(0, len(names) - 1)
    array_access(names, index)

    # Search Test
    name = names[random.randint(0, len(names) - 1)]
    array_search(names, name)

    # Insert Test
    name = "John Doe"
    names, _ = array_insert(names, name)

    # Delete Test
    name = "John Doe"
    names, _ = array_delete(names, name)
