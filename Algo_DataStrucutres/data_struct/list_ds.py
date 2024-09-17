import random
import time

# Step 1: Load random_names.txt file
def load_names(file_path):
    with open(file_path, "r") as file:
        names = file.read().splitlines()  # Read each line as an element in a list
    return names

def list_access(names, index):
    assert isinstance(names, list), "names should be a list"
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
    print(f"Average time taken to ACCESS a name: {average_time:.5f} ms | Best time: {best_time:.5f} ms | Worst time: {worst_time:.5f} ms| Length: {len(names)}")
    return names, average_time

def list_search(names, name):
    assert isinstance(names, list), "names should be a list"
    total_time = 0
    best_time = float('inf')
    worst_time = float('-inf')
    for _ in range(100):
        start_time = time.time()
        result = name in names
        end_time = time.time()
        elapsed_time = (end_time - start_time) * 1000  # Convert to milliseconds
        total_time += elapsed_time
        if elapsed_time < best_time:
            best_time = elapsed_time
        if elapsed_time > worst_time:
            worst_time = elapsed_time
    average_time = total_time / 100
    print(f"Average time taken to SEARCH a name: {average_time:.5f} ms | Best time: {best_time:.5f} ms | Worst time: {worst_time:.5f} ms | Length: {len(names)}")
    return names, average_time

def list_insert(names, name):
    assert isinstance(names, list), "names should be a list"
    total_time = 0
    best_time = float('inf')
    worst_time = float('-inf')
    for _ in range(100):
        start_time = time.time()
        names.append(name)
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

def list_delete(names, name):
    assert isinstance(names, list), "names should be a list"
    total_time = 0
    best_time = float('inf')
    worst_time = float('-inf')
    for _ in range(100):
        start_time = time.time()
        names.remove(name)
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
    # load names in a list data structure
    file_path = "../random_names.txt"
    names = load_names(file_path)
    print(type(names))
    print(len(names))

    # Access Test
    index = random.randint(0, len(names) - 1)
    names, avg_access_time = list_access(names, index)
    print(f"Average access time: {avg_access_time:.5f} ms")

    # Search Test
    name = names[random.randint(0, len(names) - 1)]
    names, avg_search_time = list_search(names, name)
    print(f"Average search time: {avg_search_time:.5f} ms")

    # Insert Test
    name = "John Doe"
    names, avg_insert_time = list_insert(names, name)
    print(f"Average insert time: {avg_insert_time:.5f} ms")

    # Delete Test
    name = "John Doe"
    names, avg_delete_time = list_delete(names, name)
    print(f"Average delete time: {avg_delete_time:.5f} ms")

    


