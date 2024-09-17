

def load_names_into_singly_linked_list(file_path, sll):
    """
    # Example usage:
    singly_linked_list = load_names_into_singly_linked_list("random_names.txt")
    singly_linked_list.display()  # Display the names in the list
    """
    with open(file_path, "r") as file:
        names = file.read().splitlines()  # Read each line as an element in a list
        for name in names:
            sll.insert_at_end(name)  # Insert each name at the end of the singly linked list
    return sll


import time
from tqdm import tqdm

def singly_linked_list_access(sll, index):
    total_time = 0
    best_time = float('inf')
    worst_time = float('-inf')

    for _ in tqdm(range(100), desc="Accessing elements"):
        start_time = time.time()
        result = sll.get(index)  # Assume get() returns the element at the given index
        end_time = time.time()

        elapsed_time = (end_time - start_time) * 1000  # Convert to milliseconds
        total_time += elapsed_time
        if elapsed_time < best_time:
            best_time = elapsed_time
        if elapsed_time > worst_time:
            worst_time = elapsed_time

    average_time = total_time / 100
    print(f"Average time taken to ACCESS element at index {index}: {average_time:.5f} ms | Best: {best_time:.5f} ms | Worst: {worst_time:.5f} ms")
    return result, average_time

def singly_linked_list_search(sll, value):
    total_time = 0
    best_time = float('inf')
    worst_time = float('-inf')

    for _ in tqdm(range(100), desc="Searching elements"):
        start_time = time.time()
        result = sll.search(value)  # Assume search() returns True if the value exists
        end_time = time.time()

        elapsed_time = (end_time - start_time) * 1000  # Convert to milliseconds
        total_time += elapsed_time
        if elapsed_time < best_time:
            best_time = elapsed_time
        if elapsed_time > worst_time:
            worst_time = elapsed_time

    average_time = total_time / 100
    print(f"Average time taken to SEARCH for {value}: {average_time:.5f} ms | Best: {best_time:.5f} ms | Worst: {worst_time:.5f} ms")
    return result, average_time

def singly_linked_list_insert(sll, value):
    total_time = 0
    best_time = float('inf')
    worst_time = float('-inf')

    for _ in tqdm(range(100), desc="Inserting elements"):
        start_time = time.time()
        sll.insert_at_end(value)  # Insert the value at the end of the list
        end_time = time.time()

        elapsed_time = (end_time - start_time) * 1000  # Convert to milliseconds
        total_time += elapsed_time
        if elapsed_time < best_time:
            best_time = elapsed_time
        if elapsed_time > worst_time:
            worst_time = elapsed_time

    average_time = total_time / 100
    print(f"Average time taken to INSERT {value}: {average_time:.5f} ms | Best: {best_time:.5f} ms | Worst: {worst_time:.5f} ms")

    return _, average_time

def singly_linked_list_delete(sll, value):
    total_time = 0
    best_time = float('inf')
    worst_time = float('-inf')

    for _ in tqdm(range(100), desc="Deleting elements"):
        start_time = time.time()
        sll.delete(value)  # Assume delete() removes the value from the list
        end_time = time.time()

        elapsed_time = (end_time - start_time) * 1000  # Convert to milliseconds
        total_time += elapsed_time
        if elapsed_time < best_time:
            best_time = elapsed_time
        if elapsed_time > worst_time:
            worst_time = elapsed_time

    average_time = total_time / 100
    print(f"Average time taken to DELETE {value}: {average_time:.5f} ms | Best: {best_time:.5f} ms | Worst: {worst_time:.5f} ms")

    return _, average_time
