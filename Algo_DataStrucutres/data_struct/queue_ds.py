import random
import time
from collections import deque  # Import deque for queue implementation


# Step 1: Load random_names.txt file
def load_names(file_path):
    with open(file_path, "r") as file:
        names = file.read().splitlines()  # Read each line as an element in a list
    return deque(names)  # Convert the list to a deque for queue behavior

def queue_access(queue, index):
    assert isinstance(queue, deque), "Queue must be of type deque"
    total_time = 0
    best_time = float('inf')
    worst_time = float('-inf')
    for _ in range(100):
        start_time = time.time()
        result = queue[index]  # Access element in the queue by index
        end_time = time.time()
        elapsed_time = (end_time - start_time) * 1000  # Convert to milliseconds
        total_time += elapsed_time
        if elapsed_time < best_time:
            best_time = elapsed_time
        if elapsed_time > worst_time:
            worst_time = elapsed_time
    average_time = total_time / 100
    print(f"Average time taken to ACCESS a name: {average_time:.5f} ms | Best time: {best_time:.5f} ms | Worst time: {worst_time:.5f} ms | Length: {len(queue)}")
    return result, average_time

def queue_search(queue, name):
    assert isinstance(queue, deque), "Queue must be of type deque"
    total_time = 0
    best_time = float('inf')
    worst_time = float('-inf')
    for _ in range(100):
        start_time = time.time()
        result = name in queue  # Search the queue
        end_time = time.time()
        elapsed_time = (end_time - start_time) * 1000  # Convert to milliseconds
        total_time += elapsed_time
        if elapsed_time < best_time:
            best_time = elapsed_time
        if elapsed_time > worst_time:
            worst_time = elapsed_time
    average_time = total_time / 100
    print(f"Average time taken to SEARCH a name: {average_time:.5f} ms | Best time: {best_time:.5f} ms | Worst time: {worst_time:.5f} ms | Length: {len(queue)}")
    return name, average_time

def queue_enqueue(queue, name):
    assert isinstance(queue, deque), "Queue must be of type deque"
    total_time = 0
    best_time = float('inf')
    worst_time = float('-inf')
    for _ in range(100):
        start_time = time.time()
        queue.append(name)  # Enqueue (add to the end of the queue)
        end_time = time.time()
        elapsed_time = (end_time - start_time) * 1000  # Convert to milliseconds
        total_time += elapsed_time
        if elapsed_time < best_time:
            best_time = elapsed_time
        if elapsed_time > worst_time:
            worst_time = elapsed_time
    average_time = total_time / 100
    print(f"Average time taken to ENQUEUE a name: {average_time:.5f} ms | Best time: {best_time:.5f} ms | Worst time: {worst_time:.5f} ms | Length: {len(queue)}")
    return name, average_time

def queue_dequeue(queue):
    assert isinstance(queue, deque), "Queue must be of type deque"
    total_time = 0
    best_time = float('inf')
    worst_time = float('-inf')
    for _ in range(100):
        start_time = time.time()
        name = queue.popleft()  # Dequeue (remove from the front of the queue)
        end_time = time.time()
        elapsed_time = (end_time - start_time) * 1000  # Convert to milliseconds
        total_time += elapsed_time
        if elapsed_time < best_time:
            best_time = elapsed_time
        if elapsed_time > worst_time:
            worst_time = elapsed_time
    average_time = total_time / 100
    print(f"Average time taken to DEQUEUE a name: {average_time:.5f} ms | Best time: {best_time:.5f} ms | Worst time: {worst_time:.5f} ms | Length: {len(queue)}")
    return name, average_time

if __name__ == "__main__":
    # Load names in a queue (implemented using deque)
    file_path = "../random_names.txt"
    queue = load_names(file_path)
    print(type(queue))
    print(len(queue))

    # Access Test
    index = random.randint(0, len(queue) - 1)
    names, avg_time_access = queue_access(queue, index)
    print(f"Average time for access: {avg_time_access:.5f} ms")

    # Search Test
    name = queue[random.randint(0, len(queue) - 1)]
    names, avg_time_search = queue_search(queue, name)
    print(f"Average time for search: {avg_time_search:.5f} ms")

    # Enqueue Test (add to the back of the queue)
    name = "John Doe"
    names, avg_time_enqueue = queue_enqueue(queue, name)
    print(f"Average time for enqueue: {avg_time_enqueue:.5f} ms")

    # Dequeue Test (remove from the front of the queue)
    names, avg_time_dequeue = queue_dequeue(queue)
    print(f"Average time for dequeue: {avg_time_dequeue:.5f} ms")
