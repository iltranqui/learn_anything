import random
import time
from collections import deque  # Import deque for queue implementation


# Step 1: Load random_names.txt file
def load_names(file_path):
    with open(file_path, "r") as file:
        names = file.read().splitlines()  # Read each line as an element in a list
    return deque(names)  # Convert the list to a deque for queue behavior

def queue_access(queue, index):
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
    return result

def queue_search(queue, name):
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
    return result

def queue_enqueue(queue, name):
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
    return queue

def queue_dequeue(queue):
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
    return queue

if __name__ == "__main__":
    # Load names in a queue (implemented using deque)
    file_path = "../random_names.txt"
    queue = load_names(file_path)
    print(type(queue))
    print(len(queue))

    # Access Test
    index = random.randint(0, len(queue) - 1)
    queue_access(queue, index)

    # Search Test
    name = queue[random.randint(0, len(queue) - 1)]
    queue_search(queue, name)

    # Enqueue Test (add to the back of the queue)
    name = "John Doe"
    queue = queue_enqueue(queue, name)

    # Dequeue Test (remove from the front of the queue)
    queue = queue_dequeue(queue)
