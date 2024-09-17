import random
import time



# Step 1: Load random_names.txt file
def load_names(file_path):
    with open(file_path, "r") as file:
        names = file.read().splitlines()  # Read each line as an element in a list
    return names

def stack_access(stack, index):
    assert isinstance(stack, list), "The stack should be a list"
    total_time = 0
    best_time = float('inf')
    worst_time = float('-inf')
    for _ in range(100):
        start_time = time.time()
        result = stack[index]  # Access element in the stack by index
        end_time = time.time()
        elapsed_time = (end_time - start_time) * 1000  # Convert to milliseconds
        total_time += elapsed_time
        if elapsed_time < best_time:
            best_time = elapsed_time
        if elapsed_time > worst_time:
            worst_time = elapsed_time
    average_time = total_time / 100
    print(f"Average time taken to ACCESS a name: {average_time:.5f} ms | Best time: {best_time:.5f} ms | Worst time: {worst_time:.5f} ms | Length: {len(stack)}")
    return stack, average_time

def stack_search(stack, name):
    assert isinstance(stack, list), "The stack should be a list"
    total_time = 0
    best_time = float('inf')
    worst_time = float('-inf')
    for _ in range(100):
        start_time = time.time()
        result = name in stack  # Search the stack
        end_time = time.time()
        elapsed_time = (end_time - start_time) * 1000  # Convert to milliseconds
        total_time += elapsed_time
        if elapsed_time < best_time:
            best_time = elapsed_time
        if elapsed_time > worst_time:
            worst_time = elapsed_time
    average_time = total_time / 100
    print(f"Average time taken to SEARCH a name: {average_time:.5f} ms | Best time: {best_time:.5f} ms | Worst time: {worst_time:.5f} ms | Length: {len(stack)}")
    return result, average_time

def stack_push(stack, name):
    assert isinstance(stack, list), "The stack should be a list"
    total_time = 0
    best_time = float('inf')
    worst_time = float('-inf')
    for _ in range(100):
        start_time = time.time()
        stack.append(name)  # Push onto the stack
        end_time = time.time()
        elapsed_time = (end_time - start_time) * 1000  # Convert to milliseconds
        total_time += elapsed_time
        if elapsed_time < best_time:
            best_time = elapsed_time
        if elapsed_time > worst_time:
            worst_time = elapsed_time
    average_time = total_time / 100
    print(f"Average time taken to PUSH a name: {average_time:.5f} ms | Best time: {best_time:.5f} ms | Worst time: {worst_time:.5f} ms | Length: {len(stack)}")
    return name, average_time

def stack_pop(stack):
    assert isinstance(stack, list), "The stack should be a list"
    total_time = 0
    best_time = float('inf')
    worst_time = float('-inf')
    for _ in range(100):
        start_time = time.time()
        name = stack.pop()  # Pop from the stack (LIFO)
        end_time = time.time()
        elapsed_time = (end_time - start_time) * 1000  # Convert to milliseconds
        total_time += elapsed_time
        if elapsed_time < best_time:
            best_time = elapsed_time
        if elapsed_time > worst_time:
            worst_time = elapsed_time
    average_time = total_time / 100
    print(f"Average time taken to POP a name: {average_time:.5f} ms | Best time: {best_time:.5f} ms | Worst time: {worst_time:.5f} ms | Length: {len(stack)}")
    return name, average_time

if __name__ == "__main__":
    # Load names in a stack (implemented using a list)
    file_path = "../random_names.txt"
    stack = load_names(file_path)
    print(type(stack))
    print(len(stack))

    # Access Test
    index = random.randint(0, len(stack) - 1)
    stack, access_avg_time = stack_access(stack, index)

    # Search Test
    name = stack[random.randint(0, len(stack) - 1)]
    stack, search_avg_time = stack_search(stack, name)

    # Push Test (equivalent to list insert in stack terms)
    name = "John Doe"
    stack, push_avg_time = stack_push(stack, name)

    # Pop Test (LIFO deletion)
    stack, pop_avg_time = stack_pop(stack)
