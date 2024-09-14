from data_struct.array_ds import array_access, array_search, array_insert, array_delete
from data_struct.queue_ds import queue_access, queue_search, queue_enqueue, queue_dequeue
from data_struct.stack_ds import stack_access, stack_search, stack_push, stack_pop
from data_struct.list_ds import list_access, list_search, list_insert, list_delete

# Step 1: Load random_names.txt file
def load_names(file_path):
    with open(file_path, "r") as file:
        names = file.read().splitlines()  # Read each line as an element in a list
    return names

# Step 2: Implement all function of the 

def list(file_path):
    names = load_names(file_path)
    # Access an element by index
    index = 2  # Access "Charlie"
    result = list_access(names, index)
    print(result)

    # Search for an element that exists
    name = "Bob"
    result = list_search(names, name)
    print(result)

    # Search for an element that doesn't exist
    name_not_in_list = "Zara"
    result = list_search(names, name_not_in_list)
    print(result)

    # Insert an element
    name = "John Doe"
    list_insert(names, name)

    # Delete an element
    name = "John Doe"
    #list_delete(names, name)

def queue(file_path):
    names = load_names(file_path)
    # Access an element by index
    index = 2  # Access "Charlie"
    result = queue_access(names, index)
    print(result)

    # Search for an element that exists
    name = "Bob"
    result = queue_search(names, name)
    print(result)

    # Search for an element that doesn't exist
    name_not_in_queue = "Zara"
    result = queue_search(names, name_not_in_queue)
    print(result)

    # Add an element to the queue
    name = "John Doe"
    queue_enqueue(names, name)

    # Remove an element from the front of the queue
    #queue_dequeue(names)

def stack(file_path):
    names = load_names(file_path)
    # Access an element by index
    index = 2  # Access "Charlie"
    result = stack_access(names, index)
    print(result)

    # Search for an element that exists
    name = "Bob"
    result = stack_search(names, name)
    print(result)

    # Search for an element that doesn't exist
    name_not_in_stack = "Zara"
    result = stack_search(names, name_not_in_stack)
    print(result)

    # Push an element onto the stack
    name = "John Doe"
    stack_push(names, name)

    # Pop an element from the top of the stack
    stack_pop(names)

def array(file_path):
    # Load names from file
    names = load_names(file_path)

    # Access an element by index
    index = 2  # Access "Charlie"
    result = array_access(names, index)
    print(result)

    # Search for an element that exists
    name = "Bob"
    result = array_search(names, name)
    print(result)

    # Insert an element
    name = "John Doe"
    array_insert(names, name)

    # Delete an element
    index = 2  # Delete "Charlie"
    array_delete(names, index)

def main():
# Load names from file
    file_path = 'random_names.txt'  # Update the path if necessary
    
    list(file_path)
    queue(file_path)
    stack(file_path)
    array(file_path)


if __name__ == "__main__":
    main()




