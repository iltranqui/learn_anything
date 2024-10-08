from data_struct.array_ds import array_access, array_search, array_insert, array_delete
from data_struct.queue_ds import queue_access, queue_search, queue_enqueue, queue_dequeue
from data_struct.stack_ds import stack_access, stack_search, stack_push, stack_pop
from data_struct.list_ds import list_access, list_search, list_insert, list_delete

from datastructures import SinglyLinkedList, DoublyLinkedList
from data_struct.single_list_ds import load_names_into_singly_linked_list, singly_linked_list_access, singly_linked_list_search, singly_linked_list_insert, singly_linked_list_delete
from data_struct.double_list_ds import load_names_into_doubly_linked_list, doubly_linked_list_access, doubly_linked_list_search, doubly_linked_list_insert, doubly_linked_list_delete
import tqdm

from tabulate import tabulate
from collections import deque
import numpy as np

import time

# Step 1: Load random_names.txt file
def load_names(file_path):
    with open(file_path, "r") as file:
        names = file.read().splitlines()  # Read each line as an element in a list
    return names

# Timing wrapper function to measure execution time
def time_operation(operation, *args):
    start_time = time.time()
    result = operation(*args)
    end_time = time.time()
    elapsed_time = (end_time - start_time) * 1000  # Convert to milliseconds
    return result, elapsed_time

# List operations
def list_operations(file_path):
    names = load_names(file_path)
    
    # Access
    _, access_time = time_operation(list_access, names, 2)
    
    # Search for an existing element
    _, search_time = time_operation(list_search, names, "Bob")
    
    # Insert an element
    _, insert_time = time_operation(list_insert, names, "John Doe")
    
    # Delete an element
    _, delete_time = time_operation(list_delete, names, "John Doe")
    
    return ["List", access_time, search_time, insert_time, delete_time]

# Queue operations
def queue_operations(file_path):
    names = deque(load_names(file_path))
    
    # Access
    _, access_time = time_operation(queue_access, names, 2)
    
    # Search for an existing element
    _, search_time = time_operation(queue_search, names, "Bob")
    
    # Enqueue (Insert)
    _, insert_time = time_operation(queue_enqueue, names, "John Doe")
    
    # Dequeue (Delete)
    _, delete_time = time_operation(queue_dequeue, names)
    
    return ["Queue", access_time, search_time, insert_time, delete_time]

# Stack operations
def stack_operations(file_path):
    names = load_names(file_path)
    
    # Access
    _, access_time = time_operation(stack_access, names, 2)
    
    # Search for an existing element
    _, search_time = time_operation(stack_search, names, "Bob")
    
    # Push (Insert)
    _, insert_time = time_operation(stack_push, names, "John Doe")
    
    # Pop (Delete)
    _, delete_time = time_operation(stack_pop, names)
    
    return ["Stack", access_time, search_time, insert_time, delete_time]

# Array operations
def array_operations(file_path):
    names = np.array(load_names(file_path))
    
    # Access
    _, access_time = time_operation(array_access, names, 2)
    
    # Search for an existing element
    _, search_time = time_operation(array_search, names, "Bob")
    
    # Insert an element
    _, insert_time = time_operation(array_insert, names, "John Doe")
    
    # Delete an element
    _, delete_time = time_operation(array_delete, names, 2)
    
    return ["Array", access_time, search_time, insert_time, delete_time]

def singly_linked_list_operations(file_path):
    sll = SinglyLinkedList()
    sll = load_names_into_singly_linked_list(file_path, sll)
    
    # Access
    _, access_time = time_operation(singly_linked_list_access, sll, 2)
    
    # Search for an existing element
    _, search_time = time_operation(singly_linked_list_search, sll, "Bob")
    
    # Insert an element
    _, insert_time = time_operation(singly_linked_list_insert, sll, "John Doe")

    # Delete an element
    _, delete_time = time_operation(singly_linked_list_delete, sll, "John Doe")
    
    return ["Singly Linked List", access_time, search_time, insert_time, delete_time]

def doubly_linked_list_operations(file_path):
    dll = DoublyLinkedList()
    dll = load_names_into_doubly_linked_list(file_path, dll)
    
    # Access
    _, access_time = time_operation(doubly_linked_list_access, dll, 2)
    
    # Search for an existing element
    _, search_time = time_operation(doubly_linked_list_search, dll, "Bob")
    
    # Insert an element
    _, insert_time = time_operation(doubly_linked_list_insert, dll, "John Doe")

    # Delete an element
    _, delete_time = time_operation(doubly_linked_list_delete, dll, "John Doe")
    
    return ["Doubly Linked List", access_time, search_time, insert_time, delete_time]

def main():
    file_path = 'random_names.txt'  # Path to your names file

    # Run the operations on all data structures
    print("Running operations on all data structures...")
    print("========= ARRAY =========")
    array_results = array_operations(file_path)
    print("========= STACK =========")
    stack_results = stack_operations(file_path)
    print("========= QUEUE =========")
    queue_results = queue_operations(file_path)
    print("========= LIST =========")
    list_results = list_operations(file_path)
    print("========= SINGLY LINKED LIST =========")
    singly_linked_list_results = singly_linked_list_operations(file_path)
    print("========= DOUBLY LINKED LIST =========")
    doubly_linked_list_results = doubly_linked_list_operations(file_path)
    print("========= RESULTS =========")
    # Combine all results
    results = [array_results, stack_results, queue_results, list_results, singly_linked_list_results, doubly_linked_list_results]

    # Define headers
    headers = ["Data Structure", "Access (ms)", "Search (ms)", "Insert (ms)", "Delete (ms)"]

    # Print the results in a table format
    print(tabulate(results, headers=headers, tablefmt="grid"))

if __name__ == "__main__":
    main()
