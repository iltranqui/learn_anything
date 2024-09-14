from datastructures import SinglyLinkedList, DoublyLinkedList, BinarySearchTree, HashTable, Stack, Queue
import time 
import random

def time_access(array, stack, queue, singly_linked_list, doubly_linked_list, bst, hash_table):

    def access_array(array, index):
        start_time = time.time()
        result = array[index]
        end_time = time.time()
        return result, end_time - start_time
    
    def access_stack(stack, index):
        start_time = time.time()
        result = stack.stack[index]
        end_time = time.time()
        return result, end_time - start_time
    
    def access_queue(queue, index):
        start_time = time.time()
        result = queue.queue[index]
        end_time = time.time()
        return result, end_time - start_time
    
    def access_singly_linked_list(singly_linked_list, index):
        start_time = time.time()
        current = singly_linked_list.head
        for i in range(index):
            current = current.next
        result = current.data
        end_time = time.time()
        return result, end_time - start_time
    
    def access_doubly_linked_list(doubly_linked_list, index):
        start_time = time.time()
        current = doubly_linked_list.head
        for i in range(index):
            current = current.next
        result = current.data
        end_time = time.time()
        return result, end_time - start_time
    
    def access_bst(bst, index):
        start_time = time.time()
        result = bst.search(index)
        end_time = time.time()
        return result, end_time - start_time
    
    def access_hash_table(hash_table, index):
        start_time = time.time()
        result = hash_table.search(index)
        end_time = time.time()
        return result, end_time - start_time
    
    # Access each data structure at a random index
    index = random.randint(0, len(array) - 1)
    array_result, array_time = access_array(array, index)
    stack_result, stack_time = access_stack(stack, index)
    queue_result, queue_time = access_queue(queue, index)
    singly_linked_list_result, singly_linked_list_time = access_singly_linked_list(singly_linked_list, index)
    doubly_linked_list_result, doubly_linked_list_time = access_doubly_linked_list(doubly_linked_list, index)
    bst_result, bst_time = access_bst(bst, index)
    hash_table_result, hash_table_time = access_hash_table(hash_table, index)

