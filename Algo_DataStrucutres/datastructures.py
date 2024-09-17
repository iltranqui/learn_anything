# Import necessary libraries
from collections import deque

# Step 2: Stack (using Python list)
class Stack:
    def __init__(self):
        self.stack = []

    def push(self, value):
        self.stack.append(value)

    def pop(self):
        return self.stack.pop()

# Step 3: Queue (using deque)
class Queue:
    def __init__(self):
        self.queue = deque()

    def enqueue(self, value):
        self.queue.append(value)

    def dequeue(self):
        return self.queue.popleft()

class Node:
    def __init__(self, data):
        self.data = data  # The value stored in the node
        self.next = None  # Reference to the next node


class SinglyLinkedList:
    def __init__(self):
        self.head = None  # The first node of the list

    def is_empty(self):
        return self.head is None

    def insert_at_beginning(self, data):
        new_node = Node(data)
        new_node.next = self.head
        self.head = new_node

    def insert_at_end(self, data):
        new_node = Node(data)
        if self.is_empty():
            self.head = new_node
        else:
            current = self.head
            while current.next:
                current = current.next
            current.next = new_node

    def delete(self, data):
        if self.is_empty():
            print("List is empty.")
            return
        
        # If the node to delete is the head
        if self.head.data == data:
            self.head = self.head.next
            return

        # Search for the node to delete
        current = self.head
        while current.next and current.next.data != data:
            current = current.next
        
        # If node was found
        if current.next:
            current.next = current.next.next
        else:
            print(f"Node with data {data} not found.")

    def display(self):
        if self.is_empty():
            print("List is empty.")
            return
        current = self.head
        while current:
            print(current.data, end=" -> ")
            current = current.next
        print("None")

    def get(self, index):
        if self.is_empty():
            print("List is empty.")
            return None
        current = self.head
        count = 0
        while current:
            if count == index:
                return current.data
            count += 1
            current = current.next
        print(f"Index {index} out of range.")
        return None
    
    def search(self, data):
        if self.is_empty():
            print("List is empty.")
            return False
        current = self.head
        while current:
            if current.data == data:
                return True
            current = current.next
        return False
# Step 5: Doubly Linked List
class DoublyNode:
    def __init__(self, data):
        self.data = data  # The value stored in the node
        self.next = None  # Reference to the next node
        self.prev = None  # Reference to the previous node


class DoublyLinkedList:
    def __init__(self):
        self.head = None  # The first node of the list

    def is_empty(self):
        return self.head is None

    def insert_at_beginning(self, data):
        new_node = DoublyNode(data)
        if self.is_empty():
            self.head = new_node
        else:
            new_node.next = self.head
            self.head.prev = new_node
            self.head = new_node

    def insert_at_end(self, data):
        new_node = DoublyNode(data)
        if self.is_empty():
            self.head = new_node
        else:
            current = self.head
            while current.next:
                current = current.next
            current.next = new_node
            new_node.prev = current

    def delete(self, data):
        if self.is_empty():
            print("List is empty.")
            return
        
        # If the node to delete is the head
        if self.head.data == data:
            if self.head.next:
                self.head = self.head.next
                self.head.prev = None
            else:
                self.head = None
            return

        # Search for the node to delete
        current = self.head
        while current and current.data != data:
            current = current.next

        # If node was found
        if current:
            if current.next:
                current.next.prev = current.prev
            if current.prev:
                current.prev.next = current.next
        else:
            print(f"Node with data {data} not found.")

    def display_forward(self):
        if self.is_empty():
            print("List is empty.")
            return
        current = self.head
        while current:
            print(current.data, end=" -> ")
            current = current.next
        print("None")

    def display_backward(self):
        if self.is_empty():
            print("List is empty.")
            return
        current = self.head
        while current.next:
            current = current.next
        while current:
            print(current.data, end=" -> ")
            current = current.prev
        print("None")
    
    def get(self, index):
        if self.is_empty():
            print("List is empty.")
            return None
        current = self.head
        count = 0
        while current:
            if count == index:
                return current.data
            count += 1
            current = current.next
        print(f"Index {index} out of range.")
        return None

    def search(self, data):
        if self.is_empty():
            print("List is empty.")
            return False
        current = self.head
        while current:
            if current.data == data:
                return True
            current = current.next
        return False


