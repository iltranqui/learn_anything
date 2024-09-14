# Import necessary libraries
from collections import deque
from faker import Faker

# Step 1: Load random_names.txt file
def load_names(file_path):
    with open(file_path, "r") as file:
        names = file.read().splitlines()  # Read each line as an element in a list
    return names

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

# Step 4: Singly Linked List
class SinglyLinkedList:
    class Node:
        def __init__(self, data=None):
            self.data = data
            self.next = None

    def __init__(self):
        self.head = None

    def insert(self, value):
        new_node = self.Node(value)
        new_node.next = self.head
        self.head = new_node

# Step 5: Doubly Linked List
class DoublyLinkedList:
    class Node:
        def __init__(self, data=None):
            self.data = data
            self.prev = None
            self.next = None

    def __init__(self):
        self.head = None

    def insert(self, value):
        new_node = self.Node(value)
        new_node.next = self.head
        if self.head:
            self.head.prev = new_node
        self.head = new_node

class SkipList:
    class Node:
        def __init__(self, key, level):
            self.key = key
            self.forward = [None] * (level + 1)

    def __init__(self, max_level, p):
        self.max_level = max_level
        self.p = p
        self.header = self.create_node(0, max_level)
        self.level = 0

    def create_node(self, key, level):
        return self.Node(key, level)

    def insert_element(self, key):
        update = [None] * (self.max_level + 1)
        current = self.header

        for i in range(self.level, -1, -1):
            while current.forward[i] and current.forward[i].key < key:
                current = current.forward[i]
            update[i] = current

        current = current.forward[0]

        if current is None or current.key != key:
            rlevel = self.random_level()

            if rlevel > self.level:
                for i in range(self.level + 1, rlevel + 1):
                    update[i] = self.header
                self.level = rlevel

            n = self.create_node(key, rlevel)

            for i in range(rlevel + 1):
                n.forward[i] = update[i].forward[i]
                update[i].forward[i] = n

    def delete_element(self, key):
        update = [None] * (self.max_level + 1)
        current = self.header

        for i in range(self.level, -1, -1):
            while current.forward[i] and current.forward[i].key < key:
                current = current.forward[i]
            update[i] = current

        current = current.forward[0]

        if current is not None and current.key == key:
            for i in range(self.level + 1):
                if update[i].forward[i] != current:
                    break
                update[i].forward[i] = current.forward[i]

            while self.level > 0 and self.header.forward[self.level] is None:
                self.level -= 1



# Step 6: Binary Search Tree (BST)
class BinarySearchTree:
    class Node:
        def __init__(self, key):
            self.left = None
            self.right = None
            self.value = key

    def __init__(self):
        self.root = None

    def insert(self, value):
        if self.root is None:
            self.root = self.Node(value)
        else:
            self._insert(self.root, value)

    def _insert(self, root, value):
        if value < root.value:
            if root.left is None:
                root.left = self.Node(value)
            else:
                self._insert(root.left, value)
        else:
            if root.right is None:
                root.right = self.Node(value)
            else:
                self._insert(root.right, value)

# Step 7: Hash Table (using Python dictionary)
class HashTable:
    def __init__(self):
        self.table = {}

    def insert(self, key, value):
        self.table[key] = value

class CartesianTree:
    def __init__(self, root):
        self.root = root

class B_Tree:
    def __init__(self, root):
        self.root = root

class AVLTree:
    def __init__(self, root):
        self.root = root

class KD Tree:
    def __init__(self, root):
        self.root = root

class SplaingTree:
    def __init__(self, root):
        self.root = root

class RedBlackTree:
    def __init__(self, root):
        self.root = root