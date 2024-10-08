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


# Example usage of SinglyLinkedList
sll = SinglyLinkedList()
sll.insert_at_end(10)
sll.insert_at_end(20)
sll.insert_at_beginning(5)
sll.display()
sll.delete(20)
sll.display()
