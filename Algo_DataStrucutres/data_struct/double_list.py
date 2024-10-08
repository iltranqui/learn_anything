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


# Example usage of DoublyLinkedList
dll = DoublyLinkedList()
dll.insert_at_end(10)
dll.insert_at_end(20)
dll.insert_at_beginning(5)
dll.display_forward()
dll.delete(20)
dll.display_forward()
dll.display_backward()
