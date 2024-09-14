from data_struct.queue_ds import load_names, queue_access, queue_search, queue_enqueue, queue_dequeue
from data_struct.list_ds import list_access, list_search, list_insert, list_delete
from data_struct.stack_ds import stack_access, stack_search, stack_push, stack_pop
from data_struct.array_ds import array_access, array_search, array_insert, array_delete

import pytest
from collections import deque
import random

# Assume the following functions are imported from the previous queue implementation
# - load_names
# - queue_access
# - queue_search
# - queue_enqueue
# - queue_dequeue

@pytest.fixture
def sample_queue():
    # A fixture that provides a queue for each test
    return deque(["Alice", "Bob", "Charlie", "David", "Eve"])

def test_access(sample_queue):
    # Test accessing an element by index
    index = 2  # Access "Charlie"
    result = queue_access(sample_queue, index)
    assert result == "Charlie", "Access did not return the expected result."

def test_search_found(sample_queue):
    # Test searching for an element that exists
    name = "Bob"
    result = queue_search(sample_queue, name)
    assert result, "Search did not find the expected element."

def test_search_not_found(sample_queue):
    # Test searching for an element that doesn't exist
    name_not_in_queue = "Zara"
    result = queue_search(sample_queue, name_not_in_queue)
    assert not result, "Search incorrectly found an element that isn't in the queue."

def test_enqueue(sample_queue):
    # Test adding an element to the queue
    name = "John Doe"
    queue_before = len(sample_queue)
    queue_after = queue_enqueue(sample_queue, name)
    assert len(queue_after) == queue_before + 1, "pEnqueue did not add the element correctly."
    assert queue_after[-1] == name, "Enqueue did not add the element at the end of the queue."

def test_dequeue(sample_queue):
    # Test removing an element from the front of the queue
    first_in_queue = sample_queue[0]
    queue_before = len(sample_queue)
    queue_after = queue_dequeue(sample_queue)
    assert len(queue_after) == queue_before - 1, "Dequeue did not remove the element correctly."
    assert first_in_queue not in queue_after, "Dequeue did not remove the first element correctly."

def test_empty_dequeue():
    # Test dequeuing from an empty queue
    empty_queue = deque()
    with pytest.raises(IndexError, match="pop from an empty deque"):
        queue_dequeue(empty_queue)


@pytest.fixture
def sample_names():
    """Fixture that provides a mock list of names."""
    return ["Alice", "Bob", "Charlie", "David", "Eve"]


def test_list_access(sample_names):
    """Test accessing an element from the list."""
    index = 2  # Access "Charlie"
    result = list_access(sample_names, index)
    assert result == "Charlie", "Access did not return the expected result."

def test_list_search_found(sample_names):
    """Test searching for an element that exists in the list."""
    name = "Bob"
    result = list_search(sample_names, name)
    assert result is True, "Search did not find the expected element."

def test_list_search_not_found(sample_names):
    """Test searching for an element that doesn't exist in the list."""
    name_not_in_list = "Zara"
    result = list_search(sample_names, name_not_in_list)
    assert result is False, "Search incorrectly found an element not in the list."

def test_list_insert(sample_names):
    """Test inserting an element into the list."""
    name = "John Doe"
    list_before = len(sample_names)
    names_after = list_insert(sample_names, name)
    assert len(names_after) == list_before + 1, "Insert did not add the element correctly."
    assert names_after[-1] == name, "Insert did not add the element at the end of the list."

def test_list_delete(sample_names):
    """Test deleting an element from the list."""
    name = "Alice"
    names_before = len(sample_names)
    names_after = list_delete(sample_names, name)
    assert len(names_after) == names_before - 1, "Delete did not remove the element correctly."
    assert name not in names_after, "Delete did not remove the correct element."

def test_list_delete_not_found(sample_names):
    """Test deleting a name not found in the list, ensuring it raises a ValueError."""
    with pytest.raises(ValueError, match="list.remove(x): x not in list"):
        list_delete(sample_names, "NonExistentName")

