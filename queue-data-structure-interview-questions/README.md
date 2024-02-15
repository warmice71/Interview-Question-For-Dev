# 55 Fundamental Queue Data Structure Interview Questions

<div>
<p align="center">
<a href="https://devinterview.io/questions/data-structures-and-algorithms/">
<img src="https://firebasestorage.googleapis.com/v0/b/dev-stack-app.appspot.com/o/github-blog-img%2Fdata-structures-and-algorithms-github-img.jpg?alt=media&token=fa19cf0c-ed41-4954-ae0d-d4533b071bc6" alt="data-structures-and-algorithms" width="100%">
</a>
</p>

#### You can also find all 55 answers here 👉 [Devinterview.io - Queue Data Structure](https://devinterview.io/questions/data-structures-and-algorithms/queue-data-structure-interview-questions)

<br>

## 1. What is a _Queue_?

A **queue** is a data structure that adheres to the **First-In-First-Out (FIFO)** principle and is designed to hold a collection of elements.

### Core Operations

-  **Enqueue**: Adding an element to the end of the queue.
-  **Dequeue**: Removing an element from the front of the queue.
- **IsEmpty**: Checks if the queue is empty.
- **IsFull**: Checks if the queue has reached its capacity.
- **Peek**: Views the front element without removal.

All operations have a space complexity of $O(1)$ and time complexity of $O(1)$, except for **Search**, which has $O(n)$ time complexity.

### Key Characteristics

1. **Order**: Maintains the order of elements according to their arrival time.
2. **Size**: Can be either bounded (fixed size) or unbounded (dynamic size).
3. **Accessibility**: Typically provides only restricted access to elements in front and at the rear.
4. **Time Complexity**: The time required to perform enqueue and dequeue is usually $O(1)$.

### Visual Representation

![Queue](https://firebasestorage.googleapis.com/v0/b/dev-stack-app.appspot.com/o/queues%2Fsimple-queue.svg?alt=media&token=976157b7-407b-4523-b8df-61f79bc1fafc)

### Real-World Examples

-  **Ticket Counter**: People form a queue, and the first person who joined the queue gets the ticket first.
-  **Printer Queue**: Print jobs are processed in the order they were sent to the printer.

### Practical Applications

1. **Task Scheduling**: Used by operating systems for managing processes ready to execute or awaiting specific events.
2. **Handling of Requests**: Servers in multi-threaded environments queue multiple user requests, processing them in arrival order.
3. **Data Buffering**: Supports asynchronous data transfers between processes, such as in IO buffers and pipes.
4. **Breadth-First Search**: Employed in graph algorithms, like BFS, to manage nodes for exploration.
5. **Order Processing**: E-commerce platforms queue customer orders for processing.
6. **Call Center Systems**: Incoming calls wait in a queue before connecting to the next available representative.

### Code Example: Queue

Here is the Python code:

```python
from collections import deque

class Queue:
    def __init__(self):
        self.queue = deque()

    def enqueue(self, item):
        self.queue.append(item)

    def dequeue(self):
        if not self.is_empty():
            return self.queue.popleft()
        raise Exception("Queue is empty.")

    def size(self):
        return len(self.queue)

    def is_empty(self):
        return len(self.queue) == 0

    def front(self):
        if not self.is_empty():
            return self.queue[0]
        raise Exception("Queue is empty.")

    def rear(self):
        if not self.is_empty():
            return self.queue[-1]
        raise Exception("Queue is empty.")

# Example Usage
q = Queue()
q.enqueue(5)
q.enqueue(6)
q.enqueue(3)
q.enqueue(2)
q.enqueue(7)
print("Queue:", list(q.queue))
print("Front:", q.front())
print("Rear:", q.rear())
q.dequeue()
print("After dequeue:", list(q.queue))
```
<br>

## 2. Explain the _FIFO (First In, First Out)_ policy that characterizes a _Queue_.

The **FIFO (First-In-First-Out)** policy governs the way **Queues** handle their elements. Elements are processed and removed from the queue in the same order in which they were added. The data structures responsible for adhering to this policy are specifically designed to optimize for this principle, making them ideal for a host of real-world applications.

### Core Mechanism

Elements are typically added to the **rear** and removed from the **front**. This design choice ensures that the earliest elements, those closest to the front, are processed and eliminated first.

### Fundamental Operations

1. **Enqueue (Add)**: New elements are positioned at the rear end.
2. **Dequeue (Remove)**: Front element is removed from the queue.

In the above diagram:

- **Front**: Pointing to the element about to be dequeued.
- **Rear**: Position where new elements will be enqueued.
<br>

## 3. Name some _Types of Queues_.

**Queues** are adaptable data structures with diverse types, each optimized for specific tasks. Let's explore the different forms of queues and their functionalities.

### Simple Queue

A **Simple Queue** follows the basic **FIFO** principle. This means items are added at the end and removed from the beginning.

#### Visual Representation

![Simple Queue](https://firebasestorage.googleapis.com/v0/b/dev-stack-app.appspot.com/o/queues%2Fsimple-queue.svg?alt=media&token=976157b7-407b-4523-b8df-61f79bc1fafc)

#### Implementation

Here is the Python code:

```python
class SimpleQueue:
    def __init__(self):
        self.queue = []
    
    def enqueue(self, item):
        self.queue.append(item)
    
    def dequeue(self):
        if not self.is_empty():
            return self.queue.pop(0)
    
    def is_empty(self):
        return len(self.queue) == 0
    
    def size(self):
        return len(self.queue)
```

### Circular Queue

In a **Circular Queue** the last element points to the first element, making a circular link.  This structure uses a **fixed-size array** and can wrap around upon reaching the end. It's more **memory efficient** than a Simple Queue, reusing positions at the front that are left empty by the dequeue operations.

#### Visual Representation

![Circular Queue](https://firebasestorage.googleapis.com/v0/b/dev-stack-app.appspot.com/o/queues%2Fcircular-queue.svg?alt=media&token=31e68b64-92a8-45f7-84c4-0ba8eb929ee8)

#### Implementation

Here is the Python code:

```python
class CircularQueue:
    def __init__(self, k):
        self.queue = [None] * k
        self.size = k
        self.front = self.rear = -1
    
    def enqueue(self, item):
        if self.is_full():
            return "Queue is full"
        elif self.is_empty():
            self.front = self.rear = 0
        else:
            self.rear = (self.rear + 1) % self.size
        self.queue[self.rear] = item
    
    def dequeue(self):
        if self.is_empty():
            return "Queue is empty"
        elif self.front == self.rear:
            temp = self.queue[self.front]
            self.front = self.rear = -1
            return temp
        else:
            temp = self.queue[self.front]
            self.front = (self.front + 1) % self.size
            return temp
    
    def is_empty(self):
        return self.front == -1
    
    def is_full(self):
        return (self.rear + 1) % self.size == self.front
```

### Priority Queue

A **Priority Queue** gives each item a priority. Items with higher priorities are dequeued before those with lower priorities. This is useful in scenarios like task scheduling where some tasks need to be processed before others.

#### Visual Representation

![Priority Queue](https://firebasestorage.googleapis.com/v0/b/dev-stack-app.appspot.com/o/queues%2Fpriority-queue.svg?alt=media&token=df8214e1-9ba6-4aaf-9f79-bd5334a234af)

#### Implementation

Here is the Python code:

```python
class PriorityQueue:
    def __init__(self):
        self.queue = []
    
    def enqueue(self, item, priority):
        self.queue.append((item, priority))
        self.queue.sort(key=lambda x: x[1], reverse=True)
    
    def dequeue(self):
        if not self.is_empty():
            return self.queue.pop(0)[0]
    
    def is_empty(self):
        return len(self.queue) == 0
```

### Double-Ended Queue (De-queue)

A **Double-Ended Queue** allows items to be added or removed from both ends, giving it **more flexibility** compared to a simple queue.

#### Visual Representation

![De-queue](https://firebasestorage.googleapis.com/v0/b/dev-stack-app.appspot.com/o/queues%2Fdouble-ended.svg?alt=media&token=acca0731-3dd3-4af0-bbad-86739fb0506a)

#### Implementation

Here is the Python code:

```python
from collections import deque

de_queue = deque()
de_queue.append(1)  # Add to rear
de_queue.appendleft(2)  # Add to front
de_queue.pop()  # Remove from rear
de_queue.popleft()  # Remove from front
```

### Input-Restricted Deque and Output-Restricted Deque

An **Input-Restricted Deque** only allows items to be added at one end, while an **Output-Restricted Deque** limits removals to one end.

#### Visual Representation

**Input-Restricted Deque**

![Input-Restricted Deque](https://firebasestorage.googleapis.com/v0/b/dev-stack-app.appspot.com/o/queues%2Fimput-restricted-deque.svg?alt=media&token=6ca0c591-ed53-40be-b410-3aea8fa519b0)

**Output-Restricted Deque**

![Output-Restricted Deque](https://firebasestorage.googleapis.com/v0/b/dev-stack-app.appspot.com/o/queues%2Foutput-restricted-deque.svg?alt=media&token=d304afbd-5ac3-47bb-b58e-759648c0761d)
<br>

## 4. What is a _Priority Queue_ and how does it differ from a standard _Queue_?

**Queues** are data structures that follow a **FIFO** (First-In, First-Out) order, where elements are removed in the same sequence they were added.

**Priority Queues**, on the other hand, are more dynamic and cater to elements with **varying priorities**. A key distinction is that while queues prioritize the order in which items are processed, a priority queue dictates the sequence based on the priority assigned to each element.

### Core Differences

- **Order:** Queues ensure a consistent, predefined processing sequence, whereas priority queues handle items based on their assigned priority levels.
  
- **Elements Removal:** Queues remove the oldest element, while priority queues remove the highest-priority item. This results in a different set of elements being dequeued in each case.

- **Queues**: 1, 2, 3, 4, 5
- **Priority Queue**: (assuming '4' has the highest priority): 4, 2, 6, 3, 1

- **Support Functions**: Since queues rely on a standard FIFO flow, they present standard methods like `enqueue` and `dequeue`. In contrast, priority queues offer means to set priorities and locate/query elements based on their priority levels.

### Implementation Methods

#### Array List
   - **Queues**: Direct support.
   - **Priority Queues**: Manage elements to sustain a specific order.

#### Linked List
   - **Queues**: Convenient for dynamic sizing and additions.
   - **Priority Queues**: Manual management of element ordering.

#### Binary Trees
   - **Queues**: Not common but a viable approach using structures like Heaps.

   - **Priority Queues**: Specifically utilized for priority queues to ensure efficient operations based on priorities.

4. **Hash Tables**
   - **Queues**: Suitable for more sophisticated, fine-tuned queues.
   - **Priority Queues**: Can be combined with other structures for varied implementations.

### Typical Use-Cases

- **Queues**: Appropriate for scenarios where "**first come, first serve**" is fundamental, such as in printing tasks or handling multiple requests.
- **Priority Queues**: More Suitable for contexts that require managing and completing tasks in an "**order of urgency**" or "**order of importance**", like in real-time systems, traffic routing, or resource allocation.
<br>

## 5. When should I use a _Stack_ or a _Queue_ instead of _Arrays/Lists_?

**Queues** and **Stacks** provide structured ways to handle data, offering distinct advantages over more generic structures like **Lists** or **Arrays**.

### Key Features

#### Queues 

- **Characteristic**: First-In-First-Out (FIFO)
- **Usage**: Ideal for ordered processing, such as print queues or BFS traversal.

#### Stacks

- **Characteristic**: Last-In-First-Out (LIFO)
- **Usage**: Perfect for tasks requiring reverse order like undo actions or DFS traversal.

#### Lists/Arrays

- **Characteristic**: Random Access
- **Usage**: Suitable when you need random access to elements or don't require strict order or data management.
<br>

## 6. How do you reverse a _Queue_?

Reversing a queue can be accomplished **using a single stack** or **recursively**. Both methods ensure the first element in the input queue becomes the last in the resultant queue. 

### Single-Stack Method

Here are the steps:

1. **Transfer Input to Stack**: While the input queue isn't empty, **dequeue** elements and **push** them to the stack.
2. **Transfer Stack to Output**: Then, **pop** elements from the stack and **enqueue** them back to the queue. This reverses their order.

### Code Example: Reversing a Queue with a Stack

Here is the Python code:

```python
def reverse_queue(q):
    if not q:  # Base case: queue is empty
        return
    stack = []
    while q:
        stack.append(q.pop(0))  # Transfer queue to stack
    while stack:
        q.append(stack.pop())  # Transfer stack back to queue
    return q

# Test
q = [1, 2, 3, 4, 5]
print(f"Original queue: {q}")
reverse_queue(q)
print(f"Reversed queue: {q}")
```

### Complexity Analysis

- **Time Complexity**: $O(n)$ as it involves one pass through both the queue and the stack for a queue of size $n$.
- **Space Complexity**: $O(n)$ - $n$ space is used to store the elements in the stack.

### Using Recursion 

To reverse a queue **recursively**, you can follow this approach:

1. **Base Case**: If the queue is empty, stop.
2. **Recurse**: Call the reverse function recursively until all elements are dequeued.
3. **Enqueue Last Element**: For each item being dequeued, enqueue it back into the queue after the recursion bottoms out, effectively reversing the order.

### Code Example: Reversing a Queue Recursively

Here is the Python code:

```python
def reverse_queue_recursively(q):
    if not q:
        return
    front = q.pop(0)  # Dequeue the first element
    reverse_queue_recursively(q)  # Recurse for the remaining queue
    q.append(front)  # Enqueue the previously dequeued element at the end
    return q

# Test
q = [1, 2, 3, 4, 5]
print(f"Original queue: {q}")
reverse_queue_recursively(q)
print(f"Reversed queue: {q}")
```

### Complexity Analysis

- **Time Complexity**: $O(n^2)$ - this is because each dequeue operation on the queue in the recursion stack is an $O(n)$ operation, and these operations occur in sequence for a queue of size $n$. Therefore, we get $n + (n-1) + \ldots + 1 = \frac{n(n+1)}{2}$ in the worst case. While this can technically be represented as $O(n^2)$, in practical scenarios for small queues, it can have a time complexity of $O(n)$.
- **Space Complexity**: $O(n)$ - $n$ depth comes from the recursion stack for a queue of $n$ elements
<br>

## 7. Can a queue be implemented as a static data structure and if so, what are the limitations?

**Static queues** use a pre-defined amount of memory, typically an array, for efficient FIFO data handling.

### Limitations of Static Queues

1. **Fixed Capacity**: A static queue cannot dynamically adjust its size based on data volume or system requirements. As a result, it can become either underutilized or incapable of accommodating additional items.

2. **Memory Fragmentation**: If there's not enough contiguous memory to support queue expansion or changes, memory fragmentation occurs. This means that even if there's available memory in the system, it may not be usable by the static queue.

    Memory fragmentation is more likely in long-running systems or when the queue has a high rate of enqueueing and dequeueing due to the "moving window" of occupied and freed space.

3. **Potential for Data Loss**: Enqueuing an item into a full static queue results in data loss. As there's no mechanism to signify that storage was exhausted, it's essential to maintain  methods to keep track of the queue's status.

4. **Time-Consuming Expansion**: If the queue were to support expansion, it would require operations in $O(n)$ time - linear with the current size of the queue. This computational complexity is a significant downside compared to the $O(1)$ time complexity offered by dynamic queues.

5. **Inefficient Memory Usage**: A static queue reserved a set amount of memory for its potential max size, which can be a wasteful use of resources if the queue seldom reaches that max size.
<br>

## 8. Write an algorithm to _enqueue_ and _dequeue_ an item from a _Queue_.

### Problem Statement

The task is to write an algorithm to perform both **enqueue** (add an item) and **dequeue** (remove an item) operations on a **queue**.

### Solution

A Queue, often used in real-world scenarios with first-in, first-out (FIFO) logic, can be implemented using an array (for fixed-size) or linked list (for dynamic size).

#### Algorithm Steps

1. **Enqueue Operation**: Add an item at the `rear` of the queue.
2. **Dequeue Operation**: Remove the item at the `front` of the queue.

#### Implementation

Here is the Python code:

```python
class Queue:
    def __init__(self):
        self.items = []

    def enqueue(self, item):
        self.items.append(item)
        
    def dequeue(self):
        if not self.is_empty():
            return self.items.pop(0)
        return "Queue is empty"

    def is_empty(self):
        return self.items == []

    def size(self):
        return len(self.items)

# Example
q = Queue()
q.enqueue(2)
q.enqueue(4)
q.enqueue(6)
print("Dequeued:", q.dequeue())  # Output: Dequeued: 2
```

In this Python implementation, the `enqueue` operation has a time complexity of $O(1)$ while the `dequeue` operation has a time complexity of $O(n)$.
<br>

## 9. How to implement a _Queue_ such that _enqueue_ has _O(1)_ and _dequeue_ has _O(n)_ complexity?

One way to achieve **$O(1)$** enqueue and **$O(n)$** dequeue times is with a **linked-list**. You can keep the tail pointer always. Enqueues don't need to perform more than a couple of cheap link operations.

However, dequeue operations on a single-ended list are costly, potentially traversing the whole list. To keep dequeue times acceptable, you might want to limit the number of elements you enqueue before you're allowed to dequeue elements.  You could define a fixed size for the list e.g. 100 or 1000, and after this limit, you would allow dequeueing. The key is to ensure the amortized time for the last operation is still $O(1)$ this way.

### Python Example

Here is a Python code:

```python
class Node:
    def __init__(self, data=None):
        self.data = data
        self.next = None

class LimitedQueue:
    def __init__(self, max_size):
        self.head = None
        self.tail = None
        self.max_size = max_size
        self.count = 0

    def enqueue(self, data):
        if self.count < self.max_size:
            new_node = Node(data)
            if not self.head:
                self.head = new_node
            else:
                self.tail.next = new_node
            self.tail = new_node
            self.count += 1
        else:
            print("Queue is full. Dequeue before adding more.")

    def dequeue(self):
        if self.head:
            data = self.head.data
            self.head = self.head.next
            self.count -= 1
            if self.count == 0:
                self.tail = None
            return data
        else:
            print("Queue is empty. Nothing to dequeue.")

    def display(self):
        current = self.head
        while current:
            print(current.data, end=" ")
            current = current.next
        print()

# Let's test the Queue
limited_queue = LimitedQueue(3)
limited_queue.enqueue(10)
limited_queue.enqueue(20)
limited_queue.enqueue(30)
limited_queue.display()  # Should display 10 20 30
limited_queue.enqueue(40)  # Should display 'Queue is full. Dequeue before adding more.'
limited_queue.dequeue()
limited_queue.display()  # Should display 20 30
```
<br>

## 10. Discuss a scenario where _dequeue_ must be prioritized over _enqueue_ in terms of complexity.

While **enqueue** typically takes $O(1)$ time and **dequeue** $O(1)$ or $O(n)$ time in a simple Queue, there are cases, like in **stacks**, where we prioritize one operation over the other.

In most traditional Queue implementations, **enqueue and dequeue** operate in $O(1)$ time. 

However, you can design special queues, like **priority queues**, where one operation is optimized at the cost of the other. For instance, if you're using a **binary heap**.

### Binary Heap & Deque Efficiency

The efficiency of both **enqueue** and **dequeue** is constrained by the binary heap's structure. A binary heap can be represented as a binary tree.

In a **complete** binary tree, most levels are fully occupied, and the last level is either partially or fully occupied from the left.

When the binary heap is visualized with the root at the top, the following rules are typically followed:

- **Maximum Number of Children**: All nodes, except the ones at the last level, have exactly two children.
- **Possible Lopsided Structure in the Last Level**: The last level, if not fully occupied from the left, can have a right-leaning configuration of nodes.

Suppose we represent such a binary heap using an array starting from index $1$. In that case, the children of a node at index $i$ can be located at indices $2i$ and $2i+1$ respectively.

Thus, both **enqueue** and **dequeue** rely on traversing the binary heap in a systematic manner. The following efficiencies are characteristic:

#### Enqueue Efficiency: $O(\log n)$

When **enqueue** is executed:

- The highest efficiency achievable is $O(1)$ when the new element replaces the root, and the heap happens to be a min or max heap.
- The efficiency can degrade up to $O(\log n)$ in the worst-case scenario. This occurs when the new child percolates to the root in $O(\log n)$ steps after comparing and potentially swapping with its ancestors.

#### Dequeue Efficiency: $O(1)$ - $O(\log n)$

When **dequeue** is executed:

- The operation's efficiency spans from $O(1)$ when the root is instantly removed to $O(\log n)$ when the replacement node needs to 'bubble down' to its proper position.
<br>

## 11. Explain how you can efficiently track the _Minimum_ or _Maximum_ element in a _Queue_.

Using a **singly linked list as a queue** provides $O(1)$ time complexity for standard queue operations, but finding exact minimum and maximum can be $O(n)$. However, there are optimized methods for improving efficiency.

### Optimal Methods

1. **Element Popularity Counter**: Keep track of the number of times an element appears, so you can easily determine changes to the minimum and maximum when elements are added or removed.
2. **Auxiliary Data Structure**: Alongside the queue, maintain a secondary data structure, such as a tree or stack, that helps identify the current minimum and maximum elements efficiently.

### Code Example: Naive Queue

Here is the Python code:

```python
class NaiveQueue:
    def __init__(self):
        self.queue = []
    
    def push(self, item):
        self.queue.append(item)
        
    def pop(self):
        return self.queue.pop(0)
    
    def min(self):
        return min(self.queue)
    
    def max(self):
        return max(self.queue)
```
This code has $O(n)$ time complexity for both `min` and `max` methods.


### Code Example: Element Popularity Counter

Here is the Python code:

```python
from collections import Counter

class EfficientQueue:
    def __init__(self):
        self.queue = []
        self.element_count = Counter()
        self.minimum = float('inf')
        self.maximum = float('-inf')
    
    def push(self, item):
        self.queue.append(item)
        self.element_count[item] += 1
        self.update_min_max(item)
        
    def pop(self):
        item = self.queue.pop(0)
        self.element_count[item] -= 1
        if self.element_count[item] == 0:
            del self.element_count[item]
            if item == self.minimum:
                self.minimum = min(self.element_count.elements(), default=float('inf'))
            if item == self.maximum:
                self.maximum = max(self.element_count.elements(), default=float('-inf'))
        return item
    
    def min(self):
        return self.minimum
    
    def max(self):
        return self.maximum
    
    def update_min_max(self, item):
        self.minimum = min(self.minimum, item)
        self.maximum = max(self.maximum, item)
```

This code has $O(1)$ time complexity for both `min` and `max` methods.


### Code Example: Dual Data Structure Queue

Here is the Python code:

```python
from queue import Queue
from collections import deque

class DualDataQueue:
    def __init__(self):
        self.queue = Queue()  # For standard queue operations
        self.max_queue = deque()  # To keep track of current maximum
        
    def push(self, item):
        self.queue.put(item)
        while len(self.max_queue) > 0 and self.max_queue[-1] < item:
            self.max_queue.pop()
        self.max_queue.append(item)
        
    def pop(self):
        item = self.queue.get()
        if item == self.max_queue[0]:
            self.max_queue.popleft()
        return item
    
    def max(self):
        return self.max_queue[0]
```

This code has $O(1)$ time complexity for `max` method and $O(1)$ time complexity for `min` method using the symmetric approach.
<br>

## 12. Discuss an algorithm to merge two or more _Queues_ into one with efficient _Dequeuing_.

**Merging multiple queues** is conceptually similar to merging two lists. However, direct merging challenges efficiency as it enforces a $\mathcal{O}(n)$ operation for each item in the queues. Utilizing a secondary queue $\text{auxQueue}$ can provide a more efficient $\mathcal{O}(n + m)$ sequence.

### Algorithm: Queue Merging

1. **Enqueue into Aux**: Until all input queues are empty, **enqueue** from the oldest non-empty queue to $\text{auxQueue}$.
2. **Move Everything Back**: For each item already in $\text{auxQueue}$, **dequeue** and **enqueue** back to the determined queue.
3. **Return $\text{auxQueue}$**: As all input queues are empty, $\text{auxQueue}$ now contains all the original elements.

### Complexity Analysis

- **Time Complexity**: The algorithm runs in $\mathcal{O}(n + m)$ where $n$ and $m$ represent the sizes of the input queues.
- **Space Complexity**: The algorithm uses $\mathcal{O}(1)$ auxiliary space.

### Code Example: Queue Merging

Here is the Python code:

```python
from queue import Queue

def merge_queues(q_list):
    auxQueue = Queue()
    
    # Step 1: Enqueue into Aux
    for q in q_list:
        while not q.empty():
            auxQueue.put(q.get())
    
    # Step 2: Move Everything Back
    for _ in range(auxQueue.qsize()):
        q.put(auxQueue.get())
    
    # Step 3: Return auxQueue
    return q
```

### Code Example: Testing Queue Merging

Here is the Python code with the test:

```python
# Creating queues
q1 = Queue()
q2 = Queue()

# Enqueueing elements
for i in range(5):
    q1.put(i)

for i in range(5, 10):
    q2.put(i)

# Merging
merged = merge_queues([q1, q2])

# Dequeuing and printing
while not merged.empty():
    print(merged.get())
```

### Code Example: Multi-Queue Merging

Here is the Python code if we merge it into single queue:

```python
def merge_queue_multi(q_list):
    merged = Queue()
    
    # Merging the queues
    for q in q_list:
        while not q.empty():
            merged.put(q.get())
    
    return merged
```

### Time Complexity of Limitation

The time complexity of this algorithm is **not as optimal** as the enqueuing to the auxiliary queue makes each item traverse more than once, increasing control time when an element is being dequeued.

For even activity, all enqueuing actions are executed approximately the same number of times, so there's still a linear-time bound.

#### Code Example: Multi-Queue Merging with Dequeuing Control

Here is the Python code:

```python
def merge_queues_on_visit_multi(q_list):
    def on_visit(visit_cb):
        for q in q_list:
            while not q.empty():
                visit_cb(q.get())
    
    merged = Queue()
    on_visit(merged.put)
    return merged
```
<br>

## 13. Name some _Queue Implementations_. Compare their efficiency.

**Queues** can be built using various underlying structures, each with its trade-offs in efficiency and complexity.

### Naive Implementations

#### Simple Array

Using a simple array for implementation requires **shifting elements** when adding or removing from the front. This makes operations linear time $O(n)$, which is **inefficient** and not suitable for large queues or real-time use.

```python
class ArrayQueue:
    def __init__(self):
        self.queue = []

    def enqueue(self, item):
        self.queue.append(item)

    def dequeue(self):
        return self.queue.pop(0)
```

#### Singly-linked List

Using a singly-linked list allows $O(1)$ `enqueue` with a tail pointer but still $O(n)$ `dequeue`.

```python
class Node:
    def __init__(self, data=None):
        self.data = data
        self.next = None

class LinkedListQueue:
    def __init__(self):
        self.head = None
        self.tail = None

    def enqueue(self, item):
        new_node = Node(item)
        if self.tail:
            self.tail.next = new_node
        else:
            self.head = new_node
        self.tail = new_node

    def dequeue(self):
        if self.head:
            data = self.head.data
            self.head = self.head.next
            if not self.head:
                self.tail = None
            return data
```

### Efficient Implementations

#### Doubly Linked List

A doubly linked list enables $O(1)$ `enqueue` and `dequeue` by maintaining head and tail pointers, but it requires **prev node management**.

```python
class DNode:
    def __init__(self, data=None):
        self.data = data
        self.next = None
        self.prev = None

class DoublyLinkedListQueue:
    def __init__(self):
        self.head = None
        self.tail = None

    def enqueue(self, item):
        new_node = DNode(item)
        if not self.head:
            self.head = new_node
        else:
            self.tail.next = new_node
            new_node.prev = self.tail
        self.tail = new_node

    def dequeue(self):
        if self.head:
            data = self.head.data
            self.head = self.head.next
            if self.head:
                self.head.prev = None
            else:
                self.tail = None
            return data
```

#### Double-Ended Queue

The `collections.deque` in Python is essentially a double-ended queue implemented using a **doubly-linked list**, providing $O(1)$ complexities for operations at both ends.

```python
from collections import deque

class DequeQueue:
    def __init__(self):
        self.queue = deque()

    def enqueue(self, item):
        self.queue.append(item)

    def dequeue(self):
        return self.queue.popleft()
```

#### Binary Heap

A binary heap with its **binary tree structure** is optimized for **priority queues**, achieving $O(\log n)$ for both `enqueue` and `dequeue` operations. This makes it useful for situations where you need to process elements in a particular order.

```python
import heapq

class MinHeapQueue:
    def __init__(self):
        self.heap = []

    def enqueue(self, item):
        heapq.heappush(self.heap, item)

    def dequeue(self):
        return heapq.heappop(self.heap)
```
<br>

## 14. Describe an array-based implementation of a _Queue_ and its disadvantages.

While **array-based Queues** are simple, they have inherent limitations.

### Key Features

- **Structure**: Uses an array to simulate a queue's First-In-First-Out (FIFO) behavior.
- **Pointers**: Utilizes a front and rear pointer/index.

### Code Example: Simple Queue Operations

Here is the Python code:

```python
class ArrayQueue:
    def __init__(self, size):
        self.size = size
        self.queue = [None] * size
        self.front = self.rear = -1

    def is_full(self):
        return self.rear == self.size - 1

    def is_empty(self):
        return self.front == -1 or self.front > self.rear

    def enqueue(self, element):
        if self.is_full():
            print("Queue is full")
            return
        if self.front == -1:
            self.front = 0
        self.rear += 1
        self.queue[self.rear] = element

    def dequeue(self):
        if self.is_empty():
            print("Queue is empty")
            return
        item = self.queue[self.front]
        self.front += 1
        if self.front > self.rear:
            self.front = self.rear = -1
        return item
```

### Disadvantages

- **Fixed Size**: Array size is predetermined, leading to potential memory waste or overflow.
- **Element Frontshift**: Deletions necessitate front-shifting, creating an $O(n)$ time cost.
- **Unequal Time Complexities**: Operations like `enqueue` and `dequeue` can be $O(1)$ or $O(n)$, making computation times less predictable.
<br>

## 15. What are the benefits of implementing a _Queue_ with a _Doubly Linked List_ versus a _Singly Linked List_?

Let's compare the benefits of implementing a **Queue** using both a **Doubly Linked List** and **Singly Linked List**.

### Key Advantages

#### Singly Linked List Queue

- **Simplicity**: The implementation is straightforward and may require fewer lines of code.
- **Memory Efficiency**: Nodes need to store only a single reference to the next node, which can save memory.

#### Doubly Linked List Queue

- **Bi-directional Traversal**: Allows for both forward and backward traversal, a necessity for certain queue operations such as tail management and removing from the end.
- **Efficient Tail Operations**: Eliminates the need to traverse the entire list to find the tail, significantly reducing time complexity for operations that involve the tail.
<br>



#### Explore all 55 answers here 👉 [Devinterview.io - Queue Data Structure](https://devinterview.io/questions/data-structures-and-algorithms/queue-data-structure-interview-questions)

<br>

<a href="https://devinterview.io/questions/data-structures-and-algorithms/">
<img src="https://firebasestorage.googleapis.com/v0/b/dev-stack-app.appspot.com/o/github-blog-img%2Fdata-structures-and-algorithms-github-img.jpg?alt=media&token=fa19cf0c-ed41-4954-ae0d-d4533b071bc6" alt="data-structures-and-algorithms" width="100%">
</a>
</p>

