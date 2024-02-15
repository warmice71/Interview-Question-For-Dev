# 46 Fundamental Stack Data Structure Interview Questions

<div>
<p align="center">
<a href="https://devinterview.io/questions/data-structures-and-algorithms/">
<img src="https://firebasestorage.googleapis.com/v0/b/dev-stack-app.appspot.com/o/github-blog-img%2Fdata-structures-and-algorithms-github-img.jpg?alt=media&token=fa19cf0c-ed41-4954-ae0d-d4533b071bc6" alt="data-structures-and-algorithms" width="100%">
</a>
</p>

#### You can also find all 46 answers here 👉 [Devinterview.io - Stack Data Structure](https://devinterview.io/questions/data-structures-and-algorithms/stack-data-structure-interview-questions)

<br>

## 1. What is a _Stack_?

A **stack** is a simple data structure that follows the **Last-In, First-Out (LIFO)** principle. It's akin to a stack of books, where the most recent addition is at the top and easily accessible.

### Core Characteristics

- **Data Representation**: Stacks can hold homogeneous or heterogeneous data.
- **Access Restrictions**: Restricted access primarily to the top of the stack, making it more efficient for certain algorithms.

### Stack Operations

1. **Push**: Adds an element to the top of the stack.
2. **Pop**: Removes and returns the top element.
3. **Peek**: Returns the top element without removing it.
4. **isEmpty**: Checks if the stack is empty.
5. **isFull** (for array-based stacks): Checks if the stack is full.

All the above operations typically have a time complexity of $O(1)$, making stack operations highly efficient.

### Visual Representation

![Stack Data Structure](https://firebasestorage.googleapis.com/v0/b/dev-stack-app.appspot.com/o/stacks%2Fstack.png?alt=media&token=74633f0a-83f7-4038-8b82-e10f0d6006b9&_gl=1*1uzhlk1*_ga*OTYzMjY5NTkwLjE2ODg4NDM4Njg.*_ga_CW55HF8NVT*MTY5NjYwNzMxNS4xNDcuMS4xNjk2NjA3NzE2LjUwLjAuMA..)

### Practical Applications

1. **Function Calls**: The call stack keeps track of program flow and memory allocation during method invocations.
2. **Text Editors**: The undo/redo functionality often uses a stack.
3. **Web Browsers**: The Back button's behavior can be implemented with a stack.
4. **Parsing**: Stacks can be used in language processing for functions like balanced parentheses, and **binary expression evaluation**.

5. **Memory Management**: Stacks play a role in managing dynamic memory in computer systems.

6. **Infix to Postfix Conversion**: It's a crucial step for evaluating mathematical expressions such as `2 + 3 * 5 - 4` in the correct precedence order. Stack-based conversion simplifies parsing and involves operators such as `push` and `pop` until the correct order is achieved.

7. **Graph Algorithms**: Graph traversal algorithms such as Depth First Search (DFS) deploy **stacks** as a key mechanism to remember vertices and explore connected components.

### Code Example: Basic Stack

Here is the Python code:

```python
class Stack:
    def __init__(self):
        self.stack = []

    def push(self, item):
        self.stack.append(item)

    def pop(self):
        if not self.is_empty():
            return self.stack.pop()

    def peek(self):
        if not self.is_empty():
            return self.stack[-1]

    def is_empty(self):
        return len(self.stack) == 0

    def size(self):
        return len(self.stack)
```
<br>

## 2. Why _Stack_ is considered a _Recursive_ data structure?

A **stack** is considered a **recursive data structure** because its definition is self-referential. At any given point, a stack can be defined as a top element combined with another stack (the remainder).

Whenever an element is pushed onto or popped off a stack, what remains is still a stack. This **self-referential nature**, where operations reduce the problem to smaller instances of the same type, embodies the essence of **recursion**
<br>

## 3. What are the primary operations performed on a _Stack_ and their time complexities?

Let's look into the fundamental operations of a **stack** and their associated time complexities.

### Stack Operations and Complexity

- **Push** (Time: $O(1)$): New elements are added at the top of the stack, making this a $O(1)$.
- **Pop** (Time: $O(1)$): The top element, and the only one accessible, is removed during this $O(1)$ operation.
- **Peek** (Time: $O(1)$): Viewing the top of the stack doesn't alter its structure, thus taking $O(1)$.
- **Size** (Time: $O(1)$): Stacks typically keep track of their size, ensuring $O(1)$ performance.
- **isEmpty** (Time: $O(1)$): Checks for stack emptiness and usually completes in $O(1)$ time.

### Code Example: Stack

Here is the Python code:

  ```python
  stack = []
stack.append(1)  # Pushes 1 onto the stack
stack.append(3)  # Pushes 3 onto the stack
print(stack.pop())  # Outputs 3; removes the top element from the stack
print(stack[-1])  # Outputs 1; peek at the top element
print(len(stack))  # Outputs 1; returns the size of the stack
print(not stack)  # Outputs False; checks if the stack is empty
<br>

## 4. When should I use _Stack_ or _Queue_ data structures instead of _Arrays/Lists_?

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

## 5. What are _Infix_, _Prefix_, and _Postfix_ notations?

In computer science, **infix**, **prefix**, and **postfix** notations are methods of writing mathematical expressions. While humans generally use infix notation, machines can more efficiently parse prefix and postfix notations.

### Infix, Prefix and Postfix Notations

- **Infix**: Operators are placed between operands. This is the most common notation for humans due to its intuitiveness.
  
  Example: $1 + 2$

- **Prefix**: Operators are placed before operands. The order of operations is determined by the position of the operator rather than parentheses.
  
  Example: $+ 1 \times 2 3$ which evaluates to $1 + (2 \times 3) = 7$

- **Postfix**: Operators are placed after operands. The order of operations is determined by the sequence in which operands and operators appear.
  
  Example: $1 2 3 \times +$ which evaluates to $1 + (2 \times 3) = 7$

### Relation to Stacks

- **Conversion**: Stacks can facilitate the conversion of expressions from one notation to another. For instance, the Shunting Yard algorithm converts infix expressions to postfix notation using a stack.
  
- **Evaluation**: Both postfix and prefix expressions are evaluated using stacks. For postfix:
  1. Operands are pushed onto the stack.
  2. Upon encountering an operator, the required operands are popped, the operation is executed, and the result is pushed back.

  For example, for the expression $1 2 3 \times +$:
  - 1 is pushed onto the stack.
  - 2 is pushed.
  - 3 is pushed.
  - $\times$ is encountered. 3 and 2 are popped, multiplied to get 6, which is then pushed.
  - $+$ is encountered. 6 and 1 are popped, added to get 7, which is then pushed. This 7 is the result.

Evaluating prefix expressions follows a similar **stack-based method** but traverses the expression differently.

### Code Example: Postfix Evaluation

Here is the Python code:

```python
def evaluate_postfix(expression):
    stack = []
    tokens = expression.split()  # Handle multi-digit numbers
    for token in tokens:
        if token.isdigit():
            stack.append(int(token))
        else:
            operand2 = stack.pop()
            operand1 = stack.pop()
            stack.append(perform_operation(operand1, operand2, token))
    return stack[0]

def perform_operation(operand1, operand2, operator):
    operations = {
        '+': operand1 + operand2,
        '-': operand1 - operand2,
        '*': operand1 * operand2,
        '/': operand1 / operand2
    }
    return operations[operator]

# Example usage
print(evaluate_postfix('1 2 + 3 4 * -'))  # Output: -7
```
<br>

## 6. Explain how _Stacks_ are used in _Function Call management_ in programming languages.

When **functions** are called in programming languages, the system typically uses a **call stack** to manage the call sequence and memory allocation. Let's take a look at how this process works.

### The Call Stack

The **call stack** maintains a record of all the active function calls that a program makes. When a new function is called, it's added to the top of the stack. Once a function finishes its execution, it's removed from the stack, and control returns to the calling function.

This "last in, first out" behavior is well-suited to **stack** data structures.

### How the Call Stack Works

1. **Function Call**: When a function is called, a **stack frame** is created and pushed onto the call stack. This frame contains important information about the state of the function, such as local variables and the return address, which points to the instruction after the function call.

2. **Local Execution**: The CPU executes the instructions within the called function. The function accesses its inputs, processes data, and calls other functions as needed.

3. **Return**: If the called function doesn't make any further function calls, it exits, and its stack frame is removed. Alternatively, if the function makes additional calls, the call stack grows further.

4. **Stack Unwinding**: Once the initial (or other topmost) function call is finished, there are no more functions to execute. The stack then shrinks, starting with the top frame, until it's empty.

### Code Example: Using the Call Stack

Here is code written in Python:

```python
def multiply(a, b):
    result = a * b
    return result

def calculate(a, b, c):
    temp = multiply(b, c)
    return a + temp

result = calculate(2, 3, 4)
print(result)  # Output: 14
```

In this example, when `calculate` is called, it first calls `multiply` and then performs addition. The call stack looks like this during the execution of `calculate`:

1. `calculate` with parameters (2, 3, 4)
2. `multiply` with parameters (3, 4)

Once `multiply` completes, its stack frame is removed, and `calculate` continues with the next line of code.

### Benefits and Limitations of Call Stack

#### Benefits

- **Automatic Memory Management**: The call stack automatically allocates memory for local variables and function parameters, simplifying memory management for developers.
- **Efficiency**: Its simple structure makes it efficient for managing function calls in most programming scenarios.

#### Limitations

- **Size Limitations**: The stack has a fixed memory size allocated at program startup, which can lead to stack overflow errors if the stack grows too large.
- **No Random Access**: Elements in the stack can only be accessed or removed in a last-in-first-out manner, limiting its use in some algorithms and data structures.
<br>

## 7. Describe an application where _Stacks_ are naturally suited over other data structures.

**Stacks** find natural utility in various practical use-cases, such as in text editors for tracking actions and providing the "undo" and "redo" functionalities.

### Code Example: Undo and Redo Stack

Here is the Python code:

```python
class UndoRedoStack:
    def __init__(self):
        self._undo_stack = []
        self._redo_stack = []

    def push(self, action):
        self._undo_stack.append(action)
        # When a new action is pushed, the redo stack needs to be reset
        self._redo_stack = []

    def undo(self):
        if self._undo_stack:
            action = self._undo_stack.pop()
            self._redo_stack.append(action)
            return action

    def redo(self):
        if self._redo_stack:
            action = self._redo_stack.pop()
            self._undo_stack.append(action)
            return action
```

### Stack-Based Undo and Redo Workflow

In a typical text editor, the user can:

- **Type Text**: Each time new text is entered, it represents an **action** that can be **undone** or **redone**.
- **Perform Undo/Redo**: The editor navigates through previous actions, whether to reverse or reinstate them.
<br>

## 8. Compare _Array-based_ vs _Linked List_ stack implementations.

**Array-based stacks** excel in time efficiency and direct element access. In contrast, **linked list stacks** are preferable for dynamic sizing and easy insertions or deletions.

### Common Features

- **Speed of Operations**: Both `pop` and `push` are $O(1)$ operations.
- **Memory Use**: Both have $O(n)$ space complexity.
- **Flexibility**: Both can adapt their sizes, but their resizing strategies differ.

### Key Distinctions

#### Array-Based Stack

- **Locality**: Consecutive memory locations benefit CPU caching.
- **Random Access**: Provides direct element access.
- **Iterator Needs**: Preferable if indexing or iterators are required.
- **Performance**: Slightly faster for top-element operations and potentially better for time-sensitive tasks due to caching.
- **Push**: $O(1)$ on average; resizing might cause occasional $O(n)$.

#### Linked List Stack

- **Memory Efficiency**: Better suited for fluctuating sizes and limited memory scenarios.
- **Resizing Overhead**: No resizing overheads.
- **Pointer Overhead**: Requires extra memory for storing pointers.

### Code Example: Array-Based Stack

Here is the Python code:

```python
class ArrayBasedStack:
    def __init__(self):
        self.stack = []
    def push(self, item):
        self.stack.append(item)
    def pop(self):
        return self.stack.pop() if self.stack else None
```

### Code Example: Linked List Stack

Here is the Python code:

```python
class Node:
    def __init__(self, data=None):
        self.data = data
        self.next = None
class LinkedListStack:
    def __init__(self):
        self.head = None
    def push(self, item):
        new_node = Node(item)
        new_node.next = self.head
        self.head = new_node
    def pop(self):
        if self.head:
            temp = self.head
            self.head = self.head.next
            return temp.data
        return None
```
<br>

## 9. Implement a _Dynamic Stack_ that automatically resizes itself.

### Problem Statement

Implement a **Dynamic Stack** that automatically resizes itself when it reaches its capacity.

### Solution

Resizing a stack involves two main operations: **shrinking** and **expanding** the stack when needed. A common strategy is to **double the stack's size** each time it reaches full capacity and **halve** it when it becomes 25% full, as this provides efficient amortized performance.

#### Key Operations

1. `push(item)`: Add an item to the stack.
2. `pop()`: Remove and return the top item from the stack.
3. `is_full()`: Check if the stack is full.
4. `is_empty()`: Check if the stack is empty.
5. `expand()`: Double the stack's capacity.
6. `shrink()`: Halve the stack's capacity.

#### Algorithm Steps

1. Start with an initial capacity for the stack. In this example, it's `2`.
2. Whenever a `push` operation encounters a full stack, call the `expand` method before the addition.
3. Whenever a `pop` operation leaves the stack 25% full, call the `shrink` method.

This ensures the stack dynamically adjusts its size based on the current number of elements.

#### Complexity Analysis

- **Time Complexity**:
  - `push`: $O(1)$ amortized. Although `expand` can take up to $O(n)$ time, it is only triggered once every $n$ `push` operations, resulting in an average of $O(1)$ per `push`.
  - `pop`, `is_full`, and `is_empty`: $O(1)$.
  - `expand` and `shrink`: $O(n)$ in the worst case, but they are infrequently called, so their amortized time is $O(1)$ per operation.
  
- **Space Complexity**: $O(n)$ where $n$ is the number of elements in the stack. This accounts for the stack itself and any additional overhead such as the temporary arrays used during resizing.

#### Implementation

Here is the Python code:

```python
class DynamicStack:
    # Initialize the stack with an initial capacity
    def __init__(self, capacity=2):
        self.capacity = capacity
        self.stack = [None] * capacity
        self.top = -1

    # Push an element to the stack
    def push(self, item):
        if self.is_full():
            self.expand()
        self.top += 1
        self.stack[self.top] = item

    # Pop the top element from the stack and return it
    def pop(self):
        if self.is_empty():
            raise IndexError('Stack is empty')
        item = self.stack[self.top]
        self.top -= 1
        if self.top < self.capacity // 4:
            self.shrink()
        return item

    # Check if the stack is full
    def is_full(self):
        return self.top == self.capacity - 1

    # Check if the stack is empty
    def is_empty(self):
        return self.top == -1

    # Double the stack's capacity
    def expand(self):
        self.capacity *= 2
        new_stack = [None] * self.capacity
        for i in range(self.top + 1):
            new_stack[i] = self.stack[i]
        self.stack = new_stack

    # Halve the stack's capacity
    def shrink(self):
        self.capacity //= 2
        new_stack = [None] * self.capacity
        for i in range(self.top + 1):
            new_stack[i] = self.stack[i]
        self.stack = new_stack
```
<br>

## 10. What are the performance implications of a _Fixed-size Array Stack Implementation_?

While **fixed-size array stacks** offer simplicity and often better performance for certain operations, such as data caching and real-time processing, several limitation are to be considered.

### Space and Memory Management

- **Limited Capacity**: Fixed-size arrays impose a maximum capacity for stacks, introducing the potential for overflow.
- **Pre-allocated Memory**: Fixed-size arrays require memory to be allocated in advance for the maximum capacity, leading to potential inefficiencies if this capacity is not fully utilized.
- **Consistent Size**: Stacks using fixed-size arrays do not auto-resize, leading to inefficient memory use if the actual size of the stack varies significantly from the allocated size.

### Time Complexity of Fixed-Size Array Stacks

- **Push Operation**: $O(1)$ (constant time), until the array is full and a resizing operation is initiated which leads to $O(n)$ in the worst case.
- **Pop Operation**: $O(1)$ - simple memory deallocation or index decrement.
- **Peek Operation**: $O(1)$ - equivalent to pop.
- **Search Operation**: $O(n)$ - in the worst case, when the element is at the top of the stack or not present.

### Data Structure Sensitivity

- **Space Sensitivity**: Stacks using a fixed-size array have predictable, constant memory requirements.
- **Performance Sensitivity**: While operations on non-fixed size stacks might have $O(1)$ average-case time complexity parameters, certain operations on fixed-size stacks can degrade in worst-case scenarios, justifying the $O(n)$ worst-case complexity.

### Practical Applications

- **Real-Time Systems**: Fixed-size arrays can be preferable for applications with strict timing requirements, as memory operations are more deterministic.
- **Embedded Systems**: In resource-constrained environments, using fixed-size arrays can help manage memory more efficiently due to their predictable memory requirements.
- **Cache Systems**: The use of fixed-size arrays is significant in caches, where the predictability of space requirements is essential.

### Code Example: Fixed-Size Stack

Here is the Python code:

```python
class FixedSizeStack:
    def __init__(self, capacity=10):
        self.stack = [None] * capacity
        self.top = -1

    def push(self, value):
        if self.top == len(self.stack) - 1:
            print("Stack is full, cannot push.")
            return
        self.top += 1
        self.stack[self.top] = value

    def pop(self):
        if self.top == -1:
            print("Stack is empty, cannot pop.")
            return
        value = self.stack[self.top]
        self.top -= 1
        return value

    def peek(self):
        if self.top == -1:
            print("Stack is empty, no top element.")
            return
        return self.stack[self.top]

    def is_empty(self):
        return self.top == -1

    def is_full(self):
        return self.top == len(self.stack) - 1
```
<br>

## 11. Design a _Stack_ that supports _Retrieving_ the min element in _O(1)_.

### Problem Statement

The goal is to design a **stack** data structure that can efficiently retrieve both the minimum element and the top element in $O(1)$ time complexity. 

### Solution

To meet the time complexity requirement, we'll maintain two stacks:

1. **Main Stack** for standard stack functionality.
2. **Auxiliary Stack** that keeps track of the minimum element up to a given stack position.

#### Algorithm Steps

1. **Pop** and **Push**
   - For each element $e$ in the **Main Stack**, check if it's smaller than or equal to the top element in the **Auxiliary Stack**. If $e$ is the new minimum, push it onto both stacks.

2. **Minimum Element Retrieval**: The top element of the **Auxiliary Stack** will always be the minimum element of the main stack.

#### Complexity Analysis

- **Time Complexity**: $O(1)$ for all operations.
- **Space Complexity**: $O(N)$, where $N$ is the number of elements in the stack.

#### Implementation

Here is the Python code:

```python
class MinStack:
    def __init__(self):
        self.stack = []
        self.min_stack = []

    def push(self, element):
        self.stack.append(element)
        if not self.min_stack or element <= self.min_stack[-1]:
            self.min_stack.append(element)

    def pop(self):
        if not self.stack:
            return None
        top = self.stack.pop()
        if top == self.min_stack[-1]:
            self.min_stack.pop()
        return top

    def top(self):
        return self.stack[-1] if self.stack else None

    def getMin(self):
        return self.min_stack[-1] if self.min_stack else None
```
<br>

## 12. How can you design a _Stack_ to be thread-safe?

Ensuring **thread safety** in a traditional stack, where operations are based on a last-in, first-out (LIFO) approach, can be achieved through a variety of techniques. I will give examples of three different approaches here:

1. **Locking Mechanism**: Where a thread synchronizes access through synchronization techniques like locks.

2. **Non-Blocking Mechanism**: Using atomic operations without explicit locks.

3. **Data Structure Selection**: Choosing inherently thread-safe data structures that mimic stack operations.

### Locking Mechanism

This approach synchronizes operations on the stack using a lock. While a thread holds the lock, other threads are blocked from entering critical sections. 

Here is the Python code:

  ```python
  import threading
  
  class LockedStack:
      def __init__(self):
          self.stack = []
          self.lock = threading.Lock()
  
      def push(self, item):
          with self.lock:
              self.stack.append(item)
  
      def pop(self):
          with self.lock:
              if self.stack:
                  return self.stack.pop()
              return None
  ```

### Non-Blocking Mechanism

This approach uses atomic operations on **primitive data types**, which are guaranteed to happen without interruption.

Here is the Python code for this approach:

  ```python
  import queue
  import threading
  
  def non_blocking_push(q, item):
      while True:
          old_q = q.queue
          new_q = old_q.copy()
          new_q.append(item)
  
          if q.queue == old_q:
              q.queue = new_q
              return
		  
  def non_blocking_pop(q):
      while True:
          old_q = q.queue
          new_q = old_q.copy()
          if q.queue:
              new_q.pop()
              if q.queue == old_q:
                  q.queue = new_q
                  return q.queue[-1] 
      return None
		  
  q = queue.LifoQueue()
  
  threading.Thread(target=non_blocking_push, args=(q, 1)).start()
  threading.Thread(target=non_blocking_push, args=(q, 2)).start()
  threading.Thread(target=non_blocking_pop, args=(q,)).start()
  ```


For the **Multi-threaded** Setup, you can run this Python code:

  ### Data Structure Selection

  Some **container classes**, like the LifoQueue in Python, are inherently designed to be thread-safe, thus making their contents and operations secure in a multi-threaded environment.

  Here is the Python code:

  ```python
  import queue
  import threading
  
  q = queue.LifoQueue()
  
  q.put(1)
  q.put(2)
  print(q.get())
  print(q.get())
  ```
<br>

## 13. Implement a _Stack_ with a _Find-Middle_ operation in _O(1)_ time.

### Problem Statement

The task is to design a **_stack_** data structure that supports **`push`**, **`pop`**, **`findMiddle`**, and **`deleteMiddle`** operations, all in constant $O(1)$ time complexity.

### Solution

We can solve this challenge using a doubly linked list where each node also includes a pointer to the middle node. This solution, though not built on arrays, has a clear control flow and keeps a consistent time complexity.

#### Algorithm Steps

1. Initialize an empty stack and set `middle` to `NULL`.
2. During the push operation, update the `middle` pointer based on the current number of nodes. If the number of nodes is odd, `middle` moves up, otherwise, it stays at the same position.
3. The pop operation, on top of removing the item, also adjusts the `middle` pointer if the removed node was pointing to it.
4. `findMiddle` and `deleteMiddle` simply involve accessing or manipulating the node that `middle` points to.

#### Complexity Analysis

- **Time Complexity**:
  - `push`: $O(1)$ -  Constant time for every node insertion.
  - `pop`: $O(1)$ -  Always removes the top node in a constant time regardless of the stack size.
  - `findMiddle` and `deleteMiddle`: $O(1)$ - Directly accessed through the `middle` pointer.
- **Space Complexity**: $O(n)$ - Additional space is required for the pointers associated with each node.

#### Implementation

Here is the Python code:

```python
class DLLNode:
    def __init__(self, x):
        self.data = x
        self.next = None
        self.prev = None

class MyStack:
    def __init__(self):
        self.head = None
        self.mid = None
        self.count = 0

    def push(self, x):
        new_node = DLLNode(x)
        new_node.next = self.head

        if self.count == 0:
            self.mid = new_node
        else:
            self.head.prev = new_node
            if self.count % 2 != 0:
                self.mid = self.mid.prev

        self.head = new_node
        self.count += 1

    def pop(self):
        if self.count == 0:
            print("Stack is empty")
            return

        item = self.head.data
        self.head = self.head.next

        if self.head is not None:
            self.head.prev = None

        self.count -= 1

        # Update mid if the removed node was the middle one
        if self.count % 2 == 0:
            self.mid = self.mid.next

        return item

    def findMiddle(self):
        if self.count == 0:
            print("Stack is empty")
            return
        return self.mid.data

    def deleteMiddle(self):
        if self.count == 0:
            print("Stack is empty")
            return
        temp = self.mid
        self.mid.prev.next = self.mid.next
        self.mid.next.prev = self.mid.prev
        if self.count % 2 != 0:
            self.mid = self.mid.prev
        else:
            self.mid = self.mid.next
        del temp
```
<br>

## 14. Implement a _Linked List_ using _Stack_.

### Problem Statement

The task is to implement **linked list** using a **stack**.

### Solution

**Linked lists** are usually built with nodes, each containing a value and a pointer to the next node. However, you can also **simulate linked lists using stacks**, which follow a "last in, first out" (LIFO) mechanism.

#### Algorithm Steps

1. **Initialize**: Two stacks: `list_stack` and `temp_stack`.
2. **Add to Head**: Push to `list_stack`.
3. **Remove from Head**: Pop from `list_stack`.
4. **Insert**: Pop items to `temp_stack` until the insertion point, then push the new item and the `temp_stack` items back to `list_stack`.
5. **Delete**: Similar to insert but pop the node to be deleted before pushing items back.

#### Complexity Analysis

- **Time Complexity**: 
  - Add to Head: $O(1)$
  - Remove from Head: $O(1)$
  - Insert at position $k$: $O(k)$
  - Delete at position $k$: $O(k)$

- **Space Complexity**: 
  - $O(n)$ for `list_stack`, where $n$ is the number of elements.
  - $O(k)$ for `temp_stack` during insert or delete operations at position $k$, with an overall space complexity still being $O(n)$ in the worst case (when $k = n$).

#### Implementation

Here is the Python code:

```python
class LinkedListStack:
    def __init__(self):
        self.list_stack, self.temp_stack = [], []

    def push(self, data):
        self.list_stack.append(data)

    def pop(self):
        return self.list_stack.pop()

    def insert(self, data, pos):
        while pos:
            self.temp_stack.append(self.list_stack.pop())
            pos -= 1
        self.list_stack.append(data)
        while self.temp_stack:
            self.list_stack.append(self.temp_stack.pop())

    def delete(self, pos):
        while pos:
            self.temp_stack.append(self.list_stack.pop())
            pos -= 1
        self.list_stack.pop()
        while self.temp_stack:
            self.list_stack.append(self.temp_stack.pop())

    def display(self):
        print(self.list_stack)

# Example
ll_stack = LinkedListStack()
ll_stack.push(1)
ll_stack.push(2)
ll_stack.push(3)
ll_stack.insert(10, 1)
print("Before deletion:")
ll_stack.display()
ll_stack.delete(2)
print("After deletion:")
ll_stack.display()
```
<br>

## 15. Implement Doubly Linked List using Stacks with min complexity.

### Problem Statement

The task is to implement a **Doubly LinkedList** using **Stacks**.

### Solution

Using two stacks, `forwardStack` and `backwardStack`, we can emulate a **doubly linked list**.

- **Insertion**:
  - Beginning: Transfer elements from `backwardStack` to `forwardStack`, then push the new element onto `backwardStack`.
  - End: Transfer from `forwardStack` to `backwardStack` and push the new element onto `backwardStack`.

- **Traversal**:
  - Forward: Pop from `backwardStack`, push onto `forwardStack`.
  - Backward: Pop from `forwardStack`, push onto `backwardStack`.

- **Deletion**: Traverse the needed stack to locate and remove the desired element, whether it's a specific item or the first/last entry.

#### Complexity Analysis

- **Time Complexity**: 
  - Operations like `insertInBeginning`, `insertAtEnd`, `moveForward`, `moveBackward`, `delete`, `deleteFirst`, and `deleteLast` have a worst-case time complexity of $O(n)$ due to the potential full traversal of a stack.
  
- **Space Complexity**: $O(n)$, where $n$ is the number of nodes in the doubly linked list, primarily occupied by the two stacks.

#### Implementation

Here is the Java code:

```java
import java.util.Stack;

class DoubleLinkedList {
    private Stack<Integer> forwardStack;
    private Stack<Integer> backwardStack;

    public DoubleLinkedList() {
        forwardStack = new Stack<>();
        backwardStack = new Stack<>();
    }

    public void insertInBeginning(int data) {
        // Move all elements to forwardStack to insert at the beginning
        while (!backwardStack.isEmpty()) {
            forwardStack.push(backwardStack.pop());
        }
        backwardStack.push(data);
    }

    public void insertAtEnd(int data) {
        // Move all elements to backwardStack to insert at the end
        while (!forwardStack.isEmpty()) {
            backwardStack.push(forwardStack.pop());
        }
        backwardStack.push(data);
    }

    public void moveForward() {
        if (backwardStack.isEmpty()) {
            System.out.println("No more elements to move forward.");
            return;
        }
        System.out.println("Moving forward: " + backwardStack.peek());
        forwardStack.push(backwardStack.pop());
    }

    public void moveBackward() {
        if (forwardStack.isEmpty()) {
            System.out.println("No more elements to move backward.");
            return;
        }
        System.out.println("Moving backward: " + forwardStack.peek());
        backwardStack.push(forwardStack.pop());
    }

    public void delete(int data) {
        Stack<Integer> tempStack = new Stack<>();
        boolean deleted = false;

        while (!backwardStack.isEmpty()) {
            if (backwardStack.peek() == data && !deleted) {
                backwardStack.pop();
                deleted = true;
                break;
            }
            tempStack.push(backwardStack.pop());
        }

        while (!tempStack.isEmpty()) {
            backwardStack.push(tempStack.pop());
        }

        if (!deleted) {
            System.out.println("Element not found.");
        }
    }

    public void deleteFirst() {
        if (backwardStack.isEmpty()) {
            System.out.println("List is empty.");
            return;
        }
        backwardStack.pop();
    }

    public void deleteLast() {
        while (!forwardStack.isEmpty()) {
            backwardStack.push(forwardStack.pop());
        }
        if (!backwardStack.isEmpty()) {
            backwardStack.pop();
        } else {
            System.out.println("List is empty.");
        }
    }
}

```
<br>



#### Explore all 46 answers here 👉 [Devinterview.io - Stack Data Structure](https://devinterview.io/questions/data-structures-and-algorithms/stack-data-structure-interview-questions)

<br>

<a href="https://devinterview.io/questions/data-structures-and-algorithms/">
<img src="https://firebasestorage.googleapis.com/v0/b/dev-stack-app.appspot.com/o/github-blog-img%2Fdata-structures-and-algorithms-github-img.jpg?alt=media&token=fa19cf0c-ed41-4954-ae0d-d4533b071bc6" alt="data-structures-and-algorithms" width="100%">
</a>
</p>

