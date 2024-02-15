# 54 Must-Know Divide and Conquer Interview Questions

<div>
<p align="center">
<a href="https://devinterview.io/questions/data-structures-and-algorithms/">
<img src="https://firebasestorage.googleapis.com/v0/b/dev-stack-app.appspot.com/o/github-blog-img%2Fdata-structures-and-algorithms-github-img.jpg?alt=media&token=fa19cf0c-ed41-4954-ae0d-d4533b071bc6" alt="data-structures-and-algorithms" width="100%">
</a>
</p>

#### You can also find all 54 answers here 👉 [Devinterview.io - Divide and Conquer](https://devinterview.io/questions/data-structures-and-algorithms/divide-and-conquer-interview-questions)

<br>

## 1. Define _Divide & Conquer_ algorithms and their main characteristics.

**Divide & Conquer** is a problem-solving approach that involves breaking a problem into smaller, more easily solvable **subproblems**, solving each subproblem independently, and then combining their solutions to solve the original problem.

The strategy is typically implemented with **recursive algorithms**, with well-defined steps that make it easy to break the problem into smaller chunks and to reassemble the solutions into a final result.

### Core Process

1. **Divide**: Break the problem into smaller, more easily solvable subproblems.
2. **Conquer**: Solve these subproblems independently, typically using recursion.
3. **Combine**: Combine the solutions of the subproblems to solve the original problem.

### Key Characteristics

- **Efficiency**: Divide & Conquer is often more efficient than alternative methods, such as the Brute-Force approach.
- **Recursiveness**: The divide & conquer approach is frequently implemented through recursive algorithms.
- **Subproblem Independence**: Efficiency is achieved through solving subproblems independently.
- **Merging**: Combining subproblem solutions into a global solution, often through operations like **merging** or **addition**, is a key component. This step might take $O(n\log n)$ or $O(n)$ time, depending on the specific problem.
- **Divide Threshold**: There's typically a base case, defining the smallest division to solve the problem directly instead of further dividing it, to avoid infinite recursion.
- **Parallelism**: Some Divide & Conquer algorithms can be efficiently parallelized, making them attractive for multi-core processors and parallel computing environments.

### Best Practices

- **Simplicity**: Choose straightforward and direct methods to solve the subproblems, whenever possible.
  
- **Optimize**: Aim to solve subproblems in such a way that their solutions are selves used in each other's solutions as little as possible. This aids in reducing overall time complexity.
  
- **Adaptation**: Algorithms implementing Divide & Conquer might incorporate tweaks based on the specific domain or system requirements for enhanced efficiency.

### Divisibility

In many cases, the **even or uneven split** of the input dataset among the subproblems can be optimized for computational efficiency. Selecting the method that best suits the nature of the problem can be crucial for performance. For example, quicksort is generally deployed with an uneven split, while merge-sort uses an even one.
<br>

## 2. Explain the difference between _Divide & Conquer_ and _Dynamic Programming_.

**Divide and Conquer** and **Dynamic Programming** (DP) are both algorithmic design paradigms that decompose problems into smaller, more manageable subproblems. The techniques are closely related, often characterized by overlapping features. However, they differ fundamentally at a granular level of problem decomposition, **solutions to subproblems**, and the **mechanism of subproblem reuse**.

### Key Distinctions

#### Problem Decomposition

- **Divide and Conquer**: Breaks the problem into independent parts, usually halves, and solves the parts individually. Examples include quicksort and binary search.

- **Dynamic Programming**: Decomposes the problem into interrelated subproblems, often along a sequence or array. Solutions to larger problems are built from smaller, overlapping subproblem solutions.

#### Subproblem Solutions

- **Divide and Conquer**: The subproblem solutions are computed independently and aren't revisited or updated. This technique relies on "no-information sharing" among subproblems.

- **Dynamic Programming**: Subproblem solutions are computed and might be updated multiple times, enabling the reusability of results across the problem space.

#### Subproblem Reuse

- **Divide and Conquer**: Does not explicitly focus on subproblem reuse. In scenarios where subproblems are solved more than once, optimality in terms of repeated computation isn't guaranteed.

- **Dynamic Programming**: **Emphasizes subproblem reuse**. The algorithm's efficiency and optimality stem from the repeated usage of computed subproblem solutions, leading to a reduced and often polynomial running time.

#### Convergence

- **Divide and Conquer**: At each step, the algorithm gains progress in solving the problem, usually by reducing the problem's size or scope. The solution is derived once the subproblems become trivial (base cases) and are solved individually.

- **Dynamic Programming**: Progress in solving the problem is achieved through the iterative resolution of overlapping subproblems, gradually building towards the solution to the main problem. The solution is obtained after solving all relevant subproblems.

### Practical Applications

- **Divide and Conquer**: Suited for problems like sorting and ordination (quicksort, mergesort), list searching (binary search), and in problems where subproblems are solved independently.
  
- **Dynamic Programming**: Ideal for optimization problems and tasks featuring overlapping subproblems, such as making change (currency), finding the most efficient route (graph theory), and sequence alignment in bioinformatics.
<br>

## 3. What is the role of recursion in _Divide & Conquer_ algorithms?

**Divide & Conquer** algorithms solve complex tasks by breaking them into easier, equivalent sub-problems.

This strategy can be defined through the following sequence, called the **DAC Triad**:

- **Divide**: Decompose the problem into independent, smaller structures.
- **Abstract**: Tailor a mechanism to quantify the structure's individual patterns.
- **Combine**: Use partial solutions to assimilate a unified answer.

Throughout this process, **recursion** stands as a key organizing principle, serving different roles at each stage of the DAC Triad.
<br>

## 4. What are the three main steps in a typical _Divide & Conquer_ algorithm?

**Divide and Conquer** algorithms aim to break down problems into **smaller, more manageable** parts before solving them. They typically follow three fundamental steps: **Divide**, **Conquer**, and **Combine**.

### Key Steps in Divide and Conquer Algorithms

- **Divide**: This step involves breaking the problem into smaller, more manageable sub-problems. Ideally, the division results in sub-problems being independent tasks that can be solved in parallel (if resources permit).

- **Conquer**: In this step, each of the smaller sub-problems is solved separately, typically using recursion.

- **Combine**: Once the smaller sub-problems are solved, the results are merged to provide the solution to the original problem.
<br>

## 5. Give an example of a _recurrence relation_ that can describe the _time complexity_ of a _Divide & Conquer_ algorithm.

The **merge sort** algorithm, which follows a **Divide & Conquer** strategy, can be characterized by the following **recurrence relation**:

$$
T(n) = \begin{cases} 2T\left(\frac{n}{2}\right) + cn, & \text{if } n > 1 \\
c, & \text{if } n = 1 \end{cases}
$$

where:
- $T(n)$ represents the time complexity of merge sort on a list of size $n$.
- The initial term represents the two partitions of the list, each being sorted recursively with time complexity $T\left(\frac{n}{2}\right)$.
- $cn$ models the linear-time combine or merge operation.

This relation simplifies to $T(n) = n \log n$ with the help of the **Master Theorem**.

### Complexity Breakdown

- **Divide**: Requires $\log_2 n$ steps to partition the list.
- **Conquer**: Each sub-list of size $\frac{n}{2}$ is sorted in $\frac{n}{2} \log \frac{n}{2}$ time, which reduces to $n \log n$.
- **Combine**: The two sorted sub-lists are merged in $O(n)$ time.

Combining these steps yields the time complexity $T(n) = n \log n$.
<br>

## 6. Explain the _Master Theorem_ and its importance in analyzing _Divide & Conquer_ algorithms.

The **Master Theorem** provides a powerful tool to analyze the time complexity of algorithms that follow a **Divide and Conquer** paradigm.

This theorem focuses on the time complexity of algorithms that perform the following steps:

1. **Divide**: Break down the problem into a smaller set of subproblems.
2. **Conquer**: Solve each subproblem recursively.
3. **Combine**: Merge the solutions of the subproblems to form the solution of the original problem.

The Master Theorem utilizes a **recursive formula**, expressed as $T(n) = aT(n/b) + f(n)$, highlighting the number of subproblems, their size relative to the original problem, and the work done outside of the divide-and-conquer component.

### Master Theorem: Equation Components

- $a$: The number of **recursive** subproblems. Divide-and-conquer algorithms often split the problem into a fixed number of subproblems.
- $b$: The factor by which the input size is **reduced** in each subproblem.
- $f(n)$: The time complexity **outside** of the recursive call, such as the time to partition the input or combine results.

### Master Theorem: Assumptions

1. **Equal division**: The problem is divided into $a$ equal subproblems.
2. **Constant work for divide and combine steps**: The divide and combine steps have constant work, such as from operations that are $O(1)$.

### Master Theorem: Three Cases

#### Case 1: $f(n)$ is $O(n^c)$ where $c < \log_b a$
If $f(n)$ grows slower than the $n^c$ term and the number of divisions ($a$) is not too large compared to the size ($n$ raised to the power of $1/\log_b a$), then the work outside of the divisions is dominated by the divisions.

#### Case 2: $f(n)$ is $O(n^c)$ where $c = \log_b a$
This term is commonly referred to as the "balanced" term. It arises when the work outside of the divide stage is of the same order as the work attributable to the divide stage.

#### Case 3: $f(n)$ is $O(n^c)$ where $c > \log_b a$
In this case, the work outside the divisions dominates the work inside the divisions.

### Master Theorem: Advantages and Limitations

- **Advantages**: It offers a swift method for determining the time complexity of many divide-and-conquer algorithms.
- **Limitations**: It's tailored to a specific problem structure and makes some simplifying assumptions, such as equal-sized subproblems. When these assumptions don't hold, the theorem may not give the most precise time complexity.

### Code Example: Merge Sort and the Master Theorem

Here is the Python code:

```python
def merge_sort(arr):
    if len(arr) > 1:
        mid = len(arr) // 2
        left = arr[:mid]
        right = arr[mid:]

        merge_sort(left)  # Recursive call on half
        merge_sort(right)  # Recursive call on half

        # Merge step
        i, j, k = 0, 0, 0
        while i < len(left) and j < len(right):
            if left[i] < right[j]:
                arr[k] = left[i]
                i += 1
            else:
                arr[k] = right[j]
                j += 1
            k += 1

        while i < len(left):
            arr[k] = left[i]
            i += 1
            k += 1

        while j < len(right):
            arr[k] = right[j]
            j += 1
            k += 1


# As we can see in the code, Merge Sort divides the array into two halves in each recursive call,
# which satisfies the divide-and-conquer requirements.
# The merge step also takes $O(n)$ time in this case.
# Therefore, using the Master Theorem, we can efficiently determine the time complexity of Merge Sort.
# We can see that $a = 2, b = 2, \text{ and } f(n) = O(n)$, which fits the second case of the Master Theorem.
# Hence the time complexity of Merge Sort is $O(n \log n)$.
```
<br>

## 7. How can the _Master Theorem_ be applied to find the _time complexity_ of a _binary search algorithm_?

The Master Theorem provides a way to determine the time complexity of algorithms that follow a specific **divide-and-conquer** pattern.

It is best applied to recursive algorithms with equal splits or near-equal splits $a = 1$ where $b \approx 2$, and it estimates the time complexity in terms of $T(n) = a \cdot T(n/b) + f(n)$.

### Master Theorem's Three Cases

1. **Case 1 (Ruled out for Binary Search)**: If $f(n)$ is polynomially smaller than $n^b$ (i.e., $f(n) = O(n^{\log_b a - \epsilon})$ for some $\epsilon > 0$), the solution is $T(n) = \Theta(n^{\log_b a})$. For binary search $f(n)$ is $\Theta(1)$, so this case doesn't apply.

2. **Case 3 (Also Ruled out for Binary Search)**: If $f(n)$ is polynomially greater than $n^b$ (i.e., $f(n) = \Omega(n^{\log_b a + \epsilon})$ for some $\epsilon > 0$) and $a \cdot f(n/b) \leq k \cdot f(n)$ for some constant $k < 1$ and $n$ sufficiently large, then, the solution is $T(n) = \Theta(f(n))$. Since $a = 1$ and $b = 2$, the condition $a \cdot f(n/b) \leq k \cdot f(n)$ for some $k$ and sufficiently large $n$ is not satisfied, so this case doesn't apply either.

3. **Case 2 (Applicable to Binary Search)**: This case is established when $f(n)$ is the same order as the divided subproblems, often represented using $f(n) = \Theta(n^{\log_b a})$. For algorithms not fitting into Cases 1&2, the time complexity is estimated to be $T(n) = \Theta(n^{\log_b a} \log n)$.

### Applying the Master Theorem to the Binary Search Algorithm

For binary search, the key recursive relationship is $T(n) = T(n/2) + 1$, where:

- $a = 1$: The number of subproblems is halved.
- $b = 2$: Each subproblem has half the size of the original problem.
- $f(n) = 1$: The work done in dividing the problem into subproblems.

Based on the key parameters, we have $f(n) = \Theta(1)$ and $n^{\log_b a} = n^{\log_2 1} = 1$. 

Since $f(n) = \Theta(1)$ is in the same order as the divided subproblems, this matches the characteristics of Case 2 of the Master Theorem. 

Therefore, we can conclude that the binary search algorithm has a time complexity of $T(n) = \Theta(\log n)$
<br>

## 8. Describe how you would use _Divide & Conquer_ to find the _maximum_ and _minimum_ of an _array_.

**Divide & Conquer** is an efficient technique for various problems, including finding the **maximum** and **minimum** values in an array.

### Divide & Conquer Algorithm

Here are the steps for the Divide & Conquer approach to find the maximum and minimum in an array $A$:

1. **Divide**: Split the array into two equal parts: $A_L$ and $A_R$.
2. **Conquer**: Recursively find the maximum and minimum in $A_L$ and $A_R$.
3. **Combine**: From the max and min values of $A_L$ and $A_R$, compare and select the universal maximum and minimum.

This algorithm works by leveraging the relationships between $A_L$ and $A_R$ and optimizing without unnecessary comparisons.

### Complexity Analysis

- **Time Complexity**: $T(n) = 2 \cdot T(n/2) + 2$ for $n \ge 2$ (two comparisons are done for the bases, $n = 1$ and $n = 2$). The solution is $O(n)$.
- **Space Complexity**: $O(\log n)$ due to the recursive call stack.

### Python Example

Here is the Python code:

```python
def find_max_min(arr, left, right):
    # Base case for 1 or 2 elements
    if right - left == 1:
        return max(arr[left], arr[right]), min(arr[left], arr[right])
    elif right - left == 0:
        return arr[left], arr[left]
    
    # Split array into two parts
    mid = (left + right) // 2
    max_l, min_l = find_max_min(arr, left, mid)
    max_r, min_r = find_max_min(arr, mid+1, right)
    
    # Combine results
    return max(max_l, max_r), min(min_l, min_r)

# Test the function
arr = [3, 2, 5, 1, 2, 7, 8, 8]
max_num, min_num = find_max_min(arr, 0, len(arr)-1)
print(f"Maximum: {max_num}, Minimum: {min_num}")
```
<br>

## 9. Illustrate how the _Merge Sort_ algorithm exemplifies the _Divide & Conquer_ technique.

**Merge Sort** is a classic algorithm that leverages the **Divide & Conquer** technique for effective sorting across different domains such as data management and external sorting. The process entails breaking down the initial problem (array of data to sort) into smaller, more manageable sub-problems. In the context of Merge Sort, this translates to repeatedly dividing the array into halves until it's not further divisible ('Divide' part). After that, it combines the sub-solutions in a manner that solves the original problem ('Conquer').

### Merge Sort: Divide & Conquer Steps

1. **Divide**: Partition the original array until individual elements remain.
2. **Conquer**: Sort the divided sub-arrays.
3. **Combine**: Merge the sub-arrays to produce a single, sorted output.

### Key Characteristics

- **Parallelizability**: Merge Sort can be optimized for efficient execution on multi-core systems due to its independent sub-array sorting.
- **Adaptability**: It's well-suited to external memory applications thanks to its "vertical" characteristic that minimizes I/O operations.
- **Stability**: This algorithm preserves the relative order of equal elements, making it valuable in certain data processing requirements.

### Complexity Analysis

- **Time Complexity**: Best, Average, Worst Case - O(n log n)
- **Space Complexity**: O(n)

### Algorithmic Steps and Visual Representation

1. **Divide**
    - Action: Recursively divide the array into two halves.
    - Visualization: Tree diagram with divided segments.

2. **Conquer**
    - Action: Sort the divided segments.
    - Visualization:  Visualize individual, sorted segments.

3. **Combine**
    - Action: Merge the sorted segments into a single, sorted array.
    - Visualization: Show the merging of sorted segments.
    
### Python Code Example: Merge Sort

Here is the code:

```python
def merge_sort(arr):
    if len(arr) > 1:
        mid = len(arr) // 2  # Finding the middle of the array
        left_half = arr[:mid]  # Dividing the array elements into 2 halves
        right_half = arr[mid:]

        merge_sort(left_half)  # Sorting the first half
        merge_sort(right_half)  # Sorting the second half

        i, j, k = 0, 0, 0

        # Merging the sorted halves
        while i < len(left_half) and j < len(right_half):
            if left_half[i] < right_half[j]:
                arr[k] = left_half[i]
                i += 1
            else:
                arr[k] = right_half[j]
                j += 1
            k += 1

        # If elements are remaining
        while i < len(left_half):
            arr[k] = left_half[i]
            i += 1
            k += 1
        
        while j < len(right_half):
            arr[k] = right_half[j]
            j += 1
            k += 1

    return arr
```
<br>

## 10. Explain how _Quicksort_ works and how it adopts the _Divide & Conquer_ strategy.

**Quicksort** is a highly efficient sorting algorithm that uses the **Divide and Conquer** strategy to quickly sort data. It does so by partitioning an array into two smaller arrays - one with elements that are less than a chosen pivot and another with elements that are greater.

### Core Algorithm Steps

#### 1. Partitioning

The algorithm selects a **pivot** from the array. Elements are then rearranged such that:
- Elements to the left of the pivot are less than the pivot.
- Elements to the right are greater than or equal to the pivot.

This process is known as partitioning.

#### 2. Recursion

After partitioning, two sub-arrays are created. The algorithm is then **recursively** applied to both sub-arrays.

### Implementation

Here is the Python code:

#### Complexity Analysis

- **Time Complexity**: 
    - Best & Average: $O(n \log n)$ - This is the time complexity of quick sort.  
    - Worst Case: $O(n^2)$ - This occurs when the array is already sorted and the last element is chosen as the pivot every time, leading to unbalanced partitions in each recursive step.
    
- **Space Complexity**: 
    - Best & Average: $O(\log n)$ -  Each recursive call uses a stack frame to maintain local variables. On average, since the algorithm is balanced after partitioning, the stack depth is $O(\log n)$.
    - Worst Case: $O(n)$ - This occurs when the partitioning process does not make a balanced split, requiring $O(n)$ stack space.
<br>

## 11. How does the _Karatsuba algorithm_ for _multiplying_ large numbers employ _Divide & Conquer_?

**Karatsuba algorithm** makes use of the **Divide & Conquer strategy** to significantly reduce the number of math operations needed for large number multiplication.

### Core Concept

When multiplying two numbers, say $X$ and $Y$, with $n$ digits, the Karatsuba algorithm partitions the numbers into **smaller, equal-sized sections** to efficiently compute the product.

Mathematically, the partitions are represented as:

$$
$$
X & = X_h \times 10^{\frac{n}{2}} + X_l \\
Y & = Y_h \times 10^{\frac{n}{2}} + Y_l
$$
$$

where:

- $X_h$ and $Y_h$ are the **high-order** digits of $X$ and $Y$ respectively.
- $X_l$ and $Y_l$ are the **low-order** digits of $X$ and $Y$ respectively.

### Divide & Conquer Strategy

The algorithm follows a set of **recursive steps** to efficiently compute $X \times Y$:

1. **Divide**: Split the numbers into high-order and low-order halves.
2. **Conquer**: Recursively compute the three products $X_h \times Y_h$, $X_l \times Y_l$, and $(X_h + X_l) \times (Y_h + Y_l)$.
3. **Combine**: Use these results to calculate the final product:
   $$
   X \times Y = X_h \times Y_h \times 10^n + (X_h \times Y_l + Y_h \times X_l) \times 10^{\frac{n}{2}} + X_l \times Y_l
   $$

By effectively employing **Divide & Conquer** in these three steps, the algorithm reduces the number of required products from four to three, resulting in a more efficient $O(n^{1.58})$ complexity as opposed to the traditional $O(n^2)$.
<br>

## 12. Describe the _Strassen's algorithm_ for _matrix multiplication using_ _Divide & Conquer_.

**Strassen's Algorithm** is a divide-and-conquer method that reduces the number of required operations for matrix multiplication.

While the standard matrix multiplication has a time complexity of $O(n^3)$, Strassen's Algorithm can achieve $O(n^{\log_2 7})$, which is approximately $O(n^{2.81})$.

### Key Concepts

- **Divide & Conquer**: The algorithm splits the input matrices into smaller submatrices, processes these recursively, and then combines them to get the result.

- **Strassen's Assumption**: The algorithm relies on 7 unique linear combinations of smaller submatrices to compute the product. Each combination only involves addition and subtraction, instead of using the conventional approach with 8 individual products.

### Algorithm Steps and Complexity

- **Step 1: Divide**: Divide the input matrix of size $n \times n$ into four submatrices of size $\dfrac n2 \times \dfrac n2$. This step has $O(1)$ complexity.

- **Step 2: Conquer**: Compute the seven matrix products of size $\dfrac n2 \times \dfrac n2$ using the four products from the previous step. This step has a time complexity of $T\left(\dfrac n2\right)$.

- **Step 3: Combine**: Combine the results from the previous step using five additions or subtractions. This step has $O(n^2)$ complexity.

### Recursive Algorithm

Here is the Python code:

```python
def strassen(matrix1, matrix2):
    n = len(matrix1)
    # Base case
    if n == 1:
        return [[matrix1[0][0] * matrix2[0][0]]]
    
    # Divide
    a11, a12, a21, a22 = split(matrix1)
    b11, b12, b21, b22 = split(matrix2)
    
    # Conquer
    p1 = strassen(add(a11, a22), add(b11, b22))
    p2 = strassen(add(a21, a22), b11)
    p3 = strassen(a11, sub(b12, b22))
    p4 = strassen(a22, sub(b21, b11))
    p5 = strassen(add(a11, a12), b22)
    p6 = strassen(sub(a21, a11), add(b11, b12))
    p7 = strassen(sub(a12, a22), add(b21, b22))
    
    # Combine
    c11 = add(sub(add(p1, p4), p5), p7)
    c12 = add(p3, p5)
    c21 = add(p2, p4)
    c22 = add(sub(add(p1, p3), p2), p6)

    return join(c11, c12, c21, c22)
```
<br>

## 13. How would you use a _Divide & Conquer_ approach to calculate the _power of a number_?

The **Divide and Conquer** technique for calculating the **power of a number** is based on breaking down even and odd cases, thus reducing the complexity to O(log n). The strategy focuses on **efficiency** and **minimizing multiplication operations**.

### Algorithm

1. **Base Case**: If the exponent is 0, return 1.
2. **Odd Exponent**: $x^m = x \cdot x^{m-1}$, e.g., If `m` is odd, call the function with $m-1$ since $x^{m-1}$ is an even exponent.
3. **Even Exponent**: $x^m = (x^{m/2})^2$, e.g., If `m` is even, call the function with $m/2$ and square the result.

### Code Example: Divide and Conquer Approach

Here is the Python code:

```python
def power(x, m):
    if m == 0:
        return 1
    elif m % 2 == 0:  # Even
        temp = power(x, m // 2)
        return temp * temp
    else:  # Odd
        temp = power(x, m - 1)
        return x * temp

# Test
print(power(2, 5))  # Result: 32
```

### Complexity Analysis

- **Time Complexity**: $O(\log m)$ - Each step reduces the exponent by a factor of 2.
- **Space Complexity**: $O(\log m)$ - Due to recursive calls.
<br>

## 14. Solve the _Tower of Hanoi_ problem using _Divide & Conquer_ techniques.

### Problem Statement

The Tower of Hanoi is a classic problem that consists of three rods and a number of disks of different sizes which can slide onto any rod. The **objective** is to move the entire stack to another rod, following these rules:

1. Only one disk can be moved at a time.
2. Each move consists of taking the top (smallest) disk from one of the stacks and placing it on top of the stack you're moving it to.
3. No disk may be placed on top of a smaller disk.

The problem can be solved with a **recursive** divide-and-conquer algorithm.

### Solution

The Tower of Hanoi problem can be elegantly solved using **recursion**. The key is to recognize the pattern that allows us to reduce the problem in a recursive form.

#### Algorithm Steps

1. **Base Case**: If there is only one disk, move it directly to the target peg.
2. **Recursive Step**:
   - Move the top $n-1$ disks from the source peg to the auxiliary peg (using the target peg as a temporary location).
   - Move the $n$th disk from the source peg to the target peg.
   - Move the $n-1$ disks from the auxiliary peg to the target peg (using the source peg as a temporary location if needed).

By breaking down the problem with this logic, we're effectively solving for smaller sets of disks, until it reduces to just one disk (the base case).

#### Complexity Analysis

- **Time Complexity**: $O(2^n)$ — Each recursive call effectively doubles the number of moves required. Though the actual number of calls is 3 for each disk, it can be approximated to $O(2^n)$ for simplicity.

- **Space Complexity**: $O(n)$ — This is the space used by the call stack during the recursive process. 

#### Implementation

Here is the Python code:

```python
def tower_of_hanoi(n, source, target, auxiliary):
    if n == 1:
        print(f"Move disk 1 from {source} to {target}")
        return
    tower_of_hanoi(n-1, source, auxiliary, target)
    print(f"Move disk {n} from {source} to {target}")
    tower_of_hanoi(n-1, auxiliary, target, source)

# Example
tower_of_hanoi(3, 'A', 'C', 'B')
```
<br>

## 15. Solve the _Closest Pair of Points_ problem using _Divide & Conquer_.

### Problem Statement

Given a set of $n$ points in the $2D$ plane, find the closest pair of points.

#### Example

Given Points: $(0, 2), (6, 67), (43, 71), (39, 107), (189, 140)$, the closest pair is $(6, 67)$ and $(43, 71)$.

### Solution

1. Sort points by $x$ coordinates, yielding left and right sets.
2. **Recursively** find the closest pairs in left and right sets.
3. Let $d = \min$ (minimum distance) from left and right sets.
4. Filter points within distance $d$ from the vertical **mid-line**.
5. Find the closest pair in this **strip**.

#### Algorithm Steps

1. Sort points based on $x$ coordinates.
2. Recursively find $d_{\text{left}}$ and $d_{\text{right}}$ in the left and right sets.
3. Set $d = \min(d_{\text{left}}, d_{\text{right}})$.
4. Construct a strip, $S$, of points where $|x - \text{midpoint}| < d$. Sort by $y$ coordinates.
5. For each point, compare with at most 7 nearby points (as they are sorted) and update $d$.

The **time complexity** is $O(n \log n)$, dominated by the sort step, while the **space complexity** is $O(n)$.

### Implementation

Here is the Python code:

```python
import math

# Calculate distance
def dist(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

# Find the closest pair of points in a strip of given size
def strip_closest(strip, size, d):
    # Initially the minimum distance is d
    min_val = d
    
    # Sort by y-coordinate
    strip.sort(key=lambda point: point[1])

    for i in range(size):
        j = i + 1
        while j < size and (strip[j][1] - strip[i][1]) < min_val:
            min_val = min(min_val, dist(strip[i], strip[j]))
            j += 1

    return min_val

# Find the closest pair of points 
def closest_pair(points):
    n = len(points)

    # If the number of points is less than 3, brute force it
    if n <= 3:
        return brute_force(points)

    # Sort points by x-coordinate
    points.sort(key=lambda point: point[0])
    
    # Midpoint
    mid = n // 2
    mid_point = points[mid]

    # Recursively solve sub-arrays
    left = points[:mid]
    right = points[mid:]
    
    # Minimum distance in left and right sub-arrays
    d_left = closest_pair(left)
    d_right = closest_pair(right)
    d = min(d_left, d_right)
    
    # Find points in the strip
    strip = [point for point in points if abs(point[0] - mid_point[0]) < d]
    
    # Compute strip distance
    return strip_closest(strip, len(strip), d)

# Brute force method
def brute_force(points):
    min_dist = float('inf')
    for i in range(len(points)):
        for j in range(i+1, len(points)):
            if dist(points[i], points[j]) < min_dist:
                min_dist = dist(points[i], points[j])
    return min_dist

# Example usage
points = [(0, 2), (6, 67), (43, 71), (39, 107), (189, 140)]
print("Closest distance is", closest_pair(points))
```
<br>



#### Explore all 54 answers here 👉 [Devinterview.io - Divide and Conquer](https://devinterview.io/questions/data-structures-and-algorithms/divide-and-conquer-interview-questions)

<br>

<a href="https://devinterview.io/questions/data-structures-and-algorithms/">
<img src="https://firebasestorage.googleapis.com/v0/b/dev-stack-app.appspot.com/o/github-blog-img%2Fdata-structures-and-algorithms-github-img.jpg?alt=media&token=fa19cf0c-ed41-4954-ae0d-d4533b071bc6" alt="data-structures-and-algorithms" width="100%">
</a>
</p>

