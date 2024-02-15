# Top 35 Fibonacci Sequence Interview Questions

<div>
<p align="center">
<a href="https://devinterview.io/questions/data-structures-and-algorithms/">
<img src="https://firebasestorage.googleapis.com/v0/b/dev-stack-app.appspot.com/o/github-blog-img%2Fdata-structures-and-algorithms-github-img.jpg?alt=media&token=fa19cf0c-ed41-4954-ae0d-d4533b071bc6" alt="data-structures-and-algorithms" width="100%">
</a>
</p>

#### You can also find all 35 answers here 👉 [Devinterview.io - Fibonacci Sequence](https://devinterview.io/questions/data-structures-and-algorithms/fibonacci-sequence-interview-questions)

<br>

## 1. What is the _Fibonacci sequence_?

The **Fibonacci Sequence** is a series of numbers where each number $F(n)$ is the sum of the two preceding ones, often starting with 0 and 1. That is:

$$
F(n) = F(n-1) + F(n-2)
$$

with initial conditions

$$
F(0) = 0, \quad F(1) = 1
$$

### Golden Ratio

The ratio of consecutive Fibonacci numbers approximates the Golden Ratio ($\phi \approx 1.6180339887$):

$$
\lim_{{n \to \infty}} \frac{{F(n+1)}}{{F(n)}} = \phi
$$

### Real-World Occurrences

The sequence frequently manifests in nature, such as in flower petals, seedhead spirals, and seashell growth patterns.

### Visual Representation

![Fibonacci Spiral](https://firebasestorage.googleapis.com/v0/b/dev-stack-app.appspot.com/o/fibonacci-sequence%2Fwhat-is-fibonacci-sequence%20%20.svg?alt=media&token=023579c2-a056-4aa8-8af5-a82082d3a621)

### Code Example: Calculating The Nth Fibonacci Number

Here is the Python code:

```python
# Using recursion
def fibonacci_recursive(n):
    if n <= 0: return 0
    elif n == 1: return 1
    return fibonacci_recursive(n-1) + fibonacci_recursive(n-2)
    
# Using dynamic programming for efficiency
def fibonacci_dynamic(n):
    fib = [0, 1]
    for i in range(2, n+1):
        fib.append(fib[-1] + fib[-2])
    return fib[n]
```
<br>

## 2. Write a function to calculate the _nth Fibonacci number_ using a _recursive_ approach.

### Problem Statement

The task is to write a function that **returns the $n$th Fibonacci number** using a **recursive approach**.

### Solution

The **naive recursive** solution for the Fibonacci series, while easy to understand, is inefficient due to its exponential time complexity of $O(2^n)$.

**Dynamic Programming** methods like memoization and tabulation result in optimized time complexity.

#### Algorithm Steps

1. Check for the base cases, i.e., if $n$ is 0 or 1.
2. If not a base case, **recursively** compute $F(n-1)$ and $F(n-2)$.
3. Return the sum of the two recursive calls.

#### Implementation

Here is the Python code:

```python
def fib_recursive(n):
    if n <= 1:
        return n
    return fib_recursive(n-1) + fib_recursive(n-2)
```

#### Complexity Analysis

- **Time Complexity**: $O(2^n)$. This is due to the two recursive calls made at each level of the recursion tree, resulting in an exponential number of function calls.

- **Space Complexity**: $O(n)$. Despite the inefficient time complexity, the space complexity is $O(n)$ as it represents the depth of the recursion stack.
<br>

## 3. Provide a non-recursive implementation for generating the _Fibonacci sequence_ to the _nth number_.

### Problem Statement

The task is to **generate** the Fibonacci sequence to the $n$th number using a **non-recursive approach**.

### Solution

While recursion offers a straightforward solution for the Fibonacci sequence, it has performance and stack overflow issues. A **non-recursive** approach, often based on a **loop** or **iterative method**, overcomes these limitations.

Here, I'll present both **binet's formula** and an **iterative method** as the non-recursive solutions.

#### Binet's Formula

$$
F(n) = \frac{{\phi^n - \psi^n}}{{\sqrt 5}}
$$

where:

- $\phi = \frac{{1 + \sqrt 5}}{2}$ (the golden ratio)
- $\psi = \frac{{1 - \sqrt 5}}{2}$

#### Algorithm Steps

1. Compute $\phi$ and $\psi$.
2. Plug values into the formula for $F(n)$.

#### Complexity Analysis

- **Time Complexity**: $O(1)$
- **Space Complexity**: $O(1)$

**Note**: While Binet's formula offers an elegant non-recursive solution, it's sensitive to **floating-point errors** which can impact accuracy, especially for large $n$.

#### Iterative Method

#### Algorithm Steps

1. Initialize $\text{prev} = 0$ and $\text{curr} = 1$. These are the first two Fibonacci numbers.
2. For $i = 2$ to $n$, update $\text{prev}$ and $\text{curr}$ to be $\text{prev} + \text{curr}$ and $\text{prev}$ respectively. These become the next numbers in the sequence.

#### Complexity Analysis

- **Time Complexity**: $O(n)$
- **Space Complexity**: $O(1)$

#### Implementation

Here's the Python code for both methods:

### Binet's Formula

```python
import math

def fib_binet(n):
    phi = (1 + math.sqrt(5)) / 2
    psi = (1 - math.sqrt(5)) / 2
    return int((phi**n - psi**n) / math.sqrt(5))

# Output
print(fib_binet(5))  # Output: 5
```

### Iterative Method

```python
def fib_iterative(n):
    if n <= 1:
        return n
    prev, curr = 0, 1
    for _ in range(2, n + 1):
        prev, curr = curr, prev + curr
    return curr

# Output
print(fib_iterative(5))  # Output: 5
```

Both functions will return the 5th Fibonacci number.
<br>

## 4. What is the _time complexity_ of the recursive Fibonacci solution, and how can this be improved?

The **naive recursive** implementation of the Fibonacci sequence has a time complexity of $O(2^n)$, which can be optimized using techniques such as **memoization** or employing an **iterative approach**.

### Naive Recursive Approach

This is the straightforward, but inefficient, method.

#### Algorithm

1. **Base Case**: Return 0 if $n = 0$ and 1 if $n = 1$.
2.  **Function Call**: Recur on the sum of the $n-1$ and $n-2$ elements.

#### Python Code

Here is the Python code:

```python
def fibonacci_recursive(n):
    if n <= 1:
        return n
    return fibonacci_recursive(n-1) + fibonacci_recursive(n-2)
```

#### Complexity Analysis

- **Time Complexity**: $O(2^n)$ - As each call branches into two more calls (with the exception of the base case), the number of function calls grows exponentially with $n$, resulting in this time complexity.
- **Space Complexity**: $O(n)$ - The depth of the recursion stack can go up to $n$ due to the $n-1$ and $n-2$ calls.

### Memoization for Improved Efficiency

Using **memoization** allows for a noticeable performance boost.

#### Algorithm

1. Initialize a cache, `fib_cache`, with default values of -1. 
2. **Base Case**: If the $n$th value is already calculated (i.e., `fib_cache[n] != -1`), return that value. Otherwise, calculate the $n$th value using recursion.
3. **Cache the Result**: Once the $n$th value is determined, store it in `fib_cache` before returning.

#### Python Code

Here is the Python code:

```python
def fibonacci_memo(n, fib_cache={0: 0, 1: 1}):
    if n not in fib_cache:
        fib_cache[n] = fibonacci_memo(n-1, fib_cache) + fibonacci_memo(n-2, fib_cache)
    return fib_cache[n]
```

#### Performance Analysis

- **Time Complexity**: $O(n)$ - Each $n$ is computed once and then stored in the cache, so subsequent computations are $O(1)$.
- **Space Complexity**: $O(n)$ - The space used by the cache.

### Iterative Method for Superior Efficiency

The **iterative approach** shines in terms of time and space efficiency.

#### Algorithm

1. Initialize `a` and `b` as 0 and 1, respectively.
2. **Loop**: Update `a` and `b` to the next two Fibonacci numbers, replacing them as necessary to ensure they represent the desired numbers.
3. **Return**: `a` after the loop exits, since it stores the $n$th Fibonacci number.

#### Python Code

Here is the Python code:
```python
def fibonacci_iterative(n):
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a
```

#### Performance Analysis

- **Time Complexity**: $O(n)$ - The function iterates a constant number of times, depending on $n$.
- **Space Complexity**: $O(1)$ - The variables `a` and `b` are updated in place without utilizing any dynamic data structures or recursion stacks.
<br>

## 5. Describe the _memoization_ technique as applied to the _Fibonacci sequence_ calculation.

**Memoization** is a technique that makes dynamic programming faster by storing the results of expensive function calls and reusing them.

To apply memoization to the **Fibonacci sequence**, a list or dictionary (**array** or **hashmap** in terms of computer science) is used to store the intermediate results, essentially turning the calculation process into a more optimized dynamic programming algorithm.

### Code Example: Memoized Fibonacci Calculation

Here is the Python code:

```python
def fibonacci_memo(n, memo={0: 0, 1: 1}):
    if n not in memo:
        memo[n] = fibonacci_memo(n-1, memo) + fibonacci_memo(n-2, memo)
    return memo[n]

# Test
print(fibonacci_memo(10))  # Outputs: 55
```
<br>

## 6. Implement an _iterative solution_ to generate the _Fibonacci sequence_, discussing its _time_ and _space complexity_.

### Problem Statement

The task is to **generate the Fibonacci sequence** using an **iterative** approach, and then analyzing its **time and space complexity**.

### Solution

#### Iterative Algorithm  

1. Initialize `a = 0` and `b = 1`.
2. Use a loop to update `a` and `b`. On each iteration:
   - Update `a` to the value of `b`.
   - Update `b` to the sum of its old value and the old value of `a`.
3. Repeat the loop 'n-1' times, where 'n' is the desired sequence length.

#### Visual Representation

Here's how the first few Fibonacci numbers are computed:

- Iteration 1: $a = 0, \, b = 1, \, b = 0 + 1 = 1$
- Iteration 2: $a = 1, \, b = 1, \, b = 1 + 1 = 2$
- Iteration 3: $a = 1, \, b = 2, \, b = 2 + 1 = 3$
- Iteration 4: $a = 2, \, b = 3, \, b = 3 + 2 = 5$
- And so on...

#### Complexity Analysis

- **Time Complexity**: $O(n)$ — This is more efficient than the recursive approach which is $O(2^n)$.
- **Space Complexity**: $O(1)$ — The space used is constant, regardless of the input 'n'.

#### Implementation

Here is the Python code:

```python
def fibonacci_iterative(n):
    if n <= 0:
        return "Invalid input. n must be a positive integer."

    fib_sequence = [0] if n >= 1 else []

    a, b = 0, 1
    for _ in range(2, n+1):
        fib_sequence.append(b)
        a, b = b, a + b

    return fib_sequence

# Example usage
print(fibonacci_iterative(10)) # Output: [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]
```
<br>

## 7. Explain the concept of _dynamic programming_ as it relates to the _Fibonacci sequence_.

**Dynamic Programming** is a powerful algorithmic technique that is widely used to optimize **recursive problems**, like computing the Fibonacci sequence, by avoiding redundant computations.

### Efficient Fibonacci Calculation

The naive method of Fibonacci computation is highly inefficient, often taking exponential time. Dynamic Programming offers better time complexity, often linear or even constant, without sacrificing accuracy.

#### Memoization and Caching

The most common way of optimizing Fibonacci computations is through **memoization**, where function calls are stored with their results for future reference. In Python, you can employ decorators or dictionaries to achieve this.

Here is the Python code:

```python
def memoize(fib):
    cache = {}
    def wrapper(n):
        if n not in cache:
            cache[n] = fib(n)
        return cache[n]
    return wrapper

@memoize
def fib(n):
    if n < 2:
        return n
    return fib(n-1) + fib(n-2)

# Test
print(fib(10))  # 55
```
<br>

## 8. Solve the _nth Fibonacci number_ problem using _matrix exponentiation_ and analyze its efficiency.

### Problem Statement

The task is to compute the $n$th Fibonacci number using **matrix exponentiation** and analyze its efficiency.

### Solution

Matrix exponentiation offers an optimal $O(\log n)$ solution for the Fibonacci sequence, in contrast to the traditional recursive method that has a time complexity of $O(2^n)$.

#### Algorithm Steps

1. Represent the **Fibonacci transformation** as $F = \begin{bmatrix} 1 & 1 \\ 1 & 0 \end{bmatrix}$.
2. Utilize **exponentiation by squaring** to efficiently compute $F^n$ for large $n$.
3. Extract the $n$-th Fibonacci number as the top right element of the resultant matrix.

#### Complexity Analysis

- **Time Complexity**: $O(\log n)$ due to the efficiency of exponentiation by squaring.
- **Space Complexity**: $O(\log n)$

#### Implementation

Here is the Python code:

```python
import numpy as np

def matrix_power(A, n):
    if n == 1:
        return A
    if n % 2 == 0:
        half = matrix_power(A, n // 2)
        return np.dot(half, half)
    else:
        half = matrix_power(A, (n - 1) // 2)
        return np.dot(np.dot(half, half), A)

def fibonacci(n):
    if n <= 0:
        return 0
    F = np.array([[1, 1], [1, 0]])
    result = matrix_power(F, n - 1)
    return result[0, 0]

# Example usage
print(fibonacci(6))  # Output: 8
```
<br>

## 9. Can the _nth Fibonacci number_ be found using the _golden ratio_, and if so, how would you implement this method?

The $n$-th term of the **Fibonacci sequence** can be approximated using the **Golden Ratio** $(\phi \approx 1.61803)$ through Binet's formula:

$$ F(n) \approx \frac{{\phi^n}}{{\sqrt{5}}} $$

### Code Example: Fibonacci with Golden Ratio

Here is the Python code:

```python
import math

def approximate_fibonacci(n):
    golden_ratio = (1 + math.sqrt(5)) / 2
    return round(golden_ratio ** n / math.sqrt(5))

# Test
print(approximate_fibonacci(10))  # Output: 55
```
<br>

## 10. Present an approach to precomputing _Fibonacci numbers_ to answer multiple _nth Fibonacci queries_ efficiently.

### Problem Statement

The objective is to compute $\text{Fib}(n)$, the $n$th Fibonacci number, efficiently for **multiple queries**.

### Solution

A **precomputation** strategy is suitable for scenarios where `Fibonacci` numbers are repeatedly requested over a **fixed range**.

#### Precomputation Table

1. Generate and store Fibonacci numbers from 0 to the highest $n$ using an **array** or **dictionary**. This operation has a time complexity of $O(n)$.
2. Subsequent queries are answered directly from the precomputed table with a time complexity of $O(1)$.

#### Complexity Analysis

- Precomputation: $O(\text{max\_n})$
- Query time: $O(1)$

#### Realization

Let's consider Python as our programming language.

#### Code

Here is a Python function that precomputes Fibonacci numbers up to a certain limit and then returns the $n$th Fibonacci number based on the precomputed table.

```python
def precompute_fibonacci(max_n):
    fib = [0, 1]
    a, b = 0, 1

    while b < max_n:
        a, b = b, a + b
        fib.append(b)

    return fib

def fibonacci(n, fib_table):
    return fib_table[n] if n < len(fib_table) else -1

# Usage
max_n = 100
fib_table = precompute_fibonacci(max_n)
print(fibonacci(10, fib_table))  # 55
```
<br>

## 11. How might the _Fibonacci sequence_ be altered to start with two arbitrary initial values? Provide an algorithm for such a sequence.

Modifying the **Fibonacci sequence** to start with arbitrary initial values still leads to a unique sequence.

The iterative approach can handle custom starting values and **compute any term** in the sequence through the specified algorithm.

### Algorithm: Custom Fibonacci Sequence

1. **Input**: Start values `a` and `b` (with a not equal to b) and target term `n`.
2. Check if `n` is 1 or 2. If it is, return the corresponding start value.
3. Otherwise, execute a loop `n-2` times and update the start values.
4. Compute the `n`-th term once the loop concludes.

### Code Example: Custom Fibonacci Sequence

Here is the Python code:

```python
def custom_fibonacci(a, b, n):
    if n == 1:
        return a
    elif n == 2:
        return b
    
    for _ in range(n-2):
        a, b = b, a+b
    return b

# Example with start values 3 and 4 for the 6th term
result = custom_fibonacci(3, 4, 6)  # Output: 10
```
<br>

## 12. Explain an algorithm to compute the _sum of the first n Fibonacci numbers_ without generating the entire sequence.

The **Fibonacci sequence** is a classic mathematical series defined by the following:

$$
F(n) = 
\begin{cases} 
0 & \text{if } n = 0 \\
1 & \text{if } n = 1 \\
F(n-1) + F(n-2) & \text{if } n > 1 
\end{cases}
$$

### Direct Formulas for Sum and Generalization

1. **Sum of First n Numbers**: The sum of the first $n$ Fibonacci numbers is equal to $F(n+2) - 1$.
  
  This can be computed using a simple, **iterative function**:

  $$ \text{Sum}(n) = F(n+2) - 1 $$

2. **n-th Fibonacci Number**: It can be calculated using **two seed values** $F(0)$ and $F(1)$. 

  The general form is:

  $$ F(n) = F(0)A + F(1)B $$

  **Matrix Representation**:
  $$
  \begin{bmatrix}
  F(n) \\
  F(n-1) \\
  \end{bmatrix}
  = 
  \begin{bmatrix}
  1 & 1 \\
  1 & 0 \\
  \end{bmatrix}^{(n-1)}
  \begin{bmatrix}
  F(1) \\
  F(0) \\
  \end{bmatrix}
  $$

  Where:
  - $A$ and $B$ vary based on the initial seed values.
  - The matrix is multiplied by itself $(n-1)$ times.

### Code Example: Sum of First $n$ Fibonacci Numbers

Here is the Python code:

  ```python
  def fibonacci_sum(n):
      fib_curr, fib_next, fib_sum = 0, 1, 0

      for _ in range(n):
          fib_sum += fib_curr
          fib_curr, fib_next = fib_next, fib_curr + fib_next

      return fib_sum
  ```

The time complexity is $O(n)$, which is significantly better than $O(n\log n)$ when computing the $n$-th Fibonacci number using matrix exponentiation.
<br>

## 13. Define the _Lucas sequence_ and detail how a program can generate it.

The **Lucas Sequence**, a variant of the Fibonacci sequence, starts with 2 and 1, rather than 0 and 1, and follows the same recursive structure as the classic sequence:

$$
\text{Lucas}(n) = \begin{cases}
2, & \text{if } n = 0, \\
1, & \text{if } n = 1, \\
\text{Lucas}(n-1) + \text{Lucas}(n-2), & \text{otherwise}.
\end{cases}
$$

### Advantages of the Lucas Sequence

Compared to the Fibonacci sequence, the Lucas sequence offers:

- **Simpler Recurrence Relation**: The Lucas sequence uses only addition for its recursive relation, which can be computationally more efficient than Fibonacci's addition and subtraction.

- **Alternate Closed-Form Expression**: While the closed-form formula for the $n$th term of the Fibonacci sequence involves radicals, the Lucas sequence provides an alternate expression that can be easier to work with.
<br>

## 14. How can the _Fibonacci sequence_ be used to solve the _tiling problem_, where you need to cover a _2xn rectangle_ with _2x1 tiles_?

The **Fibonacci sequence** is closely related to the problem of covering a 2xN rectangle with 2x1 tiles, often referred to as the "tiling problem". It in fact offers a direct solution to this problem.

### Relationship Between Tiling and Fibonacci Sequence

Covering a 2xN rectangle $R_{1}$ with 2x1 tiles can be understood in terms of the number of ways to cover the last column, whether with a single vertical tile or two horizontal tiles:

$$
W_{1}(2xN) = W(2x(N-1)) + W(2x(N-2))
$$

This is a recursive relationship similar to the one used to define the Fibonacci numbers, making it clear that there is a connection between the two.

### Code Example: Tiling a 2xN Rectangle

Here is the Python code:

```python
def tiling_ways(n):
    a, b = 1, 1
    for _ in range(n):
        a, b = b, a + b
    return a

n = 5
ways_to_tile = tiling_ways(n)
print(f'Number of ways to tile a 2x{n} rectangle: {ways_to_tile}')
```

In this example, `a, b = b, a + b` is a compact way to **update** `a` and `b`. It relies on the fact that the right-hand side of the assignment is _evaluated first_ before being assigned to the left-hand side.

This approach has a **time complexity** of $O(N)$ and a **space complexity** of $O(1)$ since it only requires two variables.
<br>

## 15. How to calculate large _Fibonacci numbers_ without encountering _integer overflow_ issues?

**Fibonacci numbers** grow at a rapid rate, which can lead to **integer overflow**. Numerous strategies exist to circumvent this issue.

### Various Techniques to Handle Overflow 

1. **Using a Data Type with Larger Capacity**:
    The `long long int` data type in C/C++ provides up to 19 digits of precision, thus accommodating Fibonacci numbers up to $F(92)$.
  
2. **Using Built-in Arbitrary Precision Libraries**:
    Certain programming languages such as Python and Ruby come with **arbitrary-precision** arithmetic support, making them suited for such computations.

3. **Implementing Custom Arithmetic**:
    Libraries like GMP (GNU Multiple Precision Arithmetic Library) and `bigInt` in Java enable the handling of arbitrary-precision operations. Additionally, rolling out a custom arithmetic procedure through **arrays, linked lists, or strings** is viable.

### Code Example: Using `long long int` for Larger Capacity

Here is the C++ code:

  ```cpp
    #include <iostream>
    #include <vector>
    using namespace std;
    
    long long int fibonacci(int n) {
        if (n <= 1) return n;
    
        long long int a = 0, b = 1, c;
        for (int i = 2; i <= n; ++i) {
            c = a + b;
            a = b;
            b = c;
        }
        return b;
    }
  
    int main() {
        int n = 100;
        cout << "Fibonacci number at position " << n << " is: " 
             << fibonacci(n) << endl;
        return 0;
    }
  ```


#### Explore all 35 answers here 👉 [Devinterview.io - Fibonacci Sequence](https://devinterview.io/questions/data-structures-and-algorithms/fibonacci-sequence-interview-questions)

<br>

<a href="https://devinterview.io/questions/data-structures-and-algorithms/">
<img src="https://firebasestorage.googleapis.com/v0/b/dev-stack-app.appspot.com/o/github-blog-img%2Fdata-structures-and-algorithms-github-img.jpg?alt=media&token=fa19cf0c-ed41-4954-ae0d-d4533b071bc6" alt="data-structures-and-algorithms" width="100%">
</a>
</p>

