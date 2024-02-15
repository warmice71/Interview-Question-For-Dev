# Top 41 Greedy Algorithms Interview Questions

<div>
<p align="center">
<a href="https://devinterview.io/questions/data-structures-and-algorithms/">
<img src="https://firebasestorage.googleapis.com/v0/b/dev-stack-app.appspot.com/o/github-blog-img%2Fdata-structures-and-algorithms-github-img.jpg?alt=media&token=fa19cf0c-ed41-4954-ae0d-d4533b071bc6" alt="data-structures-and-algorithms" width="100%">
</a>
</p>

#### You can also find all 41 answers here 👉 [Devinterview.io - Greedy Algorithms](https://devinterview.io/questions/data-structures-and-algorithms/greedy-algorithms-interview-questions)

<br>

## 1. What is a _Greedy Algorithm_?

A **greedy algorithm** aims to solve optimization problems by making the **best local choice** at each step. While this often leads to an **optimal global solution**, it's not guaranteed in all cases. These algorithms are generally easier to implement and faster than other methods like Dynamic Programming but may not always yield the most accurate solution.

### Key Features
1. **Greedy-Choice Property**: Each step aims for a local optimum with the expectation that this will lead to a global optimum.
2. **Irreversibility**: Once made, choices are not revisited.
3. **Efficiency**: Greedy algorithms are usually faster, particularly for problems that don't require a globally optimal solution.

### Example Algorithms

#### Fractional Knapsack Problem
Here, the goal is to maximize the value of items in a knapsack with a fixed capacity. The greedy strategy chooses items based on their **value-to-weight ratio**.

```python
def fractional_knapsack(items, capacity):
    items.sort(key=lambda x: x[1]/x[0], reverse=True)
    max_value = 0
    knapsack = []
    for item in items:
        if item[0] <= capacity:
            knapsack.append(item)
            capacity -= item[0]
            max_value += item[1]
        else:
            fraction = capacity / item[0]
            knapsack.append((item[0] * fraction, item[1] * fraction))
            max_value += item[1] * fraction
            break
    return max_value, knapsack

items = [(10, 60), (20, 100), (30, 120)]
capacity = 50
print(fractional_knapsack(items, capacity))
```

#### Dijkstra's Shortest Path
This algorithm **finds the shortest path** in a graph by selecting the vertex with the minimum distance at each step.

```python
import heapq

def dijkstra(graph, start):
    distances = {node: float('inf') for node in graph}
    distances[start] = 0
    priority_queue = [(0, start)]

    while priority_queue:
        current_distance, current_node = heapq.heappop(priority_queue)
        if current_distance > distances[current_node]:
            continue
        for neighbour, weight in graph[current_node].items():
            distance = current_distance + weight
            if distance < distances[neighbour]:
                distances[neighbour] = distance
                heapq.heappush(priority_queue, (distance, neighbour))
    return distances

graph = {'A': {'B': 1, 'C': 4},'B': {'A': 1, 'C': 2, 'D': 5},'C': {'A': 4, 'B': 2, 'D': 1},'D': {'B': 5, 'C': 1}}
print(dijkstra(graph, 'A'))
```

In summary, **greedy algorithms** offer a fast and intuitive approach to optimization problems, although they may sacrifice optimal solutions for speed.
<br>

## 2. What are _Greedy Algorithms_ used for?

**Greedy algorithms** are often the algorithm of choice for problems where the optimal solution can be built incrementally and **local decisions** lead to a **globally optimal solution**.

### Applications of Greedy Algorithms

#### Shortest Path Algorithms
- **Dijkstra's Algorithm**: Finds the shortest path from a source vertex to all vertices in a weighted graph. 

  Use-Case: Navigation systems.

#### Minimum Spanning Trees
- **Kruskal's Algorithm**: Finds the minimum spanning tree in a weighted graph by sorting edges and choosing the smallest edge without a cycle.

  Use-Case: LAN setup.

- **Prim's Algorithm**: Starts from a random vertex and selects the smallest edge connecting to the growing tree.

  Use-Case: Superior for dense graphs.

#### Data Compression
- **Huffman Coding**: Used for data compression by building a binary tree with frequent characters closer to the root. 

  Use-Case: ZIP compression.

#### Job Scheduling
- **Interval Scheduling**: Selects the maximum number of non-overlapping intervals or tasks.

  Use-Case: Classroom or conference room organization.

#### Set Cover
- **Set Cover Problem**: Finds the smallest set collection covering all elements in a universal set.

  Use-Case: Efficient broadcasting in networks.

#### Knapsack Problem
- **Fractional Knapsack**: A variant that allows parts of items to be taken, with greedy methods giving an optimal solution.

  Use-Case: Resource distribution with partial allocations.

#### Other Domains
- **Text Justification** and **Cache Management**.
<br>

## 3. List some characteristics that are often present in problems that are solvable by _greedy algorithms_.

**Greedy algorithms** are tailored for optimization problems where making the locally optimal choice at each step typically leads to a global optimum.

### Key Characteristics

1. **Optimum Substructure**: An overall optimal solution can be achieved by making locally sound decisions. When making a greedy choice, the algorithm doesn't need to reconsider any part of the problem that has already been solved. 

2. **Greedy Choice Property**: At each step, the algorithm makes the decision that leads to the most immediate benefit. 

3. **Irreversibility**: Decisions are final and are not undone. This is often referred to as the "one-shot" nature of greedy choices. 

4. **Lack of Dependency**: The choice made at each step is independent of previous selections.

5. **Stateless Nature**: The solution is derived by considering each input item exactly once, making the algorithm **memory efficient**.

6. **Decomposability**: Problems are broken down into smaller, self-sufficient subproblems or building blocks which are then solved using the greedy approach. 

7. **Feedback-Backward Compatibility**: Greedy algorithms perform optimally in problems that lack any negative feedback or ones that allow for occasional errors or approximations.

While problems exhibiting these characteristics are suitable for greedy algorithms, not all such problems are solvable this way. It's crucial to validate the appropriateness of the greedy approach for a particular problem setting.

### Common Applications

- **Shortest Path in Graphs**: Dijkstra's algorithm is a classic example of a greedy approach to finding the shortest path in a weighted graph.
  - Example Metric: Link Distance

- **Text Compression**: Huffman coding constructs a tree with the property that parent nodes have shorter codes than their children, optimizing for minimal code length.
  - Application in ZIP, GIF, JPEG formats

- **Set Cover Problems**: The aim is to choose the smallest collection of sets whose union covers all the elements in the universal set.
  - Example: Concert Scheduling

- **Coin Change Problem**: The objective is to make change for a given amount using the fewest coins.

- **Activity Selection**: The task is to schedule such activities so that the maximum number of activities are accomplished.

- **Min/Max/Average Scheduling**: For minimizing, maximizing, or averaging values, greedy algorithms streamline the cumbersome task.

The greedy approach simplifies complex tasks by breaking them down into smaller, more manageable components. Venice Scheduling

When faced with resource limitations, time constraints, or the need for rapid approximate solutions, it offers an efficient and often effective problem-solving strategy.
<br>

## 4. What is the difference between _Greedy_ and _Heuristic_ algorithms?

While every **greedy algorithm** is a **heuristic**, the reverse is not true. Below is a side-by-side comparison of the two.

### Key Distinctions

#### Paradigm

  - **Heuristic**: A broader strategy for problem-solving.
  - **Greedy**: A specific method prioritizing immediate returns.

#### Basis of Decisions

- **Heuristic**: Uses rules of thumb or experience-based techniques.
- **Greedy**: Chooses the most favorable option based on the current situation.
  
#### Exploration

- **Heuristic**: Offers limited exploration, primarily based on predefined heuristics.
- **Greedy**: Focuses on immediate decisions without an exhaustive overview.

#### Solution Quality

- **Heuristic**: Provides sub-optimal, yet faster solutions for complex problems.
- **Greedy**: Ensures local optimality, which might not always lead to a global optimum.

#### Backtracking

  - **Heuristic**: Can allow backtracking or even randomness in some cases.
  - **Greedy**: Doesn't permit backtracking after a decision.

### Key Takeaways

**Heuristics** offer a flexible approach suitable for complex problems but might not guarantee the best solution. **Greedy algorithms**, while straightforward, can sometimes miss better global solutions due to their focus on immediate gains.

Neither approach ensures global optimality, making their effectiveness dependent on the specific problem context.
<br>

## 5. Compare _Greedy_ vs _Divide & Conquer_ vs _Dynamic Programming_ algorithms.

Let's explore how **Greedy**, **Divide & Conquer**, and **Dynamic Programming** algorithms differ across key metrics such as optimality, computational complexity, and memory usage. 

### Key Metrics

- **Optimality**: Greedy may not guarantee optimality, while both Divide & Conquer and Dynamic Programming do.
- **Computational Complexity**: Greedy is generally the fastest; Divide & Conquer varies, and Dynamic Programming can be slower but more accurate.
- **Memory Usage**: Greedy is most memory-efficient, Divide & Conquer is moderate, and Dynamic Programming can be memory-intensive due to caching.

### Greedy Algorithms

Choose Greedy algorithms when a **local** best choice leads to a **global** best choice.

#### Use Cases
- **Shortest Path Algorithms**: Dijkstra's Algorithm for finding the shortest path in a weighted graph.
- **Text Compression**: Huffman Coding for compressing text files.
- **Network Routing**: For minimizing delay or cost in computer networks.
- **Task Scheduling**: For scheduling tasks under specific constraints to optimize for time or cost.

### Divide & Conquer Algorithms

Opt for Divide & Conquer when you can solve **independent subproblems** and combine them for the **global optimum**.

#### Use Cases

- **Sorting Algorithms**: Quick sort and Merge sort for efficient sorting of lists or arrays.
- **Search Algorithms**: Binary search for finding an element in a sorted list.
- **Matrix Multiplication**: Strassen's algorithm for faster matrix multiplication.
- **Computational Geometry**: Algorithms for solving geometric problems like finding the closest pair of points.

### Dynamic Programming Algorithms

Choose Dynamic Programming when overlapping subproblems can be **solved once and reused**.

#### Use Cases

- **Optimal Path Problems**: Finding the most cost-efficient path in a grid or graph, such as in the Floyd-Warshall algorithm.
- **Text Comparison**: Algorithms like the Levenshtein distance for spell checking and DNA sequence alignment.
- **Resource Allocation**: Knapsack problem for optimal resource allocation under constraints.
- **Game Theory**: Minimax algorithm for decision-making in two-player games.
<br>

## 6. Is _Dijkstra's algorithm_ a _Greedy_ or _Dynamic Programming_ algorithm?

**Dijkstra's algorithm** utilizes a combination of greedy and dynamic programming techniques.

### Greedy Component: Immediate Best Choice
The algorithm selects the closest neighboring vertex at each step, reflecting the greedy approach of optimizing for immediate gains.

#### Example
Starting at vertex A, the algorithm picks the nearest vertex based on current known distances. 

![Dijkstra's Algorithm](https://upload.wikimedia.org/wikipedia/commons/5/57/Dijkstra_Animation.gif)

### Dynamic Programming Component: Global Optimization
Dijkstra's algorithm updates vertex distances based on previously calculated shortest paths, embodying the dynamic programming principle of optimal substructure.

#### Example
Initially, all vertices have infinite distance from the source, A. After the first iteration, distances to neighbors are updated, and the closest one is chosen for the next step.

```plaintext
Initial State: A: 0, B: inf, C: inf, D: inf, E: inf
After first iteration: A: 0, B: 2, C: 3, D: 8, E: inf
```

### Primarily Dynamic Programming
Despite combining both strategies, the algorithm aligns more closely with dynamic programming for several reasons:

1. **Guaranteed Optimality**: It provides the best solution, a hallmark of dynamic programming.
2. **Comprehensive Exploration**: The algorithm reviews all vertices to ensure the shortest path.
<br>

## 7. Is there a way to _Mathematically Prove_ that a _Greedy Algorithm_ will yield the _Optimal Solution_?

While the answer, **whether a greedy algorithm always yields an optimal solution**, may vary depending on the specific problem, two primary properties often need to be met for a greedy algorithm to be effective:

1. **Greedy Choice Property**: Making locally optimal choices at each step should lead to a global optimum.
2. **Optimal Substructure**: The optimal solution to the problem includes optimal solutions to its subproblems.

### Matroids

Matroids are mathematical structures that help formalize the concept of "**independence**" in a set. They provide a framework for understanding when a **greedy algorithm** will yield an optimal solution. A matroid is characterized by:

1. **Hereditary Property**: If a set is in the matroid, all its subsets also belong to the matroid.
2. **Exchange Property**: For any two sets $A$ and $B$ in the matroid, where $|A| < |B|$, there exists an element $x$ in $B - A$ such that $A \cup \{x\}$ is also in the matroid.

### Example: Knapsack Problem

In the Knapsack Problem, the goal is to **maximize** the total value of items in a knapsack without exceeding its capacity. A **greedy approach** of selecting items based on the highest **value-to-weight ratio** can be optimal under certain conditions, such as when the problem aligns with a matroid structure.

### Code Example: Knapsack Problem

Here is the Python code:

```python
def knapsack_greedy(items, capacity):
    # Sort items by value-to-weight ratio in descending order
    items = sorted(items, key=lambda item: item[1]/item[0], reverse=True)
    knapsack = []
    total_value = 0

    for item in items:
        weight, value = item
        if capacity - weight >= 0:
            knapsack.append(item)
            total_value += value
            capacity -= weight

    return knapsack, total_value

# Example usage
items = [(2, 10), (3, 5), (5, 15), (7, 7), (1, 6), (4, 18), (1, 3)]
capacity = 15
print(knapsack_greedy(items, capacity))
```
<br>

## 8. Explain the _Greedy Choice Property_ and its significance in _greedy algorithms_.

The **greedy choice property** is a core feature of **greedy algorithms**, ensuring that local optima lead to global optima. Such algorithms make **decisions iteratively** on smaller parts of a larger problem, aiming to optimize towards an overall solution.

### Key Points

- **Unique Factor**: Each step is made entirely based on the local best choice without any need for future decision information. This separability between steps is what makes the algorithm "greedy."

- **Potential Limitations**: Even though exhibiting this property, a **greedy approach** isn't always ideal for problem solving, especially if the local optima do not lead to the global optima. In such scenarios, a more thorough assessment becomes necessary.

- **Optimality Unassured in General**: Greedy algorithms, though often effective and efficient, can't be universally relied upon for providing the most optimized solution. The approach is best suited for problems where local best choices cumulatively lead to the global best choice.

### Visual Example: Children at a Candy Store

Consider a group of children at a candy store, each equipped with a certain budget. The candy store has diverse treats, each with its respective cost. The task is to maximize the total number of treats bought, within individual budgets.

- If each child is left to pick the most appealing treat solely based on its cost against their budget, the act of satisfying their local preferences through its local budget can lead to a global optimum, represented by all children collectively obtaining the greatest number of treats overall.

This cascading pattern of local best choices leveraging children's individual budgets towards the most economically favorable options is emblematic of the **"greedy choice property"**.

### Common Problems Leveraging the Greedy Choice Property

1. **Minimum Spanning Tree Construction**: In the context of graph traversal, selecting the edge having the lowest weight at each step contributes to building a "minimal" tree or path.

2. **Shortest Path Algorithms**: Both Dijkstra's and the Bellman-Ford algorithms, in varying degrees, exemplify the "greedy choice" as they make momentary, local best decisions towards building the overall shortest path.

3. **Optimal Caching**: Utilized in caching mechanisms, where the most frequently or recently accessed data is selectively stored in a limited cache space.

4. **Packing Problems and Knapsack Behaviors**: Known for optimizing the utilization of area or space given certain constraints, such as in data compression or packing items in a predefined space.

5. **Huffman Coding**: A method of efficient data compression where the tree's leaf nodes, embodying individual characters and their frequencies, are built through successive two-node mergers that have the lowest frequency counts.

6. **Activity Selection**: Predicated on the idea of preferring the activity that will end first, freeing up resources and enabling additional activities to be scheduled.

7. **Text Justification**: Facilitating the fitting of words from a passage into discrete lines while minimizing surplus spaces by preferentially adding words until no more can fit.

8. **Network Routing with Dijkstra's Algorithm**: Known for its utilization in finding the shortest path from a single source to all nodes in a graph, often seen in networking logistics and beyond.

9. **Set Cover**: Strives for the most compact cover set among a given assemblage of sets, with the objective of including all elements implicated by any subset.

10. **Making Change**: The embodiment of the familiar money-denomination concept, seeking to streamline the process of providing change with the fewest coins possible.

11. **Subset Sum**: As the name implies, it concentrates on subsets that aggregate to a particular sum, and the algorithm wields a form of "greed" by preferentially catering to the summed value or discarding it until the exact match is identified.

By creatively aligning with the "greedy choice property," these techniques manage to simplify fairly intricate problems into digestible and swiftly solvable packages.
<br>

## 9. Design a _greedy algorithm_ for making change using the fewest coins possible.

### Problem Statement

Given a specific set of coins and a target amount of change, the objective is to determine the smallest number of coins needed to make the change.

### Solution

This classic problem, making change, has an optimal greedy algorithm solution for U.S. coins: quarters ($25\, ¢$), dimes ($10\, ¢$), nickels ($5\, ¢$), and pennies ($1\, ¢$).

#### Algorithm Steps

1. **Select the Largest Coin**: Start by picking the largest coin that is less than or equal to the remaining amount. Continue this process for the reduced amount, selecting the largest coin each time.

2. **Repetition**: Repeat this until the remaining amount becomes $0$. At each step, you should pick the largest coin that does not exceed the remaining amount.

3. **Termination**: If you've successfully reduced the amount to $0$, you have found the optimal solution.

#### Complexity Analysis

- **Time Complexity**: $O(n)$, where $n$ is the amount of change.
- **Space Complexity**: Constant, not depending on the change amount.

#### Proof of Correctness

The greedy algorithm selects each coin as many times as possible while staying below the remaining amount. This strategy is optimal for U.S. coins due to their denominations being multiples of the previous coin.

For general coin systems, the greedy algorithm might not be optimal. For example, with coin denominations of $\{1, 3, 4\}$ and a target amount of $6$, the optimal solution uses two coins: $3$ and $3$, while the greedy algorithm would use three coins: $3$, $3$, and $3$ (which is suboptimal).
<br>

## 10. Describe a greedy strategy for scheduling events that use the fewest number of resources.

When scheduling events to **minimize resource use**, a **greedy strategy** works by selecting the **earliest-starting** tasks that don't conflict.

This is similar to finding the Maximum Independent Set (MIS) in graph theory, which selects non-adjacent vertices to maximize the number of selected vertices. Tasks are equivalent to vertices, and overlapping schedules to edges in the graph representation.

### Greedy Scheduling Algorithm

1. **Sort tasks**: Arrange them in non-decreasing order of finishing time.
2. **Select tasks**: Starting with the earliest, pick non-overlapping tasks.

### Example

Consider the following schedule:

$$
$$
1 & : 1\to 2\\
2 & : 1\to 3\\
3 & : 2\to 4\\
4 & : 3\to 5\\
5 & : 4\to 6
$$
$$

The sorted order by finishing time is:

$$
$$
1 & : 1\to 2\\
2 & : 1\to 3\\
3 & : 2\to 4\\
4 & : 3\to 5\\
5 & : 4\to 6
$$
$$

The greedy strategy selects the following tasks:

$$
$$
1 & : 1\to 2\\
3 & : 2\to 4\\
5 & : 4\to 6
$$
$$

### Complexity

1. **Sorting**: $O(n\log n)$ for $n$ tasks.
2. **Task Selection**: Linear, $O(n)$.
   - **Total**: $O(n \log n)$

The algorithm's main advantage is its **simplicity and speed**, especially for pre-sorted tasks.

### Code Example: Greedy Scheduling Algorithm

Here is the Python code:

```python
def schedule(tasks):
    # Sort tasks by finish time
    tasks.sort(key=lambda x: x[1])
    
    # Initialize selected_tasks and last_finished
    selected_tasks = []
    last_finished = float('-inf')
    
    # Select tasks that don't overlap
    for task in tasks:
        start, finish = task
        if start >= last_finished:
            selected_tasks.append(task)
            last_finished = finish
    
    return selected_tasks

# Example tasks
tasks = {
    "Task 1": (1, 2),
    "Task 2": (1, 3),
    "Task 3": (2, 4),
    "Task 4": (3, 5),
    "Task 5": (4, 6)
}

# Print the selected tasks
print(schedule(tasks))
```
<br>

## 11. Explain how to apply a greedy approach to minimize the total waiting time for a given set of queries.

To **minimize waiting time** for a set of $n$ queries, you should prioritize shorter tasks before longer ones. This sequential approach is best optimized by **using a greedy method**.

### Greedy Strategy for Query Optimization

1. **Sort Queries**: Arrange the queries in increasing order of duration.
2. **Iterate and Sum**: Calculate the total waiting time as you process each query.
<br>

## 12. Develop a _greedy algorithm_ for the _activity selection problem_ to maximize the number of activities.

### Problem Statement

Given the **start** and **finish** times of $n$ activities, the objective of the **Activity Selection Problem (ASP)** is to select a maximum-size set of mutually compatible activities. Two activities are compatible if they don't overlap.

### Solution

The **greedy strategy** for ASP involves the following steps:

1. **Sort**: According to the finish times
2. **Select**: The activity with the earliest finish time
3. **Repeat**: Until no compatible activities remain

The key is to greedily **maximize the number of activities**. This isn't just an arbitrary choice but a proven, optimal selection based on finishing times.

#### Complexity Analysis

- **Time Complexity**: $O(n \log n)$ (from sorting)
- **Space Complexity**: $O(1)$ (excluding input storage)

#### Implementation

Here is the Python code:

```python
def activity_selection(start, finish):
    n = len(finish)
    i, j = 0, 1
    print("Selected activities:", i)

    for j in range(1, n):
        if start[j] >= finish[i]:
            print("Selected activities:", j)
            i = j
```
Here is a C++ code:
```cpp
#include <bits/stdc++.h>
using namespace std;

void activity_selection(int start[], int finish[], int n) {
    int i = 0;
    cout << "Selected activities: " << i << " ";

    for (int j = 1; j < n; j++) {
        if (start[j] >= finish[i]) {
            cout << "Selected activities: " << j << " ";
            i = j;
        }
    }
}

int main() {
    int start[] = {1, 3, 0, 5, 8, 5};
    int finish[] = {2, 4, 6, 7, 9, 9};
    int n = sizeof(start) / sizeof(start[0]);
    cout << "Following activities are selected:\n";
    activity_selection(start, finish, n);
    return 0;
}
```
<br>

## 13. How can you establish the correctness of a _greedy algorithm_?

**Greedy algorithms** make locally optimal choices with the hope that they lead to a globally optimal solution. The **key challenge** is that a locally optimal choice may not always be part of the global optimum.

To mitigate this, greedy algorithms often employ strategies like **monotonicity** or **exchange arguments**. Let's explore each strategy in detail.

### Monotonicity

When a problem exhibits the "take or leave" property, **monotonicity** can be employed to show that the greedy choice is part of the optimal solution. Monotonicity implies that if a local choice is made, it won't be reversed later.

Consider the Knapsack Problem, where you have a limited capacity knapsack and items with different weights and values. The goal is to maximize the total value of items in the knapsack without exceeding its capacity. This problem exhibits monotonicity, as adding an item cannot decrease its value or weight.

Here is how monotonicity is established for the Knapsack Problem:

- Take a locally optimal solution $S$ which includes an item $i$ in the knapsack.
- Assume $S$ is not globally optimal, and there exists an optimal solution $S'$ which does not include $i$.
- Start with the solution $S - \{i\}$ and add items from $S'$ until reaching capacity. Call this new solution $S''$.
- By the "take or leave" property, $S''$ is at least as good as $S$, contradicting the local optimality of $S$.
- Therefore, $S$ must be globally optimal, and the greedy choice of adding item $i$ is justified. 

### Exchange Arguments

In some problems, it is not practical to establish monotonicity. **Exchange arguments** can be used instead, showing that for any suboptimal solution, there exists another solution that is at least as good but has made a different choice at the greedy step.

- **Example 1: Interval Scheduling**: The goal is to select the maximum number of non-overlapping intervals from a set of intervals. The exchange argument here involves swapping an interval from the optimal solution with one outside of it, preserving the total number of intervals but potentially achieving a higher sum of lengths. This contradicts the optimality of the initial solution.

- **Example 2: Minimum Spanning Tree**: In this problem, a graph must be connected, but not all edges are included. The exchange argument is based on the notion of "safe edges" that preserve connectivity and minimize total weight. It establishes that the greedy algorithm's selection of safe edges is one among many possible minimum spanning trees.

### Formal Proofs

To establish the correctness of a greedy algorithm formally, one would ideally use techniques like **induction**, **mathematical argument**, or **structural induction**. These are often more complex to implement and may be unnecessary in many cases where the above argument strategies alone are sufficient.

### Code Example: Minimum Spanning Tree (Prim's Algorithm)

Here is the Python code:

```python
from heapq import heappop, heappush

def prim(graph):
    visited = set()
    pending = [(0, 0)]  # (weight, node)
    total = 0
    
    while pending:
        weight, node = heappop(pending)
        if node not in visited:
            visited.add(node)
            total += weight
            for neighbor, weight in graph[node]:
                heappush(pending, (weight, neighbor))
    
    return total
```
<br>

## 14. What is the role of _sorting_ in many _greedy algorithms_, and why is it often necessary?

**Sorting** is a foundational operation in many **greedy algorithms**. These algorithms make locally optimized choices at each step, ultimately aiming for a global or optimal solution.

### Primary Benefits

1. **Identifying Natural Order**: Sorting helps recognize patterns, structures, or sequences that are crucial for optimal solutions.
  
2. **Efficiency**: Many greedy approaches need sorted input to make the algorithm feasible or improve its time complexity.

3. **Algorithmic Independence**: Greedy algorithms are often modular and can leverage the power of sorting independently.

### Key Factors for Sorting in Greedy Algorithms

1. **Input Format**: Some problem statements provide input as already sorted, necessitating a modified greedy approach.
  
2. **Intermediate States**: Sorting can be done at the beginning, during, or end of the computation, leading to varied algorithm behavior.
  
3. **Interaction with Greedy Choices**: The position of sorting in the decision-making loop affects the algorithm's behavior and guarantees.

### Illustration: Fractional Knapsack

The Fractional Knapsack problem maximizes the value of items to be put into a knapsack, subject to a weight constraint. It employs a greedy strategy based on the value-to-weight ratio of items.

### Code Example

Here is the Python code:

```python
def fractional_knapsack(items, capacity):
    items.sort(key=lambda x: x[1]/x[0], reverse=True)  # Sort items by value-to-weight ratio
    result = 0
    for weight, value in items:
        if capacity >= weight:
            capacity -= weight
            result += value
        else:
            result += (value/weight) * capacity
            break
    return result
```
<br>

## 15. Can _greedy algorithms_ be used to solve _optimization problems_? Give an example.

Yes, **greedy algorithms** are designed to solve **optimization problems** by making locally best choices in the hope of obtaining a global optimum.

### Example: Activity Selection Problem

The **Activity Selection Problem** provides an apt example of a scenario where a greedy algorithm works optimally.

#### Problem Description

Consider a collection of activities, each labeled with its start and finish times. The goal is to select a maximal set of non-overlapping activities. Two activities are said to be non-overlapping if their time intervals do not intersect.

#### Greedy Decision-Making

Selecting the activity that finishes earliest seems like the right local choice since it frees up time for other activities.

#### Algorithm Steps

1. **Sort Activities**: Sort the given activities based on their finishing times.
2. **Select First Activity**: Always select the first activity from the sorted list.
3. **Choose Subsequent Activities**: For each remaining activity, if it starts after the finish of the last selected activity, add it to the selected list.

#### Code Example: Activity Selection

Here is the Python code:

```python
def activity_selection(act_list):
    # Sort activities based on finish time
    act_list.sort(key=lambda x: x[1])
    
    selected = [act_list[0]]  # First activity always selected
    last_selected = 0
    
    for i in range(1, len(act_list)):
        # If start time is after finish time of last selected activity, select it
        if act_list[i][0] >= act_list[last_selected][1]:
            selected.append(act_list[i])
            last_selected = i

    return selected

# Example usage
activities = [(1, 4), (3, 5), (0, 6), (5, 7), (3, 8), (5, 9), (6, 10), (8, 11), (8, 12), (2, 13), (12, 14)]
print(activity_selection(activities))
```

### Complexity Analysis

- **Time Complexity**: The most time-consuming step is the sorting of activities, taking $O(n \log n)$ time. After that, each activity is processed in constant time, yielding an overall complexity of $O(n \log n)$.
- **Space Complexity**: The algorithm requires $O(n)$ space to store the selected activities.
<br>



#### Explore all 41 answers here 👉 [Devinterview.io - Greedy Algorithms](https://devinterview.io/questions/data-structures-and-algorithms/greedy-algorithms-interview-questions)

<br>

<a href="https://devinterview.io/questions/data-structures-and-algorithms/">
<img src="https://firebasestorage.googleapis.com/v0/b/dev-stack-app.appspot.com/o/github-blog-img%2Fdata-structures-and-algorithms-github-img.jpg?alt=media&token=fa19cf0c-ed41-4954-ae0d-d4533b071bc6" alt="data-structures-and-algorithms" width="100%">
</a>
</p>

