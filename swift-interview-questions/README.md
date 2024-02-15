# Top 70 Swift Interview Questions

<div>
<p align="center">
<a href="https://devinterview.io/questions/web-and-mobile-development/">
<img src="https://firebasestorage.googleapis.com/v0/b/dev-stack-app.appspot.com/o/github-blog-img%2Fweb-and-mobile-development-github-img.jpg?alt=media&token=1b5eeecc-c9fb-49f5-9e03-50cf2e309555" alt="web-and-mobile-development" width="100%">
</a>
</p>

#### You can also find all 70 answers here 👉 [Devinterview.io - Swift](https://devinterview.io/questions/web-and-mobile-development/swift-interview-questions)

<br>

## 1. What is the difference between `let` and `var` in _Swift_?

In Swift, the `let` and `var` keywords are used to **declare constants** and **variables** respectively.

### Key Distinctions

- **Immutability**: Variables declared with `let` are immutable, while those with `var` can be mutated.
- **Initialization**: `let` requires immediate initialization, whereas `var` can be initialized later.
- **Safety and Clarity**: Using `let` over `var` whenever possible promotes code clarity and reduces the risk of unintended mutations.
<br>

## 2. How do you define a constant that is computed at runtime in _Swift_?

In Swift, you can use **lazy properties** and **immediately-invoked closure expressions** (IICE) to compute values at runtime, ensuring they are calculated only once. Let's look at the detailed steps for both approaches.

### Using `Get-Only Computed Properties`

A `get` only computed properties are declared using the `var` keyword but without a `set` block. They're calculated when accessed and won't change afterward, effectively becoming **constants** after the initial calculation.

```swift
struct Circle {
    let radius: Double
    lazy var area: Double = {
        return Double.pi * (2 * self.radius)
    }()
}
```

### Implementing `Set-Once Computed Properties`

This method combines a stored property with a `didSet` observer to handle computations only when the property is initially set.
- **Pros**: Offers an immediate computed value and ensures the computation is done only once.
- **Cons**: Looks lengthier than other options.

```swift
struct Circle {
    private var _diameter: Double?
    var diameter: Double? {
        set {
            guard _diameter == nil else { return }
            _diameter = newValue
            area = newValue! * newValue! * Double.pi / 4
        }
        get { return _diameter }
    }
    private(set) var area: Double?

    init(radius: Double) {
        self.radius = radius
    }
}
```
<br>

## 3. Can you explain the purpose of _optionals_ in _Swift_?

In Swift, **optionals are a special type that represents either a valid value or no value at all**. Merging the best of both Objective-C's nullable references and Swift's type safety, they contribute to modern Swift's safety and flexibility.

### Key Features of Optionals

- **Safety First**: They assist in managing nil or null pointer-related operations and prevent common programming errors, such as directly accessing a nil object or referencing an uninitialized pointer.
  
- **Clarity and Intention**: The presence of an optional in a type declaration transparently conveys the possibility of a nil value.

- **Flexibility in Initialization**: Variables are often initialized as `nil` and only assigned a value later based on certain conditions.

- **Swift Literals for Easy Instantiation**: Swift provides the convenience of `nil` to represent no value and non-nil values can be wrapped directly, for instance: `let optional: Int? = 42`

- **Compile-Time Checks**: The Swift compiler enforces careful handling of optional values, reducing runtime errors.

### The Powerful "if let"

Swift offers the elegant "if let" construct for effortless **unwrapping and optional handling**:

```swift
var optionalValue: Int? = 5

if let unwrappedValue = optionalValue {
    print("The value is \(unwrappedValue)")
} else {
    print("The optional value is nil.")
}
```

This construct ensures that `unwrappedValue` is used only if `optionalValue` contains a non-nil value.

### Implicit Unwrapping

For specific scenarios where you are certain an optional always has a value, Swift provides a mechanism for implicit unwrapping using **the `!` operator**:

```swift
let constant: String! = "Certain non-nil value."
let text = constant
```

Although convenient, it comes with the caveat that unintentionally accessing a nil value results in a runtime error.

### Nil Coalescing Operator (??)

Swift employs the **Nil Coalescing Operator** to offer a default value in place of nil:

```swift
let someText: String? = nil
let nonOptionalText = someText ?? "Default Text"
```

If `someText` is nil, "Default Text" becomes the value assigned to `nonOptionalText`.

### Chaining Optional Values

Swift supports **chaining** of optional values. When you need to access numerous nested optionals, it's cleaner and safer to do so in a single line:

```swift
let addressLength = user.address?.street?.count
```

If any part of the chain resolves to nil, subsequent parts are naturally short-circuited.

### Optional-Handling Operators

Swift provides a dedicated set of operators for efficient optional handling, enhancing readability and reducing code verbosity:

- **Nil-Coalescing Operator (\?\?:)**. Provides a default value in case of nil.
- **Forced Unwrapping (\!)**: Explicitly unwraps an optional, possibly leading to a runtime error.
- **Optional Chaining (\? and \?\?)**: Enables method and property invocation on a chained optional, continuing execution if no nil is encountered.
- **Type Casting Operators (as and as?)**: Facilitates dynamic type-casting on optionals.

### Instant Initialization with a Default Value

Using **`init(defaultValue:)`**, Swift allows you to set an initial, non-nil value for an optional, resulting in cleaner and efficient code:

```swift
var optionalBool =  Bool?(defaultValue: true)
```

### Error-Handling with Optionals

Swift's `try?` and `try!` keywords are instruments for integrating error-handling mechanisms. Both can either propagate the caught error or return an optional to signify either a result or an error, providing a standardized approach towards error management.

### Modern Observability with Optionals

Swift's property observers, enabled for both computed and stored properties, grant a well-structured path to monitor changes in optional values by way of `willSet` and `didSet`.
<br>

## 4. What are _tuples_ and how are they useful in _Swift_?

In **Swift**, a **tuple** is a compound data type that enables bundling multiple values together. Unlike arrays or dictionaries, tuples provide a lightweight, fixed-size grouping.

### Tuple Syntax

You define a tuple using **parentheses** and separate its components with **commas**:

```swift
let person: (String, Int, Bool) = ("John Doe", 25, true)
```

You can also assign type aliases to tuple elements:

```swift
typealias Book = (title: String, author: String, year: Int)
let book: Book = (title: "1984", author: "George Orwell", year: 1949)
```

### Benefits of Tuples

- **Data Grouping**: They allow you to pair or group values conveniently.

- **Lightweight**: Tuples are ideal when you need a quick, temporary grouping of values without creating a custom data structure.

- **Out-of-the-box**: They come with built-in, generic equality checks, making them easy to use.

- **Multiple Return Values**: Instead of encapsulating values within a custom type for multiple returns from a function, you can use a tuple.

- **Visual Clarity**: Can be particularly useful for returning multiple values in function definitions, adding explicitness and code readability.

- **Position-based and Labeled Access**: You can either access tuple values based on their position or using named labels, making the code more readable.

- **Type Flexibility**: Tuples can hold elements of different types, offering a quick solution for ad-hoc data groupings.

### Practical Use-Cases

- **Returning Multiple Values**: Instead of creating a `struct` just for the purpose of returning multiple values from a function, tuples provide a more lightweight alternative.

- **Intermediate Computation**: Tuples are suitable for bundling intermediate results before processing further.

- **Optional-Bind Shortcut**: When using a conditional `if let` statement, tuples offer a quick way to get an unwrapped result along with a boolean evaluation.

### Code Example: Tuples

Here is the Swift code:

```swift
// Define a simple person tuple
let person: (name: String, age: Int, isEmployed: Bool) = ("John Doe", 25, true)

// Access tuple elements by position or name
print("Age: \(person.1)")  // Output: "Age: 25"
print("Name: \(person.name)")  // Output: "Name: John Doe"

// Function returning a tuple
func createMessage(using name: String, isExcited: Bool) -> (greeting: String, punctuation: String) {
    let greet = isExcited ? "Hello, \(name)! 😀 What a pleasure to see you." : "Hello, \(name)."
    let punct = isExcited ? "!" : "."
    return (greeting: greet, punctuation: punct)
}

let welcomeMessage = createMessage(using: "Jane", isExcited: true)
print(welcomeMessage.greeting + welcomeMessage.punctuation)
// Output: "Hello, Jane! 😀 What a pleasure to see you."
```
<br>

## 5. Describe the different _collection types_ available in _Swift_.

Swift offers diverse collections, each optimized for specific use-cases.

### Common Traits Across Collections

- **Type Safety**: Each element in a Swift collection is of the same type.
- **Mutability**: Most collections can be mutable or immutable, depending on the specific type or how they are defined.
-  **Value Semantics**: Swift collections work based on value types; when a collection is copied, the elements are copied as well.
- **Indexing**: Most Swift collections support zero-based indexing, allowing direct access via the subscript `[]`. 

### Array

- **Type** : Ordered, Random Access, Duplicates
- **Performance**: $O(1)$ on average for both read and write operations
- **Initialization**:
  - **Empty**: `var myArray: [Int] = []`
  - **With Values**: `let myArray = [1, 2, 3, 4, 5]`

### Set

- **Type**: Unordered, Uniqueness
- **Performance**: $O(1)$ on average for insertions and lookups
- **Initialization**:
  - **Empty**: `var mySet: Set<Int> = []`
  - **With Values**: `let mySet: Set<Int> = [1, 2, 3, 4, 5]`

### Dictionary

- **Type**: Key-Value Pairs, Unique Keys
- **Performance**: $O(1)$ on average for both inserts and lookups, but can degrade based on hashing function's strength
- **Initialization**:
  - **Empty**: `var myDict: [String: Int] = [:]`
  - **With Values**: `let myDict: [String: Int] = ["one": 1, "two": 2, "three": 3]`

### Range

- **Type**: Sequence of Values
- **Performance**: $O(1)$ for operations
- **Initialization**: Ranges can be created using range operators or Range methods.
  - **Using Range Operators**: `let closedRange = 1...10`
  - **Using Methods**: `let halfOpenRange = Range(1..<10).`
- **Common Use**: Used in loops to iterate over a sequence of numbers or characters.
- **Indexing**: Ranges can be indexed.

### Tuple

- **Type**: Heterogeneous Collection, Fixed Size
- **Performance**: $O(1)$ for operations
- **Initialization**: Tuples can be created using parentheses and separating elements with commas.
  - **With Values**: `let myTuple = (1, "hello", true)`
  - **With Labels**: `let person = (name: "Alice", age: 24)`. Once labels are assigned, they can be used as element names for the tuple.

### Subscripts

Arrays, sets, and dictionaries can use subscripting with indices or keys to retrieve items.

- **Array and Tuple**: Use **Int** indices to access elements.
- **Set and Dictionary**: Use **keys** to access or update elements.

Here are some Swift code snippets that illustrate its use:

#### Code Example: Swift Collection Types

Here is the Swift code:

```swift
var intArray: [Int] = [1, 2, 3, 4, 5]
var mySet: Set<Int> = [1, 2, 3, 4, 5]
var myDict: [String: Int] = ["one": 1, "two": 2, "three": 3]

print(intArray[2])   // Access the 3rd element of the array using index
print(mySet.contains(3))  // Check if set contains 3
print(myDict["one"] ?? 0)  // Access the value for the key "one"

let closedRange = 1...10  // Closed Range from 1 to 10, inclusive
let halfOpenRange = 1..<10  // Half-Open Range from 1 to 9

let myTuple = (1, "hello", true)  // Tuple without labels
let person = (name: "Alice", age: 24)  // Tuple with labels for its elements
print(myTuple.0)  // Accessing the first element of a tuple without labels
print(person.name)  // Accessing the "name" element using the label
```
<br>

## 6. How do you handle flow control in _Swift_ with _loops_ and _conditions_?

In Swift, you can **control the flow of your program** using common statements such as conditionals and loops.

### Conditional Execution

- Use the `if` statement for simple conditionals.
- Employ the `switch` statement for multi-case evaluations.

Both of these constructs can include `else` clauses to define actions to take in **non-matching situations**.

#### Code Example: Conditional Execution

Here is the Swift code:

```swift
var temperature = 23
var weatherDescription: String

if temperature < 10 {
    weatherDescription = "Wintery"
} else if temperature < 25 {
    weatherDescription = "Cool"
} else {
    weatherDescription = "Warm"
}

print(weatherDescription) // Output: Cool
```
<br>

## 7. What are _enumerations_ in _Swift_ and how do they support _associated values_?

**Enumerations**, commonly referred to as enums, are a powerful feature of Swift that provide a way to define a common type for a group of related values.

Enums can be enhanced with **associated values** to present more detailed data records. 

For example, consider a pizza delivery app. An order can be in one of several states, and different states may have associated data.

### What are Associated Values?
Associated values are **Swift's feature** that permits each enum case to hold specific data.

This ability makes enums more flexible and versatile. Unlike basic enums, which store the same type of data for all their cases, enums with associated values can store different types of data for each case.

### When to Use Associated Values

Use associated values when an enum case can represent distinct states or data structures. This mechanism is a powerful way to model payload data. For instance:

- In an online clothing store app, the `ShoppingCart` can be empty, or it can contain one or more items.
- In a game, a `Player` can be in a particular state, like `fighting` or `healing`, and each state may include additional data.

### Standard Associated Values Example

Take, for example, the `PizzaDelivery` enum that covers the different stages of a pizza order, along with **associated information** like the estimated delivery time and/or any delivery issues.

Here is the Swift code:

```swift
enum PizzaOrder {
    case confirmed(estimatedDelivery: Date)
    case outForDelivery(driver: String)
    case deliveredWithIssue(String)  // Could represent an issue like 'No one was home'
    case delivered
}
```

The enum values are self-explanatory. For instance, `confirmed` indicates that the order is placed and the expected delivery time. The second case (`outForDelivery`) includes the name of the delivery driver.

You could represent these states using only a basic enum without associated values, but you'd lose the convenient packaging of associated data.

### Nested Enum Example

You can **further contextualize** enums by nesting them inside another enum or struct. This approach helps to organize closely related enums and minimizes the chance of polluting the global scope.

For instance, consider a `NetworkResult` enum that enables representing the result of a network request. This nested enum approach also includes `Recommendation` cases for showing error details and contextual recommendations to the user.

Here is how the swift code looks:

```swift
struct NetworkRequest {
    enum Status {
        case success, failure(NetworkError)
    }
    
    enum NetworkError {
        case timeout
        case serverError
        case authorizationFailure(recommendation: Recommendation)
        
        enum Recommendation {
            case refreshSession, logout
        }
    }
}
```

### Associated Values and Control Flow

The **flexibility stemming from associated values** is particularly beneficial when dealing with conditional logic. For instance, consider an enum that represents a geometric shape. It has associated values for its attributes. Using this enum data in a `switch` statement, you can access the attribute data for the specific case.

Here is the Swift code:

```swift
enum Shape {
    case circle(radius: Double)
    case square(side: Double)
    case rectangle(width: Double, height: Double)
}

let someShape: Shape = .circle(radius: 5.0)

switch someShape {
case .circle(let radius):
    print("Circle's radius: \(radius)")
case .square(let side):
    print("Square's side: \(side)")
case .rectangle(let width, let height):
    print("Rectangle's width: \(width), height: \(height)")
}
```

### Advanced Use Cases

1. **State Machines**: Define a `state` enum with associated values to build robust state machines. For instance, in a game, you can model a `player`'s `state` with associated energy points, position, and attack strategy.

2. **Functional Programming**: With Swift supporting some functional programming paradigms, you can combine associated values-powered enums with higher-order functions like `map` and `filter`.

3. **Error Handling**: Swift's `Result` type uses associated values in enums to indicate either a successful result or a failure with an associated `Error` instance.

### Limitations of Associated Values

- **Loose Type Consistency**: Due to the flexibility of holding different types, you must handle these values cautiously, frequently requiring type checks or pattern matching.

- **Case Payloads Could Be Unused**: If some cases don't always have associated data, the handling code might need to include conditional checks to avoid unintentional use of nil data.

### Summary

Enums with associated values in Swift allow you to define a data structure that can represent multiple distinct states, each with its distinct set of associated data.
<br>

## 8. In _Swift_, how are _switch statements_ more powerful compared to other languages?

Swift takes the typical switch statement to a new level by allowing for **rich pattern-matching** and asserting complex conditions in each `case`.

Several unique features make Swift's switch statement stand out:

### Enhanced Pattern Matching

- **Value Binding**: Extracts and assigns associated values or collection elements for later use.
- **Case Matching**: Handles a wide range of cases, including tuple, range, and type matching.
- **Where Clauses**: Allows custom conditions for case selection.

### Exhaustiveness and Unreachability

Swift enforces that _enum cases_ are fully handled in a switch statement. It flags any incomplete coverage as a compile-time error. This prevents against unintentional bugs arising from missing case handling.

### Compound Cases

Swift enables the grouping of cases in clever ways, improving code clarity and reducing redundancy:

- **Compound Cases**: Multiple cases that execute the same block of code. Compound cases streamline the logic when, for example, several values of an enum require the same action to be taken.
- **Compound Matching Conditions**: Combine case conditions with a single block of code that's executed when all conditions match.

### Convenient Syntax for `Optional` and `where` Clause in `switch`

Here's the Swift enum to support the code example.

**EmploymentStatus.swift**

```swift
enum EmploymentStatus {
    case employedAt(company: String, since: Date)
    case unemployed
}
```

### Code Example: Match on Associated values of Enum

Let's see the code:

```swift
let employment: EmploymentStatus = .employedAt(company: "ABC Inc", since: Date())
    
switch employment {
    case .employedAt(let company, let since):
        print("Employed at \(company) since \(since)")
    case .unemployed:
        print("Currently unemployed")
}
```

In this case, the specific condition that is being matched here is:

- `case .employedAt(let company, let since)`: Here, it's being verified whether the `employment` status is `employedAt` a company or not. If it's true, then both `company` and `since` are also "captured" for the rest of the case scope.

This is particularly useful for enums that use associated values to convey more comprehensive information. It's especially handy for enums that take on the role of a "discriminated union" in languages that support such a concept.

### Guidance on Where to Use The Advanced Capabilities

It's recommended to use these advanced features in scenarios where they enhance **readability** and **maintainability** of the code. These advanced matching capabilities really shine when working with Swift-friendly types, such as Optionals and Enums with associated values.
<br>

## 9. Describe the concept of _type inference_ in _Swift_.

**Type Inference** in Swift allows the compiler to **deduce variable types** from their assigned values, thereby reducing the need for explicit type annotations.

### Benefits of Type Inference

- **Conciseness and Readability**: Code is streamlined and often clearer without superfluous type declarations. 
- **Flexibility**: Swift's strong type system lets you toggle between inferred and explicit types for better precision.

### How Type Inference Works

- **Initialization Assignments**: The initial value assigned to a variable or constant informs its type. Any subsequent value assigned must be of the same type for inference to occur.


- **Arithmetic and Comparison Operators**: Swift infers variable types based on the type of literals or variables used in operations.


- **Contextual Clues**: Inferred types can be influenced by surrounding expressions or expected parameter types in function calls.

### Code Example: Type Inference

Here is the Swift code:

```swift
// Compiler infers 'name' as 'String'
var name = "John"

// The type of 'age' is deduced as 'Int'
var age = 25

// 'balance' is inferred as 'Double' due to the decimal value
var balance = 134.56

// Type for 'isApproved' is derived as 'Bool'
var isApproved = true

// Inferred as an array of strings
let shoppingList = ["Eggs", "Milk", "Bread"]

// Compiler recognizes 'count' as 'Int' based on the array's type
let count = shoppingList.count

// If a floating point number is involved, the result is always inferred as 'Double'
let amount: Double = balance / Double(age)

// The type of 'user' is inferred as a string since it's being printed with a string literal
print("User: \(name)")
```
<br>

## 10. What is _type casting_ in _Swift_ and how is it implemented?

**Type casting** in Swift refers to the ability to check a variable's type and conditionally convert it to a different type (upcasting or downcasting).

### Upcasting: Converting to a Superclass or Protocol Type

- **Superclass Type**: You might cast a Dog instance to its superclass, Animal, for example, to treat all animals the same way, whether they are cats, dogs, or any other animal type that is a subclass of Animal.
  
  - **Code Example**:
    Here we have a set of animals (`[Animal]`) and a Dog instance. We treat both as `Animal` using Upcasting.

  ```swift
  class Animal {}
  class Dog: Animal {}
  let someAnimal = Dog()
  let animals: [Animal] = [someAnimal] // Upcasting to [Animal]
  ```

- **Protocol Type**: With protocols, you can group different types that conform to the same protocol interface.
  
  - **Code Example**:
    A `Flying` protocol is represented with the `upCasted` object:

  ```swift
  protocol Flying {}
  class Bird: Animal, Flying {}
  let upCasted: Flying = Bird()
  ```

### Performing Downcasting

- **Definition**: Downcasting refers to the act of safely converting a variable to a specific type inheriting from a superclass or conforming to a protocol.

- **Applicability**: It is commonly used after upcasting when you need to access the original specific type.

- **Approach**: Swift provides two mechanisms for downcasting:

  1. "**as?**": Conditional downcast that returns an optional. It's used with an if-let or guard-let statement to check and unwrap the result.

  2. "**as!**": Forced downcast. Use only when you are certain about the target type, and its failure would be a programming error.

- **Code Example**:
  Here, a `Bird` instance is downcast from its `Animal` form obtained through upcasting.

```swift
let anotherAnimal = Bird()
if let someBird = anotherAnimal as? Bird {
    print("It's a bird!")
} else {
    print("It's not a bird!")
}
```

### The `is` Operator

- **Role**: It's used for type checking. The operator returns a Boolean value, indicating whether a variable is of a specific type or conforms to a certain protocol.

- **Code Example**:
  With the `is` operator, you can validate the type before performing any specific type-related operations:
 
  ```swift
  if anotherAnimal is Bird {
    print("It's a bird!")
  }
  ```

### Whole Number Types

- **Validity**: The is and as operators in Swift do not support them for whole number types such as Int, UInt, or even Float. You must use other type-checking methods for these.
<br>

## 11. How do you define a _class_ in _Swift_?

In Swift, a **class** serves as a blueprint for creating objects. **Classes** enable you to use both reference types and value semantics.

### Basics of Classes in Swift

- **Reference Type**: Objects are shared by reference.
- **Inheritance**: Both single and multi-level inheritance is supported.
- **Type Casting**: You can check and interpret types as needed.
- **Deinitialization**: Classes support the process of deinitialization.

### Code Example: Basic Class Structure

Here is the Swift code:

```swift
class Vehicle {
    var wheels: Int // Property declaration
    
    init(wheels: Int) {
        self.wheels = wheels
    }
    
    func startEngine() {
        print("Vroom Vroom!")
    }
}
```

### Properties

- **Stored Properties**: 
    - Class instances can have variables and constants to store values.
    - Use **lazy** to create properties when first accessed.
    - Use **static** for type-level properties. 

- **Observer**: 
    - Monitor and respond to changes in property values.
    - **willSet** and **didSet** are used to define actions.

### Code Example: Properties in a Class

Here is the Swift code:

```swift
class Car: Vehicle {
    var brand: String
    static let headlightCount = 2

    lazy var registration: String = {
        let reg = "ABC" // Simulate registration generation
        return reg
    }()

    var mileage: Double = 0 {
        willSet(newMileage) {
            print("About to update mileage to \(newMileage)")
        }
        didSet {
            if mileage > oldValue {
                print("Mileage increased!")
            }
        }
    }

    init(brand: String, wheels: Int) {
        self.brand = brand
        super.init(wheels: wheels)
    }
}
```

### Methods

- **Instance Methods**:
    - Belong to an instance of a class.
    - Accessed and manipulated using the `self` keyword.
  
- **Type Methods**:
    - Associated with the class itself.
    - Defined with the `static` keyword.

### Code Example: Methods in a Class

Here is the Swift code:

```swift
class Bicycle: Vehicle {
    var hasBasket: Bool = false

    override func startEngine() {
        print("No engine, just pedal!")
    }

    func ringBell() {
        print("Ding ding!")
    }

    class func changePedals() {
        print("Pedals changed.")
    }
}
```

### Method Dispatch

Swift supports **dynamic dispatch** by default for methods. You can opt for **static dispatch** for performance reasons using the `final` keyword.

### Code Example

Here is the Swift code:

```swift
class Shape {
    final func displayName() {
        print("I am a shape.")
    }

    func area() -> Double {
        return 0.0
    }
}

class Square: Shape {
    var sideLength: Double

    init(sideLength: Double) {
        self.sideLength = sideLength
    }

    override func area() -> Double {
        return sideLength * sideLength
    }
}

let shape: Shape = Square(sideLength: 5.0)
shape.area()  // Ultimately dispatches to Square's overwritten method for computing the area.
```
<br>

## 12. Explain the difference between _classes_ and _structures_ in _Swift_.

In Swift, both **classes** and **structures** serve to define custom data types, like in most object-oriented programming languages. The language also introduces **enums** and **tuples** as first-class citizens for managing data.

### Key Distinctions

- **Inheritance**: Swift only permits classes to inherit from other classes, forming a parent-child hierarchy. Structures offer no inheritance mechanism.

- **Memory**: Class instances are reference types and reside on the heap. In contrast, structures are value types and are generally stack-allocated.

- **Mutability**: For class instances, you can have constants (let) that can still change, as data inside the instance can be mutable (vars). In structures, if an instance is defined as a constant, all its properties are also defined as constants.

- **Identity vs. Equivalence**: Classes possess identity, recognizable through references. Two references point to the same object if they share an identity referred to by the same memory location. Structures equate based on their complete data match.

### What Makes Structures Unique?

Swift structures provide a range of features bringing additional functionality and purpose beyond simple data encapsulation, often found in languages like C.

- **Protocols Adoption**: Structures can implement protocols and define conformance to standard functionality sets.

- **Computed Properties**: Unlike classes, structures can offer calculated properties without storage backing them.

- **Memberwise Initializers**: Structures automatically provide parameter list for property-based construction, which must be manually handled in classes.

- **Immutable Properties**: Structures can have **constant properties** even if the instance itself is variable. This ensures the properties don't change after initialization.

### Modern Best Practices with Structures

The use of structures is often advocated for, especially when constructing lightweight, immutable data types. Their predictable copy-on-write behavior can boost performance in scenarios involving temporary data storage or multithreading.

Swift Standard Library essentials, such as `String`, `Array`, `Dictionary`, and `Set`, lean on structures.

### When to Choose Classes over Structures

- **Reference Semantics**: When you need multiple instances to point to the same data, influencing shared state across an app or system.
- **Inheritance Requirements**: When the design warrants a need for class inheritance, for example, in deviseen customized user interfaces.
<br>

## 13. What are the key principles of _inheritance_ in _Swift_?

**Inheritance** is a central concept in Object-Oriented Programming (OOP) that allows classes to inherit characteristics from other classes. Swift has a single-inheritance model. 

### Key Principles

#### Limited to a Single Superclass

In Swift, each class is explicitly derived from one, and only one, **superclass**. However, it's still possible to have multiple subclasses associated with the same superclass.  

This helps keep the class hierarchy easily understandable, preventing complexities that arise from multiple inheritance.

#### Unavoidable Inheritance Chain

Even if a class is explicitly declared with no superclass, it still indirectly inherits from Swift's **base class**, Any.

#### Fundamental Base Class: "Any"

Swift's "Object" counterpart, **Any**, is the base class for all other classes. This relationship exists, whether it's stated explicitly or implicitly.

#### Inheritance-Enabling Keyword: class

The **class** keyword identifies the superclass of a derived class. 

- It's mandatory for superclasses.
- It's optional for classes without a superclass, aligning with Swift's single-inheritance model.
<br>

## 14. How does _Swift_ enable _encapsulation_ within classes and _structs_?

Both **classes** and **structs** in Swift provide encapsulation through access control modifiers. Swift has several access control levels, such as **open**, **public**, **internal**, **fileprivate**, and **private**, to determine the visibility and accessibility of properties and methods.

### Key Access Control Levels

- **Open** (for classes and their members) means the entity is accessible and can be subclassed outside its defining module. This level allows the most freedom but is also the most open.
  
- **Public** indicates that the entity can be used and accessed outside the defining module, but not subclassed.

- **Internal** is the default. It signifies that properties and methods are accessible within the defining module but not from outside.

- **Fileprivate** makes the entity visible only within the same file.

- **Private** is the most restrictive: members with this level are accessible only within the defining declaration.

### Strike a Balance for Better Encapsulation

The access levels form a hierarchy, and it's better for encapsulation and ease of maintenance to **expose only what's necessary**. Even in a tightly knit team, following these access control best practices can lead to more maintainable code:

- If possible, use **private** access for properties and methods that are internal to a type.
- Opt for **fileprivate** if these properties or methods need to be accessed from within the same file but are not meant for external use. Use **internal or public** only if absolutely necessary.
<br>

## 15. Can _Swift_ classes have _multiple inheritance_?

While **Swift** does not support **multiple inheritance** for classes, you can still achieve similar functionality using protocols and extensions. This design approach, known as "Multiple Inheritance with Protocols", elegantly resolves many of the classic issues associated with multiple inheritance in other languages.

### Key Components

- **Classes**: Remain inherently single-inheritance. However, by adhering to multiple protocols, they can incorporate functionalities from various sources.

- **Protocols**: Allow for method and property definitions without specifying their actual implementation. This sets the stage for multiple functionalities to be "inherited" through protocol adoption.

- **Extensions**: Provide implementations for the methods and properties defined within the adhered protocols. Each extension essentially acts as a bridge between the class and a specific protocol, supplying the required behaviors.

### Code Example: Multiple Inheritance with Protocols

Here is the Swift code:

  ```swift
  protocol A {
      func methodA()
  }
  
  extension A {
      func methodA() {
          print("Method A!")
      }
  }
  
  protocol B {
      func methodB()
  }
  
  extension B {
      func methodB() {
          print("Method B!")
      }
  }
  
  class MyClass: A, B {
      // No need to provide an implementation for methodA or methodB
      // They are inherited from protocols A and B and are implemented in the protocol extensions.
  }
  
  let obj = MyClass()
  obj.methodA()
  obj.methodB()
  ```

 In this example:

- **Protocol `A`** requires `methodA()` and has an extension conforming to that protocol and providing a default implementation for `methodA()`.

- **Protocol `B`** requires `methodB()` and has an extension conforming to that protocol and providing a default implementation for `methodB()`.

- The `MyClass` class adopts both protocols **`A`** and **`B`**. Due to this adoption and the default implementations provided by the protocol extensions, instances of **`MyClass`** will have both `methodA()` and `methodB()` available. When invoking `methodA()` or `methodB()` on an object of `MyClass`, the implementations provided by the protocol extensions will be called.

This approach allows **Swift** to retain its focus on safety, clarity, and ease of maintenance. It resolves complexities associated with diamond problems and method collisions that can arise in traditional multiple-inheritance models.
<br>



#### Explore all 70 answers here 👉 [Devinterview.io - Swift](https://devinterview.io/questions/web-and-mobile-development/swift-interview-questions)

<br>

<a href="https://devinterview.io/questions/web-and-mobile-development/">
<img src="https://firebasestorage.googleapis.com/v0/b/dev-stack-app.appspot.com/o/github-blog-img%2Fweb-and-mobile-development-github-img.jpg?alt=media&token=1b5eeecc-c9fb-49f5-9e03-50cf2e309555" alt="web-and-mobile-development" width="100%">
</a>
</p>

