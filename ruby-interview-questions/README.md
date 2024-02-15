# 100 Core Ruby Interview Questions

<div>
<p align="center">
<a href="https://devinterview.io/questions/web-and-mobile-development/">
<img src="https://firebasestorage.googleapis.com/v0/b/dev-stack-app.appspot.com/o/github-blog-img%2Fweb-and-mobile-development-github-img.jpg?alt=media&token=1b5eeecc-c9fb-49f5-9e03-50cf2e309555" alt="web-and-mobile-development" width="100%">
</a>
</p>

#### You can also find all 100 answers here 👉 [Devinterview.io - Ruby](https://devinterview.io/questions/web-and-mobile-development/ruby-interview-questions)

<br>

## 1. What is _Ruby_, and why is it popular for web development?

**Ruby** is a dynamic, object-oriented programming language known for its simplicity and focus on developer productivity. Its main claim to fame in web development is the **web application framework**, **Ruby on Rails** (RoR), which transformed the way web applications are built by promoting convention over configuration.

### Key Features & Contributions to Web Development

- **Language Syntax**: Ruby's syntax has an appeasing natural language style. This, paired with its dynamic typing, powerful metaprogramming features, and absence of semicolons, results in clean and expressive code.

- **Gems**: Ruby's package manager, **RubyGems**, simplifies library management, making it easy to integrate numerous third-party extensions.

- **Database Integration**: ActiveRecord, a popular object-relational mapping system, aids in managing database records via a natural, object-oriented interface.

- **MVC Pattern**: Rails, in particular, is famed for its adherence to the Model-View-Controller pattern, offering a clear separation of concerns.

- **Code Rearrangement**: The **auto-loading mechanism** allows for seamless navigation between related files and classes while coding.

- **Ecosystem Consistency**: RoR brought about a `many`-`to`-`many` relationship with databases, streamlining and simplifying existing **development patterns**.

- **Strong Community**: The language's supportive community and its commitment to clean, readable code are evident in guiding principles like "Mediterranean" quality and "Matz's kindness."

- **Test-Driven Development**: RoR promotes best-testing practices from the project's inception, ensuring reliability.

- **Giant Corporations' Indulgence**: Prominent organizations such as GitHub, Shopify, and Airbnb have successfully tapped into the potential of Ruby on Rails.

### Code Example: Ruby on Rails (RoR) Routing

Here is the Ruby code:

```ruby
# config/routes.rb
Rails.application.routes.draw do
  root 'welcome#index'
  get 'products/:id', to: 'products#show'
  resources :articles
end
```

This file configures routes for different URLs, specifying which controllers and actions to invoke. For instance, upon receiving a `GET` request for `products/5`, RoR would route it to the `show` action in the `ProductsController` with the ID parameter set to `5`. Such straightforward setups contribute to RoR's appeal.
<br>

## 2. How do you create a _Ruby script file_ and execute it on a _command line_?

First, you create a `Ruby` script file with a `.rb` extension that contains your Ruby code. You can then **execute** this script using the `ruby` command in your command line.

### Basic Steps for Creating and Running a Ruby Script in a File

1.  **Create a File**: Use any text editor to write your Ruby code, and save the file with a `.rb` extension, e.g., `my_ruby_script.rb`.

2. **Write Ruby Code**: Here is a simple example.

    ```ruby
    # Filename: my_ruby_script.rb
    puts "Hello, Ruby Script!"
    ```

3. **Run the Ruby Script**: Go to your command line and navigate to the folder where the Ruby file is saved. Then, type the following command:

   ```bash
   ruby my_ruby_script.rb
   ```

   After pressing enter, you will see the output:
   
   ```bash
   Hello, Ruby Script!
   ```

### Getting More Advanced

#### Command-Line Arguments

You can access **command-line arguments** using special variables called `ARGV`.

Here is the code:

```ruby
# Filename: script_with_args.rb
puts "Arguments: #{ARGV.join(', ')}"
```

In the command line, you would run this script as:

```bash
ruby script_with_args.rb arg1 arg2 arg3
```

The output would be:

```bash
Arguments: arg1, arg2, arg3
```

#### Interactive Scripts

Ruby scripts can engage with users using the `gets` method.

Here is an example:

```ruby
# Filename: interactive_script.rb
puts "What is your name?"
name = gets.chomp
puts "Hello, #{name}!"
```

When you run this script using `ruby interactive_script.rb`, it will prompt you for your name, and after you provide it, it will greet you.

#### Background Processing

If you want a script to run in the background without blocking your command line, you can use the `&` character.

For instance, to run a script called `background_script.rb` in the background, you can use:

```bash
ruby background_script.rb &
```

#### Ruby Shell

For more complex shell operations, `Ruby` offers the `shell` library.

Here is the sample code:

```ruby
require 'shell'

# Use 'open' to open a URL in your default browser.
sh = Shell.new
sh.open "https://example.com"
```
<br>

## 3. What are the basic _data types_ in _Ruby_?

**Ruby** is claimed to treat "everything as an object". But like many languages, Ruby has both **primitive** and **abstract data types**.

### Primitive Types

- **Numbers**:
  - **Integers** can be of any size (limited by system memory).
  - **Floating-Point** numbers follow the IEEE 754 standard.
- **Booleans**: Represented by `true` and `false`.
- **Symbols**: Unique, immutable identifiers represented with a `:` followed by a name.

### Abstract Types

- **Strings**: Unicode with multiple encodings.
- **Arrays**: Ordered, indexed collections.
- **Hashes**: Key-value pairs, also known as dictionaries or maps in other languages.

### Others assimilated Primitive Types

Ruby, despite its philosophy of being completely object-oriented, has some underlying **primitive paradigms** due to its performance concerns and efficiency considerations.

- **nil**: Represents 'nothing' or 'empty'. It's the only instance of `NilClass`.

- **Booleans**: While `true` and `false` are themselves keywords, any other value in Ruby is considered truthy in a conditional context.

### Advanced Types

- **Rational Numbers**: Represented as a fraction (e.g., `1/3r`).
- **Complex Numbers**: Have real and imaginary parts (e.g., `2 + 3i`).
- **Dates and Times**: Provide various built-in classes like `Time` for dealing with date and time values.

### Ruby Uniqueness

Ruby shuns a "strictly-typed" system. **Variables need not be declared upfront** and can be reassigned to different types during execution. This freedom, although liberating, can lead to unexpected behavior, especially in larger codebases.
<br>

## 4. Explain the difference between _symbols_ and _strings_ in _Ruby_.

**Ruby** features both strings and **symbols**, each with distinct use cases.

### Key Distinctions

- **Type**: Strings are of class `String`, while symbols are instances of `Symbol`.
- **Mutability**: Strings are mutable, symbols are not.
- **Memory**: Symbols are stored as a single, unique object in memory, while each string is unique.
- **Performance**: As symbols are immutable, **lookups** are faster than for equivalent strings.

### Primary Usages

- **Strings**: For text and dynamic data that may change or be unique across different objects or occurrences.
- **Symbols**: Typically used as keys for hashes or unique identifiers in the program. They're advantageous for **lookup efficiency** and when the actual content of the identifier is less relevant than its unique identity.

### Memory Considerations

- As symbols are **stored only once** in memory, they are memory-efficient in certain scenarios, like using the same symbol across different objects or operations. Be cautious, though, as unnecessarily creating a large number of symbols can lead to memory bloat.
- Strings may be more memory-intensive, especially when there are numerous unique strings. However, they are the right choice when dealing with data that genuinely varies or where mutability is required.

### Code Example: String vs Symbol

Here is the Ruby code:

```ruby
# Strings
str_1 = "Hello"
str_2 = "Hello"
puts str_1.object_id == str_2.object_id  # Output: false

# Symbols
sym_1 = :hello
sym_2 = :hello
puts sym_1.object_id == sym_2.object_id  # Output: true
```
<br>

## 5. How are _constants_ declared and what is their _scope_ in _Ruby_?

In Ruby, you declare a constant by using all uppercase letters. Constants are subject to lexical scoping. While **reassignment** is technically possible (spawning a warning), it should be avoided as a practice.

### Constant Declaration

You can declare a Ruby constant using `Object::CONSTANT` notation or by assigning a **value directly to an identifier**.

#### Code Example: Constant Declaration 

```ruby
# Using Object::CONSTANT notation
Math::PI 

# Direct assignment
RADIUS = 5.0
```

### Constant Scope

Constants have a **global scope**, but their visibility can be restricted within classes and modules.

#### Global VS. Local Scope

- **Global Scope**: Constants are accessible throughout the entire application.
  ```ruby
  A = 1     # Top level
  module M
    puts A  # Outputs: 1
  end
  ```

- **Local Scope**: Constants are defined within a module or a class.
  ```ruby
  module M
    A = 2
    A = 3
    puts A  # Outputs: 3
  end
  ```

### Best Practices

- **Avoid** re-assigning constants. Although this generates a warning, the reassignment can still take place, which can lead to unintended behavior.
- For areas where you want to have a **constant's value remain unchanged**, use `.freeze` on the constant or variable storing the constant's value.
  
#### Code Example: Avoiding Reassignment

```ruby
require "warning"

# Generates a warning: already initialized constant
A = 1
A = 2 

warning 'constant reassignment'

puts A  # Outputs: 2
```

#### `Object#freeze`

```ruby
CIRCLE_AREA = Math::PI * (RADIUS ** 2)
CIRCLE_AREA.freeze

# Any reassignment will generate an error
# CIRCLE_AREA = 100 

puts CIRCLE_AREA
```
<br>

## 6. Explain the use of '_require_' and '_include_' in _Ruby_.

**Ruby** uses both **Require** and **Include** to manage dependencies and to mix modules into classes.

### Require

- **Purpose**: Loads external libraries, enabling access to their defined classes and modules.

- **Execution**: Done at the top of the file or script.

- **Trigger**: A `LoadError` is raised if the required file is not found.

- **State Management**: Tracks loaded libraries, subsequently ignoring further `require` calls for the same library.

### Example: Using Require

Here is the Ruby code:

```ruby
# In file application.rb
require 'my_library'

# In file my_library.rb
class MyLibrary
  # ...
end
```

### Include

- **Purpose**: Integrates a module's methods within a class, giving the class access to those methods.

- **Execution**: On the specific class that necessitates the module's functionality.

- **State**: Not applicable for classes, as they can include multiple modules.

#### Why is it Used?

- **Require**: Ensures the presence of the external library before continuing, a basic necessity for external code.
- **Include**: Mixes in module functionality only when needed, aligning with Rails' convention of using it in the classes contextually.
<br>

## 7. What are _Ruby iterators_ and how do they work?

When it comes to Ruby, iterators **allow for easy, streamlined data manipulation**. Whether you're working with arrays, ranges, or other data structures, iterators help you efficiently apply operations to each element without needing to manage loop counters.

### Most Common Ruby Iterators

- **Each**: The most basic iterator, it goes through each element.
- **Each with index**: Similar to each, but it also gives the index of the current element.

### Code Example: Each & Each with Index

Here is the Ruby code:

```ruby
arr = [5, 4, 3, 2, 1]

# Iterating with Each
arr.each { |num| puts num }

# Output:
# 5
# 4
# 3
# 2
# 1

# Iterating with Each with Index
arr.each_with_index { |num, index| puts "#{index}: #{num}" }

# Output:
# 0: 5
# 1: 4
# 2: 3
# 3: 2
# 4: 1
```

### Common Usage

- **Each Char**: Often used with strings, this iterator loops through each character.
- **Each Line**: Handy for reading files, it processes lines one at a time.

### Code Example: Each Char & Each Line

Here is the Ruby code:

```ruby
str = "Hello, World!"

# Iterating Each Character
str.each_char { |char| puts char }

# Output:
# H
# e
# l
# l
# o
# ,
# ...
```

```ruby
File.open('example.txt').each_line do |line|
  puts line
end
```

### Predicative Iterators

These iterators select elements from a collection that match specific conditions. They are typically used in combination with **blocks**.

Examples include `select`, `reject`, and `grep`. Each is designed to achieve specific selection goals:

- `select` returns elements that yield true in the block.
- `reject` returns elements that yield false in the block.
- `grep` returns elements that match a specified pattern.

### Code Example: `select`, `reject`, and `grep`

Here is the Ruby code:

```ruby
# Select even numbers
numbers.select { |num| num.even? }

# Reject short names
names.reject { |name| name.length < 5 }

# Grep to find email addresses
text = "Email me at user@example.com"
text.grep(/\b\w+@\w+\.\w+\b/)
```

### Chase & Transform

These iterators process the elements and return a result. They include `map`, `collect`, and `partition`.

- `map`: Transforms each input and returns a new array.
- `collect`: Identical to map, but ops include the return value.
- `partition`: Separates elements into two groups based on whether the block returns true or false.

### Code Example: `map`, `collect`, and `partition`

Here is the Ruby code:

```ruby
# Double each number
numbers.map { |num| num * 2 }

# Names all uppercase
names.collect { |name| name.upcase }

# Split numbers based on even or odd
numbers.partition { |num| num.even? }
```

### Execute Operations

These iterators modify their elements or perform side effects. Examples include `each` and `each_with_index`.

Often used for their simplicity, do exercise caution as these functions can have unexpected results, especially when combined with unintended side effects.

- `each`: Processes each element but does not return anything.
- `each_with_index`: Similar to each, but also gives the index of the current element.

### Sort-Related Operations

When working with ordered collections like arrays or ranges, Ruby provides various sorting options. Common sorting iterators include `sort`, `sort_by`, and `reverse_each`. They all work with blocks to customize the sorting or iteration behavior.

### Repetition-Based Iterators

These Ruby constructs are particularly useful in the context of text manipulation, allowing you to repeat characters (such as hyphens for formatting headers) for a specified number of times.

- `each_line`: Useful when processing multi-line strings or files.
- `each_char`: Ideal for character-level processing in strings or enumerations.
- `downto`: Iterates downwards to a specified value.
- `upto`: Iterates upwards to a specified value.
- `times`: Repeats the associated block a predetermined number of times.
- `step`: Indents a set number of times, confined by a range.
- `cycle`: Used primarily with blocks, it repeatedly moves through the specified range.

### Code Example: Repetition-Based Iterators

Here is the Ruby code:

```ruby
# Print a line of stars
'*'.upto('*****') { |s| puts s }

# Output:
# *
# **
# ***
# ****
# *****

# Print numbers from 5 to 10, then their squares
5.upto(10) { |num| puts num }
5.upto(10).each { |num| puts num**2 }
```
<br>

## 8. How are _errors_ handled in _Ruby_?

Ruby's **exception hierarchy** enables developers to manage different kinds of errors. The two main exception types cater to a multitude of issues:

- **StandardError**: For generic problems that occur during code execution.
- **SystemCallError**: Specifically deals with errors originating from system calls.

### Ways to _Handle Exceptions_ in Ruby

#### Top-Level Exception Handling

Ruby leverages the `at_exit` method for centralized error handling. This approach is mainly useful for logging errors before program exit or for cleaning up resources.

```ruby
at_exit do
  puts $!.message if $!
end
```

#### Single Statement Unwind

Utilize **inline rescue**, marked by the `begin` and `end` block. If an exception arises during the evaluation of the enclosed expression, it's caught.

```ruby
result = begin
  raise StandardError, "This is an error"
rescue StandardError => e
  "Rescued: #{e.message}"
end

puts result  # Output: Rescued: This is an error
```

### Custom Exception Handling

Developers benefit from creating their custom exception classes or identifying specific exception types to tailor their error management strategies.

#### Defining Custom Exception Classes

The `Exception` class or, more preferably, its subclass, `StandardError`, are parents to all user-defined exceptions. This inheritance ensures that all custom exceptions are catchable via `rescue`.

```ruby
class MyCustomError < StandardError
  # Additional behavior or settings
end

raise MyCustomError, "Something went wrong!"
```

#### Identifying the Right Exception

An error's distinct nature often demands a corresponding exception. For instance, consider a method handling file operations:

```ruby
def read_file(file_path)
  raise ArgumentError, "File path is empty" if file_path.to_s.empty?
  raise IOError, "File not found" unless File.exist?(file_path)

  File.read(file_path)
end
```

Upon calling `read_file`, any exception correlated to an invalid file path can be reliably caught and addressed with a targeted `rescue` block.

### Error Handling Best Practices

- **Keep it Precise**: Make use of granular `rescue` blocks or `case` statements to align the corrective measures with the specific error.
  
- **Maintain a Balance**: Overuse of exceptions can convolute code and hinder its readability. Carefully select the exceptions likely to surface and necessitate special attention.

- **Locale Transparency**: Choose either a local exception resolver that terminates in the current method or a global one that percolates up the call stack, but aim for consistency.

### Performance Considerations

While exceptions can be invaluable for isolated and unexpected mishaps, triggering and managing them incurs a performance cost. It's typically wiser to leverage them predominantly in such scenarios and not as part of conventional program flow.
<br>

## 9. Describe the difference between _local_, _instance_, and _class variables_.

Let's set the record straight on the differences between **local**, **instance**, and **class variables** in Ruby.

### Common Features

All three variable types support:

- **naming**: ![A-Za-z0-9_](2, 50)
- **assignment**: `variable_name = value`
- **access control**: `public`, `private`, and `protected`
- **immediacy**: their scope begins from where they are initialized and exists until the scope ends.

### Local Variables

- **Scope**: Limited to the block where they are defined.
- **Life Cycle**: Created when the program reaches their definition and destroyed when the block is exited.

#### Example: Local Variable

Here is the Ruby code:

```ruby
def hello
  name = "Ruby"
  puts "Hello, #{name}!"  # Output: Hello, Ruby!
end

# Accessing name outside its defined block will cause an error.
# puts name  # Will raise an error
```

### Instance Variables

#### Naming Convention

An instance variable is **prefixed with a single '@' symbol**.

- **Scope**: Primarily within the class, but is accessible from outside the class if the class is instantiated.
- **Life Cycle**: Created when an object is instantiated and remains available until that particular object is destroyed.

#### Example: Instance Variable

Here is the Ruby code:

```ruby
class Greeting
  def set_name(name)
    @name = name
  end

  def display_greeting
    puts "Hello, #{@name}!"  # Output: Hello, Ruby!
  end
end

greeting_instance = Greeting.new
greeting_instance.set_name("Ruby")
greeting_instance.display_greeting
```

### Class Variables

#### Naming Convention

A class variable is **prefixed with two '@' symbols**.

- **Scope**: Within the class and its inheritors but not accessible from outside.
- **Life Cycle**: Created when assigned within the class or its inheritors and accessible as long as the class or one of its inheritors is in memory.

#### Example: Class Variable

Here is the Ruby code:

```ruby
class Employee
  @@company_name = "ABC Corporation"

  def self.company_name=(name)
    @@company_name = name
  end

  def display_company_name
    puts @@company_name
  end
end

employee1 = Employee.new
employee2 = Employee.new

# Output: "ABC Corporation" for both employee1 and employee2.
employee1.display_company_name
employee2.display_company_name

Employee.company_name = "New Company"  # changes the class variable

# After changing, outputs for both employee1 and employee2 will be "New Company".
employee1.display_company_name
employee2.display_company_name
```
<br>

## 10. What are _Ruby's accessor methods_?

In Ruby, **accessor methods** allow you to manipulate object attributes. There are three types of accessor methods: **`attr_reader`**, **`attr_writer`**, and **`attr_accessor`**, each serving a specific role in the attribute's lifecycle

### Attribute Methods

- **`attr_reader`**: Generates a simple getter method for an attribute. It can be accessed but not modified externally.
- **`attr_writer`**: Generates a basic setter method. The attribute can be modified but not read externally.
- **`attr_accessor`**: Combines both getter and writer methods in one. This creates a full-fledged getter and setter for the attribute.

### Code Example: Accessor Methods

Here is the Ruby code:

```ruby



class Person
  attr_reader :name, :age
  attr_writer :name, :age
  def initialize(name, age)
    @name = name
    @age = age
  end
end

person = Person.new("Alice", 30)

person.name # Returns "Alice"
person.name = "Bob" # Error: undefined method 'name='

person.age # Returns 30
person.age = 35 # Error: undefined method 'age='

person.instance_variables # Returns [:@name, :@age]

```
<br>

## 11. How does _garbage collection_ work in _Ruby_?

**Ruby** employs automatic memory management, which is primarily influenced by **garbage collection techniques**. Let's understand the specifics.

### Mark-and-Sweep Algorithm

- **Step 1 - Mark**: The process starts from the **root** of object references. The GC traverses memory, marking referenced objects.
  
- **Step 2 - Sweep**: It scans for **unmarked** objects and reclaims their memory, making it available for future use.

### Generational Garbage Collection

To optimize the Mark-and-Sweep approach, Ruby introduces generational garbage collection.

- **Focused on Object Age**: Objects are classified based on their age.

- **Young vs. Old Objects**: 
  - New objects start in the **Young Generation**.
  - Objects that persist multiple GC cycles move to the **Old Generation**.

- **Collection Frequency**: The Young Generation is collected more frequently.

- **Short- and Long-Lived Object Management**: It's easier to collect younger objects, reducing the scope and overhead of a complete garbage collection cycle.

### Reference-Counting and `ObjectSpace`

Although CPython uses reference-counting to track object lifespans, Ruby typically does not. 

- **ObjectSpace**: It's a module that allows retrieval of all objects.

  However, note that modern Ruby versions represent a hybrid system, sensitive to object types and context.

### Code Example: Garbage Collection in Ruby

Here is the Ruby code:

```ruby
# Enable trashcan (Ruby 2.6 onwards)
ObjectSpace::count_objects[:FREE] > 100_000 && GC.start

# Ruby versions before 2.6
GC.start
```
<br>

## 12. Explain the difference between '_gets.chomp_' and '_gets.strip_'.

Let me go through the major difference.

### Key Distinctions

- **Input Requirement**:
  - `gets.chomp` removes **all trailing whitespace** and the newline character.
  - `gets.strip` eliminates **all leading and trailing whitespace**, including the newline character.

- **Use Cases**:
  - **`gets.chomp`**: Suited when you anticipate or require specific trailing characters to be preserved.
  - **`gets.strip`**: Ideal for scenarios where you need to sanitize or validate user input by removing any extra leading or trailing spaces.

### Code Sample: `gets.chomp` & `gets.strip`

Here is the Ruby code:

```ruby
# Using the gets.chomp method
puts "Enter your name (including a trailing space): "
name_chomp = gets.chomp
puts "Name with trailing space: #{name_chomp}"

# Using the gets.strip method
puts "Enter your name: "
name_strip = gets.strip
puts "Name without trailing space: #{name_strip}"
```
<br>

## 13. What is the role of '_self_' in _Ruby_?

In Ruby, **`self`** serves as a "mirror" that reflects the current context. Depending on where it's used, `self` can represent different objects.

Here's the breakdown:

### `Self` in Different Contexts

#### 1. Instance Methods

In this context, `self` refers to the instance of the object on which the method is called.

Consider the following code:

```ruby
class MyClass
  def instance_method
    puts self
  end
end

object = MyClass.new
object.instance_method
```

The output would be the object reference `#<MyClass:0x007fb4fa869358>`.

#### 2. Class Methods

Within a class definition, `self` refers to the class itself. This is why you use `self.method_name` to define class methods.

For instance:

```ruby
class MyClass
  def self.class_method
    puts self
  end
end

MyClass.class_method
```

The output will be the class `MyClass`.

#### 3. Method Invocation

When a method is missing due to a typo or other reason, Ruby executes `method_missing` which can help handle such cases.

Consider this example:

```ruby
class MyClass
  def method_missing(m, *args)
    puts "There's no method called #{m}"
  end

  def test_method
    method_thaat_doesnt_exist
  end
end
```

Calling `test_method` will invoke `method_missing` with the method name `"method_thaat_doesnt_exist"`.
<br>

## 14. Explain the principle of '_Convention over Configuration_' in the context of _Ruby_.

**Convention over Configuration** (CoC) is a software design principle that simplifies development by reducing the number of decisions developers need to make.

In its essence, CoC means that frameworks come with **best practice defaults** or "conventions" that are automatically applied unless explicitly configured to behave differently.

### Practical Application

1. **Code-Base Structures**: Many Ruby web frameworks, like Ruby on Rails or Sinatra, expect a certain directory structure that groups related files.
2. **Naming Conventions**: Specially designed rules for naming classes, methods, and databases to help in identification and automatic linking.
3. **API Endpoints**: Through consistent naming, it's possible to infer routing information in web applications.
4. **Database Schemas**: Named fields and tables allow the ORM to deduce relationships and configurations.

### Example: CRUD Actions in RoR

In Ruby on Rails, the "conventions" for a resourceful route automatically map HTTP verbs to CRUD actions:

```ruby
# config/routes.rb
resources :articles

# Routes:
# HTTP   Path                Controller#Action    Used For
# ------------------------------------------------------------
# GET    /articles           articles#index       Display a list
# GET    /articles/:id       articles#show        Display a specific article
# GET    /articles/new       articles#new         Display a form to create a new article
# POST   /articles           articles#create      Add a new article to the database
# GET    /articles/:id/edit  articles#edit        Display a form to edit an existing article
# PATCH  /articles/:id       articles#update      Update an existing article in the database
# PUT    /articles/:id       articles#update      (Alternate for update)
# DELETE /articles/:id       articles#destroy     Remove a specific article from the database
```

Here, the convention to map action names to routes frees the developer from configuring each route manually.

### Benefits

- **Speed**: It streamlines development and reduces boilerplate.
- **Interoperability**: CoC enables consistency across different projects and teams.

### Risks and Challenges

- **Over-optimization**: While it's efficient for simple, well-understood requirements, it can make advanced configurations and customizations cumbersome.
- **Learning Curve**: Newcomers might find it challenging to adapt to these standard conventions.
- **Magic**: Over-reliance on CoC can make the system seem like it has hidden, unexplained behaviors.
<br>

## 15. How does _Ruby_ support _metaprogramming_?

**Ruby** offers powerful **metaprogramming** capabilities, enabling developers to write flexible, dynamic code. Key to Ruby's metaprogramming are `class` methods such as `define_method` and language features like `Open Classes` leading to advanced techniques like `Dynamic Dispatch`.

### Dynamic Dispatch Mechanism

- **Dynamic Dispatch**: Methods can be called at runtime, based on the object's context, using `send`. This makes it easier to manage method invocations in metaprogrammed code.

```ruby
class MathOperations
  def do_operation(operator, x, y)
    send(operator, x, y) # Dynamic dispatch
  end

  private

  def add(x, y)
    x + y
  end

  def subtract(x, y)
    x - y
  end
end

result = MathOperations.new.do_operation(:add, 10, 5) # 15
```

### Class Modifications with `Open Classes`

- **Open Classes**: Ruby allows changing a class's definition dynamically, even after its initial declaration.
  
  This example adds a `reverse` method to the `String` class.

  ```ruby
  class String
    def reverse
      chars.reverse.join
    end
  end
  ```

### Code Evaluation and Execution

- **Code Evaluation**: Code strings can be executed within a bound context, enabling runtime code execution and evaluation.

  This is an example using `eval` to define a method at runtime, equivalent to `def double(x) x * 2; end`, but the method signature is constructed dynamically.

  ```ruby
  method_signature = 'double(x)'
  method_body = 'x * 2'
  eval("def #{method_signature}; #{method_body}; end")
  ```

- **Binding Tasks**: `proc` objects capture both the method (or block) and its associated context. They can be transferred across lexical scopes, allowing delayed execution of code.

  ```ruby
  context = binding
  task = Proc.new { eval 'some_method', context }
  ```

- **Context Toggling**: By toggling a method's visibility, you can control its access scope.

  ```ruby
  class MyClass
    def some_method
      "Public method"
    end

  private

    def toggle_method_visibility(visibility)
      # `send` here is being used for dynamic dispatch
      send(visibility, :some_method)
    end
  end

  instance = MyClass.new
  instance.toggle_method_visibility(:private)
  ```

### Internationalization: Advanced Use of `send` and `eval`

- **Localizing Method Calls**: In internationalization tasks where method calls need to be localized, `send`, `public_send`, or even the more general `eval` can be suitable.

  ```ruby
  def greeting(language)
    eval("#{language}_greeting")
  end

  def spanish_greeting
    "Hola Mundo"
  end
  ```

### Method Missing and Missing Method Feature

- **Method Missing**: This feature is the heart of Ruby's duck typing. It allows classes and objects to respond to method calls even when their definitions are absent, rather than resorting to method-not-found errors.

  This example cleans up a method call, removing spaces or underscores.

  ```ruby
  def method_missing(name, *args, &block)
    cleaned_name = name.to_s.delete(' ').delete('_')
    send(cleaned_name, *args, &block)
  end
  ```

- **`respond_to_missing?`**: This method is often used in conjunction with `method_missing`, providing a way for a class to communicate whether it handles a method call beyond what is statically defined.
<br>



#### Explore all 100 answers here 👉 [Devinterview.io - Ruby](https://devinterview.io/questions/web-and-mobile-development/ruby-interview-questions)

<br>

<a href="https://devinterview.io/questions/web-and-mobile-development/">
<img src="https://firebasestorage.googleapis.com/v0/b/dev-stack-app.appspot.com/o/github-blog-img%2Fweb-and-mobile-development-github-img.jpg?alt=media&token=1b5eeecc-c9fb-49f5-9e03-50cf2e309555" alt="web-and-mobile-development" width="100%">
</a>
</p>

