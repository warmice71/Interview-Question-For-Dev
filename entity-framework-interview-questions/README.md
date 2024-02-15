# Top 100 Entity Framework Interview Questions

<div>
<p align="center">
<a href="https://devinterview.io/questions/web-and-mobile-development/">
<img src="https://firebasestorage.googleapis.com/v0/b/dev-stack-app.appspot.com/o/github-blog-img%2Fweb-and-mobile-development-github-img.jpg?alt=media&token=1b5eeecc-c9fb-49f5-9e03-50cf2e309555" alt="web-and-mobile-development" width="100%">
</a>
</p>

#### You can also find all 100 answers here 👉 [Devinterview.io - Entity Framework](https://devinterview.io/questions/web-and-mobile-development/entity-framework-interview-questions)

<br>

## 1. What is an ORM and how does _Entity Framework_ function as one?

**Object-Relational Mapping** (ORM) bridges the gap between object-oriented code and relational databases.

Key components include:

- **Model**: Defines the data structure.
- **Context**: Acts as an in-memory database, allowing CRUD operations.
- **Mappings**: Specifies relationships between classes and database tables.

### Working of the Entity Framework

  - **Model First**: Design the model, then generate the database.
  - **Database First**: Use an existing database to generate a model.
  - **Code First**: Define classes and relationships first, then generate a database.

### Core Concepts of EF

- **DbContext**: Represents a session with the database.
  
- **Entity**: An object that you map to a table or a view in the database.

- **DbSet\<T>**: Represents a table or a view.

- **Entity State**: Describes the state of an entity in the context.

  - Added
  - Modified 
  - Deleted 
  - Detached 
  - Unchanged

- **Query**: Describes how data is retrieved through LINQ.

### Coordination Between Database and Objects

- **Change Tracking**: Records modifications made to entity objects during their lifecycle.

- **Relationship Management**: Manages relational data by using techniques like lazy loading, eager loading, and explicit loading.

- **Transaction Management**: Handles unit of work operations within the database.

- **Caching**: Maintains an in-memory state of entities for enhanced performance.

### Code Example: Entity Framework

Here is the Csharp code:

```csharp
// Define the model
public class Customer {
    public int Id { get; set; }
    public string Name { get; set; }
    public ICollection<Order> Orders { get; } = new List<Order>();
}

public class Order {
    public int Id { get; set; }
    public decimal Amount { get; set; }
}

// Define the context
public class MyDbContext : DbContext {
    public DbSet<Customer> Customers { get; set; }
    public DbSet<Order> Orders { get; set; }
}

// Usage
using (var context = new MyDbContext()) {
    // Insert
    var customer = new Customer { Name = "John Doe" };
    context.Customers.Add(customer);

    // Query
    var customersWithOrders = context.Customers.Include(c => c.Orders).ToList();

    // Update
    customer.Name = "Jane Doe";

    // Remove
    context.Customers.Remove(customer);

    // Save changes
    context.SaveChanges();
}
```
<br>

## 2. Can you explain the architecture of _Entity Framework_?

**Entity Framework** (EF) provides an abstraction layer for developers, allowing them to work with **object-oriented programming** (OOP) constructs for database operations. It's a powerful **Object-Relational Mapping** (ORM) tool that offers a range of features, including querying, change tracking, and more.

### Key Components

- **Context**: Acts as a gatekeeper between your application and the database. It represents a session and holds an **object cache**.
- **Entity**: A plain-old CLR object (POCO), reflective of a database table or view.
- **Storage Model**: Represents the database schema, including tables, relationships, and views.
- **Mapping**: Establishes the relationships between entities and the storage model.

### 3 Fundamental Aspects

- **Model**: Describes the entities, their properties, and relationships. The model is defined using Code-First, Database-First, or Model-First approaches. EF can also infer the model from an existing database.

- **Database**: Represents the storage where the entity data is persisted.

- **Data Services**: Provide mechanisms to query and perform CRUD (Create, Read, Update, Delete) operations on entities in the database.

### EF Architecture Styles

#### Code-First

- **Development Approach**: Begins with the definition of your model classes and relationships, and EF generates the database schema accordingly.
- **Use Case**: Ideal for scenarios where you have an existing database but want to develop and maintain the database schema using C# or VB.NET.

#### Database-First

- **Development Approach**: Involves designing the database schema first and then creating the EF model based on the schema.
- **Use Case**: Suited when you need to work with an existing database and want to generate the model classes based on that database.

#### Model-First

- **Development Approach**: Implement the model graphically via the EF designer or XML-based EDMX file, and then generate the database schema and model classes.
- **Use Case**: Typically used in legacy applications or for rapid application development where the model is defined first and is then used to generate the database schema and classes.

#### Hybrid Approaches

In many real-world applications, the clear distinction between the development approaches might not always hold true. For instance, an application that began with a Database-First approach might over time introduce new features via the Code-First style. This evolution creates a **hybrid** design, combining the strengths of the various approaches.
<br>

## 3. What are the main differences between _Entity Framework_ and _LINQ to SQL_?

**Entity Framework** and **LINQ to SQL** are both Object Relational Mapping (ORM) frameworks designed for .NET applications.

However, the two differ in various aspects.

### Background

- **Entity Framework**: It was developed by Microsoft as part of the ADO.NET family. It supports visual designers and can generate code from databases and vice-versa.
  
- **LINQ to SQL**: While also developed by Microsoft, it's more lightweight compared to EF. It's specifically designed for modeling databases using objects, methods, and LINQ.

### Relationship Capabilities

- **Entity Framework**: It offers better support for complex relationships, including many-to-many relationships.

- **LINQ to SQL**: It supports basic relationships, but isn't as robust in managing complex ones.

### Code and Database Synchronization

- **Entity Framework**: It features database-first, code-first, and model-first approaches. It can synchronize the code with the database schema.

- **LINQ to SQL**: It's primarily a database-first approach. Changes to the database must be reflected in the code manually.

### Schema Evolution

- **Entity Framework**: It supports automatic database migration through code-first approaches, making it convenient for evolving schemas.

- **LINQ to SQL**: It requires manual updates to the model, and these changes need to be explicitly propagated to the database. It does not support automatic migration.

### Querying Capabilities

- **Entity Framework**: Its querying capabilities are broader due to its ability to work with objects outside the database context, like in-memory datasets.

- **LINQ to SQL**: With a focus on the database, it's optimized for translating LINQ queries directly to SQL, but it's not as versatile as EF.

### Performance and Overhead

- **Entity Framework**: With its superior feature set, it can introduce more overhead, especially in complex scenarios.

- **LINQ to SQL**: As a more focused and lighter framework, it can sometimes provide better performance in specific use cases.

### Data Integrity and Transactions

- **Entity Framework**: It offers better data integrity management and **transactional support** due to its broader feature set.
  
- **LINQ to SQL**: It's not as robust in managing transactions and data integrity.

### Customization and Fine-Tuning

- **Entity Framework**: Given its feature-rich nature, it offers more options for fine-tuning, especially relating to caching, data loading mechanism, etc.

- **LINQ to SQL**: While potentially providing better performance in simpler scenarios, it offers limited options for fine-tuning and optimization. 

### Suitable Use Cases

- **Entity Framework**: It's a comprehensive ORM framework suited for complex enterprise systems, multi-tiered applications, or applications where the database schema evolves frequently.

- **LINQ to SQL**: Due to its lightweight nature, it's better suited for simple applications or those where extensive ORM features aren't required.
<br>

## 4. What is _DbContext_ and how is it used in _EF_?

**DbContext**, a key part of **Entity Framework**, functions as an intelligent bridge between the application and the database. It encapsulates the database session and acts as a hub where entities are tracked, changes are managed, and datasets are queried and persisted.

### Setup and Best Practices

- **Database Context**: Establish a database connection and provide database operations.
- **Entity Sets**: Represent tables, their records, and relationships.

### Key Features

- **Change Tracking**: Alerts on any modifications to entities.
- **Lazy Loading**: On-demand loading of related entities.
- **Early Loading**: Immediate retrieval of data, including related entities.

### Management and Persistence

- **Inserts**
- **Updates**
- **Deletions**
- **Transactions**: Ensures atomic operations, safeguarded by rollback mechanisms.

### Context Lifecycle

- **Transient**: New instances are used per request.
- **Scoped**: Corresponds to a unit of work or an HTTP request.
- **Singleton**: A single instance shared across the entire application.

### Code Example: DbContext

Here is the C# code:

```csharp
using System;
using Microsoft.EntityFrameworkCore;

public class Product {
    public int Id { get; set; }
    public string Name { get; set; }
}

public class StoreContext : DbContext {
    public DbSet<Product> Products { get; set; }

    protected override void OnConfiguring(DbContextOptionsBuilder optionsBuilder) {
        optionsBuilder.UseSqlServer("connection string");
    }
}

// Context usage
public class Program {
    static void Main() {
        using (var context = new StoreContext()) {
            // New product
            var newProduct = new Product { Name = "Laptop" };
            context.Products.Add(newProduct);

            // Update existing product
            var existingProduct = context.Products.Find(1);
            if (existingProduct != null) {
                existingProduct.Name = "Desktop";
            }

            // Delete a product
            var productToDelete = context.Products.Find(2);
            if (productToDelete != null) {
                context.Products.Remove(productToDelete);
            }

            // Commit changes
            context.SaveChanges();
        }
    }
}
```
<br>

## 5. What is the purpose of _DbSet_ in _Entity Framework_?

**DbSet** in **Entity Framework** acts as a gateway to relational databases, enabling you to interact with database tables using object-oriented programming.

### Core Functions

1. **Entity Tracking**: DbSet monitors changes made to the entities during their lifespan. Modifications are categorized as Added, Deleted, or Updated, based on which Entity Framework controls how the changes are reflected in the database.

2. **LINQ Queries**: You can harness the power of Language-Integrated Query (LINQ) to extract, manipulate, and in some cases, construct new model entities from database tables. DbSet acts as a LINQ query provider.

3. **CRUD Operations**: DbSet provides straightforward methods to: insert new entities (**Add**), update their values (**Update**), remove them (**Remove**), retrieve entities by their primary key (**Find**), and also execute bulk operations such as deleting all entities from the table in memory (**RemoveRange**).

4. **Data Binding**: It offers data binding capabilities, making it easier to integrate entities seamlessly with UI components in supported platforms like WPF and WinForms. This ensures that changes in UI maps back to the entities and vice versa, and these changes are tracked effortlessly by the DbSet. This aids in handling more extensive workflows with many entities and updates. **However, Data Binding is not supported in the most recent versions of Entity Framework Core**.

### Code Example: Working with DbSet

Here is the C# code:

```csharp
public class MyDbContext : DbContext
{
    public DbSet<Employee> Employees { get; set; }
}

public static void Main()
{
    using (var context = new MyDbContext())
    {
        // Retrieve an employee by their unique identifier
        var employee = context.Employees.Find(1);

        // Update the title of the employee
        employee.Title = "Senior Engineer";

        // Mark the entity as Modified
        context.Employees.Update(employee);

        // Save changes to the database
        context.SaveChanges();

        // Delete an employee
        var employeeToDelete = context.Employees.Find(2);
        context.Employees.Remove(employeeToDelete);
        context.SaveChanges();

        // Query for a specific set of employees
        var engineeringEmployees = context.Employees.Where(e => e.Department == "Engineering").ToList();
    }
}
```
<br>

## 6. Can you describe the concept of _migrations_ in _EF_?

**Entity Framework** (EF) **Migrations** streamline the management of database schema changes, offering an automated approach to keep the schema in sync with your model.

### Why Use Migrations

  - **Schema Evolution**: Migrations facilitate seamless updates to the database schema as the underlying model evolves.

  - **Collaboration**: By keeping schema changes as code, the entire development team can track, review, and apply them using source control management.

  - **Automation**: Migrating databases can be a near-automated task in environments such as build pipelines or during application upgrades.

  - **Reproducibility**: With each schema change being versioned, rollback and forward/backward compatibility becomes clearer and more manageable.

  - **Multistep Migrations**: Complicated updates that involve several smaller changes can be broken down into sequential migrations, ensuring that at each intermediate step, the database remains in a consistent state.

  - **Code Paradigm**: Developers can remain in the code-first perspective, designing models and letting EF take care of the database details.

### Key Concepts

- **Migration**: A script or set of instructions that transforms the database from one state to another.

- **Migration History Table**: A system table in the database that stores the chronology of applied migrations. This allows EF to determine the current state of the database and what migrations, if any, are pending.

- **Migration Configuration**: Migrations can be tuned and customized through a dedicated configuration class.

- **Model Snapshot**: Migrations are backed by a "snapshot" of the model at each evolutionary stage, enabling EF to compare the existing database with the model and generate required scripts.

### The Workflow

Entity Framework organizes the migration workflow into distinct steps:

1. **Create Initial Migration**: This step is about establishing the starting point—creating a migration that encompasses the existing model.

2. **Code and Model Changes**: Whenever you make changes to your code-first model, you sync these changes using Migrations. This typically involves a few commands provided by your development environment, such as **add-migration** in Visual Studio or **dotnet ef migrations add** via the .NET CLI.

3. **Database Update**: Once the migration script is generated, you apply the changes to the database. This is done using **update-database** in Visual Studio, **dotnet ef database update** in the .NET CLI, or programmatically in your code.

   For example, in C#, you would call Context.Database.Migrate() during the database initialization process.

### Best Practices

- **Consistent Naming**: Maintain a unified and meaningful naming convention for your migrations to ensure clarity, especially in team settings.
  
- **Continuous Migration**: Invest in a reflexive approach where, as part of the development pipeline, databases are continuously and seamlessly updated.

- **Source Control**: Migrations are essentially code that should be tracked, versioned, and deployed along with your application code.

- **Validation**: Before applying a migration, consider running automated tests to ensure systemic stability.

- **Periodic Cleanup**: Over time, your project may accumulate numerous migrations. Consolidate or remove obsolete ones to keep the codebase manageable.

### Code Example: Data Annotations for Migrations

Here is the C# code:

```csharp
public class Customer
{
    [Key] // Identifies the primary key of the entity.
    public int CustomerId { get; set; }
    
    [Required] // Specifies that a value is required for the property.
    public string Name { get; set; }
}
```
<br>

## 7. How does _EF_ implement _code-first_, _database-first_, and _model-first_ approaches?

Let's explore how **Entity Framework (EF)** implements three primary design methodologies: **Code-First**, **Database-First**, and **Model-First**.

### Database-First

In a **Database-First** approach, the data model *initially* resides in an existing database. EF then generates the corresponding model and code.

#### Process Steps

1. **Generate Object Context**: Tools such as `Entity Framework Designer` in Visual Studio or CLI-based `ef` commands build an **Object Context** derived from the database schema. This context forms a bridge between database entities and the application's domain model.

2. **Create Entity Classes**: EF creates entity classes corresponding to database tables, along with necessary properties and relationships. These classes map closely to the existing schema.

3. **Compile and Use**: Developers integrate the generated classes with their applications and invoke the database via these entities.

#### Benefits

- Quick adaptability to existing databases for legacy systems or applications with strict schema requirements.
- Automatic code generation streamlines the development process.

#### Limitations

- **Extra Work for Complex Changes**: Efficiently handling complex updates to the database structure can be challenging.

### Code-First

In a **Code-First** approach, developers define the data model using accessible classes. This model acts as the primary resource for schema creation and database persistence.

#### Process Steps

1. **Write POCO Classes**: Developers craft plain, POCO (Plain Old CLR Object) classes representing the business entities of the application. Annotations or a Fluent API configuration guide EF in understanding how these classes map to the database.

2. **Create Data Context**: A context class, derived from `DbContext`, serves as an access point to the database and a tracker for entity changes.

3. **Refine Model as Necessary**: Refinement of the model is consistent with the application's evolving requirements. Migrations, for instance, allow for the sequential remodeling of the database.

4. **Database Generation/Application Launch**: The database schema is either generated or updated when the application runs. This can be achieved using migrations or by explicitly invoking the database initialize method in code.

#### Benefits

- **Flexible Model Evolution**: The data model adapts directly to the evolving needs of the application.
- **Coherent Code and Database Schema Maintenance**: Simplified schema management from the codebase, supporting version control and collaborative development.

#### Limitations

- **Potential Synchronization Issues**: Developers must ensure that the application's classes and the underlying database schema stay aligned.

### Model-First

In the **Model-First** approach, developers define the conceptual model using a designer tool. This high-level model is refined, and database schema and code are then generated accordingly.

#### Process Steps

1. **Design Conceptual Model**: An EF Designer or Model Builder tool such as Visual Studio's **Entity Data Model Designer** is used to define a conceptual model. This encompasses entities, relationships, and other data-related features.

2. **Generate Database from EF Model**: After establishing the conceptual model, configuration tools in Visual Studio or `Entity Data Model Wizard` generate the underlying database schema.

3. **Code Generation**: EF also facilitates the automatic creation of classes that mirror the model, using tools like **T4 Templates** or designers.

#### Benefits

- **Visual Modeling First**: Offers an intuitive approach for initial model creation, often beneficial for understanding business requirements.
- **Unified Development Environment**: Developers can manage the complete process, from model design to code generation, within a single tool such as Visual Studio.

#### Limitations

**Model-Code Mismatch Potential**: Designs may not perfectly match generated code, and manual code changes can impact future model-based code generation.

The Database-First approach is particularly helpful for legacy databases and was widely used in earlier EF versions. In contrast, Code-First has become a preferred choice for new projects due to its better control over data models and support for model evolution through features like Migrations. Visual Studio's designer is suitable for projects that require rapid visualization, but its use is limited in continuous integration and delivery scenarios.
<br>

## 8. What is the role of the _EdmGen_ tool in _EF_?

The `_EdmGen` tool plays a vital role in **Entity Framework**, especially during design and build phases.

### What is \_EdmGen\_?

The EdmGen tool is a command-line utility that comes with Entity Framework. Its main purpose is to generate storage model from the conceptual model and mapping files.

### Key Functions

- **Generates** three types of files: .csdl (Conceptual Schema Definition Language), .ssdl (Storage Schema Definition Language), and .msl (Mapping Schema Language). These represent the three components of the Entity Data Model (EDM): conceptual model, storage model, and mapping files.
- **Helps** in database-first, code-first, and model-first development paradigms.

### Code Example: Running \_EdmGen\_

The following command generates .csdl, .ssdl, and .msl files and specifies the output directory:

```bash
EdmGen /mode:GenerateArtifact 
       /outDir:"C:\MyProject" 
       /nameSpace:MyApp
       /project:MyModelProject
       /language:CSharp
       /connectionString:"metadata=res://*/MyModel.csdl|res://*/MyModel.ssdl|res://*/MyModel.msl; providerName=System.Data.SqlClient;provider connection string='data source=.;initial catalog=MyDatabase;integrated security=True;multipleactiveresultsets=True;App=EntityFramework'" 
       /entityContainer:MyEntities
```

### From Command Line to Visual Studio

Over time, the use of the **EdmGen** tool has diminished, thanks to the enhanced integration of Entity Framework in Visual Studio, especially for database-first and model-first workflows. Visual Studio's "Update Model Wizard" or the package manager console's scaffold commands are commonly used in modern EF-based projects, reducing the direct need for EdmGen.
<br>

## 9. Can you describe the _Entity Data Model (EDM)_ in _EF_?

The **Entity Data Model** (EDM) is a multi-layered, conceptual framework that facilitates data management in Entity Framework.

### Core Components

1. **Entity Type**: Represents an object or data, such as a person or product. An entity type corresponds to a table in a database. 

   Example:
   ```C#
   public class Product
   {
       public int ProductID { get; set; }
       public string Name { get; set; }
       public decimal Price { get; set; }
   }
   ```

2. **Entity Set**: This is a collection or group of entity instances of a specific entity type. Internally, they map to tables in the database. 

   Example:
   ```C#
   public DbSet<Product> Products { get; set; }
   ```

3. **Association**: Specifies the relationship between two or more entity types. For instance, in a one-to-many relationship between `Order` and `Product`, an `Order` can have multiple `Products`. 

   Example:
   ```C#
   public class Order
   {
       [Key]
       public int OrderID { get; set; }
       public ICollection<Product> Products { get; set; }
   }
   ```

4. **Complex Type**: Represents an object with a composite state, possibly comprising various related entities. It doesn't have a key attribute and cannot exist independently.

   Example:
   ```C#
   [ComplexType]
   public class Address
   {
       public string Street { get; set; }
       public string City { get; set; }
       public string ZipCode { get; set; }
   }
   ```

5. **Association Set**: This corresponds to a group of related entities. It's primarily used for tracking relationships in the database.

6. **Function Import**: Maps stored procedures or user-defined functions in the database to corresponding methods in the context.

7. **Scalar Property**: Represents simple, individual properties of an entity type. 

8. **Navigation Property**: Enables navigation from one end of an association to another. 

9. **Entity Container**: This acts as a container for all the objects used within the model, like entity types, complex types, and entity sets.

10. **Inheritance**: Allows for object-oriented concepts such as **inheritance** and **polymorphism** in the model. You can define **base** and **derived** entity types. When you create a hierarchy, EF organizes the entities in a database table to mirror this relationship.

11. **Child Entity Type**: When using the TPH strategy, child types come into play. They represent types that inherit from a parent entity type and exist in a TPH configuration.

### Database-First Design

The EDM provides for **Database-First** design, where the EDM and entities are generated from an existing database schema. This method offers a parallel advantage where when the database schema undergoes changes, the modifications are mirrored in the model.

### Code-First Approach

With the Code-First approach, which is commonly known and preferred by developers because of the flexibility and ease of sharing with the teams, the EDM is derived from the code representation of the data model. Developers write classes to represent the model and establish relationships within them, and EF generates the database based on these classes.

### Model-First Strategy

The **Model-First** strategy allows developers to create the EDM graphically using designer tools such as Visual Studio's EDM Designer. This method is especially favored where intricate models are in play.

**What's noteworthy** is EDM's capability to cater to multiple storage schemas. Whether the source is a relevant database, an XML document, or various data sources, **EDM** is versatile and adaptable.
<br>

## 10. How does _lazy loading_ work in _EF_?

**Lazy loading** allows related objects to be fetched from the database only when they are accessed. This reduces the initial data load, making the system more efficient.

### Key Components

- **Proxy Generation**: When a navigation property is virtual, Entity Framework generates a dynamic proxy at runtime. 

- **Interception Mechanism**: Access to the navigation property triggers a database query through a proxy instance. The process is monitored by EF to maintain data consistency.

- **Underlying Context Connection**: The context maintains a virtual link to related entities. Actual data fetching occurs once there's a navigation property access for the first time within a context session.

### Performance Considerations

While lazy loading can enhance efficiency, it may also introduce performance overheads. If misused, it can result in the N+1 problem, where many additional queries are executed, leading to performance degradation.

Also, when used in disconnected scenarios, such as within a web application, late queries can cause unexpected issues.

### Code Example: Sales Context

Here is the C\# code:

```csharp

public class SalesContext : DbContext
{
    public DbSet<Order> Orders { get; set; }
    public DbSet<Customer> Customers { get; set; }
    public DbSet<Product> Products { get; set; }
}

public class Order
{
    public int OrderId { get; set; }
    public int CustomerId { get; set; }
    public virtual Customer Customer { get; set; }
    public virtual ICollection<Product> Products { get; set; }
}

public class Customer
{
    public int CustomerId { get; set; }
    public string Name { get; set; }
    public virtual ICollection<Order> Orders { get; set; }
}

public class Product
{
    public int ProductId { get; set; }
    public string Name { get; set; }
    public decimal Price { get; set; }
    public virtual ICollection<Order> Orders { get; set; }
}
```
<br>

## 11. How do you install or upgrade _Entity Framework_ in a _.NET_ project?

To **install** or **upgrade** Entity Framework (EF) in your .NET project, you can use **NuGet Package** Manager in Visual Studio or the Command Line Interface (CLI).

### Visual Studio (VS): NuGet Package Manager

1. **Access Package Manager**: Go to `Tools` > `NuGet Package Manager` > `Manage NuGet Packages for Solution...`.
2. **Install**: In the `Browse` tab, look for `EntityFramework`, then click `Install`.
3. **Upgrade**: Navigate to the `Installed` tab, select `EntityFramework`, and choose `Update`.

### VS Code: NuGet Package Manager

1. **Install**: In the terminal, use `dotnet add package EntityFramework`.
2. **Upgrade**: Run `dotnet add package EntityFramework --version 6.x`.

### Command Line Interface (CLI)

1. **Install**: Run `dotnet add package EntityFramework`.
2. **Upgrade**: Specify the version using `dotnet add package EntityFramework --version 6.x`.

### Benefits of Multi-Level Flexibility

- **Error Handling**: NuGet provides feedback on potential errors as you type, reducing the likelihood of version conflicts or wrong selections.
- **Version Control**: You can specify exact versions, providing stability in your project, or opt for dynamic updates.
- **Efficiency**: Multiple install or upgrade tasks can be executed in one command, streamlining workflows.
<br>

## 12. What is the difference between _local_ and _global configuration_ in _EF_?

**Entity Framework** operates with both global and local configurations to handle the mapping between your data schema and the domain model.

### Global vs Local Configurations

- **Global Configuration**: Embodies the primary mapping logic between classes and database tables. Global configurations are implemented during **Model creation**.

- **Local Configuration**: Offers more granular control and at times, can override global setups. This happens during the **initializer's seeding phase**.

### Code First Vs Database First

- **Code First**: In the Migrations model, the `DbContext` offers an `OnModelCreating` method, which is the location for both global and local configurations.

- **Database First**: In this model, the `.edmx` file encompasses global definitions (the main `.edmx` file) and may have local definition files (model-specific `.edmx` files).
<br>

## 13. What is the purpose of the _Entity Framework connection string_?

The Entity Framework connection string, typically stored in a project's `app.config` or `web.config`,  is necessary to **establish a connection between the application and the database**.

### Key Elements

- **Data Source**: Essential for server location, and can be a literal source or a path to a file or database.
- **Initial Catalog**: Specifies the database to target at the start.
- **User ID** and **Password**: Required for an ID-assisted secure connection.
- **Integrated Security**: A true/false flag often employed with Windows authentication.

### Configuration Examples

- **Database File**: Ideal for simpler applications leveraging local storage.
- **Windows Security**: When paired with `Integrated Security=true`, uses Windows credentials.
- **Provide Both**: Acceptable when a precise database, ID, and password are necessary.

### Potential Issues

- **Hardcoded Connection Strings**: Resists configuration modifications or environment-specific adjustments.
- **Security Risks**: Publicly available IDs or passwords could compromise the database's safeguarding.

### Best Practices

- **Externalize Connection Strings**: Leverage app or web.config files to hold the strings. This externalization promotes maintainability and diminishes security threats.
- **Parameterized Constructs**: Use SQL parameters to fortify the link's integrity while thwarting potential assaults.
- **Secure Storage Strategies**: Tactics such as an encrypted configuration file or a safe data storeroom ensure enhanced security levels.
<br>

## 14. How do you switch between different _databases_ using _EF_?

While developing **ASP.NET** applications, you might need to **switch between different databases** in an **Entity Framework** (EF) context.

Here are two common approaches to accomplish this:

1. **Code-based Selection**: Perfect for instances where the choice of database is known at compile-time. You use an `app.config` or `web.config` to specify the database connection.

2. **Run-time Database Selection**: Ideal for scenarios where the database to be used can only be determined at runtime.

### Code-Based Selection

In environments where the database choice is known at compile-time, you can use **Conditional Compilation Directives** to select the appropriate EF context, using tools such as `#if DEBUG` to differentiate between, say, a development and a production environment.


  - **Code CSHarp**

  ```csharp
   #if DEBUG
       using (var db = new DevelopmentDbContext()) 
       {
          // Database logic for development environment
       }
   #else
       using (var db = new ProductionDbContext()) 
       {
          // Database logic for production environment
       }
   #endif
  ```

### Run-time Database Selection

 You can **set the EF context** at runtime, allowing the application to **dynamically switch** between different databases based on user inputs or other factors. This approach gives more flexibility and is often used in applications where the database might change at runtime.

  - **Code CSharp**

```csharp
  using System.Data.Entity;

  public class DbContextFactory
  {
      public DbContext GetDbContext(DatabaseType type)
      {
          switch(type)
          {
             case DatabaseType.Production:
                 return new ProductionDbContext();
             case DatabaseType.Staging:
                 return new StagingDbContext();
             default:
                 return new DevelopmentDbContext();
          }
      }
  }

  public enum DatabaseType
  {
      Production,
      Staging,
      Development
  }

  // Somewhere in your code...
  // var dbFactory = new DbContextFactory();
  // var db = dbFactory.GetDbContext(DatabaseType.Production);
```
<br>

## 15. Can you configure the _pluralization_ and _singularization conventions_ in _EF_?

**Entity Framework** allows developers to tailor singular-to-plural and plural-to-singular naming conventions using the **PluralizationService**.

### PluralizationService in EF

The PluralizationService uses a set of rules and applies them in reverse for singularization. For instance, "**dogs**" would be singularized to "**dog**".

EF's pluralization and singularization rules exist in the `System.Data.Entity.Design` namespace. Custom rules can be adopted by inheriting from the `PluralizationService` base class. Several approaches, such as modifying or replacing rules, are available for advanced customization.

### Using `System.Data.Entity.Design`

If you choose to work with `System.Data.Entity.Design`, you can apply customized conventions by accessing the singleton service **Default** and replacing or augmenting rules using standard methods. Taking "child" as an example, the method to "singularrize" it will act as:

```csharp
var singular = Default.PluralizationService.Singularize("child");
```

### Using NuGet Package: "System.Data.Entity.Design" for older EF versions

EF Core doesn't support the PluralizationService method natively, but for older versions, you can install the NuGet package **System.Data.Entity.Design**.

For EF Core, you can use **EF Core Power Tools**, which equips EF Core with pluralization support.

### Code Example: Singularizing "Children"

Here is the C# code:

```csharp
using System.Data.Entity.Design.PluralizationServices;
using System.Globalization;

var pluralService = PluralizationService.CreateService(CultureInfo.GetCultureInfo("en-us"));
var singularChild = pluralService.Singularize("children");
```

Upon running this code, **singularChild** will hold the value "**child**".
<br>



#### Explore all 100 answers here 👉 [Devinterview.io - Entity Framework](https://devinterview.io/questions/web-and-mobile-development/entity-framework-interview-questions)

<br>

<a href="https://devinterview.io/questions/web-and-mobile-development/">
<img src="https://firebasestorage.googleapis.com/v0/b/dev-stack-app.appspot.com/o/github-blog-img%2Fweb-and-mobile-development-github-img.jpg?alt=media&token=1b5eeecc-c9fb-49f5-9e03-50cf2e309555" alt="web-and-mobile-development" width="100%">
</a>
</p>

