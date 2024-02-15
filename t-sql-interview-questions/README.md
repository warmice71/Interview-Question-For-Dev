# 100 Essential T-SQL Interview Questions

<div>
<p align="center">
<a href="https://devinterview.io/questions/web-and-mobile-development/">
<img src="https://firebasestorage.googleapis.com/v0/b/dev-stack-app.appspot.com/o/github-blog-img%2Fweb-and-mobile-development-github-img.jpg?alt=media&token=1b5eeecc-c9fb-49f5-9e03-50cf2e309555" alt="web-and-mobile-development" width="100%">
</a>
</p>

#### You can also find all 100 answers here 👉 [Devinterview.io - T-SQL](https://devinterview.io/questions/web-and-mobile-development/t-sql-interview-questions)

<br>

## 1. What is _T-SQL_ and how is it different from standard _SQL_?

**Transact-SQL** (T-SQL) is an extension of **SQL** that's specific to Microsoft SQL Server. It includes functionalities such as procedural programming, local variables, and **exception handling** through `TRY...CATCH` blocks. These features are not found in standard SQL.

### Key T-SQL Features

1.  **Stored Procedures**: T-SQL supports server-side scripts, known as **stored procedures**, for better security, performance, and encapsulation.

2.  **User Defined Functions (UDFs)**: These custom, reusable functions can help in tasks not directly supported by built-in SQL functions.

3.  **Common Table Expressions (CTEs)**: With the `WITH` clause, T-SQL offers an efficient way to define temporary result sets.

4.  **Triggers**: **T-SQL** can be used to define **triggers** that automatically execute in response to certain database events.

5.  **Table Variables**: These are variable collections, especially useful for temporary data storage during complex queries.

6.  **Transaction Control**: T-SQL allows **finer-grained control** over transactions with commands like `BEGIN TRAN`, `ROLLBACK`, and `COMMIT`.

### Code Example: TRY-CATCH Block in T-SQL

Here is the T-SQL code:

```sql
BEGIN TRY
  -- Generate a divide by zero error intentionally
  DECLARE @num1 INT = 10, @num2 INT = 0;
  SELECT @num1 / @num2;
END TRY
BEGIN CATCH
  -- Provides details of the error
  PRINT 'Error Number: ' + CAST(ERROR_NUMBER() AS VARCHAR);
  PRINT 'Error Severity: ' + CAST(ERROR_SEVERITY() AS VARCHAR);
  PRINT 'Error State: ' + CAST(ERROR_STATE() AS VARCHAR);
  PRINT 'Error Line: ' + CAST(ERROR_LINE() AS VARCHAR);
  PRINT 'Error Message: ' + ERROR_MESSAGE();
END CATCH;
```
<br>

## 2. Explain the use of the _SELECT_ statement in _T-SQL_.

**SELECT** is the fundamental statement in SQL for retrieving data from databases.

### Key Components

- **SELECT**: Keyword to indicate data retrieval.
- **DISTINCT**: Optional keyword to remove duplicate records.
- **FROM**: Keyword to specify data source (table or view).
- **WHERE**: Optional keyword for setting conditions.
- **GROUP BY**: Optional keyword for grouping data.
- **HAVING**: Optional keyword for filtering grouped data.
- **ORDER BY**: Optional keyword for sorting data.
- **TOP (or OFFSET-FETCH)**: Optional keyword(s) to limit the number of rows returned.

### SELECT Query Structure

Here is the structure of the _SELECT_ statement:

### Sample Query

Here is the T-SQL code:

```sql
SELECT 
    EmployeeID, 
    FirstName, 
    LastName
FROM 
    Employees
WHERE 
    Department = 'IT'
ORDER BY 
    HireDate DESC;
```

### Commonly Used Components in SELECT Statements

#### WHERE

- **Purpose**: Filters records based on one or more conditions.
- **Examples**:
    - `WHERE Age > 30`: Selects employees older than 30.
    - `WHERE JoinDate >= '2021-01-01'`: Selects employees who joined after January 1, 2021.

#### GROUP BY and HAVING

- **Purpose**: Groups records based on the specified column(s). HAVING acts as a filter after grouping.
- **Examples**:
    - `GROUP BY Department`: Groups employees based on their departments.
    - `GROUP BY Department HAVING COUNT(*) > 10`: Groups departments with more than 10 employees.

#### ORDER BY

- **Purpose**: Sorts records based on the specified column(s).
- **Examples**:
    - `ORDER BY Salary DESC`: Sorts employees in descending order of salary.
    - `ORDER BY HireDate ASC, Salary DESC`: Sorts employees ascending by hiring date and descending by salary.

#### DISTINCT

- **Purpose**: Selects unique records.
- **Example**:
    - `SELECT DISTINCT Department FROM Employees`: Retrieves distinct department names where employees work.

#### TOP, OFFSET, and FETCH

- **Purpose**: Limits the number of rows returned. Commonly used for pagination.
- **Examples**:
    - `SELECT TOP 5 * FROM Orders`: Retrieves the first 5 orders.
    - `SELECT * FROM Orders ORDER BY OrderDate OFFSET 10 ROWS FETCH NEXT 5 ROWS ONLY`: Retrieves 5 orders starting from the 11th (ordered by date).
<br>

## 3. What are the basic components of a _T-SQL_ query?

A **T-SQL** Query consists of the following components:

1. **SELECT Statement**: Selects columns or computed values.
   ```sql
   SELECT Name, Age FROM Users
   ```

2. **FROM Clause**: Specifies the data source.
   ```sql
   FROM Users
   ```

3. **WHERE Clause**: Filters rows based on a condition.
   ```sql
   WHERE Age > 18
   ```

4. **GROUP BY Clause**: Groups rows based on common values.
   ```sql
   GROUP BY Country
   ```

5. **HAVING Clause**: Applies a filter on grouped data.
   ```sql
   HAVING SUM(Sales) > 10000
   ```

6. **ORDER BY Clause**: Sorts the result set.
   ```sql
   ORDER BY Age DESC
   ```

7. **Set Operators**: Enables combining results of two or more SELECT statements. The most common set operators are `UNION`, `INTERSECT`, and `EXCEPT`.

   **Example**:  
   ```sql
   SELECT Name FROM Students
   UNION
   SELECT Name FROM Teachers
   ```

8. **JOINs:** Constructs a relationship between tables, combining data points. Common joins are `INNER JOIN`, `LEFT (OUTER) JOIN`, and `RIGHT (OUTER) JOIN`.
   **Example**:  
   ```sql
   SELECT Orders.OrderID, Customers.CustomerName
   FROM Orders
   INNER JOIN Customers ON Orders.CustomerID = Customers.CustomerID
   ```

9. **Subqueries**: A query nested within another query.
   **Example**:  
   ```sql
   SELECT Name  
   FROM Users  
   WHERE ID IN  
   (SELECT UserID  
   FROM UserRoles  
   WHERE RoleID = 1)  
   ```

10. **Common Table Expressions (CTE)**: A temporary result set that can be referenced multiple times in a query.
    **Example**:  
    ```sql
    WITH cteProducts AS (
        SELECT ProductID, ProductName, UnitPrice
        FROM Products
        WHERE CategoryID = (SELECT CategoryID FROM Categories WHERE CategoryName = 'Beverages')
    )
    SELECT ProductName, UnitPrice
    FROM cteProducts
    WHERE UnitPrice > 20
    ```

11. **Window Functions**: Perform calculations across a set of table rows.
    **Example**:  
    ```sql
    SELECT orderNumber, orderDate,  
       ROW_NUMBER() OVER(ORDER BY orderDate) AS 'RowNumber'  
       FROM orders
    ```

12. **Aggregation Functions**: Functions that operate on a set of input values and return a single value. Common functions include `SUM`, `COUNT`, `AVG`, `MIN`, and `MAX`.
    **Example**:  
    ```sql
    SELECT Country, COUNT(*) AS 'UserCount'
    FROM Users
    GROUP BY Country
    ```

13. **Pivoting**: Transforms data from rows to columns. Common functions used are `PIVOT` and `UNPIVOT`.

14. **Unions**: Combines the results of two or more SELECT statements into one result.

15. **CASE Statement**: Offers logical evaluations and assigns values based on conditions.
    **Example**:  
    ```sql
    SELECT 
        ProductName,
        UnitPrice,
        CASE
            WHEN UnitPrice < 10 THEN 'Inexpensive'
            WHEN UnitPrice >= 10 AND UnitPrice < 50 THEN 'Moderate'
            ELSE 'Expensive'
        END AS 'PriceCategory'
    FROM Products
    ```
<br>

## 4. How do you write a _T-SQL_ query to _filter_ data using the _WHERE_ clause?

**Transact-SQL** (T-SQL) offers the **WHERE clause** for filtering data from tables prior to displaying or manipulating them.

### Syntax

```sql
SELECT column1, column2,...
FROM table_name
WHERE condition;
```

Here, `condition` specifies the filtration criterion, such as `age > 25` or `name = 'John'`.

If multiple conditions are required, use logical **AND** or **OR** operators and parentheses for clear precedence.

### Examples

#### Basic Usage

1. Get names of students older than 25:
   
    ```sql
    SELECT name
    FROM Students
    WHERE age > 25;
    ```

2. Retrieve titles of all IT books:

    ```sql
    SELECT title
    FROM Books
    WHERE category = 'IT';
    ```

#### Logical Operators

1. Find students older than 25 who have not completed their degrees:

    ```sql
    SELECT name
    FROM Students
    WHERE age > 25 AND degreeCompletionYear IS NULL;
    ```

2. Obtain all data of products either in 'Electronics' or 'Mobile' category:

    ```sql
    SELECT *
    FROM Products
    WHERE category = 'Electronics' OR category = 'Mobile';
    ```

#### Using `IN` Operator

1. Retrieve records of students from 'History' and 'Biology' courses:

    ```sql
    SELECT *
    FROM Students
    WHERE course IN ('History', 'Biology');
    ```

#### Applying `BETWEEN` Operator

1. Retrieve books published between the year 2010 and 2020:

    ```sql
    SELECT *
    FROM Books
    WHERE publishYear BETWEEN 2010 AND 2020;
    ```

#### Using `LIKE` for Pattern Matching

1. Find customers with phone numbers starting with area code '123':

    ```sql
    SELECT *
    FROM Customers
    WHERE phone LIKE '123%';
    ```

2. Locate users whose email addresses end with '.com':

    ```sql
    SELECT *
    FROM Users
    WHERE email LIKE '%.com';
    ```

#### Negating Conditions with the `NOT` Keyword

1. Get details of books not published by 'Penguin':

    ```sql
    SELECT *
    FROM Books
    WHERE publisher NOT LIKE 'Penguin%';
    ```

#### Filtering Null Values

1. Retrieve all students who have not yet determined their completion year:

    ```sql
    SELECT *
    FROM Students
    WHERE degreeCompletionYear IS NULL;
    ```

2. Obtain names of all employees without assigned managers:

    ```sql
    SELECT name
    FROM Employees
    WHERE managerID IS NULL;
    ```

#### Using Complex Conditions with Parentheses

1. Display books that are either in the 'Fiction' category or published after 2015:

    ```sql
    SELECT *
    FROM Books
    WHERE category = 'Fiction' OR publishYear > 2015;
    ```
<br>

## 5. Describe how to _sort_ data using the _ORDER BY_ clause in _T-SQL_.

**Order By** in T-SQL arranges query results according to specified criteria, such as unique identifiers or columns.

### Basic Order By Operations

- Sort by ID:
  ```sql
  SELECT * FROM Employees ORDER BY EmployeeID;
  ```

- Sort by multiple criteria:
  ```sql
  SELECT * FROM Users ORDER BY LastName, FirstName, BirthDate;
  ```

- Order By Position Descriptor (Ordinal Number):
  ```sql
  SELECT TOP 5 WITH TIES * FROM Users ORDER BY 5;
  ```

### Directional Sorting

- Ascending (default):
  ```sql
  SELECT * FROM Orders ORDER BY OrderID ASC;
  -- or shorthand:
  SELECT * FROM Orders ORDER BY OrderID;
  ```

- Descending:
  ```sql
  SELECT * FROM Products ORDER BY Price DESC;
  ```

### NULL Placement

- **First**: Nulls first, then non-nulls.
  ```sql
  SELECT * FROM Students ORDER BY GPA DESC NULLS FIRST;
  ```

- **Last**: Nulls last, after non-nulls.
  ```sql
  SELECT * FROM Products ORDER BY ExpiryDate ASC NULLS LAST;
  ```

### Ordering on Expressions

- Calculate criteria for sorting:
  ```sql
  SELECT Price, Discount, (Price - Discount)
  AS SalePrice FROM Products ORDER BY (Price - Discount);
  ```

### Advanced Techniques

- Specific Character Set Order:
  ```sql
  SELECT * FROM Players ORDER BY DisplayName
  COLLATE Latin1_General_BIN;
  ```

- Excluding Sort Copies:
  ```sql
  SELECT DISTINCT City FROM Addresses ORDER BY City;
  ```
<br>

## 6. What are _JOINs_ in _T-SQL_ and can you explain the different types?

**Joins** in T-SQL are critical for combining data from multiple tables. The different types of joins offer flexibility in data retrieval.

### Common Join Types

1. **Inner Join**: Retrieves records that have matching values in both tables.

2. **Left (Outer) Join**: Retrieves all records from the left table, and the matched records from the right table. If no match is found, NULL is returned from the right side.

3. **Right (Outer) Join**: Similar to the Left Join but retrieves all records from the right table and matched records from the left table. Unmatched records from the left table return NULL.

4. **Full (Outer) Join**: Retrieves all records when there is a match in either the left or right table. If there is no match, NULL is returned for the opposite side.

5. **Cross Join**: Produces the Cartesian product of two tables, i.e., each row from the first table combined with each row from the second table. This join type doesn't require any explicit join conditions.

6. **Self Join**: This is when a table is joined to itself. It's useful when a table has a 'parent' and 'child' relationship, such as an employee's hierarchical structure.

7. **Anti Join**: This type of join is similar to a LEFT JOIN, but it returns only the rows where there is no match between the tables.

8. **Semi Join**: It's a special type of join that can be used with EXISTS and IN. This type of join is usually optimized by the query processor to improve performance.

9. **Equi Join**:  This is similar to the Inner Join and joins tables based on a specific column that has equivalent values in both tables.

10. **Non-Equi Join**: Differs from the Equi Join because the join condition doesn't use the equality operator.

### SQL Queries

Here are some example SQL queries to help understand the different join types:

#### Inner Join

```sql
SELECT Orders.OrderID, Customers.CustomerName
FROM Orders
INNER JOIN Customers ON Orders.CustomerID = Customers.CustomerID;
```

#### Outer Joins

- **Left Join**
```sql
SELECT Employees.LastName, Employees.DepartmentID, Departments.DepartmentName
FROM Employees
LEFT JOIN Departments ON Employees.DepartmentID = Departments.DepartmentID;
```

- **Right Join**
```sql
SELECT Orders.OrderID, Customers.CustomerName
FROM Orders
RIGHT JOIN Customers ON Orders.CustomerID = Customers.CustomerID;
```

- **Full Join**
```sql
SELECT A.Column1, B.Column 2
FROM TableA A
FULL JOIN TableB B ON A.Column1 = B.Column2;
```

- **Anti Join**
```sql
SELECT Customers.CustomerName
FROM Customers
WHERE NOT EXISTS (SELECT 1 FROM Orders WHERE Customers.CustomerID = Orders.CustomerID);
```

- **Semi Join**
```sql
SELECT LastName, FirstName, Title
FROM Employees
WHERE EmployeeID IN (SELECT ManagerID FROM Employees);
```
<br>

## 7. How do you implement _paging_ in _T-SQL_ queries?

In SQL Server, **paging** results involves using the `ORDER BY` and `OFFSET-FETCH` clauses. `OFFSET` specifies the number of rows to skip, and `FETCH` limits the number of rows to return.

### Query Syntax

```sql
SELECT 
    columns 
FROM 
    table_name
ORDER BY 
    ordering_column
    [ASC | DESC] -- Optional
OFFSET 
    n ROWS
FETCH NEXT 
    m ROWS ONLY;
```

- **n**: The number of initial rows to skip.
- **m**: The number of subsequent rows to return.

### Code Example: Simple Paging

The below query will return rows 6 to 10 out of 15 items.

```sql
SELECT 
    *
FROM 
    user_profiles
ORDER BY 
    sign_up_date
OFFSET 
    5 ROWS
FETCH NEXT 
    5 ROWS ONLY;
```

### Using Variable and OFFSET

In scenarios needing **dynamic paging**, `OFFSET...FETCH` and `ORDER BY` must also be dynamic. **Common Table Expressions (CTE)**, together with `ROW_NUMBER()`, facilitate dynamic sorting and limiting.

### Code Example: Dynamic Paging

The following code sets up dynamic paging to fetch the second page with 10 rows per page.

```sql
DECLARE @PageSize INT = 10;
DECLARE @PageNumber INT = 2;

WITH user_cte AS 
(
    SELECT
        *,
        RowNum = ROW_NUMBER() OVER (ORDER BY sign_up_date)
    FROM 
        user_profiles
)
SELECT 
    *
FROM 
    user_cte
WHERE 
    RowNum > (@PageNumber - 1) * @PageSize 
    AND RowNum <= @PageNumber * @PageSize;
```

In this example, `ROW_NUMBER() OVER (ORDER BY sign_up_date)` assigns a row number based on the defined order, and the CTE `user_cte` helps filter rows within **dynamic boundaries**.
<br>

## 8. What is the difference between _UNION_ and _UNION ALL_?

Both **UNION** and **UNION ALL** are used to combine the results of two or more SELECT statements, albeit with key distinctions.

### Core Principle

- **UNION**: Performs a set-union, which eliminates any duplicate rows across the SELECT statements.
- **UNION ALL**: **ALL** retrieves all rows from each SELECT statement, including duplicates, and combines them into the result set.

### Pros and Cons

- **UNION** requires extra processing to identify and remove duplicates, making it slightly slower. However, it's often more suitable for data consolidation tasks.
 
- **UNION ALL** is faster since it doesn't have to perform any duplicate checks. Use it when you want to preserve all records, even those that might be duplicates.

### Query Examples

Consider the following tables:

```sql
CREATE TABLE Employees (
    ID INT,
    Name NVARCHAR(100)
);


CREATE TABLE Customers (
    ID INT,
    Name NVARCHAR(100)
);

INSERT INTO Employees (ID, Name) VALUES (1, 'John'), (2, 'Mary'), (3, 'John');
INSERT INTO Customers (ID, Name) VALUES (4, 'Peter'), (1, 'John');
```

#### Union All

- **Syntax**:

```sql
SELECT * FROM Employees
UNION ALL
SELECT * FROM Customers;
```

- **Result**: 

| ID | Name   |
|----|--------|
| 1  | John   |
| 2  | Mary   |
| 3  | John   |
| 4  | Peter  |
| 1  | John   |


#### Union

- **Syntax**:

```sql
SELECT * FROM Employees
UNION
SELECT * FROM Customers;
```

- **Result**: 

| ID | Name   |
|----|--------|
| 1  | John   |
| 2  | Mary   |
| 4  | Peter  |
<br>

## 9. How are _aliases_ used in _T-SQL_ queries?

**Aliases** in T-SQL are temporary labels for tables, views, or columns. They streamline queries and improve readability. They are applied using the `AS` keyword $optionally$ and can be declared for tables/views $table aliases$ and columns $column aliases$.

### Table & View Aliases

With **table aliases**, you simplify syntax, especially for self-joins and subqueries. Use them when dealing with complex and large datasets to keep queries clear and compact.

### Column Aliases

**Column aliases** come in handy for customizing column headings in result sets or for using intermediate calculations.
Here is the `SELECT` query for both cases.

#### Column Aliases
```sql
SELECT OrderID AS OrderNumber,
       Quantity * UnitPrice AS TotalCost
  FROM OrderDetails;
```

#### Table Aliases
```sql
SELECT c.CustomerID, o.OrderID
  FROM Customers AS c
       JOIN Orders AS o ON c.CustomerID = o.CustomerID;
```
### Code Example

Here is an example of a more complex query that uses table and column aliases for clarity:

#### SQL Query

```sql
SELECT e1.LastName AS ManagerLastName, 
       e1.FirstName AS ManagerFirstName, 
       e2.LastName AS EmployeeLastName, 
       e2.FirstName AS EmployeeFirstName
  FROM Employees e1
       JOIN Employees e2 ON e1.EmployeeID = e2.ReportsTo;
```
<br>

## 10. Can you explain the _GROUP BY_ and _HAVING_ clauses in _T-SQL_?

**GROUP BY** and **HAVING** work in tandem to filter data after grouping has taken place, as well as on aggregated data.

### Key Differences

- **Grouping**: `GROUP BY` arranges data into groups based on common column values.
- **Filtering**: `HAVING` filters grouped data based on specific conditions, much like `WHERE` does for ungrouped data.
- **Aggregation**: Since `HAVING` operates on grouped and aggregated data, it's often used in conjunction with aggregate functions like `COUNT`, `SUM`, etc.

### Common Scenarios

- **Aggregate Filtering**: Tasks that require a group-level condition based on aggregated values. For example, to identify `SUM(Sales)` values greater than 100.
  
- **Post-Aggregation Filtering**: Restrictions on grouped data that can only be determined after applying aggregate functions.

### Code Example: Using GROUP BY and HAVING

Here is the T-SQL code:

```sql
SELECT
    OrderDate,
    COUNT(*) AS OrderCount,
    SUM(OrderTotal) AS TotalSales
FROM
    Orders
GROUP BY
    OrderDate
HAVING
    COUNT(*) > 1
ORDER BY
    TotalSales DESC;
```

In this example, we're trying to retrieve all `OrderDates` with more than one order and their corresponding `TotalSales`. As a reminder, `HAVING` limits results based on group-level criteria, which is why the `COUNT(*)` of orders is used here.
<br>

## 11. What are the _T-SQL commands_ for _inserting_, _updating_, and _deleting_ data?

Let's look at the essential **T-SQL commands** for data **insertion**, **updating**, and **deletion**.

### T-SQL Data Modification Commands

- **INSERT**: Adds new rows to a table.
- **UPDATE**: Edits existing rows in a table based on specified conditions.
- **DELETE**: Removes rows from a table based on certain conditions.


### Features

- **Integrity Constraints**: Such as **Primary Key**, **Foreign Key**, and **Unique Key** are used for data validation and maintenance.
- **Transaction Management**: With `BEGIN TRANSACTION`, `COMMIT`, and `ROLLBACK`, T-SQL helps ensure the ACID (Atomicity, Consistency, Isolation, Durability) properties are met.
- **Logging and Data Recovery**: Changes are logged, allowing for data recovery in case of accidents.

### Code Example: INSERT

Use `INSERT INTO` to add data to a table. If you've defined an auto-incrementing primary key, you don't need to specify its value.

For tables without identity columns:

```sql
INSERT INTO TableName (Column1, Column2, ...)
VALUES (Value1, Value2, ...);
```

For tables with identity columns:

```sql
SET IDENTITY_INSERT TableName ON; -- Turn identity insert on
INSERT INTO TableName (ID, Column1, Column2, ...)
VALUES (NewID, Value1, Value2, ...);
SET IDENTITY_INSERT TableName OFF; -- Turn identity insert off
```


### Code Example: UPDATE

The `UPDATE` statement allows you to modify existing rows that match specific criteria.

```sql
UPDATE TableName
SET Column1 = NewValue1, Column2 = NewValue2, ...
WHERE Condition;
```

For instance:

```sql
UPDATE Employees
SET Salary = 50000
WHERE Department = 'Marketing';
```

### Code Example: DELETE

Use `DELETE` to remove rows from a table based on specified conditions.

For instance, to delete all employees who joined before 2015:

```sql
DELETE FROM Employees
WHERE JoinDate < '2015-01-01';
```
<br>

## 12. How do you perform a _conditional update_ in _T-SQL_?

**Conditional updates** in T-SQL leverage the `UPDATE` statement and `WHERE` clause to modify existing data under specific conditions.

### Method: Using `WHERE` Clause for Conditional Updates

The `WHERE` clause restricts updates based on specified conditions.

Here is the SQL Query:

```sql
UPDATE table_name
SET column1 = value1, column2 = value2, ...
WHERE condition;
```

### Example: Updating Employee Salaries

Let's assume the task is to increase salaries by **10%** for employees **older than 30**:

The corresponding SQL query is:

```sql
UPDATE Employees
SET Salary = Salary * 1.1  -- increasing salary by 10%
WHERE Age > 30;
```
<br>

## 13. What is the purpose of the _COALESCE_ function?

The **COALESCE** function is a versatile SQL tool that serves several key functions, all tailored around the concept of handling **NULL** values in a query. 

### Fundamental Role

**COALESCE** allows you to select values from a set of arguments in the order they are specified, until a non-NULL value is encountered. This makes it particularly useful in scenarios such as data transformation, default value selection, and in filter criteria.

### Common Use-Cases

- **Default Value Specification**: If a column can have a null value and you want to display a non-null value in result set, you can provide a default via COALESCE.

  ```sql
  SELECT COALESCE(NullableField, 'Default') AS FieldWithDefault
  FROM YourTable;
  ```

- **Filtering**: Using **COALESCE** in your WHERE clause can help in a variety of scenarios, like when dealing with nullable parameters.

  ```sql
  SELECT *
  FROM YourTable
  WHERE Column1 = COALESCE(@Input1, Column1) 
    AND Column2 = COALESCE(@Input2, Column2);   
  ```
  
- **Data Transformation**: You can use **COALESCE** to map **NULL** values to non-null values during result set retrieval.

  ```sql
  SELECT COALESCE(SalesRep, 'No Sales Rep Assigned') AS SalesRepDisplay
  FROM Sales;
  ```

### Alternative Approaches

While **COALESCE** is the most direct way to handle **NULL**s and is highly portable across SQL implementations, there are other methods that can achieve the same results.

- **ISNULL**: A SQL Server-specific function that serves the same purpose as **COALESCE**, but is limited to only handling two parameters.
- **NVL**: Common in Oracle SQL, this function serves the same role as **COALESCE** but is more limited in terms of syntax and features.
<br>

## 14. Explain how to _convert data types_ in _T-SQL_.

Data type conversion in T-SQL can take place explicitly or implicitly.

### Implicit Conversion

SQL Server performs implicit data type conversions when it can reasonably infer the target data type. For instance, in the expression `3 + 4.5`, the integer `3` gets converted to a `numeric` type to allow for the addition.

### Special Cases of Implicit Conversion

-   **Character Types to Numeric Values**: Conversions from character types to numeric ones can be implicitly handled in specific scenarios. For example, a query like `SELECT '10' + 5` treats the character `'10'` as a numeric `10`.

-   **Date and Time Types**: Implicit conversions work among different date and time types, too. If you add an `int` to a `datetime` type, SQL Server takes the `int` as the number of days to add to the date.

-   **String to Date and Time**: T-SQL can convert string literals representating dates and times to their respective data types. For instance, '10 JAN 2018' or '2018-01-10' will be converted to a `datetime` type.

### Explicit Data Type Conversion

You can assert control over data type conversions with explicit casting and conversion functions. 

#### CAST and CONVERT Functions

-   **CAST**: Universally supported, its syntax is `CAST(expression AS data_type)`.
-   **CONVERT**: Offers additional formatting options for date and time data, text, and is RDBMS-specific. Its syntax is `CONVERT(data_type, expression, style)`. The `style` parameter, where applicable, permits customization of the conversion result. 

#### Rounding and Truncating Numerical Values

-   **ROUND**: This function rounds a numeric value to a specified digit. For example, `ROUND(123.4567, 2)` results in `123.46`.
-   **FLOOR**: Rounds up to the nearest integer less than or equal to the numeric expression. For instance, `FLOOR(123.4)` becomes `123`.

#### Working with Strings

-   **LEFT**: Extracts a specific number of characters from the beginning of a string.
-   **UCASE/LCASE**: Transforms all characters to upper or lower case.

#### Binomial Data Types in SQL Server for Number Handling

-   **TINYINT**: Represents an 8-bit unsigned whole number from 0 to 255.
-   **SMALLINT**: Typically an 16-bit integer from -32,768 to 32,767.
-   **BIGINT**: Represents an 8-byte integer from -9,223,372,036,854,775,808 to 9,223,372,036,854,775,807.
<br>

## 15. How do you handle _NULL values_ in _T-SQL_?

In **T-SQL**, **NULL** represents the absence of a value, and it brings certain considerations for data management and query execution.

### Dealing with NULLs

#### IS NULL / IS NOT NULL

Use the `IS NULL` and `IS NOT NULL` predicates to evaluate NULL values:

```sql
SELECT * FROM users WHERE phone IS NULL;
```

#### COALESCE

To substitute NULLs with a defined value, use `COALESCE`:

```sql
SELECT COALESCE(salary, 0) AS Salary FROM employees;
```

This retrieves `0` if `salary` is NULL. You can chain `COALESCE` for multiple replacements:

```sql
COALESCE(salary, bonus, 0)
```

#### NULLIF

`NULLIF` compares two values and returns NULL if they are equal, otherwise the first value:

```sql
SELECT NULLIF(column1, column2) FROM table;
```

#### Common Functions Handling NULLs

- Use `ISNULL` to replace NULL with a defined value.
- `NVL` and `NVL2` are Oracle equivalents of `ISNULL` and `COALESCE` respectively.

### Indexing and Performance Implications

- Queries including NULL-dependent conditions can be more resource-intensive, potentially resulting in a full scan of the dataset.

- Standard indexes include NULL values, but you can employ Filtered Indexes to exclude or include them, achieving performance enhancements in specific scenarios.

- For efficient JOINs, consider nullable columns that often contain non-NULL values but are, technically, not mandatory, by employing `UNION ALL`.

### Using DISTINCT

`DISTINCT` can yield unexpected results with NULLs. Duplicates or NULLs might not be removed as anticipated. To address this, consider using `GROUP BY` or additional logic in your queries.
<br>



#### Explore all 100 answers here 👉 [Devinterview.io - T-SQL](https://devinterview.io/questions/web-and-mobile-development/t-sql-interview-questions)

<br>

<a href="https://devinterview.io/questions/web-and-mobile-development/">
<img src="https://firebasestorage.googleapis.com/v0/b/dev-stack-app.appspot.com/o/github-blog-img%2Fweb-and-mobile-development-github-img.jpg?alt=media&token=1b5eeecc-c9fb-49f5-9e03-50cf2e309555" alt="web-and-mobile-development" width="100%">
</a>
</p>

