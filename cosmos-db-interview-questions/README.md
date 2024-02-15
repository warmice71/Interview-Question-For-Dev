# Top 55 Cosmos DB Interview Questions

<div>
<p align="center">
<a href="https://devinterview.io/questions/web-and-mobile-development/">
<img src="https://firebasestorage.googleapis.com/v0/b/dev-stack-app.appspot.com/o/github-blog-img%2Fweb-and-mobile-development-github-img.jpg?alt=media&token=1b5eeecc-c9fb-49f5-9e03-50cf2e309555" alt="web-and-mobile-development" width="100%">
</a>
</p>

#### You can also find all 55 answers here 👉 [Devinterview.io - Cosmos DB](https://devinterview.io/questions/web-and-mobile-development/cosmos-db-interview-questions)

<br>

## 1. What is _Cosmos DB_ and what are its _core features_?

**Azure Cosmos DB**, a globally distributed, multi-model database, is designed to provide elasticity, high availability, and low-latency access in any data model.

### Key Features

#### Multi-Model

Unlike traditional databases that are usually limited to one data model, Cosmos DB provides support for **document**, **key-value**, **graph**, and **column-family** data models.

#### Global Distribution

With a single click/tap, you can replicate your data across **Azure regions** or even on **multiple continents**. This strategic feature ensures high availability and low-latency access.

#### Automated Indexing

To enable efficient and secure data retrieval, Cosmos DB offers automatic indexing without the need for manual configuration.

#### Multi-API Support

This database accommodates different **APIs**, such as SQL, Cassandra, Gremlin (Graph), Azure Table Storage, and MongoDB. This way, developers can use familiar data models and SDKs without learning new ones. 

#### ACID Transactions

Cosmos DB provides multi-document transactions guaranteeing Atomicity, Consistency, Isolation, and Durability (ACID) to ensure data integrity.

#### Scalability

You can scale your database throughput or storage independently according to your application's changing needs. It supports horizontal as well as vertical scaling.

#### SLA-Backed Performance

The **Service-Level Agreements (SLAs)** ensure predictable and guaranteed latency.

#### Data Security

With built-in security features, such as transparent data encryption (TDE) and Role-Based Access Control (RBAC), you can rest assured about the safety of your data.

#### Consistency Levels

Choose a consistency model among five offered by Cosmos DB: Strong, Bounded Staleness, Session, Consistent Prefix, or Eventual. 

Each provides a different trade-off between consistency, availability, and latency.

#### Compatibility with Azure Services

Seamless integration with other Azure components like Azure Search, Azure Stream Analytics, and HDInsight helps in data exploration, real-time analytics, and other operations.
<br>

## 2. Explain the different _APIs_ provided by _Cosmos DB_. How do you choose the _right one_ for your application?

**Azure Cosmos DB** offers multiple APIs to cater to various data models, versatility requirements. Each API is purpose-built to cater to specific **NoSQL** paradigms

### API Offerings

#### 1. Core (SQL)

- SQL API is designed for **JSON documents**.
- It provides a **SQL-like** language for querying.
- It serves as the base for other Cosmos DB APIs.

#### 2. MongoDB

- This API **emulates the MongoDB server**, allowing you to use your existing MongoDB code and experience with Cosmos DB.

#### 3. Cassandra

- The Cassandra API is compatible with **CQL** (Cassandra Query Language).
- This is advantageous if you are migrating or leveraging existing Cassandra applications.

#### 4. Gremlin (Graph)

- For **graph data**, you can use the Gremlin API to traverse and manage graph structures.

#### 5. Table

- The Table API is compatible with **Azure Table Storage**, designed for companies transitioning to Cosmos, allowing them to leverage their existing structures and platforms.

#### 6. Etcd

- Etcd API aims to be compatible with the Consistency, Availability, Partition tolerance (CAP) guarantees of Etcd, a distributed key-value store, offering strong CP consistency.

### Factors for API Selection

- **Data Model**: Identify whether your data is schema-less (JSON/BSON) or follows a structured format. For example, if you are working with graph structures, the **Gremlin** API is the best choice.
  
- **Existing Ecosystem**: If your applications and teams are already familiar with certain ecosystems like MongoDB or Cassandra, it makes sense to choose the respective APIs to streamline operations and minimize disruptions.

- **Querying Flexibility**: The SQL and Gremlin APIs give in-depth querying capabilities, while Cassandra and Table APIs have their query languages. If you have unique querying needs, choose an API that aligns with them.

### Key Considerations for API Selection

- **Cost and Scale**: Different APIs might have distinct scaling and pricing implications.

- **Geographical Distribution**: Local data compliance, latency requirements, and distribution strategies may differ among APIs.

For more context, **Cosmos DB** offers API-specific documentation, and their support team can guide you through the selection process.
<br>

## 3. What is the _data model_ used by _Cosmos DB_ and how does it differ from _relational databases_?

Even though related databases mainly utilize a tabular data structure, **Cosmos DB** leverages a **document-oriented model** that's specifically build to handle unstructured, semi-structured and structured data.

### Overview of Document-Oriented Data Model

Each record in a **document database** is a self-contained, hierarchical data unit termed a **"document"**. These documents are often serialized in familiar formats such as JSON or XML. Key benefits of this approach include enhanced data handling for objects and lower I/O requirements.

### Key Characteristics:
- **Self-Contained**: Any necessary references or relationships are embedded within the document, avoiding the need for complex, multi-table queries.
- **Schema Flexibility**: Mappings from application objects to database documents are straightforward, and Cosmos DB allows for adaptable data models by supporting multi-dimensional schemas through the use of varied formats and content in documents.
- **Atomicity at Document Level**: The database management system assures that operations on a single document are all-or-nothing.
- **Consistency Models**: Cosmos DB offers various global distributed architectures for data consistency, enabling customizations based on specific application necessities.

### Code Example: Document in JSON Format

Here is the JavaScript code:

```json
{
  "id": "1",
  "name": "John Doe",
  "addresses": [
    {
      "street": "123 Main St",
      "city": "Anytown",
      "state": "CA"
    },
    {
      "street": "456 Elm St",
      "city": "Othertown",
      "state": "NY"
    }
  ],
  "contact": {
    "email": "john@example.com",
    "phone": "123-456-7890"
  }
}
```
<br>

## 4. How does _Cosmos DB_ ensure _data durability_?

**Cosmos DB** ensures **data durability** using a few key mechanisms: **replication**, **transactions**, and **backups**.

### Write Ahead Logging (WAL)

- Before changes are written to disk, all updates are first recorded in a **log**. This is known as **Write Ahead Logging (WAL)**.
- The log is continually synced to disk to guarantee that changes are **persisted** even if the system crashes.

### Log Structured Storage and Compaction Logs

- In Cosmos DB, data is stored in a **log-structured** format. A **Compaction Log** is used to manage the merging and cleanup of data segments. This process ensures that data remains up-to-date, even as it's compacted and reorganized over time.

### Multi-Version Concurrency Control (MVCC)

- **Under MVCC**, whenever data is updated, the old version isn't immediately discarded. Instead, it's supplanted by a new version. This mechanism supports **snapshot isolation** for transactions.
- When reading data, a transaction sees a consistent snapshot of the data, even if other transactions are modifying it concurrently.

### Automated Backup

- Cosmos DB offers **point-in-time backups** to support data loss recovery. This feature automatically captures backups of your data at regular intervals. Should data need to be restored or recovered, these backups are available for that purpose.
- You have control over the retention period of these backups, allowing you to define the duration for which data is kept in the backup store.

### Replication and Multi-Region Data Centers

- One of the core features of Cosmos DB is its **multi-region replication**. This not only allows for high availability by replicating data across different locations but also ensures data durability in the face of regional disasters.
- All regions housing a Cosmos DB account operate in sync, and any changes made to data are replicated across all regions. Should a region become inaccessible, counterparts are instantly available to ensure continued data integrity and accessibility.
<br>

## 5. What is the _Request Unit (RU)_ in _Cosmos DB_ and how is it _used_?

In Cosmos DB, **Request Units (RUs)** are a measure of the resources needed to perform read and write operations.

### Understanding Request Units

RUs serve as a unit of measure for database operations. Performing an **"Item Read" or "Point Query"** requires 1 RU, while heavy operations like **"Query Based on Several Index Ranges"** can demand up to 10 RUs or more.

### RUs Allocation Modes

1. **Fixed RUs Mode**: Offers predictable pricing by allotting a fixed quantity of RUs to each operation.
2. **Provisioned RUs Mode**: The classic mode where RUs are provisioned in advance and billed hourly.

### What Is Under the Hood?

Cosmos DB uses a sophisticated infrastructure to manage resources, which can be further tuned using RUs:

- **Storage**: Request Units are associated with the storage and retrieval of data.
- **Indexing**: They help maintain indexes for efficient querying.
- **Compute**: The operations to process the queries need RUs.

### Code Example: RUs in Action

Here is the C# code:

```csharp
var query = client.CreateDocumentQuery<User>(collectionUri, new SqlQuerySpec("SELECT * FROM c"), queryOptions).AsDocumentQuery();
var response = await query.ExecuteNextAsync();
Console.WriteLine($"Request Charge: {response.RequestCharge} RUs");
```

Here is the JavaScript code:

```javascript
const querySpec = {
    query: "SELECT * FROM c"
};
const { result: users, rucharge: charge } = await container.items.query(querySpec).fetchAll();
console.log(`Request Charge: ${charge}`);
```

Make sure to test the quota by using the settings in the portal. An optimal usage can provide cost advantages.
<br>

## 6. Can you describe the _multi-model capabilities_ of _Cosmos DB_?

**Azure Cosmos DB** combines the best qualities of both SQL and NoSQL databases, offering multi-model capabilities to serve a diverse range of applications.

### Key Multi-Model Aspects

#### Comprehensive Multiple Data Models

- **Documents**: Offered through the core SQL API for JSON data format.
- **Key-Value Pair**: Using SQL API, you can store and retrieve data based on simple key-value pairs.

#### Globally Distributed

Hashing algorithms help in distributing the data across various Azure data centers efficiently. This process is called Partitioning.

### Code Example: Key-Value Pair Operations

Here is the C# code:

```csharp
var container = cosmosClient.GetContainer("databaseId", "containerId");

// Adding a Key-Value Pair
await container.UpsertItemAsync<dynamic>("myPartitionKey", new { id = "myKey", value = "myValue" });

// Retrieving a Value by Key
var iterator = container.GetItemQueryIterator<dynamic>(
    new QueryDefinition("SELECT * FROM T WHERE T.id = @id")
    .WithParameter("@id", "myKey"));

var result = await iterator.ReadNextAsync();
var myValue = result.First();
Console.WriteLine(myValue.value);
```
<br>

## 7. Outline the types of _indexing_ available in _Cosmos DB_. How does _indexing_ affect _performance_?

**Azure Cosmos DB** provides several indexing strategies to optimize database performance for specific data access patterns.

### Types of Indexing in Cosmos DB

- **Range Index**: Supports range queries, equality filters, order by, and various built-in functions like `IS_DEFINED()` and `EXISTS()`. 
- **Spatial Index**: Optimizes geospatial queries. Requires a special index for properties that represent geospatial data.
- **Composite Index**: Merges multiple single-property indexes into a comprehensive compound index, ideal for queries with more than one filter.
- **Hash Index**: Tailored for arrays, and assists queries that target particular array elements or index ranges.

### Indexing and Performance

- **Consistency**: Maintaining a high level of indexing can lead to more consistent reads, but it also incurs additional indexing overhead.
- **RUs (Request Units)**: Writing and updating indexed attributes consumes RUs. An effective index design is pivotal to **manage RU expenditures**.
- **Query Performance**: Up-to-date, well-maintained indexes ensure snappier query responses.

### Code Example: Indexing Policies

Here is the C# code:

```csharp
public class MyEntity
{
    public int Id { get; set; }
    public string Name { get; set; }

    [JsonProperty("location")]
    public Point Location { get; set; }

    public List<int> Tags { get; set; }
}

// Define an Indexing Policy tailored for the MyEntity type
IndexingPolicy policy = new IndexingPolicy
{
    Automatic = true,
    IndexingMode = IndexingMode.Consistent,
    IncludedPaths =
    {
        new IncludedPath
        {
            Path = "/Name/?",
            Indexes = new Collection<Index> { new RangeIndex(DataType.String) }
        },
        new IncludedPath
        {
            Path = "/location/?",
            Indexes = new Collection<Index> { new SpatialIndex() }
        },
        new IncludedPath
        {
            Path = "/Tags/?",
            Indexes = new Collection<Index> { new HashIndex(DataType.Number) }
        }
    },
    ExcludedPaths =
    {
        new ExcludedPath { Path = "/*" }
    }
};
```
<br>

## 8. Discuss the _ACID properties_ in the context of _Cosmos DB transactions_.

**Azure Cosmos DB** offers ACID transactions, which ensure the integrity of data across distributed systems. Let's discuss each of the ACID properties in the context of Azure Cosmos DB and refine this topic.

### ACID Properties in Azure Cosmos DB

1. **Atomicity**: All or Nothing

   Atomic transactions in Cosmos DB are like a light switch: they are instantaneous and can only be fully on or fully off.

2. **Consistency**: Valid State

   Cosmos DB ensures the consistency of data, upholding its defined schema and the all-or-nothing principle. If a transaction fails, the system is restored to its state before the transaction began.

3. **Isolation**: Independent Actions

   During a transaction, Cosmos DB is designed to isolate the involved resources, preventing their concurrent access and guarding the transaction's integrity.

4. **Durability**: Permanent Changes

   When a transaction is committed, Cosmos DB guarantees that the changes will be permanently stored and not lost, even in the face of failures. The system ensures this using a combination of synchronous replication, logging, and storage mechanisms.

### Multi-Master Configurations

In multi-master configurations, Cosmos DB handles complex scenarios like conflicting changes with mixtures of auto-merge and deterministic merge control. These features make multi-master deployments ideal for geographically distributed applications.

### Code Example: ACID Transactions in Cosmos DB

Here is the C# code:

  ```csharp
  using Microsoft.Azure.Cosmos;
  using System;
  using System.Threading.Tasks;

  public class CosmosDBService
  {
      private CosmosClient _cosmosClient;

      public CosmosDBService(string connectionString)
      {
          _cosmosClient = new CosmosClient(connectionString);
      }

      public async Task SaveDocumentAsync<T>(string databaseId, string containerId, T document)
      {
          var container = _cosmosClient.GetContainer(databaseId, containerId);
          var transactionalBatch = container.CreateTransactionalBatch(new PartitionKey("partitionKey"));

          transactionalBatch.UpsertItem<T>(document);
          
          try
          {
            await transactionalBatch.ExecuteAsync();
          }
          catch (CosmosException ex)
          {
              Console.WriteLine("Exception: " + ex.ToString());
              throw;
          }
      }
  }
```
<br>

## 9. What is a _partition key_ in _Cosmos DB_ and how is it _used_?

In **Azure Cosmos DB**, the **partition key** serves as a fundamental design element that dictates the **physical data distribution** and **performance characteristics** of your dataset.

With a well-chosen partition key, you can ensure balanced data distribution, efficient data access, and optimized scalability.

### Design Importance

Selecting an appropriate partition key is one of the most critical design decisions in Cosmos DB. It is crucial for:

- **Performance**: Ensuring that your data access patterns, such as queries and write operations, perform optimally.
- **Scalability**: Allowing your container to expand and contract effectively as data volumes and throughputs fluctuate.

### Physical Data Organization

Partition keys are used to divide data into **logical partitions**. Cosmos DB then takes care of distributing these logical partitions across **physical partitions**, a transparent process called **physical partitioning**.

- Each logical partition maps to one or more physical partitions.
- Data within a logical partition is co-located within the same physical partition.
- Queries that target a single partition are served from a single physical partition, optimizing performance.

### Performance Considerations

- **Throughput**: The provisioned throughput of a container is shared among all its logical partitions. If you compact a high volume of data and activity into a single partition, it could lead to **hot partitions**.
  
- **Latency and Resource Consumption**: Queries, updates, or deletes that span logical partitions may require distributed coordination, potentially affecting latency and resource usage.

### Common Partition Key Strategies

#### Natural and Intrinsic Keys

Many datasets possess **built-in candidates** for partition keys, such as `customer-id`, `device-id`, or `location`. 

Such keys are referred to as **natural keys** that are already inherent to the data. They are often the ideal choice as they align with common access patterns and facilitate efficient data placement.

#### Synthetic or Auto-Generated Keys

In some scenarios, a natural key could be unavailable or inappropriate. In these cases, a **synthetic key**, generated by the application, might be preferred.

Examples include keys based on time ranges, such as the **CreatedAt** property of a document, or keys that use a hash of another property for uniform distribution.

- **Time-Based Keys**: Implementing **time-slicing** using a well-known algorithm can be useful for **telemetry data** to ensure **recent data is frequently accessed and modified**.

  For example, a partition key based on the year and month could look like `202204`.

- **Balanced Property-Based Keys**: When natural keys exist but may not ensure uniform data distribution, a **hashed version of the key** can be used for **balance**. In the case of a `product-id`, you can take a consistent hash of it for distribution uniformity.

### Code Example: Generating a Time-Based Key

Here is the Python code:

```python
from datetime import datetime

def generate_time_partition_key():
    return datetime.utcnow().strftime('%Y-%m')
```
<br>

## 10. Explain the concept of _logical partitions_ vs _physical partitions_ in _Cosmos DB_.

**Cosmos DB** is distinguished by its distinctive partitioning strategy. To better understand its partitioning mechanics, it's crucial to differentiate **physical** from **logical partitions**.

### Key Distinctions

- **Physical Partitions**: These are the units of storage and throughput. Every **physical partition** is associated with a dedicated set of system resources. Cosmos DB dynamically and transparently manages these **physical partitions**.
  
- **Logical Partitions**: These exist within each **physical partition** and are the organizational units for your data. **Logical partitions** are essential for understanding how your data is distributed across **physical partitions** for optimal query performance and throughput utilization.

In a nutshell, **physical partitions** manage the backend physical resources, while **logical partitions** help optimize data access efficiency.

### When to Use Logical Partitions

- **Scenario**: Data retrieved together often.
- **Benefit**: Minimized cross-partition queries.
 

### Scalability and Throughput

-  An individual **logical partition** is subject to the throughput limits and **storage quota** of the parent **physical partition**.
- The target is to distribute workloads across **logical partitions** to harness the full throughput capacity offered by **physical partitions**.

Cosmos DB encompasses a range of mechanisms to facilitate this distributive workflow, including automatic **indexing**, partitioning, and intra-partition parallelism.
<br>

## 11. What are the best practices for choosing a _partition key_ in a _Cosmos DB container_?

Selecting an optimal **partition key** is crucial for designing scalable and performant **Cosmos DB containers**. Let's look at the best practices to ensure efficient data distribution and partition management.

### Key Considerations

The ideal partition key should:

- **Distibute Data Evenly**: Preventing hot partitions is essential for balanced performance.
- **Facilitate Data Access**: The partition key should align with common query patterns and enable logical grouping.
- **Allow for Scale**: It should offer adequate room for data growth without frequent splits or merges.

### Best Practices

#### Data Characteristics

- **Volume of Data**: If a container has a high number of potential partitions or documents, consider a selective and diverse partition key.
  
- **Data Distribution**: If your data is naturally grouped or clustered around certain values, selecting such a value as a partition key can be beneficial. For example, in an e-commerce application, customer ID or order ID can be a good partition key if most queries are specific to a customer or an order.

- **Access Patterns**: Identify the primary querying and data insert/update patterns. Choose a key that aligns with these patterns.

  For example, in a social media application where users frequently post status updates, a partition key of userID along with timestamp can be beneficial as most queries revolve around the user's activities.

#### Cardinality

- **High Cardinality**: A high-cardinality partition key, such as a unique identifier or timestamp, can distribute data more evenly, avoiding hot partitions. However, make sure it aligns with your query patterns.

- **Low or Moderate Cardinality**: These types of partition keys can work well if the data within each partition is relatively small and evenly distributed.

#### Partition Autopilot

Cosmos DB provides a feature called **Autopilot**, which automates the process of selecting partition keys. It's particularly useful for workloads that are hard to predict.

#### Avoid these Traps

- **Keys That Lead to Skews**: Certain keys, like those representing boolean values, may skew data distribution if most of the data falls under one value.

- **Keys That Lead to Hot Partitions**: If a specific key is frequently accessed or modified, it can result in a hot partition. This problem can arise with monotonically increasing keys, such as timestamps.

### Dynamically Changing Partition Keys

Changing an existing partition key in a Cosmos DB container is non-trivial and often requires creating a new container and migrating the data.

Remember, a well-chosen partition key is **central to both data management and performance tuning**, making it crucial for laying a strong foundation for your Cosmos DB setup.
<br>

## 12. What considerations should be taken into account when _modeling data_ for a _Cosmos DB instance_?

When **modeling data** for **Cosmos DB**, several best practices can enhance both performance and efficiency.

### Key Considerations

- **Access Patterns**: Accurately identify and define your primary document types and how they relate to one another.

- **Data Structure**: Since Cosmos DB is schema-agnostic, the choice of data types (such as nested arrays, object properties, references) should align with expected data output and input formats.

- **Model Granularity**: Opt for a fine balance between minimizing document size for focused queries and avoiding excessive JOINS.

- **Data Distribution**: Utilize partition keys to ensure data is evenly distributed, maximizing performance and cost-effectiveness.

- **Index Policies**: Fine-tune indexing to reflect query patterns and optimize performance.

- **Consistency Levels**: Choose the most suitable level to balance consistency needs and performance.

- **Data Types and Partition Keys**: Carefully choose distribution strategies for data types other than strings and partition keys.

- **Latency Considerations**: Take into account the specific workload requirements, as Cosmos DB supports a mix of low and high latency operations.

- **Throughput Efficiency**: Balance request units for cost-efficiency; this is critical to avoiding over-provisioning.

### Partition Keys

Understanding **Partition Keys** is vital for designing high-performance data models in Cosmos DB. Typically, you should choose a partition key:

- With a high cardinality.
- That distributes data evenly.
- That's frequently used in queries.
- That minimizes the need for cross-partition queries.

### Cosmic Rules

- **Data Duplication for Speed**: Duplicating data across documents can eliminate the need to perform JOIN operations, boosting query performance. This technique, often called "denormalization," is common in NoSQL databases.

- **Size Matters**: Keep individual document sizes below 2MB to maximize efficiency. If you frequently require larger documents, consider offloading them to a linked storage solution like Blob storage.

- **Selective Indexing**: Cosmos DB allows you to exclude properties from the index. This is useful for data that's rarely or never queried.

- **Chip Away**: Through throttling and other cost-related mechanisms, Cosmos DB can meter resource usage. Regularly monitor and, when necessary, adjust throughput to ensure efficient data handling.

- **TTL Leanings**: Leveraging the Time to Live (TTL) feature can automatically remove expired data, a critical consideration for regulatory compliance and data hygiene.

### Code Example: Partition Key Selection

Here is the C# code:

```csharp
public class Customer
{
    public string Id { get; set; }
    public string Name { get; set; }
    public string Email { get; set; }
    public string CountryCode { get; set; }

    // Derive the partition key from the country code for even data distribution
    [JsonProperty(PropertyName = "/CountryCode")]
    public string PartitionKey => $"/{CountryCode}";
}
```

In this example, `CountryCode` serves as the **Partition Key**.
<br>

## 13. How does _partitioning_ impact the _scalability_ and _performance_ of a _Cosmos DB_ application?

In **Azure Cosmos DB**, partitioning plays a key role in the database's scalability and performance. To make the most out of partitioning, it's important to understand its nature and characteristics.

### Key Benefits

- **Data Distribution**: Partitioning optimally distributes data across multiple physical partitions.
- **Performance Isolation**: Enables performance and throughput isolation at the partition-level.
- **Scalability**: The database as a whole can be easily auto-scaled using the provisioned throughput system.

### How it Works

- **Logical Entities**: In Cosmos DB, entities like documents are grouped logically into containers (formerly known as collections). Each container has one or more logical partitions.
- **Physical Partitions**: These logical partitions are then distributed across multiple physical partitions. A physical partition represents the unit of scale and the maximum RU/s that can be allocated to a logical partition.
- **Scalability**: During _Provisioned Throughput_ mode, RUs are distributed among all the logical partitions within a specific container, providing fair access. In _Serverless_ mode, throughput is consumed on an as-needed basis, accommodating traffic spikes more flexibly.

### Impact on Query Efficiency

- **Single-Partition Queries**: Queries that target a specific logical partition are the most efficient. They stay within a single physical partition, which optimizes both latency and throughput. Scalability, in this case, is limited based on the RU/s allocated to the specific physical partition.

- **Cross-Partition Queries**: These include queries that don't specify the partition key and those involving JOINs. They execute across all partitions and thus may consume more RUs, potentially impacting performance and throughput.

### Best Practices

- **Select an Appropriate Partition Key**: The key should ensure even, predictable data distribution while supporting the majority of your queries. This is a crucial design consideration and can impact all the points mentioned earlier.
- **Use Optimal Patterns**: Leverage patterns like one-to-few and one-to-many relationships to minimize cross-partition queries.
- **Track Request Units**: Monitor RU consumption, especially for cross-partition queries, to ensure proper resource allocation.
- **Be Mindful of Limitations**: Understand restrictions around transactions, consistency levels, and sizing to make effective use of partitioning.
<br>

## 14. How might you handle _hot partitions_ in _Cosmos DB_?

**Hot partitions** can arise in non-relational databases like **Cosmos DB** when a disproportionate share of read and write operations are targeted to specific partition keys. This leads to performance issues.

To mitigate this, use techniques such as partition key selection, data and query design, SDK configurations, and horizontal scaling.

### Addressing Hot Partitions

#### Partition Key Selection

- **Selectivity**: Prioritize partition keys with high selectivity to distribute data evenly across logical partitions.
- **Traffic Distribution**: Prefer partition keys with more balanced or predictable traffic to avoid hot partitions.

#### Data and Query Design

- **Code for Even Data Distribution**: Design your application to store and access data in a way that promotes even distribution across partition keys.

- **Aim for Concurrency**: Minimize transactions that involve data from multiple partitions. This approach can help improve performance.

#### SDK Configuration

- **Request Rate Limiting**: Adjust request rate limits in Cosmos DB based on the size and nature of the workloads.

#### Horizontal Scaling

- **Multiple Collections or Databases**: If needed, consider using multiple collections or databases to separate workloads, providing better isolation and performance.

### Code Example: Partition Key Selection

Here is the C# code:

```csharp
// Define the Document class with required attributes
public class Document
{
    [DataMember(Name = "id")]
    public string Id { get; set; }

    [DataMember(Name = "partitionKey")]
    public string PartitionKey { get; set; }

    [DataMember(Name = "data")]
    public string Data { get; set; }
}

// Create an instance of Document for storing data
Document document = new Document
{
    Id = "unique-document-id",
    PartitionKey = "selected-partition-key",  // Use a carefully chosen partition key
    Data = "sample data"
};

// Use the DocumentClient to create new documents in the collection
await documentClient.CreateDocumentAsync(collectionLink, document);
```
<br>

## 15. Describe some common _anti-patterns_ in _data modeling_ for _Cosmos DB_.

**Cosmos DB** is a powerful NoSQL database, but **flexibility** in **schema design** can lead to potential pitfalls. Here are some common anti-patterns to watch for.

### Common Anti-Patterns

#### Volatile Partition Key

A change to the **partition key** across huge datasets can cause significant operational overhead, downtime, and compromise on performance. It's best to select a **partition key** that doesn't need frequent updates.

#### Fan-Out, Cross-Partition Reads

Performing a **fan-out** is an action where a single API call retrieves data across multiple partitions, leading to inefficient and slower operations. Try to structure data to avoid the need for fan-outs.

#### Missing or Wasteful Indexing

Cosmos DB comes with **automatic indexing**. While it simplifies many setup requirements, it's crucial to understand when manual tweaking becomes necessary, such as for covering query's or size optimization.

#### Excess Writes in High-Throughput Scenarios

Over-utilization in high-throughput situations can result in the costly consumption of **Request Units** (RUs). Actions like frequent document overwrites and stack pushes should be minimized.

#### Resource-Intensive Operations

Certain operations in Cosmos DB, such as the **JOIN** mechanism, can be costly in terms of resource usage. Efficient data modeling and avoidances of operations that demand CPU or memory will help your applications to run smoother.

#### Overuse of Suboptimal Data Types

Selecting the most appropriate data types in Cosmos DB is essential. For instance, using inappropriately large data types could consume additional storage space and potentially lead to inflated retrieval costs.

#### Inefficient Transactions

Cosmos DB, like other NoSQL databases, offers atomicity and consistency at the level of the single document, but not across multiple documents in a transaction. Modeling data to avoid cross-document transactions is recommended for more efficient operations.

### Tip for Better Data Models

- **Multiple Collections**: Utilize multiple collections in Cosmos DB to isolate big or infrequently accessed resources and optimize throughput usage.
- **Hierarchical Data**: Leverage the hierarchy in JSON to reduce instances of cross-document reference, enhancing the efficiency of your data operations.

### Code Example: Efficient Data Modeling

Here is the C# code:

```csharp
public class Student
{
    public string Id { get; set; }
    public string Name { get; set; }
    public List<CourseEnrollment> CourseEnrollments { get; set; }
}

public class CourseEnrollment
{
    public string CourseId { get; set; }
    public string Grade { get; set; }
}

```
<br>



#### Explore all 55 answers here 👉 [Devinterview.io - Cosmos DB](https://devinterview.io/questions/web-and-mobile-development/cosmos-db-interview-questions)

<br>

<a href="https://devinterview.io/questions/web-and-mobile-development/">
<img src="https://firebasestorage.googleapis.com/v0/b/dev-stack-app.appspot.com/o/github-blog-img%2Fweb-and-mobile-development-github-img.jpg?alt=media&token=1b5eeecc-c9fb-49f5-9e03-50cf2e309555" alt="web-and-mobile-development" width="100%">
</a>
</p>

