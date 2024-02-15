# 100 Fundamental Spring Interview Questions

<div>
<p align="center">
<a href="https://devinterview.io/questions/web-and-mobile-development/">
<img src="https://firebasestorage.googleapis.com/v0/b/dev-stack-app.appspot.com/o/github-blog-img%2Fweb-and-mobile-development-github-img.jpg?alt=media&token=1b5eeecc-c9fb-49f5-9e03-50cf2e309555" alt="web-and-mobile-development" width="100%">
</a>
</p>

#### You can also find all 100 answers here 👉 [Devinterview.io - Spring](https://devinterview.io/questions/web-and-mobile-development/spring-interview-questions)

<br>

## 1. What is the _Spring Framework_ and what are its core features?

The **Spring Framework** is a comprehensive software platform that provides infrastructure support for developing enterprise-level applications. It's known for its robust features that simplify complex tasks and for its support of best coding practices.

### Core Features

1. **Inversion of Control (IoC)**: Centralized bean management ensures loose coupling leading to easy maintenance and testing.

2. **Aspect-Oriented Programming (AOP)**: Modularizes cross-cutting concerns such as logging and caching, promoting reusability.

3. **Validation and Data Binding**: Offers powerful validation and data binding mechanisms compatible with JavaBeans components, thereby ensuring data integrity.

4. **JDBC Abstraction and Transactions**: Provides a consistent data access layer and unified transaction management across various data sources.

5. **ORM Support**: Simplifies Object-Relational Mapping with tools like Spring Data JPA and Hibernate.

6. **MVC Web Framework**: Facilitates the development of flexible web applications and RESTful services.

7. **REST Support and Content Negotiation**: Streamlines building RESTful web services and content negotiation for better client/server communication.

8. **Security Features**: Offers a robust security framework for web applications, covering authentication, authorization, and access-control decisions.

9. **Internationalization and Localization**: Facilitates creating multi-lingual applications by providing extensive support for different languages and regions.

10. **Dynamic, Strong-Typed Property Configuration**: The Spring EL (Expression Language) simplifies dynamic resolution of property values in annotations or XML configuration files.

11. **Runtime Polymorphism and Dependency Lookup**: Spring provides lightweight, built-in dependency lookup strategies aiding late binding of dependencies.

12. **Support for Different Development Styles**: Offers support for various enterprise application patterns like Singleton, Factory, Adapter, and so on.

13. **In-Depth Testing Support**: Spring's testing modules provide classes and configurations for thorough unit and integration testing.

14. **Java Configuration and Annotation**:
    - Spring allows using plain Java classes to define beans and their dependencies, reducing XML configuration overhead.
    - Annotations like `@Autowired` enable autowiring of dependencies, promoting developer productivity.

15. **Extensive Documentation and Community Support**: Spring has rich, comprehensive documentation and a highly active user community, ensuring reliable support and guidance.

16. **Modularity**: Spring, being a modular framework, allows using only the needed modules, minimizing runtime overhead.
<br>

## 2. How do you create a simple _Spring_ application?

To create a **simple Spring application**, you need to handle the following tasks:

1. Choose the right Structure for Project
2. Set the Maven or Gradle configurations
3. Add the Essential Dependencies
4. Implement the Main Application
5. Write a simple Controller or Service for the Application 

### Project Setup

Choose your favorite  ```Build Management System``` such as Maven or Gradle.

**Maven Example** - pom.xml:

```xml
<dependencies>
    <!-- Core Spring Context -->
    <dependency>
        <groupId>org.springframework</groupId>
        <artifactId>spring-context</artifactId>
        <version>5.3.10</version>
    </dependency>
    <!-- Web Support -->
    <dependency>
        <groupId>org.springframework</groupId>
        <artifactId>spring-web</artifactId>
        <version>5.3.10</version>
    </dependency>
    <!-- Bootstrap Spring Web Application -->
    <dependency>
        <groupId>org.springframework</groupId>
        <artifactId>spring-webmvc</artifactId>
        <version>5.3.10</version>
    </dependency>
    <!-- Javax Servlet API -->
    <dependency>
        <groupId>javax.servlet</groupId>
        <artifactId>javax.servlet-api</artifactId>
        <version>4.0.1</version>
        <scope>provided</scope>
    </dependency>
</dependencies>
```

**Gradle Example** - build.gradle:

```gradle
dependencies {
    // Core Spring Context
    implementation 'org.springframework:spring-context:5.3.10'
    // Web Support
    implementation 'org.springframework:spring-web:5.3.10'
    // Bootstrap Spring Web Application
    implementation 'org.springframework:spring-webmvc:5.3.10'
    // Javax Servlet API
    providedCompile 'javax.servlet:javax.servlet-api:4.0.1'
}
```

### Key Components for a Web Application

1. **DispatcherServlet**: Captures HTTP requests and directs them to the right controllers.
2. **ApplicationContext**: The core container for Spring's IoC and dependency injection features.

### Spring Application Main Class

Here is the Java code:

```java
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

@SpringBootApplication
public class MySpringApp {
    public static void main(String[] args) {
        SpringApplication.run(MySpringApp.class, args);
    }
}
```

With the `@SpringBootApplication` annotation, Spring Boot takes care of intricate configuration, and you can be started with plain old `public static void main`.

### WebController

Create a simple Controller that listens to GET requests. Here is the Java code:

**MainController**.java:

```java
import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.RequestMapping;

@Controller
public class MainController {

    @RequestMapping("/")
    public String home() {
        return "index";
    }
}
```
<br>

## 3. What is _Inversion of Control (IoC)_? How does _Spring_ facilitate _IoC_?

**Inversion of Control** is a coding pattern where the control of flow is transferred to a framework or an external system. This mechanism serves as the foundation for **Dependency Injection** and is an integral part of the Spring framework.

### How does Spring Support IoC?

1. **Bean Management**: Spring manages Java objects, known as beans, by detailing their creation, configuration and deletion through **Bean Factories**.

2. **Configurations**: Spring employs both XML and annotations for defining bean configurations.

3. **Dependency Resolution and Injection**: Spring ensures that **inter-bean dependencies** are resolved and injected, thereby reducing tight coupling and enhancing testability and flexibility.

4. **Lifecycle Management**: Using Spring's **Bean Lifecycle**, you can manage the instantiation, modification and disposal of beans in a systematic manner.

5. **AOP and Declarative Services**: Spring aids IoC by offering an Aspect Oriented Programming (AOP) system and allowing declarative services via Java annotations, enabling you to externalize cross-cutting concerns.

6. **Externally Managed Resources**: You can bring in non-bean resources like data sources and template files under Spring IoC management, promoting resource sharing and centralization.

### Java Example: IoC with Spring

#### Maven Dependency

Add the following Spring Core dependency to your Maven `pom.xml`:

```xml
<dependencies>
  <dependency>
    <groupId>org.springframework</groupId>
    <artifactId>spring-core</artifactId>
    <version>5.3.10</version>
  </dependency>
</dependencies>
```

#### Bean Configuration (XML)

Define your beans in an XML file, typically named `applicationContext.xml`:

```xml
<beans xmlns="http://www.springframework.org/schema/beans"
  xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
  xsi:schemaLocation="http://www.springframework.org/schema/beans
    http://www.springframework.org/schema/beans/spring-beans.xsd">
  <bean id="customer" class="com.example.customer.Customer">
    <property name="itemName" value="Laptop" />
  </bean>
  <bean id="invoice" class="com.example.billing.Invoice" />
</beans>
```

#### Bean Configuration (Annotations)

In addition to XML-based configuration, you can use annotations. Add this to your XML bean container:

```java
@Configuration
public class AppConfig {
  @Bean
  public Customer customer() {
    Customer cust = new Customer();
    cust.setItemName("Laptop");
    return cust;
  }
  @Bean
  public Invoice invoice() {
    return new Invoice();
  }
}
```

#### Application Entry Point: Main

In your main application, retrieve beans from the Spring container:

```java
public class Main {
  public static void main(String[] args) {
    ApplicationContext context = new ClassPathXmlApplicationContext("applicationContext.xml");
    // XML-based instantiation
    Customer c1 = (Customer) context.getBean("customer");
    
    AnnotationConfigApplicationContext context = new AnnotationConfigApplicationContext(AppConfig.class);
    // Annotation-based instantiation
    Customer c2 = context.getBean(Customer.class);
    
    // Perform operations with beans
  }
}
```

In this example, both XML-based and annotation-based IoC and bean management are depicted.
<br>

## 4. What is the _ApplicationContext_ in _Spring_?

The `ApplicationContext` is at the core of Spring, serving as a **container** for managing and configuring the application's **beans** (components) and their lifecycle. The `ApplicationContext` is crucial for Inversion of Control (IoC) and Dependency Injection (DI).

### How It Works

- **Configuration Management**: This is a central repository for bean definitions, either specified in XML configuration, Java annotations, or XML configuration.

- **Bean Instantiation and Injection**: The `ApplicationContext` is responsible for creating and wiring beans based on the provided configuration.

- **Lifecycles**: The container manages bean lifecycles, initializing and destroying them when the application starts or shuts down.

### Bean Scopes

- **Singleton**: The default. The `ApplicationContext` creates and manages a single instance of a bean.
- **Prototype**: Each request or lookup results in a new bean instance.

There are also less commonly used scopes, such as `request`, `session`, `global session`, and `application`.

### Common `ApplicationContext` Implementations

- **ClassPathXmlApplicationContext**: Loads the context configuration from an XML file located in the classpath. 
- **FileSystemXmlApplicationContext**: Similar to `ClassPathXmlApplicationContext`, this loads from an XML file but requires a file system path.
- **AnnotationConfigApplicationContext**: Reads the configuration classes produced with Java annotations.
- **GenericWebApplicationContext**: Designed for web-aware applications and compatible with Servlet 3.0 environments.

### Close vs. Refresh

- **Close**: Shuts down the container, releasing resources and triggering bean destruction if necessary.
- **Refresh**: Usually used with web applications to perform a manual refresh of the `ApplicationContext` after it's been initialized.

### Code Example: Initializing the `ApplicationContext`

Below is the Java code:

```java
import org.springframework.context.ApplicationContext;
import org.springframework.context.support.ClassPathXmlApplicationContext;

public class MyApp {
    public static void main(String[] args) {
        ApplicationContext context = new ClassPathXmlApplicationContext("beans.xml");
        MyBean myBean = context.getBean(MyBean.class);
        myBean.doSomething();
        ((ClassPathXmlApplicationContext) context).close(); // Shuts down the context
    }
}
```
In modern Spring applications, Java configuration and annotations, such as `@Configuration` and `@Bean` methods, are typically used for initializing the `ApplicationContext`. 

Here is the corresponding code:

```java
import org.springframework.context.annotation.AnnotationConfigApplicationContext;
import org.springframework.context.support.AbstractApplicationContext;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

public class MyApp {
    public static void main(String[] args) {
        ApplicationContext context = new AnnotationConfigApplicationContext(AppConfig.class);
        MyBean myBean = context.getBean(MyBean.class);
        myBean.doSomething();
        ((AbstractApplicationContext) context).close(); // Shuts down the context
    }

    @Configuration
    public static class AppConfig {
        @Bean
        public MyBean myBean() {
            return new MyBean();
        }
    }
}
```

In this example, both the XML-based and Java-based configurations illustrate the setup of the `ApplicationContext`.

### Best Practices

- **Keep It Lightweight**: Overly complex configurations can lead to decreased application performance.
- **Know Your Bean Lifecycle**: Be mindful of singleton versus prototype scopes and how beans are initialized or destroyed.
- **Tread Carefully with Bean Scopes**: Common mistakes with bean scopes can lead to unanticipated behavior.
- **Use the Best-suited Configuration for Your Application**: Assess the requirements of your project to select the most efficient means of configuring the `ApplicationContext`.
<br>

## 5. Explain _Dependency Injection_ and its types in the _Spring_ context.

**Dependency Injection** (DI) is a fundamental concept in the **Spring Framework** and serves as a key advantage over traditional, direct dependencies management.

### Benefits

- **Easy to Manage and Test**: Dependencies can be swapped for testing and changed at runtime.
- **Decoupling**: Promotes separation of concerns and reduces inter-class coupling.
- **Increased Reusability**: Makes components more reusable across systems.

### What DI Solves

Conventional approach: 

- When a class $A$ needs an instance of class $B$, $A$ is responsible for creating $B$ (e.g., using `new`) and introducing a hard dependency, hindering flexibility and testability.

### Core Components

1. **Client**: The class requesting the dependency.
2. **Injector**: Often, it's a framework or container supplying the dependency.
3. **Service**: The class fulfilling the dependency requirement.

### Types of Dependency Injection

1. **Constructor Injection**: The \textit{Injector} provides all the necessary dependencies through the constructor.
  
   **Pros**: 
   - The object is always in a valid state when returned.
   - It's clearer which dependencies are needed.

   **Cons**: 
   - Constructors can grow in size.
   - May not be suitable when there are optional or too many dependencies.

2. **Setter Method Injection**: The \textit{Injector} uses setter methods to inject the dependencies after the object is created.

   **Pros**: 
   - No need to extend classes to pass dependencies.
   - Doesn't inject dependencies that are not needed.

   **Cons**: 
   - The object can be in an inconsistent state until all dependencies are set.
   
3. **Field Injection**: Dependencies are provided directly into the fields or properties of the class.

   **Pros**: 
   - Often requires less boilerplate.
   - Can be the simplest form of DI setup.

   **Cons**: 
   - The class's dependencies are not immediately visible from its constructor or methods.
   - Difficult to ensure that dependencies are not `null`.
   - Breaks the encapsulation principle.

4. **Method Injection (Less Common)**: Dependencies are injected through methods that are called after object construction. While this fulfills the Dependency Injection criteria, it's less frequently seen in the Spring context.

   **Pros**: 
   - Allows for flexibility in when a dependency is needed.

   **Cons**: 
   - Increases complexity as clients need to manage when to call these methods.
   - Breaks encapsulation.

### Code Example: Dependency Injection Types

Here is the Java code:

```java
/* Constructor Injection */
public class ReportService {
    private final DatabaseRepository databaseRepository;

    public ReportService(DatabaseRepository databaseRepository) {
        this.databaseRepository = databaseRepository;
    }
}

/* Setter Method Injection */
public class ReportService {
    private DatabaseRepository databaseRepository;

    public void setDatabaseRepository(DatabaseRepository databaseRepository) {
        this.databaseRepository = databaseRepository;
    }
}

/* Field Injection - Directly assigns the dependency. */
@Component
public class ReportService {
    @Autowired
    private DatabaseRepository databaseRepository;
}

/* Method Injection - Injects the dependency via a method call post-construction. */
public class ReportService {
    private DatabaseRepository databaseRepository;

    public void injectDependency(DatabaseRepository databaseRepository) {
        this.databaseRepository = databaseRepository;
    }
}
```
<br>

## 6. What are _Bean Scopes_ in _Spring_? Name them.

**Spring** allows you to specify different **bean scopes**. Each scope serves a unique lifecycle and context interaction.

### Types of Bean Scopes

1. **Singleton** (default): A single bean instance is managed per container. This scope is suitable for stateless beans.

2. **Prototype**: A new instance per bean reference or look-up. This scope is beneficial for stateful beans.

3. **Request**: A single bean instance is tied to an **HTTP** request in a **web-aware** container. This scope is ideal for beans that are required within an HTTP request, such as web controllers.

4. **Session**: Physically represents a single user session in a **web-aware** container. Objects in this scope exist for as long as the HTTP **session** endures.

5. **Global Session**: Functions similarly to the **Session** scope, but is meant for **portlet-based environments**.

6. **Application**: Deprecated. Used to create a bean for the **lifetime of a **`ServletContext`** in **web-aware** containers.

7. **WebSocket**: Introduced in Spring 5.0, this scope is associated with the lifecycle of a WebSocket connection. The bean will remain in scope as long as the WebSocket connection is active. It's often used for managing attributes and operations related to the WebSocket session.
<br>

## 7. How do you configure a _bean_ in _Spring_?

In Spring, a **bean** represents an object that's managed by the Spring IoC (Inversion of Control) Container. Developers can configure beans using annotations, XML, or Java code.

### Key Bean Configuration Elements

- **Class**: Identifies the bean's type.
- **ID/Name**: Unique identifier for the bean within the IoC container.
- **Scope**: Describes the bean's lifecycle.
- **Dependencies**: Bean dependencies and corresponding wiring mode.
- **Additional Settings**: Custom properties and configurations.

### Annotation-Based Bean Configuration

Use the `@Component` family of annotations combined with `@Autowired`.

- **@Component**: Indicates the class as a Spring-managed component.
- **@Repository**: Specialized for data access.
- **@Service**: For business services.
- **@Controller**: Designed for MVC web applications.

**Example: Using @Component**

Here is the Java code:

```java
import org.springframework.stereotype.Component;

@Component
public class MyService {
    // Bean logic here
}
```

### XML-Based Bean Configuration

The traditional method uses XML configuration.

- **bean**: The XML element that defines a Spring bean.
- **id**: Unique identifier.
- **class**: Specifies the bean's class.

**Example: XML-Configured Bean**

Here is the XML code:

```xml
<beans xmlns="http://www.springframework.org/schema/beans"
       xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
       xsi:schemaLocation="http://www.springframework.org/schema/beans
       http://www.springframework.org/schema/beans/spring-beans.xsd">
       
    <bean id="myService" class="com.example.MyService"/>
</beans>
```

### Java Configuration

Modern Spring applications often favor Java-based configuration using `@Configuration` and `@Bean` annotations.

- **@Configuration**: Marks a class as providing bean configurations.
- **@Bean**: Identifies a method that returns a bean instance.

**Example: Java-Based Configuration**

Here is the Java code:

```java
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

@Configuration
public class AppConfig {
    @Bean
    public MyService myService() {
        return new MyService();
    }
}
```
<br>

## 8. Describe the role of the _Spring Core container_.

**Spring Core container** serves as the foundation for the Spring Framework, providing the **inversion of control** (IoC) and **dependency injection** (DI) mechanisms.

### Key Responsibilities

- **Lifecycle Management**: Ensures components' lifecycle through configuration and specific hooks (such as `@PostConstruct` and `@PreDestroy` annotations).
- **Configuration Management**: Enables bean definitions through XML, annotations, or Java-based configurations.
- **Context Metadata Management**: Houses metadata, like scoped dependencies and convenient mechanisms, such as expressions and environment configuration.
  
### Primary Components

- **BeanFactory**: The basic IoC container, which is primarily responsible for the instantiation and management of beans. It provides basic features of DI and doesn't support advanced features such as AOP.
- **ApplicationContext**: This is an advanced version of the BeanFactory. It incorporates all the functionalities of the BeanFactory and adds context-specific behavior. It's the prevalent choice for most scenarios and stands as the client's gateway into the Spring world. It provides additional features such as AOP, message source, and event publication.

### Registration and Location of Beans

- **Explicit Registration**: Developers can define beans either in XML configuration files or using annotations (like `@Component`, `@Service`, `@Repository`, `@Controller`) and Java-based configurations.
- **Implicit Registration**: Some classes in Spring are automatically detected and registered if the necessary annotations are present.

### Bean Scopes

The container manages a bean's lifecycle and its visibility from other beans based on the bean scope.

#### Primary Scopes

- **Singleton**: The default scope, where a single instance is managed per container.
- **Prototype**: Defines that a new instance should be created and managed each time it's requested.

#### Extended Scopes

- **Request**: In web-aware applications, a bean is created on each HTTP request.
- **Session**: Like 'Request' but scoped to an HTTP session.
- **Global Session**: Similar to 'Session' but applies to portlet-based applications.

### Custom Scopes

Developers can **define their own scopes** to handle specific requirements.

### IoC and DI Mechanisms

- **Inversion of Control (IoC)**: Refers to the pattern in which objects delegate the responsibility of their creation and management to another party (the IoC container).
- **Dependency Injection (DI)**: Describes the process of providing the dependencies (collaborating objects) to a component from an external source.

### Practical Benefits

- **Loose Coupling**: Components are less dependent on each other, which enhances system flexibility and maintainability.
- **Simplified Unit Testing**: Easier with singletons and DI, as you can mock or provide test-specific dependencies.
- **Centralized Configuration**: Configuration details are consolidated, simplifying management and reducing the likelihood of redundancy or inconsistencies.
- **Lifecycle Control**: Accurate and centralized bean lifecycle management.
- **Reduced Boilerplate**: Annotations and configurations streamline bean definitions and wiring.
<br>

## 9. What is a _Spring configuration file_?

**Spring configuration files** provide a way to configure Spring applications. These files, often written in XML, contain **bean definitions** and other configuration elements.

### Key Elements

- **Bean Definitions**: XML files define beans using `bean` elements, or annotation-based configurations can be used.

- **Module Configurations**: These files specify Spring modules to use, such as `context`, `mvc`, or `aop`.

- **External Configurations**: XML files can import other Spring configurations, often used in larger projects.

  ```xml
  <import resource="xyz.xml" />
  ```

### Roles and Responsibilities

- **Central Configuration Repository**: Provides a central place for application-specific and infrastructure-level configurations.

- **Dependency Injection Configuration**: Specifies dependencies and how they should be injected into beans.

- **Both In-Built Configuration and Custom Configurations**: Accommodates Spring's in-built configurations and custom configurations for applications and components.

- **External Configuration Import**: Allows the modular composition of application configurations.

### Best Practices

- **Separation of Concerns**: Divide configuration files based on the functional parts they control, such as one for data access and another for web components.

- **Consistency and Standardization**: Establish best practices across the team. This ensures that configurations are maintained uniformly.

- **Minimize Global Settings**: While Spring offers a global application context, it’s often better to have smaller, more focused contexts for specific application layers or modules.

### Code Example: Spring Configuration File

Here is the XML configuration:

```xml
<beans>
    <bean id="customerService" class="com.example.CustomerService" />
    <bean id="customerDAO" class="com.example.CustomerDAO">
        <property name="cassandraTemplate" ref="cassandraTemplate" />
    </bean>
    <bean id="cassandraTemplate" class="org.springframework.data.cassandra.CassandraTemplate">
        <constructor-arg name="session" ref="cassandraSession" />
    </bean>
    <bean id="cassandraSession" class="com.datastax.driver.core.Session" factory-method="connect" />
</beans>
```
<br>

## 10. How do you create an _ApplicationContext_ in a _Spring_ application?

The **ApplicationContext** serves as the core of the Spring IoC container and is fundamental to the setup of any Spring-based application.

### Ways to Create ApplicationContext

1. **`ClassPathXmlApplicationContext`**: Loads the XML file from the classpath.

    ```java
    ApplicationContext context = new ClassPathXmlApplicationContext("applicationContext.xml");
    ```

2. **`FileSystemXmlApplicationContext`**: Loads the XML file from the filesystem.

    ```java
    ApplicationContext context = new FileSystemXmlApplicationContext("path/to/applicationContext.xml");
    ```

3. **`XmlWebApplicationContext`**: Designed for web applications and loads XML from a specified web application context.

    ```java
    ApplicationContext context = new XmlWebApplicationContext();
    ```

4. **`AnnotationConfigApplicationContext`**: When employing Java configuration with `@Configuration` classes.

    ```java
    ApplicationContext context = new AnnotationConfigApplicationContext(AppConfig.class);
    ```

5. **`GenericApplicationContext`**: A flexible option for advanced customizations.

    ```java
    GenericApplicationContext context = new GenericApplicationContext();
    context.refresh(); // Call refresh manually
    ```

6. **`GenericXmlApplicationContext`**: Lets you control how the XML is read.

    ```java
    GenericXmlApplicationContext context = new GenericXmlApplicationContext();
    context.load("path/to/applicationContext.xml");
    context.refresh();
    ```

7. **`SilentModeApplicationContext`**: Quiets any startup messages.

    ```java
    ApplicationContext context = new SilentModeApplicationContext();
    ```

### Best Practices

- **Type Safety**: Opt for `AnnotationConfigApplicationContext` or Java-based `ApplicationConfig` whenever possible.
  
- **XML vs. Java-based Config**: XML is well-suited for larger, stable applications, while **Java-based** configurations offer better refactoring tools and compile-time safety.

- **Application Type Considerations**: Select the appropriate method based on the application's specifics - such as web-based apps or when flexibility for customizations is required.

- **Startup Control**: Certain `ApplicationContext` implementations allow for custom startup modes.
<br>

## 11. What is _Aspect-Oriented Programming (AOP)_?

**Aspect-Oriented Programming** (AOP) complements **Object-Oriented Programming** (OOP) by addressing **cross-cutting concerns**, like logging or security settings, that cut across disparate modules and classes.

AOP achieves modularity by:

- Identifying **cross-cutting concerns**
- Specifying **join points** in the code, where these concerns can be applied
- Defining **advice**, which provides actions to be taken at these points

For example, you might have logging code scattered throughout your application, triggered at various points. With AOP, you can consolidate this logic separately, marking points in the code (e.g., method calls) where logging should occur.

### Core Concepts

- **Aspect**: A module of classes containing related advice and join point definitions.

- **Join Point**: A point in the execution of the application (such as a method call or exception being thrown) that can be targeted by advice.

- **Pointcut**: A set of one or more join points targeted by a piece of advice.

- **Advice**: The action that should be taken at a particular join point (e.g., the code to execute before a method is called).

- **Introduction**: Allows adding new methods or attributes to existing classes dynamically.

- **Weaving**: The process of linking aspects to the execution of an application.

  - **Compile-Time Weaving**: Modifications are applied at compile time.
  - **Load-Time Weaving**: The aspect is applied when the class is loaded, often by a special class loader.
  - **Run-Time Weaving**: Changes are made during the execution of the code.

  The three weaving mechanisms can be further categorized into four strategies:

  - **Singleton weaving**: The aspect is a singleton and is woven into the client at most once.
  - **Per-instance weaving**: The aspect is woven into each object before it is returned.
  - **Single-time weaving**: The aspect is woven into the client the first time it is instantiated.
  - **Combination-of-above weaving**: A combination of the above strategies is used to achieve weaving.

- **Decoration**: Uses a **proxy** or **wrapper** to intercept method calls and apply cross-cutting concerns.

  - **Dynamic Proxy**: Java's `java.lang.reflect.Proxy` is often used.
  - **CGLIB**: A code generation library for high-performing and customized proxies.

### Benefits of AOP

- **Modularity**: AOP allows you to isolate cross-cutting concerns, reducing code redundancy and increasing maintainability.
- **Flexibility**: Concerns like security and auditing often require fine-grained control. AOP provides that.

- **Simplicity**: The application's core components can remain clean and focused on their main tasks.

### AOP in Spring

  - In Spring, AOP is integrated using **proxies** and **aspect J annotations** or XML configurations.
  - Spring relies on a **proxy-based** system for AOP, using either standard Java interfaces for proxy creation or bytecode modification with CGLIB.
  - You can choose **declarative or programmatic** AOP configuration styles.
<br>

## 12. How does _Spring_ support _AOP_?

**Spring** provides a powerful **Aspect-Oriented Programming (AOP)** framework. This framework simplifies the development process by enabling the separation of cross-cutting concerns from the main business logic. This leads to modular and more maintainable code.

### Core Concepts

#### Aspect
An **aspect** is a module encapsulating concerns such as logging or security. These typically cross-cut multiple application modules.

#### Join Point
A **join point** is a point during the execution of a program such as a method invocation.

#### Advice
The action taken by an aspect at a particular join point. Different types of advice include *before*, *after*, *around*, *after-returning*, and *after-throwing*.

#### Pointcut
A rule in AOP that defines the join points where advice should be applied. For example, you might specify that a particular advice applies to methods whose name begins with "get".

#### Introduction
The introduction allows adding new methods or attributes to a class.

#### Target object
The object being advised by one or more aspects.

### AOP Concepts in Practice

- **Proxy**: Spring AOP uses JDK dynamic proxies or CGLIB to generate a proxy for the target object. The proxy provides an opportunity to intercept the method invocations to apply the aspects. Spring applies the most appropriate proxy type, based on the context and the configuration.

- **Weaving**: This is the process of linking the aspects with the application objects. Weaving can be achieved at different points in the application life cycle, providing flexibility. Spring supports three weaving mechanisms:
  - **Compile-time weaving**: Aspects are incorporated into the application code during compilation.
  - **Load-time weaving**: Weaving takes place at the class loading time using a class loader to load the modified classes.
  - **Run-time weaving**: Weaving happens during runtime, either programmatically or using an agent.

### Code Example: AspectJ Around Advice

Here is the Java code:

```java
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.aspectj.lang.annotation.Aspect;
import org.aspectj.lang.annotation.Around;
import org.aspectj.lang.ProceedingJoinPoint;

@Aspect
@Configuration
public class SecurityAspect {

    @Bean
    public SecurityManager securityManager() {
        return new SecurityManager();
    }

    @Around("execution(* com.example.app.service.*.*(..))")
    public Object applySecurity(ProceedingJoinPoint joinPoint) throws Throwable {
        SecurityManager securityManager = securityManager();
        if (securityManager.isAuthenticated()) {
            return joinPoint.proceed();
        } else {
            throw new SecurityException("User is not authenticated.");
        }
    }
}
```
<br>

## 13. Can you explain a _Pointcut_ and an _Advice_ in _Spring AOP_?

A **Pointcut** in Spring AOP defines the join points matched for advice and determines the regions in an application where cross-cutting concerns are applied.

In AOP, a Pointcut serves as a **predicate** for identifying the locations in an application where advice is to be applied. These locations are primarily method executions but can also include field references and updates.

Certain attributes are fundamental to Pointcuts. For example, the **scope** of the application and the type of join points, for instance, the Start of the Method of execution or even well within its body.

### Using Pointcuts

To apply AOP in Spring, mark the points in your application where cross-cutting is required. From there, define methods using annotations such as `@Before`, `@After`, or `@Around` to apply the advice.

### Code Examples: Pointcuts & Advice

Here is the Java code:

```java
@Aspect
@Component
public class LoggingAspect {

    // Define Pointcuts
    @Pointcut("execution(* com.example.service.*.*(..))")
    public void serviceLayerExecution() {}

    @Pointcut("args(Long,..) && serviceLayerExecution()")
    public void serviceLayerMethodsWithLongArg() {}

    @Before("serviceLayerMethodsWithLongArg()")
    public void logServiceMethodWithLongArg(JoinPoint joinPoint) {
        System.out.println("Before method execution: " + joinPoint.getSignature().getName());
    }
}
```

In this example:

- `serviceLayerExecution()`: Matches the execution of any public method in a class present in the `com.example.service` package.
  
- `serviceLayerMethodsWithLongArg()`: Refines the previous pointcut, ensuring it only matches on methods with a Long parameter as the first argument.

- `logServiceMethodWithLongArg()`: This is the advice that's applied before the defined join points.

### Aspect-Oriented Programming in Spring

- **@AspectJ-style**: Employs aspect-oriented programming to define **Aspects**, The units that collect related **Pointcuts** and **Advices**.

- **@Annotation-based**: Permits selection of join points using annotations on the relevant modules. This method does not involve any pointcut expressions or direct exposure of AspectJ.

- **<a>XML-based</a> Configuration**: Flexibility for hooking up aspects in codebases without annotations.

### Type of Advices

**Join Points**: The specific times during program execution when **Advices** execute.

- **Before**: Runs the advice before the selected join point.
  
- **After**: Executes the advice after the join point; applicable for both successful and failing conditions.

- **Around**: Provides control over when and if the method proceeds to its natural execution. This enables advice to be run before and after the method execution.

- **AfterReturning**: Only runs when the join point method is successfully completed.

- **AfterThrowing**: Only applies when the join point method throws an exception.

### Additional Mechanisms to Aid Debugging

- **This()**: Refers to the current executing object.

- **Target()**: The target object for the method being advised.

- **Bean()**: The bean that owns the method being advised.

- **References to Execution Methods**: A shorthand for creating pointcuts. An example is `execution(public * com.example.SomeInterface.*(..))` to match the execution of any public method in classes implementing the `SomeInterface`.
<br>

## 14. What is a _Join Point_ in _Spring AOP_?

A **Join Point** in **Spring AOP** represents a specific point during the execution of a program, which could be targeted for additional functionality, such as before, after, or around method calls.

When a join point is intercepted by a Spring AOP advice, the advice can perform certain actions. The availability of join points differs across diverse AOP methodologies. For example, AspectJ provides more extensive join points than proxy-based AOP in Spring.

### Common Join Points

- **Method Execution**: This join point signifies execution of a method.
- **Method Call**: Denotes when the method is called from another location in the code.
- **Class Initialization**: Marks when a class is initialized.
- **Field Access**: Represents read or write actions on a field.
- **Instance Construction**: Indicates the instantiation of an object through a constructor.

These join points are consistent across AOP frameworks.

### AspectJ-Exclusive Join Points

- **Executions Involving Annotated Methods**: Targets the execution of methods marked with a specific annotation.
- **Executions in Specific Layers**: Directs actions to methods situated in defined layers or packages.

### Code Example: Join Points

Here is the Java code:

```java
public class SampleClass {
    private int sampleField;

    public void sampleMethod() {
        // Join Point: Method Execution or Method Call
        System.out.println("Sample Method Executed.");
    }

    public int getSampleField() {
        // Join Point: Field Access
        System.out.println("Getting Sample Field: " + sampleField);
        return sampleField;
    }

    public void setSampleField(int value) {
        // Join Point: Field Access
        System.out.println("Setting Sample Field: " + value);
        sampleField = value;
    }

    static {
        // Join Point: Class Initialization
        System.out.println("SampleClass Initialized.");
    }

    public SampleClass() {
        // Join Point: Instance Construction
        System.out.println("SampleClass Instance Created.");
    }
}
```
<br>

## 15. What is the difference between a _Concern_ and a _Cross-cutting Concern_ in _Spring AOP_?

**Cross-cutting concerns** are aspects of software development that affect the entire application, yet are largely kept separate from the core business logic. This separation improves the modularity, maintainability, and reusability of the codebase.

`Spring Aspect-Oriented Programming` (**AOP**) is tailored for managing cross-cutting concerns.

While a "concern" is a more general term, referring to anything that requires the application's attention, a "cross-cutting concern" specifically relates to the aspects that cut across different modules or layers of a software system.

### Examples of Cross-Cutting Concerns

- **Logging**: The need to log method invocations or business operations.
- **Security**: Centralized control for authentication and authorization mechanisms.
- **Caching**: Optimizing performance by caching the results of expensive operations.
- **Exception Handling**: Providing a consistent way to handle exceptions across the application.

The AOP approach of managing such concerns employs join points, pointcuts, and advice, and is separate from method-specific or local object concerns.

### Code Example: Cross-Cutting Concerns in Spring AOP

Here is the Java code:

**Bean Class:**

```java
public class MyBook {
    private String bookName;

    public String getBookName() {
        return bookName;
    }
    public void setBookName(String bookName) {
        this.bookName = bookName;
    }
}
```

**LogAspect:**

```java
@Aspect
public class LogAspect {

    @Before("execution(* MyBook.getBookName())")
    public void logMethodName(JoinPoint joinPoint) {
        System.out.println("Method invoked: " + joinPoint.getSignature());
    }

    @AfterReturning(pointcut = "execution(* MyBook.getBookName())", returning = "result")
    public void logReturnValue(JoinPoint joinPoint, Object result) {
        System.out.println("Returned: " + result);
    }

}
```

**AppConfig:**

```java
@Configuration
@EnableAspectJAutoProxy
public class AppConfig {

    @Bean
    public MyBook myBook() {
        return new MyBook();
    }

    @Bean
    public LogAspect logAspect() {
        return new LogAspect();
    }

}
```
<br>



#### Explore all 100 answers here 👉 [Devinterview.io - Spring](https://devinterview.io/questions/web-and-mobile-development/spring-interview-questions)

<br>

<a href="https://devinterview.io/questions/web-and-mobile-development/">
<img src="https://firebasestorage.googleapis.com/v0/b/dev-stack-app.appspot.com/o/github-blog-img%2Fweb-and-mobile-development-github-img.jpg?alt=media&token=1b5eeecc-c9fb-49f5-9e03-50cf2e309555" alt="web-and-mobile-development" width="100%">
</a>
</p>

