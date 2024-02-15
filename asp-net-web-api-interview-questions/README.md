# 100 Must-Know ASP.NET Web API Interview Questions

<div>
<p align="center">
<a href="https://devinterview.io/questions/web-and-mobile-development/">
<img src="https://firebasestorage.googleapis.com/v0/b/dev-stack-app.appspot.com/o/github-blog-img%2Fweb-and-mobile-development-github-img.jpg?alt=media&token=1b5eeecc-c9fb-49f5-9e03-50cf2e309555" alt="web-and-mobile-development" width="100%">
</a>
</p>

#### You can also find all 100 answers here 👉 [Devinterview.io - ASP.NET Web API](https://devinterview.io/questions/web-and-mobile-development/asp-net-web-api-interview-questions)

<br>

## 1. What is _ASP.NET Web API_ and what is it used for?

**ASP.NET Web API** is a lightweight, framework that's integrated with **ASP.NET MVC** to create RESTful HTTP services, reaching a broad range of client devices. Utilizing **ASP.NET Web API** is an adept selection to serve the specific needs of your web application.

### Use Cases

- **Data Services**: It's perfect for applications that need to expose and manipulate data over the web.
- **Mobile Applications**: Ideal for back-end support of multi-platform applications, especially REST APIs.
- **Single Page Applications (SPAs)**: Effortlessly integrate with modern JavaScript frameworks for SPAs.
- **Real-time Applications**: Services like signalR provides real-time communication. Web API can be used to build a real-time API, which apps can consume for real-time data.

### Core Advantages

- **Ease of Development**: Utilizes familiar features from ASP.NET, rendering it simpler to develop.
- **Loose Coupling**: Supports **HTTP** services, forming a loosely coupled framework.
- **Robust Routing**: Employs in-depth routing mechanisms for **HTTP requests**.
- **Content Negotiation**: Automatically selects the most fitting response format.
- **Model Binding**: Directly binds incoming HTTP requests to the specified model or action parameters.
- **Action Results**: Provides numerous kinds of action results for handling different responses.
- **Authentication**: Offers multiple levels of data access security.
- **Testing**: Facilitates direct testing of the API in a dedicated test client.

### Code Example: ASP.NET Web API 

The standard **GET** operation for a Web API controller to retrieve a product list from a server:

```csharp
public IEnumerable<Product> GetProducts()
{
    return productsRepository.All;
}
```

The code snippet below displays the creation of an **ASP.NET Web API** controller. 

```csharp
public class ProductsController : ApiController
{
    public IEnumerable<Product> GetProducts()
    {
        return productsRepository.All;
    }
    public Product GetProductById(int id)
    {
        return productsRepository.Find(id);
    }
    public HttpResponseMessage PostProduct(Product product)
    {
        productsRepository.Add(product);
        var response = Request.CreateResponse(HttpStatusCode.Created, product);
        string uri = Url.Link("DefaultApi", new { id = product.Id });
        response.Headers.Location = new Uri(Request.RequestUri, uri);
        return response;
    }
    public void PutProduct(int id, Product product)
    {
        product.Id = id;
        if (!productsRepository.Update(product))
        {
            throw new HttpResponseException(HttpStatusCode.NotFound);
        }
    }
    public void DeleteProduct(int id)
    {
        productsRepository.Remove(id);
    }
}
```

### Key Takeaways

- **ASP.NET Web API**, an adaptable framework, excels in developing RESTful services for a wide array of consumer devices, applications, and platforms.
- Its **robust protocol support** and **manifold libraries** ensure the best possible service for developers who seek to implement modern web service principles.
<br>

## 2. How does _ASP.NET Web API_ differ from _WCF_ and _ASP.NET MVC_?

**ASP.NET Web API**, **WCF**, and **ASP.NET MVC** are all web frameworks with distinct purposes and target audiences.

### Key Distinctions

#### ASP.NET Web API

- **Purpose**: Designed for building HTTP services accessed by various clients, including browsers and mobile devices.
- **Data Format**: Typically deals with JSON, offering flexibility in handling data formatting. It can convey data through XML as well.
- **Routing**: Emphasizes RESTful services; routes often mirror the structure of resources.
- **State Management**: Functions independently of View State, implementing statelessness.


#### ASP.NET MVC

- **Purpose**: Intended for web application development, targeting browsers as the primary client.
- **Data Format**: Directs data flow to Views, which typically receive data in ViewModel classes. While it can implement AJAX for JSON, it is more oriented toward HTML,
- **Routing**: Capable of supporting both RESTful routing and more traditional URL-based routing.
- **State Management**: Utilizes View State for maintaining state during requests.

#### WCF

- **Purpose**: Focused on building distributed systems using various communication protocols, of which HTTP is just one.
- **Data Format**: Offers diversified data encodings and supports data contracts for tailoring message formats.
- **Routing**: Provides more extensive routing capabilities, suitable for diverse communication strategies.
- **State Management**: Features comprehensive state management tools, such as session handling.

### Choose the Right Tool for the Job

When it comes to building modern, HTTP-centric applications, **ASP.NET Web API** is usually the go-to platform. For scenarios necessitating robust message-level control and support for numerous communication protocols, **WCF** is the better choice. If your aim is to construct web applications that primarily interact with browsers and emphasize data-driven Views, **ASP.NET MVC** remains a strong contender.
<br>

## 3. Explain _RESTful services_ and how they relate to _ASP.NET Web API_.

**ASP.NET Web API** acts as a communication bridge between clients and servers through **RESTful** services, offering a web-friendly framework for data manipulation.

### What is RESTful Service?

**REST** establishes a set of constraints that focus on stateless connections and standardized operations across resources. Services are accessed via the standard HTTP methods:

- **GET**: Retrieve data.
- **POST**: Submit new data.
- **PUT**: Update existing data.
- **DELETE**: Remove data.

### Common Elements

#### Resources

Resources such as APIs are identified by **Uniform Resource Identifiers (URIs)**, serving as unique "addresses" for each resource.

#### Verbs

Standard HTTP verbs define basic operations. **GET** retrieves data, **POST** creates new records, **PUT** updates them, and **DELETE** removes them.

#### Representation

Data within a service is represented in a **format**, such as JSON or XML.

### ASP.NET Web API for RESTful Services

- **Attribute-Based Routing**: The framework lets developers define URI structures through attributes.
- **Model Binding**: Automatically extracts parameters from the request, enhancing ease of use.
- **Content Negotiation**: Allows for dynamic response formatting based on client needs, enabling multiple output formats like JSON or XML.
- **ActionResult**: A flexible return type that simplifies response handling, e.g., returning different HTTP status codes.

### Code Example: RESTful Services

In this example, a music library supports all standard CRUD (Create, Read, Update, Delete) operations.

```csharp
public class MusicController : ApiController
{
    private List<Music> musics = new List<Music>();

    // GET: api/Music
    public IEnumerable<Music> GetAll()
    {
        return musics;
    }

    // GET: api/Music/5
    public Music Get(int id)
    {
        return musics.FirstOrDefault(m => m.Id == id);
    }

    // POST: api/Music
    public IHttpActionResult Post(Music music)
    {
        musics.Add(music);
        return CreatedAtRoute("DefaultApi", new { id = music.Id }, music);
    }

    // PUT: api/Music/5
    public void Put(int id, Music music)
    {
        var existing = musics.FirstOrDefault(m => m.Id == id);
        if (existing != null)
        {
            existing = music;
        }
    }

    // DELETE: api/Music/5
    public IHttpActionResult Delete(int id)
    {
        var music = musics.FirstOrDefault(m => m.Id == id);
        if (music != null)
        {
            musics.Remove(music);
            return Ok();
        }
        return NotFound();
    }
}

public class Music
{
    public int Id { get; set; }
    public string Title { get; set; }
    public string Artist { get; set; }
    public int Duration { get; set; }
}
```

**Note**: This is a highly simplified example. In a production scenario, a service like this would be backed by an actual data store or database.

Here is the ASP.NET Razor form:

```csharp
   public IActionResult Index()
        {
            return View();
        }
```

- For **GET**, detail of all music objects can be obtained from the service by calling `GET: /api/Music`.
- For **POST**, new `Music` objects can be added by calling `POST: /api/Music` with the details of the object in the request body.
- For **PUT**, the ID of the existing `Music` object is specified in the URL, and the updated `Music` object is sent in the request body, using URI path segment (`/api/music/5`) and the HTTP method `PUT`.
- For **DELETE**, the ID of the `Music` object to be deleted is specified in the URL, using the URI path segment and the HTTP method `DELETE`.

 The `ApiController`s actions are automatically routed based on the HTTP method and the structure of the requested URI.

For example, if a client sends an HTTP GET request to `/api/music/5`, the framework will invoke the `Get` action method on the `MusicController`, passing the `id` value of `5` from the request URI. Similarly, if a client sends a POST request to `/api/music`, the framework will invoke the `Post` method, and so on.
<br>

## 4. What are _HTTP verbs_ and how are they used in _Web API_?

**HTTP Verbs**, also known as methods, dictate the type of action to be taken on a resource by a web server or Web API. They convey semantical meanings and operate on standard CRUD operations.

### Commonly Used Verbs

- **GET**: Fetches one or more resources.
- **POST**: Creates new resources, often with server-defined IDs.
- **PUT**: Updates an existing resource or creates a new one.
- **PATCH**: Partially updates an existing resource.
- **DELETE**: Removes the specified resource.

### Code Example: Web API Controller Endpoints

Here is the C# code:

```csharp
using System.Net;
using System.Web.Http;

public class ProductsController : ApiController
{
    // GET /api/products
    public IHttpActionResult Get()
    {
        // Retrieve and return all products.
    }

    // GET /api/products/5
    public IHttpActionResult Get(int id)
    {
        // Retrieve and return the product with the specified ID.
    }

    // POST /api/products
    public IHttpActionResult Post(Product product)
    {
        // Create a new product using POST data and return its location.
        return CreatedAtRoute("DefaultApi", new { id = product.Id }, product);
    }

    // PUT /api/products/5
    public IHttpActionResult Put(int id, Product product)
    {
        if (id != product.Id)
            return BadRequest("ID mismatch");

        // Update the product with the specified ID.
    }

    // PATCH /api/products/5
    public IHttpActionResult Patch(int id, Delta<Product> product)
    {
        // Apply partial updates to the product with the specified ID.
    }

    // DELETE /api/products/5
    public IHttpActionResult Delete(int id)
    {
        // Delete the product with the specified ID.
    }
}
```
<br>

## 5. How do you create a basic _Web API controller_?

To create a Web API controller, you need to follow these steps:

1. **Use the Right Class Attribute**
   The `ApiController` attribute is essential to differentiate a regular controller from a Web API controller.

2.  **Define Class and Methods**
    Use methods like `Get`, `Post`, `Put`, and `Delete` to map HTTP verbs to controller actions.

3. **Code Example**: Web API Controller
    
   Here is the C# code:

   ```csharp
   using System.Collections.Generic;
   using System.Web.Http;

   public class ProductsController : ApiController
   {
       // GET: api/Products
       public IEnumerable<string> Get()
       {
           return new string[] { "product1", "product2" };
       }

       // GET: api/Products/5
       public string Get(int id)
       {
           return "product with ID " + id;
       }

       // POST: api/Products
       public void Post([FromBody]string value)
       {
           // add a product
       }

       // PUT: api/Products/5
       public void Put(int id, [FromBody]string value)
       {
           // update a product
       }

       // DELETE: api/Products/5
       public void Delete(int id)
       {
           // delete a product
       }
   }
   ```
<br>

## 6. Describe _routing_ in _ASP.NET Web API_.

**Routing in ASP.NET Web API** enables you to map HTTP requests to specific Controller and Action methods, much like traditional MVC routing. It typically uses the `WebApiConfig` and benefits from attribute-based routing.

### Route Setup

1. **Controller Route**: The route to the entire controller. This is set up in the `WebApiConfig.cs` file.

   **Example**: `api/{controller}/{id}`. In URI it would look like: `/api/products/10`.

2. **Action Route**: The route specific to an action method. This is configured via attributes on the action methods.

    **Example**: `[HttpGet("action/{id}")]`. In URI it would look like `/api/products/action/10`.

3. **HTTP Verb & Route**: Both the HTTP Verb and the route need to match for the request to be dispatched to the corresponding action.

    **Example**: `[HttpGet("specificAction/{id}")]`.

4. **Default Values**: They are useful for providing defaults for route parameters and differentiating the route from others. 

    **Example**: `[HttpGet("actionWithDefault/{id:int=5}")]`.

5. **Additional Constraints**: These can be added to route parameters for **value pattern matching** and are especially useful for avoiding ambiguity between different routes.

    **Example**: `[Route("{id:int:range(1, 10)}")]`.

6. **Route Prefix**: This is used via the `[RoutePrefix]` attribute at the controller level. It allows you to set up a common route prefix for all action methods within the controller. This is often used for versioning.

    **Example**: `[RoutePrefix("api/V2/products")].`

### Route Mapping Scenarios

#### Controller Routes

- **Default**: All action methods within the controller will use this default route.
- **Custom**: You can use the `[Route]` attribute to specify a custom route for the entire controller.

#### Action Routes

- **Custom**: Assign a custom route to the action method by using the `[Route]` attribute.
- **Override**: Using the `[Route]` attribute provides the ability to override any route conventions set at the controller level.

#### Code Example: Route Setup

Here are the code examples:

Controller class:

```csharp
[RoutePrefix("api/products")]
public class ProductsController : ApiController
{
    //Matches GET api/products/1
    [Route("{id:int:min(1)}")]
    public HttpResponseMessage GetProduct(int id)
    {
    }
    
    //Matches GET api/products/category/123
    [Route("category/{id}")]
    public HttpResponseMessage GetByCategory(string id)
    {
    }
    
    [HttpGet]
    public List<Product> GetAllProducts()
    {
    }
}
```

`WebApiConfig.cs`:

```csharp
public static class WebApiConfig
{
    public static void Register(HttpConfiguration config)
    {
        config.MapHttpAttributeRoutes();
        config.Routes.MapHttpRoute(
            name: "DefaultApi",
            routeTemplate: "api/{controller}/{id}",
            defaults: new { id = RouteParameter.Optional }
        );
    }
}
```
<br>

## 7. How are requests mapped to actions in _Web API_?

In **ASP.NET Web API**, actions are identified using combination of HTTP method and request URL.

### Action Selection

Web API action selection occurs in two stages:

  1. **Mapping to Resource**: The request URL, including query string, if any, is used to route to a particular resource.
  2. **Mapping to Action**: The HTTP verb (GET, POST, etc.) of a request further directs the routing to a specific action on the resource.

This two-tiered approach is designed to mirror the RESTful resource structure and the HTTP method semantics.

#### Route Generation and Matching

Web API leverages the powerful ASP.NET routing engine to **parse the request URL** and match the URL against a Route Table to identify the relevant resource.

#### Attribute and Conventional Routing

Web API supports both attribute-based and conventional routes for mapping URL segments to actions and controllers.

- **Attribute Routing**: Methods are adorned with `[Route]` attributes, explicitly defining the URL template.
- **Conventional Routing**: URL paths are matched using a set of conventions.

While **Attribute Routing** provides more granular control, **Conventional Routing** is often simpler to set up.

### Code Example: Action Selection

Here is the C# code:

```csharp
public class EmployeesController : ApiController
{
    [Route("api/employees/{id}")]
    public Employee GetEmployee(int id)
    {
        // Retrieve employee based on ID
    }

    [Route("api/employees")]
    public IEnumerable<Employee> GetEmployees()
    {
        // Retrieve all employees
    }

    [Route("api/employees")]
    public IHttpActionResult PostEmployee(Employee employee)
    {
        // Add new employee
    }
}
```
<br>

## 8. What is _content negotiation_ in the context of _Web API_?

**Content negotiation** and **media type formatter** together allow **ASP.NET Web API** to serve multiple responses based on the requesting client's preferences. Content negotiation also permits matching requested media types to the configured formatters and serializing the response data accordingly.

You can choose between two approaches for **content negotiation**: **media type mapping** and **accept headers**. 

### Media Type Mapping

This straightforward method associates a media type with its corresponding `JsonFormatter` or `XmlFormatter`:

- **Pros**  
  - It's simple and explicit.
  - Useful when clients cannot or do not provide `Accept` headers.
- **Cons**  
  - It can't handle complex client requests that involve `q-values` (quality values indicating client preferences).

### Accept Headers

This approach uses HTTP `Accept` headers sent by the client:

- **Pros**  
  - It can handle more advanced client preferences.
- **Cons**  
  - Requires client support.

### Code Example: Content Negotiation

Here is the C# code:

The `MapHttpRoute` method specifies:

- **routeTemplate**: The URI to handle (`api/products/{id}`)
- **defaults** object: the route's default values, including the controller (`Products`).

The `config.Formatters.JsonFormatter` and `config.Formatters.XmlFormatter` lines handle media type formatting for JSON and XML, respectively.

```csharp
public static void Register(HttpConfiguration config)
{
    // Web API configuration and services
    config.MapHttpRoute(
        name: "DefaultApi",
        routeTemplate: "api/{controller}/{id}",
        defaults: new { id = RouteParameter.Optional }
    );

    // Remove the XML formatter
    config.Formatters.Remove(config.Formatters.XmlFormatter);
}
```
<br>

## 9. What data formats does _Web API_ support by default for response data?

**ASP.NET Web API** primarily operates with **JSON** and **XML** data formats. While JSON is more common due to its lighter-weight and growing popularity, both formats offer distinct advantages.

### JSON 

- **Simplicity**: Ideal for fast and straightforward data transfer.
- **Data-Type Flexibility**: Values can be strings, numbers, arrays, or objects.
- **Readability**: Human-readable and easily parsed.

### XML

- **Data Structure**: Especially suited for intricate data hierarchies due to its tree-like structure.
- **Data Verification**: XML supports schemas for automatic data verification.
- **Support for Unstructured Data**: XML can handle unstructured datasets better than JSON.
<br>

## 10. How do you secure a _Web API_?

Ensuring **security** for your Web API is crucial for safeguarding the data and resources it exposes.

### Key Security Measures

1. **Authentication**: Verify the identity of the client making the API request. Common methods include:
    - **HTTP Basic Authentication**: Employing a username and password, but it's less secure as credentials travel with every request.
    - **Token-based Authentication**: Using a short-lived token (such as JWT) that the client presents with each request. This is preferred for stateless and mobile applications. The token can be obtained through a separate authentication process.
    - **Auth0, IdentityServer, or Custom Providers**: Advanced platforms or custom solutions that offer various authentication methods and workflows.

2. **Authorization**: After authenticating the client, ensure they have the necessary permissions to access specific resources. Common strategies include:
    - **Role-Based Access Control (RBAC)**: Define roles such as 'admin', 'user', and 'guest'. Each role has specific permissions.
    - **Attribute-Based Access Control (ABAC)**: Access is determined based on attributes (such as user age or region) rather than predefined roles.

3. **Data Protection**: Ensure the confidentiality and integrity of the data transmitted. This can be achieved using HTTPS.

4. **IP Whitelisting**: For added security, limit API access to specific IP addresses or IP ranges.

### Code Example: Token-Based Authentication in ASP.NET Core Web API

```csharp
public void ConfigureServices(IServiceCollection services)
{
    services.AddAuthentication(JwtBearerDefaults.AuthenticationScheme)
       .AddJwtBearer(options =>
       {
           options.TokenValidationParameters = new TokenValidationParameters
           {
               ValidateIssuer = true,
               ValidateAudience = true,
               ValidateLifetime = true,
               ValidateIssuerSigningKey = true,
               ValidIssuer = Configuration["Jwt:Issuer"],
               ValidAudience = Configuration["Jwt:Audience"],
               IssuerSigningKey = new SymmetricSecurityKey(Encoding.UTF8.GetBytes(Configuration["Jwt:Key"]))
           };
       });
}

[Authorize]
[ApiController]
[Route("api/[controller]")]
public class ValuesController : ControllerBase
{
    // Controller actions
}
```

In this example, the `[Authorize]` attribute specifies that only authenticated requests are allowed. The `AddJwtBearer` method configures JWT token validation parameters.
<br>

## 11. How can you host an _ASP.NET Web API_ application?

When it comes to hosting an **ASP.NET Web API** application, the main options include **IIS**, **Self-Hosting**, and **Cloud Platforms**.

### IIS Hosting

- **Benefits**: Easy to Secure, Well-Integrated with Windows Identity Foundation, and Offers GUI Management Tools.
- **Considerations**: Requires Admin Rights, Only Available on Windows Servers, May Have Performance Overhead.
- **How?**: Simply create a Web API application using Visual Studio and then deploy it on an IIS server. Configure web.config settings based on the server environment.

### Self-Hosting

- **Benefits**: Can Be Hosted on Non-Windows Platforms, Such as Linux, or Embedded Systems, Does Not Require Admin Rights.
- **Considerations**: Manual Management Needed, Requires Sufficient System Resources for Ensured Uptime, a Bit Risky for Production Environments.
- **How?**: Use `Owin` or `WebListener` to self-host. The application can be kept running through the `HttpSelfHostServer`. Ensure server resources are sufficient to keep the application running without interruptions.

### Cloud Platforms

- **Benefits**: Scalability, High Availability, Reduced Maintenance, Multi-Region Deployment for Global Accessibility.
- **Considerations**: May Increase Dependency on Provider, Data Location and Compliance Issues, Requires Internet Connectivity.
- **How?**: Deploy on cloud platforms such as Microsoft Azure or AWS by creating cloud services or using container technologies like Kubernetes and Docker. Cloud platforms often have integration with VS for easy deployments. Just select the Cloud platform in Visual Studio Publishing wizard and it will handle the deployment steps.
<br>

## 12. What is _OWIN_ and how does it relate to _Web API_?

**OWIN** (Open Web Interface for .NET) is a standard for creating **web servers** and **web applications**. Unlike traditional web servers, OWIN applications are decoupled from the server hosting them.

**Katana** is the OWIN-compliant middleware from Microsoft that enables web applications and hosts to communicate using standard interfaces.

### Key Concepts

- **Middleware**: Core OWIN component that processes HTTP requests.
- **Application**: Comprises the middleware pipeline and processes these requests.
- **Server**: Listens for HTTP requests and passes them to the OWIN application.

### Benefits

- **Flexibility**: Middleware is interchangeable, allowing for ecosystem freedom.
- **Scalability**: Performant servers and web applications are possible due to OWIN's streamlined interface.

### OWIN and Web API

OWIN empowers Web API with modularity and more streamlined request handling. 

**Web API 2** and **ASP.NET Identity** benefit from OWIN integration.
- Simplified configuration thanks to OWIN startup classes.
- Enhanced request/response pipeline control via middlewares.

### OWIN vs. Katana

While OWIN is the interface specification, Katana is a concrete implementation by Microsoft. It most notably introduces several OWIN components :

- **Microsoft.Owin.dll**: Provides OWIN core libraries.
- **Microsoft.Owin.Hosting.dll**: Enables hosting in different environments.
- **Microsoft.Owin.Host.HttpListener.dll**: A simple self-hosting option using the `HttpListener` class.

### Code Example: OWIN Startup

Here is the C# code:

```csharp
using Owin;

public class Startup
{
    public void Configuration(IAppBuilder app)
    {
        app.UseWelcomePage();
    }
}
```
<br>

## 13. Explain the difference between _self-hosting_ and _IIS hosting_ in _Web API_.

When deploying a **Web API**, you have the option of **self-hosting** it using a custom application or using the more common **IIS hosting**. Each approach has its unique advantages and considerations.

### Self-Hosting

In this approach, you bypass IIS and run the Web API through a **custom process**.

#### Advantages

- **Platform Independence**: It can be run on any host supporting the .NET framework.
- **Complete Control**: You can perform finer configuration and monitoring tailored to your application.
- **Portability**: Useful for packing up an application into a private, self-executable unit.

#### Considerations

- **Maintenance Responsibility**: You are accountable for the application's lifetime and resource management.
- **Missing IIS Features**: Some services offered by IIS, such as load balancing, aren't available.
- **Security Configuration**: You need to manage security context for the application manually.

#### Sample Code: Self-Hosted Web API

Here is the C# code:

```csharp
public class Program
{
    static void Main()
    {
        using (WebApp.Start<Startup>("http://localhost:9000"))
        {
            Console.WriteLine("Web API hosted on http://localhost:9000/");
            Console.ReadLine();
        }
    }
}

public class Startup
{
    public void Configuration(IAppBuilder appBuilder)
    {
        var config = new HttpConfiguration();
        config.Routes.MapHttpRoute("default", "{controller}/{id}", new { id = RouteParameter.Optional });
        appBuilder.UseWebApi(config);
    }
}
```

### IIS Hosting

In IIS hosting, the Web API runs as part of the **IIS worker process**. This means it's managed by IIS in several ways.

#### Advantages

- **Enhanced Security**: IIS takes care of security features, allowing you to concentrate on developing API logic.
- **Configurable**: Configuration settings, such as load balancing and scaling, can be easily managed through IIS.

#### Considerations

- **IIS Dependency**: This restricts the environment to Windows servers with IIS installed.
- **Less Customization**: Fine-tuned customizations, especially for monitoring, can be more complex.

#### Sample Code: IIS Hosted Web API

C# Code for Web API:

```csharp
[Route("api/[controller]")]
[ApiController]
public class ValuesController : ControllerBase
{
    [HttpGet]
    public IEnumerable<string> Get()
    {
        return new string[] { "value1", "value2" };
    }
}
```

Configure IIS for Web API:

- Create a new website in IIS.
- Point the website to the Web API project's root folder.
- Ensure the Application Pool selected for the API is at least .NET 4.5 Integrated.
<br>

## 14. How do you configure _CORS_ in _Web API_?

**Cross-Origin Resource Sharing (CORS)** is essential for secure, client-server communication in web applications, especially when dealing with resources from different origins.

In the context of **ASP.NET Web API**, enabling CORS involves a few steps.

### Configuration Steps

1. **Install the `Microsoft.AspNet.WebApi.Cors` NuGet package** to your project if not already installed.

   ```powershell
   Install-Package Microsoft.AspNet.WebApi.Cors
   ```

2. **Configure CORS in Web API**: You can do this in either WebApiConfig.cs or directly in the controller.

    Here is the C# code:
    ```csharp
    public static class WebApiConfig
    {
        public static void Register(HttpConfiguration config)
        {
            var cors = new EnableCorsAttribute("http://localhost:8080", "*", "*");
            config.EnableCors(cors);
            
            // Other Web API configuration code...
        }
    }
    ```

    or directly in the controller:

    ```csharp
    [EnableCors(origins: "http://example.com", headers: "*", methods: "*")]
    public class YourController : ApiController
    {
        // Controller methods...
    }
    ```

    Let me know if you are looking for a specific one.

3. **Test Your CORS Configuration**: After making necessary changes, consider testing the server's CORS configuration using a web client such as Postman, cURL, or a browser console.
<br>

## 15. What is _attribute routing_ and how does it improve the _Web API_?

**Attribute routing** enables you to define Web API routes directly in the controller with attributes, giving you a more tailored, readable, and maintainable Web API infrastructure.

### Why Use Attribute Routing?

Traditional Web API route configuration relies on centralized route mapping, typically defined in a separate `RouteConfig` file or within the `WebApiConfig` file.

While this approach is both familiar and easy to set up, it can become cumbersome as your application scales. Attribute routing offers **improved flexibility**, **granular control**, and **maintenance simplicity** by allowing for route-to-action mapping at the method level, directly within the controller, using intuitive and dedicated attributes.

### Key Attributes for Route Configuration

- **RoutePrefix**: Serves as a prefix for all routes defined within a specific controller. This makes it easy to group related actions under a shared route segment.
- **Route**: Marks individual actions with specific route templates.

### Code Example: Attribute Routing

Here is the controller code:

```csharp
[RoutePrefix("api/books")]
public class BooksController : ApiController
{
    // Route will be: api/books
    [Route("")]
    public IHttpActionResult GetBooks() { /* ... */ }

    // Route will be: api/books/5
    [Route("{id:int}")]
    public IHttpActionResult GetBookById(int id) { /* ... */ }

    // Route will be: api/books/5/author
    [Route("{id:int}/author")]
    public IHttpActionResult GetBookAuthor(int id) { /* ... */ }

    // Route will be: api/books
    [Route("")]
    public IHttpActionResult PostBook([FromBody] Book book) { /* ... */ }

    // Route will be: api/books/5
    [Route("{id:int}")]
    public IHttpActionResult PutBook(int id, [FromBody] Book book) { /* ... */ }

    // Route will be: api/books/5
    [Route("{id:int}")]
    public IHttpActionResult DeleteBook(int id) { /* ... */ }
}
```

In the above example:

- The `RoutePrefix` attribute, on the `BooksController`, declares a common prefix for all methods within `BooksController`.
- The `Route` attributes, on individual methods, define specific route templates relative to the prefix set by `RoutePrefix`.

### Combining Attribute and Conventional Routing

You can use both **attribute routing** and **conventional routing** in the same application, but it's generally preferred to stick to one method for consistency and to avoid potential confusion.
<br>



#### Explore all 100 answers here 👉 [Devinterview.io - ASP.NET Web API](https://devinterview.io/questions/web-and-mobile-development/asp-net-web-api-interview-questions)

<br>

<a href="https://devinterview.io/questions/web-and-mobile-development/">
<img src="https://firebasestorage.googleapis.com/v0/b/dev-stack-app.appspot.com/o/github-blog-img%2Fweb-and-mobile-development-github-img.jpg?alt=media&token=1b5eeecc-c9fb-49f5-9e03-50cf2e309555" alt="web-and-mobile-development" width="100%">
</a>
</p>

