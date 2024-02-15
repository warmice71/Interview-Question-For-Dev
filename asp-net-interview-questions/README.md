# Top 100 ASP NET Interview Questions

<div>
<p align="center">
<a href="https://devinterview.io/questions/web-and-mobile-development/">
<img src="https://firebasestorage.googleapis.com/v0/b/dev-stack-app.appspot.com/o/github-blog-img%2Fweb-and-mobile-development-github-img.jpg?alt=media&token=1b5eeecc-c9fb-49f5-9e03-50cf2e309555" alt="web-and-mobile-development" width="100%">
</a>
</p>

#### You can also find all 100 answers here 👉 [Devinterview.io - ASP NET](https://devinterview.io/questions/web-and-mobile-development/asp-net-interview-questions)

<br>

## 1. What is _ASP.NET_ and how does it differ from classic _ASP_?

**Active Server Pages** (ASP) and **ASP.NET** are web application frameworks from Microsoft, each with its unique characteristics and features.

### Commonalities

- Serve as a middleware between web servers and dynamic web content.
- Use server-side languages for dynamic content generation and database interaction.
- Facilitate creation of interactive interfaces and sophisticated web applications.

### Key Features

#### Classic ASP

- **Code Execution**: Relies on an interpreter for languages like VBScript or JScript.
- **Performance Monitoring**: Lacks built-in mechanisms for performance tracking.
- **Multilanguage Support**:  Limited to languages supported by the scripting engines.
- **Object Model**:  Employs a limited set of objects such as `Request` and `Response` for web-related tasks.
- **Data Handling**: Requires direct interaction with ADO (ActiveX Data Objects).
- **Security**: Prone to potential vulnerabilities such as SQL injection.

#### ASP.NET

- **Code Execution**: Translates high-level languages like C# or Visual Basic into an intermediate language (IL) that's executed by the .NET Common Language Runtime (CLR).
- **Performance Monitoring**: Offers rich tools like Application Insights for real-time monitoring and diagnostics.
- **Multilanguage Support**:  Comprehensive support for all languages in the .NET ecosystem.
- **Object Model**: Employs a rich set of classes in the .NET Framework, enabling modular, object-oriented development.
- **Data Handling**: Simplifies data access with technologies like Entity Framework and LINQ.
- **Security**: Integrates with ASP.NET Identity, mitigating common security risks.
<br>

## 2. Describe the _ASP.NET page life cycle_.

The **ASP.NET Page Life Cycle** governs the stages a web form undergoes, from initialization to loading and rendering, and finally disposal.

### Key Events in the Page Lifecycle

1. **Page Request**: This sets in motion the entire page lifecycle.
2. **Start**: The page's structure is established.
3. **Initialization**: Components, such as master pages, are initialized.
4. **Load**: Data binding and loading occur.
5. **Postback**: Handles form submissions and validations.
6. **Rendering**: The page is transformed to HTML.
7. **Unload**: The page is disassociated from the server.

### Understanding the Stages

- **PreInit**: During this step, the page initializes components like master pages and themes.
- **Init**: The page sets properties that might be modified or reset later.
- **InitComplete**: Any post-init tasks are completed.
- **PreLoad**: Actions that need to be completed before the page and its controls are loaded can be executed during this step.
- **LoadComplete**: After this stage, all page life cycle events might have completed.
- **PreRender**: Actions before the page or any of its controls are rendered are performed here.
- **SaveStateComplete**: ViewState, form data, and other data relevant to the page are saved. After this stage, if a postback is performed, any data from the current page instance is lost, and the page reverts to the version that was saved during this lifecycle stage.

### Code Example

Here is the C# code:

```csharp
public partial class PageLifeCycle: System.Web.UI.Page
{
    protected void Page_PreInit(object sender, EventArgs e)
    {
        // Perform PreInit tasks
    }

    protected void Page_Init(object sender, EventArgs e)
    {
        // Perform Init tasks
    }

    // Other lifecycle event methods

    protected void Page_PreRenderComplete(object sender, EventArgs e)
    {
        // Perform tasks before complete page rendering
    }
}
```
<br>

## 3. What are the different types of _state management_ in _ASP.NET_?

ASP.NET provides several techniques for managing state across web applications. These approaches ensure that the web server can retain data pertaining to a specific user session, request, or application-wide settings.

### Implicit State Management

**Browsers**: Primary way of state management. When users visit web pages, their browsers maintain session details that help in state preservation.

**HTTP Protocol**: Largely stateless with GET and POST methods.

### Server-side Techniques

#### Code-Behind Files

The traditional ASP.NET model relies on separate .aspx and .aspx.cs files. While the .aspx file represents the view and structure, the .aspx.cs file contains server-side code related to the view. Data binding and state management strategies are often unified within these code-behind files.

This approach simplifies separation of concerns but relies on a server's performance and memory for state management.

#### HttpContext

The `HttpContext` class offers an intuitive way to **access request and session state**. Further, it inherits from `HttpContextBase`, which is testable in isolation, aiding in unit testing.

```csharp
// Set a value in session
HttpContext.Current.Session["UserID"] = "123";

// Access the same value
var userID = HttpContext.Current.Session["UserID"];
```

### Client-side Techniques

#### ViewState

`ViewState` is a client-side state management mechanism that retains the state of server-side controls across postbacks. Under the hood, its data is stored in a hidden form field.

ViewState is useful for maintaining small to moderately sized data specific to a page, such as control values and state.

Enabling ViewState at the control or page level lets ASP.NET take care of the rest, making it convenient but potentially less efficient in terms of network traffic and HTML size.

#### Hidden Fields

A straightforward and low-overhead method of client-side state management is through hidden form fields. By adding a hidden input element to the form and setting its value server-side, data can be persisted across postbacks.

Here's a basic example:

```html
// In the .aspx file
<input type="hidden" id="userIDHidden" runat="server" />

// In the .aspx.cs file
userIDHidden.Value = "123";
```

#### Cookies

Cookies operate at the browser level, allowing the server to send small pieces of data to be stored on the client's device. ASP.NET provides built-in methods for cookie creation, reading, and deletion using `HttpRequest` and `HttpResponse` objects.

Their small storage capacity makes them suitable for limited state management schemes or to hold session IDs that link each client request to a server session.

The following code sets a cookie:

```csharp
Response.Cookies["UserID"].Value = "123";
```

#### Query Strings

Query strings are URL parameters that enable state to be passed across different pages or requests. Their ease of use makes them a simple choice for state management, especially for parameters unique to a specific web page.

Here's an example URL:

```
https://example.com/Account/Login?userID=123
```

#### Local Storage

Modern browsers support local storage, providing an alternative to cookies. Local storage allows larger data volumes to be stored (typically up to several megabytes) on the client's side, persisting even after the browser is closed and reopened.

Its client-side persistence lets developers implement multi-page applications without server round trips for state management.

Here's an example of setting an item in local storage:

```javascript
// Set a key-value pair
localStorage.setItem('userID', '123');
```
<br>

## 4. Explain the difference between _server-side_ and _client-side code_.

**Server-side code** runs on the web server. It's responsible for managing and storing data, processing requests, and generating dynamic content that's then sent to the client.

For web applications, this is commonly achieved through a technology stack that includes a web server, a server-side scripting language like C# or PHP, and a database.

#### Server-Side Technologies

- **ASP.NET** is the predominant choice.
- **Node.js**, powered by JavaScript, is also widely used.
- **Python's Django** and **Flask** are popular with Python developers.

#### Pros and Cons

- **Pros**: Offers tight control over security, data integrity, and business logic. Suitable for applications that require strong data validation and security protocols.
  
- **Cons**: Might be slower for tasks that can be done on the client, potentially resulting in a less responsive user experience.

### Client-Side Code

**Client-side code**, on the other hand, runs directly in the web browser or the client.

It is primarily responsible for **presenting the User Interface (UI)** and often employs **asynchronous calls** to interact with server-side resources, providing a more dynamic user experience.

#### Common Technologies and Languages

  - **JavaScript (JS)** is the primary language used worldwide.
  - HTML5 and CSS3 are essential companions to JS for web-page building.
  - **Ajax** facilitates asynchronous requests.
  
  #### Pros and Cons

-  **Pros**: Can offload processing from the server, making web pages more responsive. Can reduce server load and improve user experience.
  
- **Cons**: Might be vulnerable to security risks and isn't the best option when strict data integrity is required.
<br>

## 5. What is a _PostBack_ in _ASP.NET_?

In ASP.NET, a **PostBack** refers to the client-server communication that takes place when a user submits a web form. Unlike traditional web pages that are stateless and require manual data syncing for user inputs, ASP.NET pages use PostBacks to manage state and keep the form's data up-to-date.

### Core Components of a PostBack

- **Control State**: Individual components, like textboxes or dropdowns, store their data during a PostBack.
- **View State**: The page itself retains state through a hidden field, allowing non-control data to persist across PostBacks.
- **IsPostBack Property**: This bool flag helps distinguish between the initial page load and subsequent PostBack events.

### Benefits and Drawbacks of PostBacks

#### Benefits

- **Simplicity**: PostBacks offer an easy-to-understand, event-driven model for web forms.
- **State Management**: Built-in mechanisms help maintain component and page state.
- **Familiarity**: It mirrors desktop application behavior, making it intuitive for developers.

#### Drawbacks

- **Network Traffic**: Complete page data is sent back and forth, leading to potentially slower performance.
- **Limited Flexibility**: PostBacks can at times inhibit the implementation of complex client-side interactions.

### Recommendations for PostBack Optimization

- **UpdatePanel**: This controls asynchronous PostBacks, sending only the relevant data and HTML, hence improving performance.
- **ScriptManager**: This JScript management tool can be used with UpdatePanel to minimize data transfers and decrease jittery web page transitions.
- **Client-Side Validation**: Employing client-side validation prompts prevents unnecessary PostBacks, reducing network overhead.
<br>

## 6. What are _WebForms_ in _ASP.NET_?

WebForms, a core feature of ASP.NET, enable **rapid web application development** through a variety of visual tools. These tools offer an intuitive method to build rich, interactive web applications while **handling low-level plumbing** tasks automatically.

WebForms fundamentally **abstract the stateless nature of the web**. They encapsulate web pages as stateful entities, mirroring desktop application behavior.

### Central Components

#### Pages

ASP.NET WebForms consist of **.aspx** pages, combining HTML, server-side controls, and code-behind files written in either C# or VB.NET. Thanks to event-driven architecture, server and user interactions are seamless.

#### User Controls

Modular and reusable, user controls (**.ascx**) help structure larger applications efficiently. These controls pack visual elements and server logic, perfect for shared UI components.

#### Master Pages

Offering a consistent layout across a site, master pages provide a template for content pages. This way, design and structure remain unified throughout the application.

### State Management

WebForms, in contrast to classic web apps that lack state, endow web pages with **built-in state management**. Users can preserve state through techniques like View State and Session State, making the web experience akin to desktop applications.

### Event Cycle

The event cycle of WebForms pages comprises several stages, each playing a specific role. The **initiation phase** involves page construction, setting properties and declaring controls. Following that, if the request is a postback, the page enters the **postback phase**. Here, control events are processed and server controls' state is reloaded. This phase is critical for providing user interaction and form submissions.

Finally, during the **render phase**, the server creates the HTML response before rendering it onto the client.

### Visual Studio Integration

Visual Studio infuses WebForms with a potent suite of drag-and-drop, tools, and visual designers. These capabilities expedite UI building, permitting developers to focus on logic.

### Code Sample: Page Lifecycle

Here is the C# code:

```csharp
public partial class MyWebForm : System.Web.UI.Page
{
    protected void Page_Load(object sender, EventArgs e)
    {
        if (!IsPostBack)
        {
            // Perform initialization tasks
        }
        else
        {
            // Execute tasks related to postback
        }
    }
}
```
<br>

## 7. What is _MVC_ in _ASP.NET_ and how does it work?

**MVC** (Model View Controller) in **ASP.NET** is a software design pattern that organizes an application into three interconnected components:  **Model**, **View**, and **Controller**.

### Key Components

- **Model**: Represents the application's data and business rules, exposing these to other components. 

- **View**: Provides the user interface, presenting data from the Model and collecting user input. Views are not aware of the Model's structure, only the data it needs to display.

- **Controller**: Acts as an intermediary, handling user input, modifying the Model as needed, and selecting the appropriate View to present.

### Data Flow in MVC

1. **User Interaction**: A user performs an action, typically through the View. For example, the user clicks a button.

2. **Routing and Controller Handling**: The routing mechanism identifies the corresponding Controller based on the incoming request. The Controller takes this user action, processes it, and updates the Model as necessary.

3. **Model Update**: The Controller updates the Model if required.

4. **View Selection**: After the Model is updated, the Controller selects an appropriate View to render and provides the updated Model to that View.

5. **Presentation and User Feedback**: The View, rendered with the updated Model, is presented to the user, possibly with updated interface elements like text and images.

### Benefits of MVC

- **Separation of Concerns**: Each component has a specific role, making the codebase easier to manage and less prone to bugs from interdependent logic.
- **Testability**: Components like the Model and Controller can be unit tested in isolation, facilitating quality control.
- **Code Reusability**: Both Views and Controllers can be reused in multiple parts of the application, enhancing the development efficiency.
- **SEO Friendly**: MVC promotes cleaner URLs, benefits search engine optimization, and improves website accessibility.

### MVC vs. WebForms

- **Event-Driven Model**: WebForms use a page-based event-driven model whereas MVC is action-based.
- **Control Over HTML and URLs**: MVC offers more control over HTML markup and URL structures.
- **Reusability vs. Control**: WebForms emphasize the design principle of reusability, offering server controls that can be dropped onto any page. MVC, on the other hand, emphasizes control and flexibility.

### Familiar Use Cases

- **Model**: Manages the behavior and data of the application domain. For instance, a Model might encapsulate the logic for retrieving a list of products.
  
- **View**: Renders the Model data, potentially allowing users to interact with it. For a shopping website, a View could display product details and allow users to add items to their cart.

- **Controller**: Interacts with the Model and selects the View to present. In a shopping website, the Controller might handle actions like adding an item to the cart and then select the appropriate View to confirm the addition.   

### Code Example: MVC Structure

Here is the C# code:

#### Model (Product.cs)

```csharp
public class Product
{
    public string Name { get; set; }
    public decimal Price { get; set; }
}
```

#### View (Views/Product/ProductDetails.cshtml)

```html
@model Product

<!DOCTYPE html>
<html>
<head>
    <title>Product Details</title>
</head>
<body>
    <h1>@Model.Name</h1>
    <p>Price: @Model.Price</p>
</body>
</html>
```

#### Controller (Controllers/ProductController.cs)

```csharp
public class ProductController : Controller
{
    public ActionResult Details(int id)
    {
        var product = GetProductById(id);

        if (product != null)
            return View(product);

        return HttpNotFound();
    }

    private Product GetProductById(int id)
    {
        // Retrieve product from data source
        // Example: using Entity Framework or querying a database
    }
}
```
<br>

## 8. Describe the role of a _master page_ in _ASP.NET WebForms_.

**Master Pages** in ASP.NET WebForms serve as templates for designing consistent web applications, offering **centralized control** over layout and style. Master Pages are especially beneficial in multi-page applications and for parallel development.

### Key Features

- **Consistent Layout**: Master Pages ensure uniform design across web pages.
- **Separation of Concerns**: Division between Master Pages and content pages allows independent editing or updating.
- **Selective Inheritance**: Different content pages can leverage distinct Master Pages.
- **UI Core Elements**: Master Pages provide essential elements such as headers, footers, navigation, and placeholders.

### When to Use Master Pages

- **Corporate Branding**: Employ a consistent design theme reflecting the corporate identity.
- **Large Applications**: For simplified maintenance and updates in multi-page apps.
- **Centralize Management**: Where core design features or controls need to be managed from a single point.
- **Uniformity**: To ensure visual consistency throughout the application.

### Master Page and Viewstate

While **Viewstate** tracks changes across postbacks, Master Pages  have their specific Viewstate objects.

- **Content Pages**: They don't directly access the Viewstate in the Master Page. Instead, they interact via the `this` keyword (C#) or `Me` (VB.NET).

- **Cross-Page Postbacks**: If a user control on a Master Page needs to initiate a postback on a content page, it's accomplished using the `FindControl` method.

### Code Example: Accessing Controls

Here is the C# code:

```csharp
// Access a Label control within a Master Page
protected void Page_Load(object sender, EventArgs e)
{
    Label lbl = (Label)Master.FindControl("lblFooter");
    if (lbl != null)
    {
        lbl.Text = "Updated from content page!";
    }
}
```
<br>

## 9. What is a _web.config file_ and what are its uses?

The `web.config` file is crucial to configuring **ASP.NET web applications**, providing a range of customizations and settings.

### Core Elements

#### Configuration

- Specifies the root-level elements in the `web.config` file, inclusive of custom settings for components such as modules, handlers, and security mechanisms.

#### System.Web

- Consists of numerous child elements for setting up aspects like `<authentication>`, `<authorization>`, `<compilation>`, and `<httpRuntime>`, among others.

#### AppSettings

- Houses app-specific key-value pairs which can be accessed through `ConfigurationManager.AppSettings`.

#### ConnectionStrings

- Stores data about database connections, incorporated into the application via `ConfigurationManager.ConnectionStrings`.

#### System.WebServer

- Tailors IIS behavior, including elements such as `<modules>`, `<handlers>`, and `<security>`.

#### Third-Party Libraries

- Can include proprietary sections from external libraries like Entity Framework, helping with data persistence or cache configurations.

### Additional Functions

- **Error Handling**: Unveils better error pages for development or guides users to custom pages.
  
- **HTTP Modules**: Activates global event handlers for all requests.
  
- **HTTP Handlers**: Specifies distinct action handlers for unique URI requests.

- **Local Desktop Development**: Can configure development settings that deviate from the production server or cloud settings.

- **Debug Mode**: Enabling debugging for detailed system information when issues arise.

- **Performance Monitoring**: Adjusts settings for application performance monitoring and debugging.

- **Security Settings**: Customizes access control, authentication mechanisms, such as Windows or custom user checks, and sets SSL requirements.

- **Database Connection and Authorization Handling**: Secures and configures data sources like SQL Server.

- **Session and Caching Control**: Provides settings for in-memory session data and cache management, customizable with external caching tools.

### File Nesting in Visual Studio 2019

In **Visual Studio 2019**, the `web.config` file is automatically associated with web projects, ensuring proper configuration settings when deployed to a hosting server.

While usually located in the root directory, developers can add **multiple `web.config`** files to subdirectories for fine-grained configuration control. However, only the `web.config` file in the root takes precedence.

### Code Example: Using custom settings from `web.config`

Here is the C# code:

```csharp
using System;
using System.Configuration;

public class ConfigurationManagerExample
{
    public static void Main()
    {
        // Accessing custom app settings from web.config
        string settingValue = ConfigurationManager.AppSettings["CustomSettingKey"];
        Console.WriteLine($"Custom setting value: {settingValue}");

        // Accessing a connection string
        string connectionString = ConfigurationManager.ConnectionStrings["MyDatabase"].ConnectionString;
        Console.WriteLine($"Database connection string: {connectionString}");
    }
}
```
<br>

## 10. Explain the concept of _ViewState_ in _ASP.NET_.

**ViewState** enables you to persist state information across **page requests** in ASP.NET web forms. It is especially useful for retaining data during **round-trip** events like button clicks and form submissions.

### How ViewState Works

- **Locality**: The ViewState property, present in the Page class, is scoped to specific instances of a web form and ensures that state data is kept related to that form only.

- **Persistence**: The view state data, including its type, can be preserved across postback events either in the page or in a hidden input of the page.

- **Connection and Data Integrity**: It relies on hidden fields and state information that is usually embedded in web forms. It ensures coordinated communication and synchronization between the server and different positioned form fields.

- **Obfuscation**: The data in ViewState is encoded, but not encrypted, offering a level of security against tampering.

- **Round-Trip**: This mechanism preserves state during a round trip from client to server and back, obviating the need to retrieve or recompute the data.

### When to Use ViewState

- **Scope Management**: To safeguard values within a form across postbacks without the need for requesting them from the client or the database.

- **Data Preservation**: For maintaining control state even during validation or subsequent round-trip events.

- **Sensitivity**: For storing non-sensitive data, as ViewState is client-side, readable, and modifiable.

### Security Considerations

- **Limited Sensitivity**: While data is obfuscated, it's not encrypted and can be decoded. Therefore, avoid storing highly confidential or sensitive information.

- **Potential Attack Point**: ViewState tampering is a known attack vector. Developers should closely monitor and validate data in the ViewState.

### Code Example: ViewState

Here is the C# code:

```csharp
  protected void Page_Load(object sender, EventArgs e)
    {
        // On initial load, set a ViewState value
        if (!IsPostBack)
        {
            ViewState["Count"] = 0;
        }
    }

    protected void IncreaseCountButton_Click(object sender, EventArgs e)
    {
        // Increment count in ViewState and display
        int count = (int)ViewState["Count"];
        count++;
        ViewState["Count"] = count;
        CountLabel.Text = "Count: " + count;

        // Other form elements are not affected
    }
```
<br>

## 11. What is a _server control_ in _ASP.NET_?

**Server controls** in ASP.NET are essential for building dynamic, interactive web applications. These controls are backed by server-side code and can interface with client-side technologies such as HTML, CSS, and JavaScript.

### Categories of Server Controls

1. **HTML Server Controls**: An enriched version of HTML elements with server-side functionalities.
   
2. **Web Server Controls**: Advanced, specialized controls that abstract both server-side and client-side logic.

3. **User Controls**: A grouping of controls that can be reused across different web pages. These are primarily established from existing controls, paired with programmatic or declarative logic.

### Key Benefits

- **Familiar Abstraction**: Simplifies web development, especially for those transitioning from desktop applications.
- **IDE Support**: Provides extensive design-time features and Visual Studio compatibility.
- **Event-Handling**: Offers easy-to-use event models, reducing the complexity of client/server communication.
- **State Management**: Assists in managing state between server requests and client interactions.

### Controls Hierarchy

- **Component Control**: This serves as the base for all server controls. Most properties and methods are defined in this directive.
- **WebControl**: Adds extensive layout, design, and content formatting capabilities.
- **DataBoundControl**: Enhances data-related capabilities, particularly for data operations like querying.
- **Composite Control**: Utilizes multiple embedded controls to compose a single unified control.

### Key Features

- **Built-in Validations**: Offers server-side validations, often a more secure alternative to client-side validations.
- **Rich Visuals**: Ensures a consistent look and feel through predefined styles and templates.
- **Data Binding**: Simplifies database integration and data display.
- **State Management**: Capable of managing application, session, and control-specific state.

### Challenges of Server Controls

- **Complex Rendering Logic**: Interweaving server and client code can sometimes lead to convoluted, difficult-to-follow rendering flows.
- **Limited Extensibility**: The parent-child relationship of controls can impose restrictions on how they can be manipulated and extended.
- **Verbose HTML Output**: Due to the server-client interplay, the generated HTML can be cluttered with ASP.NET-centric attributes.

### Code Example: Server Controls

Here is the C# code:

```csharp
// TextBox that appears only after a button click
<asp:Button ID="btnShowText" runat="server" Text="Show Text" OnClick="btnShowText_Click" />
<asp:TextBox ID="txtDynamic" runat="server" Visible="false" />

// Backend Code
protected void btnShowText_Click(object sender, EventArgs e) {
    txtDynamic.Visible = true;
}
```
<br>

## 12. What are _user controls_ and _custom controls_?

**User Controls** and **Custom Controls** offer modularity and code reusability in ASP.NET, but on different scales. While User Controls are confined to single web forms, Custom Controls can be used broadly across the website.

### User Controls

A **User Control** is like a mini web form that you can build separately and then embed into your main web forms. This makes it ideal for creating standalone, reusable components.

#### Use-Cases

- Repeated sections in a website (headers, footers, sidebars).
- Elements specific to a section of the website (like a product catalog).

#### ASPX & Code-Behind

- **ASPX**: A user control is generally composed of an `.ascx` file, which defines the control's structure and elements.
- **Code-Behind**: For behaviors, you can have a separate code-behind file (`.ascx.cs`), or you can embed your code in the `.ascx` file itself.

#### Code Example: User Control

Here is the C# code:

```csharp
// Sample UC: MyUserControl.ascx

<%@ Control Language="C#" AutoEventWireup="true" CodeBehind="MyUserControl.ascx.cs" Inherits="MyNamespace.MyUserControl" %>

<!-- UC elements go here. -->

// Sample UC Code-behind: MyUserControl.ascx.cs

public partial class MyUserControl : System.Web.UI.UserControl
{
    protected void Page_Load(object sender, EventArgs e)
    {
        // Logic here.
    }
}
```

### Custom Controls

A **Custom Control** in ASP.NET is a more sophisticated type of control that you can design and configure from scratch to suit your specific needs.

#### Design & Implementation

Custom controls are more aligned with the look and feel of standard ASP.NET server controls but are highly customizable.

- **HTML Output**: You have total control of the HTML and can dynamically generate it.
- **Embedded Resources**: You can package CSS, JavaScript, and images as part of your custom control.

#### Types of Custom Controls

- **User-Created**: These are controls you create by extending existing controls or building from scratch.
- **Composite Controls**: They are composed of other controls or custom controls.
- **Rendered Content Controls**: They render UI elements from code or a template.

#### Registered vs Unregistered

- **Unregistered**: You can use these controls on a single page.
- **Registered**: Once registered in the web.config or page, they can be used across the site.

#### Code Example: Custom Control

Here is the C# code:

```csharp
// Sample CC: MyCustomControl.cs

using System;
using System.Web.UI;
using System.Web.UI.WebControls;

[assembly: TagPrefix("MyNamespace", "cc")]

namespace MyNamespace
{
    public class MyCustomControl : CompositeControl
    {
        private TextBox textBox;
        private Button button;

        protected override void CreateChildControls()
        {
            textBox = new TextBox();
            button = new Button();

            this.Controls.Add(textBox);
            this.Controls.Add(button);
        }

        protected override void Render(HtmlTextWriter writer)
        {
            textBox.RenderControl(writer);
            button.RenderControl(writer);
        }
    }
}

// Using the Custom Control in ASPX

<%@ Register Assembly="MyAssembly" Namespace="MyNamespace" TagPrefix="cc" %>
<cc:MyCustomControl runat="server" />
```
<br>

## 13. Explain how to create a _custom control_ in _ASP.NET_.

There are **two primary methods** for creating **custom controls** in ASP.NET:

### Visual and Code-Behind Approach

Here are the steps:

1. **Control Design**: Create the visual representation of your control using the designer or directly building the HTML in your `.aspx` file.

2. **Code-Behind Logic**: Use the `.ascx.cs` file to implement the control's functionalities, such as event handling or complex UI manipulations.

3. **Control Packaging**: The `.ascx` file and its code-behind form the control, which you can distribute as a reusable unit.

### Code-Behind and Control Reference Approach

This method is particularly helpful when more advanced control is needed. It's distinct from the aforementioned method as the control **doesn't have a visual design in the `.ascx` file**.

Here are the steps:

1. **Control Design**: Call `Controls.Add()` method in the `.ascx.cs` file to create a control and add it to the control's hierarchy. This is done in lieu of a visual design in the `.ascx` file itself.

2. **Control Packaging**: Just like the Visual and Code-Behind method, this one sees distribution in a packaged format.

### Best Practices for Building Custom Controls

- **Separation of Concerns**: For better maintainability, keep your control's presentation logic in `.ascx`, and the business and event handling logic in the `.ascx.cs`.
- **Custom Control vs. User Control**: Both custom controls and user controls can provide modularity. User controls are easier to build, while custom controls offer more fine-tuned control. The choice depends on your project's requirements and your development team's expertise. User controls are easier for separating concerns.
- **Documentation**: Whether you opt for a code-behind approach or the control reference method, your control must be well-documented for reusability.
<br>

## 14. What are _validation controls_ in _ASP.NET_?

**Validation controls** in **ASP.NET** provide a server-side mechanism for data validation. Using these extensively can significantly improve development efficiency and code maintainability.

### Key Features

- **Client-Side Validation**: Validation controls incorporate both client-side JavaScript and server-side validation.

- **Automatic Error Messages**: Once a validation event fails, predefined error messages, which you can customize, alert the user.

- **State Management Integration**: These controls integrate seamlessly with ViewState, which simplifies maintaining control states across multiple round-trips.

### Types of Validation Controls

- **RegExp**: Using regular expressions, you can verify that user input matches a specific pattern.
- **RequiredField**: Confirms that users provide necessary data in certain fields.
- **CompareField**: Used to compare two fields for equality, such as in password confirmation forms.

### Benefits

- **Reusability**: You can deploy validation controls across multiple forms and controls, promoting a consistent user experience.
- **Rapid Development**: These controls accelerate development, especially for client-side validation, as they're incorporated through an established process.
- **Versatility**: The controls are flexible and can be utilized across various input contexts.
  - **Custom Validation**: If the built-in controls aren't sufficient, you can always craft a tailored solution using custom methods.

### Why, When & How to use Validation Controls

- **When to Implement**: Use validation controls whenever user input needs verification before processing.
- **Underlying Mechanism**: Both client and server-side resources form validation checks, providing consistent safeguarding at both levels.
- **Hooking into Validation Events**: By handling specific validation events, such as `IsValid`, you can choose when to trigger overall validation procedures. For instance, you might determine that validation should happen on a button click.
- **Custom Error Messages**: These can be set globally or tailored per control, letting you deliver context-aware warnings. This is particularly useful in multi-lingual settings.
- **Styling and Appearance**: The look and feel of error messages can be tuned to match the visual scheme of the application. You also have the flexibility to present messages through various on-screen controls, like labels or pop-up alerts.

### Best Practices

- **Holistic Verification** : Combine client-side and server-side validation for strong data security.
- **Form Validity**: Ensure the form has concluded as valid on the server side before proceeding with data storage or any subsequent actions.

**Note**: Avoid over-reliance on client-side validation, as this can be manipulated. Server-side validation provides a secure offense.
<br>

## 15. Describe the different types of _validation controls_ available in _ASP.NET_.

**ASP.NET** provides a range of validation controls to ensure accurate user inputs. These controls, part of the broader Validation Control family, play an integral role in promoting data integrity.

### List of Commonly Used Validation Controls

1. **RequiredFieldValidator**: Forcing a field to be populated.
2. **RangeValidator**: Setting upper and lower bounds for numeric input.
3. **RegularExpressionValidator**: Enforcing a specific text format using regular expressions.
4. **CompareValidator**: Aligning two input fields.
5. **CustomValidator**: Implementing custom validation logic with server-side or client-side scripts.
6. **ValidationSummary**: Displaying an aggregate summary of all validation errors.

### Setting Up Validation

Regardless of specific validator types, you typically find a common sequence for integrating these elements:

1. **Attach Validators to Components**: Validators are usually associated with controls that demand validation. This link is built through the `ControlToValidate` parameter.
2. **Connecting Validators to Triggers**: Certain validators, especially the `CustomValidator`, might require connections to corresponding actions. Use the `ControlToCompare`, `ControlToMatch`, or `ControlToValidate` and `ValidationGroup` attributes for such pairings.
3. **Delineate Error Output**: Validation errors are manifest through mechanisms like styling alterations, alerts, or dedicated positioning, stipulated via `ErrorMessage`, `Text`, or `Display` properties.

### Enhanced User Experience with Client-Side Validation

ASP.NET also provides mechanisms for **client-side validation**. When activated, these improve UX by catching errors as soon as they happen, without waiting for server communication. This version of verification, faster and more seamless, relies on JavaScript. Nonetheless, server-side validation serves as a crucial backup, ensuring fault-tolerant, secure data handling.
<br>



#### Explore all 100 answers here 👉 [Devinterview.io - ASP NET](https://devinterview.io/questions/web-and-mobile-development/asp-net-interview-questions)

<br>

<a href="https://devinterview.io/questions/web-and-mobile-development/">
<img src="https://firebasestorage.googleapis.com/v0/b/dev-stack-app.appspot.com/o/github-blog-img%2Fweb-and-mobile-development-github-img.jpg?alt=media&token=1b5eeecc-c9fb-49f5-9e03-50cf2e309555" alt="web-and-mobile-development" width="100%">
</a>
</p>

