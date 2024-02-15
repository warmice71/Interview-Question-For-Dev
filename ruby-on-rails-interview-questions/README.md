# 100 Must-Know Ruby on Rails Interview Questions

<div>
<p align="center">
<a href="https://devinterview.io/questions/web-and-mobile-development/">
<img src="https://firebasestorage.googleapis.com/v0/b/dev-stack-app.appspot.com/o/github-blog-img%2Fweb-and-mobile-development-github-img.jpg?alt=media&token=1b5eeecc-c9fb-49f5-9e03-50cf2e309555" alt="web-and-mobile-development" width="100%">
</a>
</p>

#### You can also find all 100 answers here 👉 [Devinterview.io - Ruby on Rails](https://devinterview.io/questions/web-and-mobile-development/ruby-on-rails-interview-questions)

<br>

## 1. What is _Ruby on Rails_, and why is it popular for _web development_?

**Ruby on Rails**, often termed simply as **Rails**, is a popular web application framework known for its simplicity and productivity.

It is built on the **Ruby** programming language and adheres to the **MVC** (Model-View-Controller) architectural pattern. Rails takes a strong stance on conventions over configurations, reducing the need for boilerplate code.

### Key Components

- **Active Record**: Simplifies data handling from databases, treating tables as classes and rows as objects.
- **Action View**: Manages user interface elements like forms and templates.
- **Action Controller**: Handles requests, processes data, and manages flow between models and views.

Rails also comes with an integrated testing suite and a robust security system.

### Notable Advantages

- **Developer-Friendly Syntax**: Ruby on Rails prioritizes readability, allowing easy collaboration across teams.
- **Unmatched Ecosystem**: The open-source community continually provides modules, referred to as gems, for rapid feature integration.
- **Enhanced Productivity**: The framework's emphasis on best practices automates various tasks, reducing development time.
- **Scalability**: While initially receiving criticism in this area, the framework has evolved to handle large-scale applications effectively.

### Code Example: Ruby on Rails Core Components

Here is the Ruby code:

```ruby
# ActiveRecord model
class User < ApplicationRecord
  has_many :posts
end

# Controller
class UsersController < ApplicationController
  def show
    @user = User.find(params[:id])
  end
end
```

### More About Its Philosophy

Rails' philosophy, in a nutshell, can be described with the term **Optimistic Assumptions**. This means, the default setup works well for most cases, but developers can override these assumptions when necessary.

This design philosophy alongside a strong focus on **developer happiness** is why Ruby on Rails has been one of the leading web frameworks for almost two decades.
<br>

## 2. Describe the _MVC_ architecture in _Rails_.

**Model-View-Controller** (MVC) is a design pattern embraced by Ruby on Rails. It organizes code into three interconnected components to improve maintainability, separation of concerns, and ease of modification.

### Components of MVC

- **Model**: Manages the application's data, logic, and rules. It directly interacts with the database. In Rails, models are typically their own Ruby class.
- **View**: Handles the displaying of data to the user. In Rails, views are often characterized by ERB templates, which combine HTML with embedded Ruby for dynamic content.
- **Controller**: Acts as an intermediary between the model and view, interpreting data to provide the necessary visualization. It responds to user input and performs tasks following the instructions defined in the model. Each **public** method in a Rails controller corresponds to an **action** that can be invoked by an incoming HTTP request.

### Code Example: Basic MVC Relationship

Here is the Ruby code:

```ruby
# Controller
class ArticlesController < ApplicationController
  def index
    @articles = Article.all
  end
end

# Model
class Article < ApplicationRecord
end

# View (ERB)
<% @articles.each do |article| %>
  <li><%= article.title %></li>
<% end %>
```

In this example:

- The `Article` model represents a table in the database and is defined by inheriting from `ApplicationRecord`.
  
- The `ArticlesController` has a single action, `index`, which retrieves all articles using the `Article` model and makes them available to the associated view.

- The View iterates through the list of articles, displaying their titles.

### Responsibilities of MVC Components

#### Model

- Represents the structure and behavior of the application, independent of the user interface. ([](#MVC))
- Performs various tasks like data validation, relationships, and business logic. For instance, it ensures that a 'User' object should have a unique email id.

#### View

- Displays information to the user, with a layout and style. It might contain conditional logic or loops but should remain free of data retrieval and complex data manipulation..

#### Controller

- Coordinates the flow of data between the model and view. It receives input from the users (HTTP requests) and directs that input to the model or view as needed. ([](#MVC))

  For example, when a user initiates a new article creation in the web app, the `new` action (a public method in the controller) will present a form (via the view) to the user. Any data input from the form post will be handled by the `create` action, which passes the data to the model for storage.
<br>

## 3. What are _Gems_ and how do you use them in a _Rails_ project?

**Gems** are pre-built, reusable packages in Ruby on Rails, serving similar purposes as libraries do in other languages. They streamline common software development tasks and add functionality to your application. RubyGems, the package manager for Ruby, is responsible for managing these gems.

- **Install Gems**: Use the `gem` command-line tool to install a gem into your system, making it globally accessible:

  ```bash
  gem install specific_gem
  ```

- **Require Gems**: When a gem is installed, you need to `require` it in your code. However, when using Rails, many gems are automatically loaded, removing the need for an explicit `require` statement:

  ```ruby
  require 'specific_gem'
  ```

- **Filter Gems**: You can specify the versions of gems to use within your application by including them in the `Gemfile`. Upon doing so, the `bundle` tool installs gems that match the defined criteria:

  ```ruby
  # In Gemfile
  gem 'specific_gem', '~> 3.0.1'
  ```

- **Load Gems**: After defining the gems in the `Gemfile`, execute the `bundle` command to load the dependencies:

  ```bash
  bundle install
  ```

  Then use the `Bundler.require` directive in your application:

  ```ruby
  # In config/application.rb. This line is generally already present.
  Bundler.require(*Rails.groups)
  ```

This process minimizes version conflicts and ensures a consistent development environment across different machines.

### Advanced Gem Management

1. **Development Mode**: Rather than installing the entire set of gems for an application, the `group` declaration in the `Gemfile` lets you segregate them for specific environments.

   Choose from predefined groups like `:development` or `:test`:

   ```ruby
   # In Gemfile
   group :development, :test do
     gem 'specific_gem', require: false
   end
   ```

   Then, install gems without these groups in the development or production environment:

   ```bash
   bundle install --without development test
   ```

   Gems enclosed in the development block won't be accessible in the `production` environment.

2. **Single Responsibility**: Adhere to the Single Responsibility Principle with packaging gems that tackle a specific concern, offering focused functionality.

3. **Gem Naming Conventions**: Comprehensible gem names make them more discoverable and easier to understand.

4. **Exposing Functionality**: Reveal gem features through a straightforward API and comprehensive documentation.

5. **Avoiding Duplication**: Determine if a gem introduces functionality that's already available in another installed gem, to prevent redundancy.
<br>

## 4. What is the purpose of the _Gemfile_ in a _Rails_ application?

The `Gemfile` in a Ruby on Rails application acts as a manifest of all required gems for the project.

It serves as the foundation for the **Bundler** tool, which automates the process of managing and versioning gem dependencies.

### Bundler: Gem Management and Version Control

Bundler tracks gem versions to ensure project consistency across different environments and team members. It also keeps a record of specific gem versions in the project's `Gemfile.lock`.

#### Mechanism of **Gemfile.lock**

When Bundler installs or updates gems, it incorporates them into the project with precision according to the versions recorded in `Gemfile.lock`.

This feature is particularly beneficial for collaboration, as it preserves a consistent gem "snapshot" for all contributors.

#### Gems Grouping

The `Gemfile` segments gems into groups, each accommodating specific needs. For instance, the `:development` group hosts gems necessary during the development phase.

```ruby
group :development, :test do
  gem 'rspec-rails'
  gem 'pry'
end
```

#### Advantages

- **Ease of Setup**: New project collaborators can swiftly onboard by leveraging the `Gemfile` and `Gemfile.lock`.
- **Version Consistency**: Bundler enforces uniformity in gem versions, minimizing potential compatibility issues.
- **Optimized Deployment**: Bundler ensures that only required gem versions are transferred in production environments.

### Best Practices

- **Regular Updates**: Periodically update gems to access new features, improvements, and security patches while checking for compatibility with updated versions.
- **Shared Understanding**: Collaborators should be familiar with the gems enlisted in the `Gemfile`.
<br>

## 5. How do you install a _Ruby gem_?

To install a **Ruby gem**, start by using either of the following:

- Directly from RubyGems.org: `$ gem install gem_name`
- From a local file: `$ gem install /path/to/gem.gem`

To select a specific version, add the `--version` flag, and for developer tools, use the `--dev` or `--development` flags.

For development or testing, it's best to install gems in your project's directory, a process known as **bundling**.

### Common bundler commands

#### Bundle Install

The `bundle install` command installs all gem dependencies in the local project, as specified in the `Gemfile`.

#### Bundle Update

Running `bundle update gem_name` updates the specified gem to its latest version. To update all gems, omit the gem name.

### Using Gemfile and Gemfile.lock

- **Gemfile**: This file lists all the gems your project depends on and any rules for their usage.
- **Gemfile.lock**: This companion file records the specific gems and versions that are currently in use.

Always check both files into version control. This practice ensures a consistent development and production environment for your team.

### Steps to bundle gems

#### Initialize a Gemfile

If your project doesn't have a `Gemfile` yet, run `bundle init` in its directory to create one.

#### Add Gems

In the `Gemfile`, list all your project's gem dependencies and any version or source restrictions.

For example:

```ruby
# Gemfile
source 'https://rubygems.org'

gem 'rails', '~> 6.0'
gem 'sqlite3'
```

#### Install Dependencies

Finally, to install the defined gem set from the `Gemfile` into the local directory, run `bundle install`. This action creates or updates `Gemfile.lock`, locking down precise gem versions for your project.
<br>

## 6. Explain how to generate a new _Rails_ application.

When starting a **Rails** project, you use the `rails new` directive in a terminal shell or command prompt.

### The Command

Here is the command you can use to create a new Rails application:

```bash
rails new myapp
```

Where `myapp` is the name of your new application.

### Essential Flags

- **Template**: Use `-m` to specify a pre-defined template or script that customizes the initial structure and content of your app.
- **Database**: `-d` specifies your preferred database; use either `sqlite3`, `mysql2`, `postgresql`, or `oracle`.

### Advanced Configurations

You can tailor your `rails new` command further by interfacing with the following tools:

- **Bundler**: Configure your **Gemfile** settings using `-B` or `--skip-bundle` to avoid running Bundle Install.
- **JavaScript Runtime**: Utilize one of the available JavaScript runtimes by specifying `-j <runtime>`. The typical choices are `coffee` (CoffeeScript), `esm` (ESM), or `importmap` (Importmap).
- **Web Server**: Indicate your preferred web server for the development environment with `--dev <server>`. Common choices include `puma` (Puma) and `thin` (Thin).
- **API**: Create an API-only app via `--api` to exclude traditional middleware.

### Sample Command

```shell
rails new myapp -m template.rb -d postgresql
```

This command generates a new application called `myapp`. It customizes the initial setup utilizing a template from `template.rb` and sets PostgreSQL as the database.

### Common Mistakes

- **Improper Installation**: Ensure Rails is installed in your system before executing `rails new`. If it isn't, first run `gem install rails`.
- **Compatibility Issues**: Make sure your system has the requisite versions of Ruby and Rails for the application you wish to build. You can verify this by referencing the project's documentation.

### Further Considerations

The `rails new` directive creates an initial project structure according to a standardized configuration, known as "convention over configuration". This approach aims to minimize the need for extensive setup procedures, enabling you to start developing your project right away.
<br>

## 7. What is the _convention over configuration_ principle?

**Convention over Configuration** is a foundational principle in **Ruby on Rails** that streamlines development through pre-defined rules, reducing the need for explicit configurations.

This design philosophy centers around the concept of **defaults** and **predictable naming conventions**, effectively cutting down on repetitive code and providing a more unified structure to Rails projects.
<br>

## 8. How do you define a _route_ in _Rails_?

In Ruby on Rails, **routes** define how an HTTP request is processed, matching it to a controller action.

### Components

1. **HTTP Verb**: The action (GET, POST, PUT, DELETE) that triggers the route.
2. **Path**: The URL that the matching request accesses.
3. **Controller#Action**: Specifies the controller and its method to handle the request.

### Syntax

Here is how to define the route:

- **Simple Route**: Uses the match method to map a path to a specific controller and action.

  - Ruby on Rails 3.0 and above
    
    ```ruby
    match '/articles/:id', to: 'articles#show', via: 'get'
    ```

  - Ruby on Rails 4.0 and above

    ```ruby
    get '/articles/:id', to: 'articles#show'
    ```

- **Resource Routes**: Utilizes a RESTful interface to define multiple routes for a resource.

  ```ruby
  resources :articles
  ```

- **Redirect**: Sends a 301 or 302 HTTP status code and a URL to redirect the request to.

  ```ruby
  get '/stories', to: redirect('/articles')
  ```

- **Non-Resourceful Routes**: For custom actions that do not fit REST conventions, use the add_route method.

  ```ruby
  DynamicRouter::Routing.build do
      add_route 'futurama/:planet', :controller => 'tasks', :action => 'futurama'
  end
  ```

### Route Helpers

For a cleaner and more robust approach, use **route helpers**.

- They auto-generate paths and URLs. 
- They're standardized, reducing errors and making the code more maintainable.

#### Example: Route Helpers in Action

The code could look like this:

```ruby
# routes.rb
resources :articles do
  collection do
    get :published
  end
end

# article_controller.rb
class ArticlesController
  def publish
    @articles = Article.where(published: true)
  end
end
```

Using route helpers, the matching URL for the `publish` action in the `ArticlesController` will be `/articles/published`.
<br>

## 9. Explain the use of _yield_ in _Rails_ layouts.

**Yielding** in Ruby on Rails allows for partial **content injection**, making layouts versatile and adaptable to specific view needs.

### Mechanism

- **Placeholder**: The `yield` command in the layout creates a temporary placeholder for the content that views will inject.

  ```erb
  <header>
    <%= yield :header_content %>
  </header>
  <nav>
    <%= yield :navigation %>
  </nav>
  <main>
    <%= yield %>
  </main>
  ```

- **Injection**: In views, corresponding `content_for` commands designate where they should inject content within the layout.

  ```erb
  <% content_for :header_content do %>
    <h1>Welcome to My Site</h1>
  <% end %>

  <% content_for :navigation %>
    <ul>
      <li><%= link_to 'Home', root_path %></li>
      <li><%= link_to 'Blog', blog_path %></li>
    </ul>
  <% end %>
  ```

- **Render Flow**: When views render, the `content_for` content is placed into the named **yield sections** (if provided) or, by default, into the main one.

  ```erb
  # => Layout Output
  # <header>
  #   <h1>Welcome to My Site</h1>
  # </header>
  # <nav>
  #   <ul>
  #     <li><%= link_to 'Home', root_path %></li>
  #     <li><%= link_to 'Blog', blog_path %></li>
  #   </ul>
  # </nav>
  # <main>...</main>
  ```

### Benefits

- **Separation of Concerns**: Promotes clear distinctions between layout and content.
- **Dynamic Layouts**: Allows customization based on view-specific needs.
- **Concise Content Definition**: Content definition in views is done close to the action it's associated with.
- **Reusable Components**: Named yield sections permit reusability across various views.

### Code Example: Yield in Layout and Views

#### User Layout

Here's the layout that provides placeholders for specific pieces of content.

##### Layout: user.html.erb

```erb
<!DOCTYPE html>
<html>
<head>
  <title>User Management</title>
  <%= stylesheet_link_tag 'user' %>
  <%= javascript_include_tag 'user' %>
  <%= csrf_meta_tags %>
</head>
<body>
  <header>
    <%= yield :user_header %>
  </header>
  <nav>
    <ul>
       <li><%= link_to 'Users', users_path %></li>
       <li><%= link_to 'Settings', user_settings_path %></li>
    </ul>
    <%= yield :user_nav %>
  </nav>
  <main>
    <%= yield %>
  </main>
</body>
</html>
```

#### User List View

This view tailors content for the user layout, populating specific sections.

##### View: users/index.html.erb

```erb
<% content_for :user_header do %>
  <h1>Manage Users</h1>
<% end %>

<% content_for :user_nav do %>
  <li><%= link_to 'Roles', user_roles_path %></li>
<% end %>

<%= render 'user_table', users: @users %>
```

#### Administrative Layout

If there is a need for a different look and feel for administrative pages, a separate layout can also be defined, like so:

##### Layout: admin.html.erb

```erb
<!DOCTYPE html>
<html>
<head>
  <title>Administration</title>
  <%= stylesheet_link_tag 'admin' %>
  <%= javascript_include_tag 'admin' %>
  <%= csrf_meta_tags %>
</head>
<body>
  <header>
    <h1>Welcome, Admin!</h1>
  </header>
  <nav>
    <ul>
       <li><%= link_to 'Dashboard', admin_dashboard_path %></li>
       <li><%= link_to 'Users', admin_users_path %></li>
       <li><%= link_to 'Settings', admin_settings_path %></li>
    </ul>
  </nav>
  <main>
    <%= yield %>
  </main>
</body>
</html>
```

#### Administrative List View

```erb
<% content_for :title, "User Management - Admin" %>

<% content_for :header, "Admin Dashboard" %>

<% content_for :main do %>
  <% @users.each do |user| %>
    <p><%= user.name %> - <%= user.role %></p>
  <% end %>
<% end %>

<%= render 'admin', users: @users, roles: @roles %>
```
<br>

## 10. What is _CRUD_, and how is it implemented in _Rails_?

In **Ruby on Rails**, the **CRUD** actions (Create, Read, Update, Delete) profoundly simplify database interactions through a Model's active record representation.

### Key Components

- **Model**: Represents the data and its rules.
- **View**: Presents data to the user.
- **Controller**: Acts as an intermediary that processes user input and interacts with the model and view.

### Simplified Workflow

1. **Create**: `POST` an entity data to the server.
2. **Read**: Fetch all entities with `GET` or a specific one by ID.
3. **Update**: `PUT` the modified entity.
4. **Delete**: Remove an entity with a `DELETE` request.

### Code Example: Rails Routes

The following code illustrates RESTful routes often used in **Rails**:

```ruby
# config/routes.rb
resources :posts
```
This statement generates routes for all **CRUD** actions to interact with the `Post` model:

- **Create**: `/posts/new`
- **Read**: `/posts` for all, `/posts/:id` for one
- **Update**: `/posts/:id/edit`
- **Delete**: `delete 'posts/:id'`

### Advanced Techniques

#### Dynamic Actions

For more detailed procedures, custom routes can be defined in conjunction with elaborate controller methods.

#### Partial Resources

In cases where access to only a subset of resource actions is necessary, the `:only` and `:except` options can be employed to limit defaults.

### Final Thoughts

**RESTful** design, championed by **Rails**, has proven instrumental in establishing standard interactions across web platforms, fostering interoperability and predictability. When building applications, ensure a methodical approach is employed to optimize accessibility and security of resources.
<br>

## 11. What is the purpose of a _controller_ in _Rails_?

The **controller** in Ruby on Rails serves as the processing hub for HTTP requests and determines the application's response. It plays a central role in implementing the **MVC** architecture.

By collecting user input from the view, handling data from the model, and finally responding back to the view, the controller acts as the fulcrum for application logic and external interaction.

### Key Responsibilities

#### Input Handling

The controller is responsible for processing and validating data received from the client. Any data manipulation or validation logic, such as form input sanitation, goes through the controller before updating the model.

#### Model Interaction

The controller mediates interactions between the view and the model. It retrieves and updates data from the model as needed.

#### View Selection and Rendering

Upon completing its processing, the controller determines which view to display to the user and provides the necessary data for rendering. A typical flow involves the controller making a request to the view for the necessary data to be displayed, after which the controller sends this data to the users for display in the view.

#### HTTP Response Generation

The controller is responsible for generating an appropriate HTTP response, be it a redirect, a successful data submission message, or an error notification.

### Code Example: Controller Responsibilities

Here is the code:

```ruby
class ArticlesController < ApplicationController
  # Input Handling: article_params is a method that ensures only specific attributes are accepted from the request.
  def create
    @article = Article.new(article_params)
    if @article.save
      redirect_to @article, notice: 'Article was successfully created.'
    else
      render :new
    end
  end

  # Model Interaction: Fetches all articles to be displayed in the index view.
  def index
    @articles = Article.all
  end

  # View Selection and Rendering: Routes to the Edit article form through the view.
  def edit
    @article = Article.find(params[:id])
  end

  # HTTP Response Generation: Destroys the article and redirects to the index showing a notice.
  def destroy
    article = Article.find(params[:id])
    article.destroy
    redirect_to articles_url, notice: 'Article was successfully destroyed.'
  end

  private
  def article_params
    params.require(:article).permit(:title, :text)
  end
end
```
<br>

## 12. How do you pass data from a _controller_ to a _view_?

In **Ruby on Rails**, the primary way to transfer data from the **controller** to the **view** is by using **instance variables**.

### Using Instance Variables

#### Setting Data in the Controller

```ruby
class ArticlesController < ApplicationController
  def show
    @article = Article.find(params[:id])
  end
end
```

Here, `@article` is accessible within the `show` view.

#### Accessing Data in the View

You can directly display data in views:

```erb
<%= @article.title %>
```

### Limitations

- **Global visibility**: All instance variables set in a controller's action are available to the corresponding view. This can be a cause of accidental data leakage and can make code harder to follow.
- **Lack of Context**: Misuse of instance variables can sometimes lead to difficulty in understanding which action set a variable currently being used in the view.

### Tips and Best Practices

- **Keep it As Needed**: Only fetch and set data needed for the specific view being rendered. Reduce unnecessary database queries.
- **Isolate State**: Minimize the use of shared state, especially if views can be rendered in a concurrent fashion. Try to make views independent of each other. If shared state is unavoidable, ensure any mutations are localized and minimized to specific actions for consistency and predictability.

### Code Implementation: Data Leakage Vulnerability

Consider this example:

```ruby
class UsersController < ApplicationController
  def show
    @user = User.find(params[:id])
    @orders = @user.orders
  end

  def refresh_orders
    @user = User.find(params[:id])
    @orders = @user.orders.where(status: 'pending')
    render :show
  end
end
```

If `refresh_orders` is called from the `show` view, the view will now display filtered orders from the `refresh_orders` method. This is not the expected behavior, and it's a classic instance of data leakage.
<br>

## 13. Explain the _Rails controller action lifecycle_.

The **Rails controller action lifecycle** involves a sequence of steps that handle web requests. Let's look at each step in this process, from initial request to the returned response.

### Overview of the Controller Action Lifecycle

1. **Routing**: Determines the appropriate controller and action based on the incoming URL.
2. **Controller Initialization**: Sets up the controller and any associated parameters or named routes.
3. **Before Actions**: Executes any methods configured as "before actions". Commonly used for tasks like authentication and authorization.
4. **Action Execution**: Invokes the method named after the action (e.g., `show` for the GET "show" route).
5. **View Rendering**: The controller prepares data to be sent to the view for rendering.
6. **After Actions**: Carries out any specified tasks after the action method and before the response is sent.
7. **Response Construction**: Combines data and any established HTTP headers to create the response.

### Code Example: Request-Response Lifecycle

Here is the Rails code:

```ruby
# app/controllers/articles_controller.rb
class ArticlesController < ApplicationController
  # Step 3: Before actions
  before_action :find_article, only: [:show, :edit, :update, :destroy]

  # Step 4: Action execution
  def show
    # Step 5: View rendering
    # Renders the 'show' template by default
  end

  private

  # Example of a 'before' method
  def find_article
    @article = Article.find(params[:id])
  end
end
```
<br>

## 14. How do you handle _parameters_ in a _controller action_?

When you work in Ruby on Rails, most of the data is **accessed through ActiveRecord models**, either by direct retrieval or through relationships.

However, there are cases where you need to **directly handle parameters**, especially for more complex data flows or for actions that don't involve models.

### Parameter Handling

Rails' `ActionController::Parameters` ensures that **incoming parameters are secure**, and it structures them, providing tools for effortless processing. 

This takes away the need for manual sanitation code or direct manipulation of the `params` hash.

#### Security Benefits

- **Strong Parameters**: Specify which parameters are permitted. This improves security by explicitly disallowing unwanted parameters.
- **Mass-Assignment Protection**: Provide extra control over which attributes can be assigned, reducing the risk of overwriting sensitive data.
- **Automatic Type Casting**: The system attempts to cast parameters to the expected types, increasing predictability and security. For instance, if you expect an integer, non-integer strings would be rejected.

### Common Patterns

#### Parameter-Driven Logic

In some cases, actions are required based on specific parameters. While it can be more favorable to approach this through RESTful routes and action methods, there might be cases where a single action has varied responsibilities based on input. 

However, it's advisable to use such patterns sparingly, favoring clear and predictable semantics that are easier to maintain.

#### Direct Parameter Manipulation

Direct `params` hash manipulation presents challenges and isn't recommended in most scenarios due to:

- **Security Vulnerabilities**: Manual sanitation can overlook potential threats.
- **Code Maintainability**: If the app grows, it becomes increasingly challenging to track and manage direct hash manipulations.

### Code Example: Accepting User Input as Parameters

Here is the Python code:

```python
class UsersController():
    # Adding or updating user via RESTful methods.
    
    def create(self, request):
        # Example URL: /users
        new_user = User.create(user_params(request))
        return HttpResponse(f"User {new_user.name} created successfully.")
    
    def update(self, request, user_id):
        # Example URL: /users/123
        user = User.find(user_id)
        user.update(user_params(request))
        return HttpResponse(f"User {user.name} updated successfully.")
    
    # Additional action to suspend a user based on parameters.
    def suspend(self, request):
        # Example URL: /users/suspend
        user = User.find_by(email: params['email'])
        user.suspend
        return HttpResponse(f"User {user.email} suspended successfully.")

    def user_params(self, request):
        return {
            'name': request['name'],  # example: from a form or JSON payload
            'email': request['email'],  # More fields can be added
        }
```

### Recommendations

- **Utilize RESTful Routing**: Whenever possible, let routing and clear naming suit the action types. This simplifies your codebase and makes it more predictable for others.
- **Leverage Form Objects and Service Objects**: For complicated or non-standard operations, consider using form or service objects to encapsulate your logic and keep your controllers slim.
- **Follow Conventions**: Unless you have compelling reasons to do otherwise, sticking to the conventions of Rails leads to a more maintainable and cohesive codebase.
<br>

## 15. What _filters_ are available in _Rails controllers_ and how do you use them?

**Filter** methods in **Ruby on Rails** provide a powerful means to run code either before or after specific controller actions, or as a "catch-all" for multiple actions.

### Prefilters

Prefilters, which include `before_action`, `prepend_before_action`, define methods that execute before the associated action(s).

#### Code Example: Using `before_action`

```ruby
class OrdersController < ApplicationController
  before_action :authenticate_user!
  
  def show
    @order = Order.find(params[:id])
  end
end
```

In this example, the `authenticate_user!` method is run before `OrdersController#show`.

### Postfilters

Postfilters, including `after_action` and `prepend_after_action`, run methods after the respective action is executed.

#### Code Example: Using `after_action`

```ruby
class InvoicesController < ApplicationController
  after_action :log_invoice_creation

  def create
    @invoice = Invoice.create(invoice_params)
    redirect_to @invoice
  end
end
```

Here, `log_invoice_creation` is invoked after the `InvoicesController#create` action.

### Around Filters

Around filters, defined using the `around_action` method, encapsulate the associated action, providing a mechanism for code to run both before and after.

#### Code Example: Using `around_action`

```ruby
class PaymentsController < ApplicationController
  around_action :log_payment_process

  def process
    # Payment processing logic
  end
end
```

The `log_payment_process` method would wrap around `PaymentsController#process`, enabling both pre- and post-action code execution.

### Functional Use-Cases

- **Authorization and Authentication**: Execute before actions to validate user credentials.
- **Parameter Processing**: Use as a "catch-all" for standardizing input data.
- **Caching**: Employ after actions to cache certain results, optimizing performance.
- **Log Maintenance**: Run methods before and after actions for detailed logging.

### Application

By integrating filters within the **Rails** controllers, one can ensure a centralized, systematic workflow across actions, fortifying the consistency and efficiency of the application.

Using ActionController::Live live streaming is possible.
Collects browser connection specific data, capturing segments of the activities.
An application is initially identified by its **controller** and **action**, both of which can be customized. For instance, a **PostsController** typically has actions for creating, reading, updating, and deleting posts. Each of these actions comprises a series of steps aimed at handling the request, including interaction with the model and view layers.

### Method Descriptions

#### Parameters Requiring Authentication

- `current_user`: Resolves the current user object, often used for user-specific operations.
- `authenticate_user!`: Checks for user authentication and halts the chain if authentication fails.
- `user_signed_in?`: Provides a boolean indicative of user presence.

#### Public Methods

- `public`: Marks an action as available publicly without requiring any pre-action steps.
- ``External Redirects``: `redirect_to` or `redirect_back` can guide request flow externally.

#### Response Handling

- `respond_to`: Determines response format based on the associated MIME type. Helps when handling different content types like HTML, XML, or JSON. Handles AJAX requests more gracefully.
- `head`: Sends an empty header-contained response, often used as a status verifier.

#### About Sessions

- `reset_session`: Wipes the session clean, helping to terminate user's session expressly. Can be useful when implementing logout functionalities.

#### Redirects

- `redirect_to`: Transfers request to a different URL, typically another action.
  Example: `redirect_to dashboard_url, alert: "Operation failed!"`
- `redirect_back`: Directs to the previous location known to the application, useful for things like cancellations or dynamically tracked pages like error pages or search results.

### Action Defaults

By default, an action is set up to respond to an HTTP GET request. This renders an HTML view generated by the details obtained via the action. Standard naming practices and RESTful routing lead to predictable action and view associations.

#### Common Action Naming Conventions

- **New Record Creation**: Present a form for resource creation. Method: GET. Action: `new`.
- **Record Creation**: Complete the process of creating a new resource using data from `params`. Method: POST. Action: `create`.
- **Record Viewing**: Render the view of a previously created record. Method: GET. Action: `show`.
- **Record Editing**: Present a form for updating an existing resource, often pre-populated with the current data. Method: GET. Action: `edit`.
- **Record Update**: Complete the process of modifying an existing record using data from `params`. Method: PATCH or PUT. Action: `update`.
- **Record Deletion**: Remove an existing resource. Method: DELETE. Action: `destroy`.

These mapping conventions between standard controller actions and RESTful CRUD operations not only are insightful for any developer following these conventions, but also they enable a standardized and intuitive use of resources.

### Rake Tasks : Making Use of it

'Rake' tasks, such as those used for database setup or clean-up tasks, can be fused directly into controller actions, providing a mechanism to manage or observe these tasks through URL requests. This strategy aligns well when ensuring a controlled setting for such tasks.

#### Code Example: Integration of Rake Task within a Controller Action

```ruby
class DatabaseTasksController < ApplicationController
  def reset
    ActiveRecord::Tasks::DatabaseTasks.reset_database
    redirect_to root_path, notice: 'Database reset complete.'
  end
end
```

In this example, navigating to `localhost:3000/database_tasks/reset` triggers the reset action.
<br>



#### Explore all 100 answers here 👉 [Devinterview.io - Ruby on Rails](https://devinterview.io/questions/web-and-mobile-development/ruby-on-rails-interview-questions)

<br>

<a href="https://devinterview.io/questions/web-and-mobile-development/">
<img src="https://firebasestorage.googleapis.com/v0/b/dev-stack-app.appspot.com/o/github-blog-img%2Fweb-and-mobile-development-github-img.jpg?alt=media&token=1b5eeecc-c9fb-49f5-9e03-50cf2e309555" alt="web-and-mobile-development" width="100%">
</a>
</p>

