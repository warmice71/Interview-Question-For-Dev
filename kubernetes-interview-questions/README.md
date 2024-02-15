# 42 Important Kubernetes Interview Questions

<div>
<p align="center">
<a href="https://devinterview.io/questions/software-architecture-and-system-design/">
<img src="https://firebasestorage.googleapis.com/v0/b/dev-stack-app.appspot.com/o/github-blog-img%2Fsoftware-architecture-and-system-design-github-img.jpg?alt=media&token=521fd2a9-0d56-49c0-a723-9bd6ca081893" alt="software-architecture-and-system-design" width="100%">
</a>
</p>

#### You can also find all 42 answers here 👉 [Devinterview.io - Kubernetes](https://devinterview.io/questions/software-architecture-and-system-design/kubernetes-interview-questions)

<br>

## 1. What is _Kubernetes_, and why is it used for container _orchestration_?

**Kubernetes** provides a robust platform for deploying, scaling, and managing containerized applications. Unlike Docker Swarm, which is specifically designed for Docker containers, Kubernetes is container-agnostic, supporting runtimes like Containerd and CRI-O. Moreover, Kubernetes is more feature-rich, offering features such as service discovery, rolling updates, and automated scaling.

### Key Kubernetes Features

#### Pod Management

- **Why**: Pods are the smallest deployable units in Kubernetes, encapsulating one or more containers. This brings flexibility and makes it effortless to manage multi-container applications.
- **Core Concepts**: Deployments, Replication Controllers, and ReplicaSets ensure that a specific number of replicas for the Pod are running at all times.
- **Code Example**: In YAML, a Pod with two containers might look like:

  ```yaml
  apiVersion: v1
  kind: Pod
  metadata:
    name: pod-with-two-containers
  spec:
    containers:
    - name: container1
      image: image1
    - name: container2
      image: image2
  ```

#### Networking

- **Why**: Kubernetes assigns each Pod its unique IP address, ensuring communication across Pods. It also abstracts the underlying network, simplifying the deployment process.
- **Core Concepts**: Services, Ingress, and Network Policies provide different levels of network abstraction and control.

#### Persistent Storage

- **Why**: It offers a straightforward, standardized way of managing storage systems, making it a great solution for databases and stateful applications.
- **Core Concepts**: Storage classes, Persistent Volumes (PVs) and Persistent Volume Claims (PVCs) abstract the underlying storage technologies and provide dynamic provisioning and access control.

#### Cluster Scaling

- **Why**: Kubernetes can automate the scaling of Pods based on CPU or memory usage, ensuring optimal resource allocation and performance.
- **Core Concepts**: Horizontal Pod Autoscaler (HPA) dynamically scales the number of replica Pods in a Deployment, ReplicaSet, or StatefulSet. Cluster autoscaler scales underlying infrastructure, including nodes.

#### Resource Management

- **Why**: Kubernetes makes it easy to manage and allocate resources (CPU and memory) to different components of an application, ensuring that no single component degrades overall performance.
- **Core Concepts**: Resource Quotas and Limit Ranges help define the upper limits of resources that each object can consume.

#### Batch Execution

- **Why**: For quick, efficient tasks, Kubernetes provides a Job and a CronJob API to manage such tasks.
- **Core Concepts**: Jobs and CronJobs manage the execution of Pods over time, guaranteeing the desired state.

#### Clusters Maintenance

- **Why**: Kubernetes enables non-disruptive updates, ensuring cluster maintenance without downtime.
- **Core Concepts**: Rolling updates for Deployments and Pod disruption budgets during updates help maintain availability while updating.

#### Health Checks

- **Why**: Kubernetes actively monitors and checks the health of containers and workloads, ensuring quick remediation in case of issues.
- **Core Components**: Liveness and Readiness probes are used to determine the health of container-based applications. Kubernetes restarts containers that don't pass liveness probes and stops routing traffic to those that don't pass readiness probes.

#### Secrets Management

- **Why**: Kubernetes provides a secure way to manage sensitive information, such as passwords, OAuth tokens, and SSH keys.
- **Core Components**: Secrets and ConfigMaps are different types of objects used to centrally manage application configuration and secrets.

#### Automated Deployments

- **Why**: Kubernetes facilitates gradual, controlled updates of applications, reducing the risk of downtime or failure during deployment.
- **Core Concepts**: Deployments, with their built-in features like rolling updates and rollbacks, provide a mechanism for achieving zero-downtime deployments.

#### Metadata and Labels

- **Why**: Labels help organize and select and workloads in a Kubernetes cluster, and annotations can attach arbitrary non-identifying information to objects.
- **Core Components**: Labels are key/value pairs that are attached to Kubernetes resources to give them identifying attributes. Annotations are used to attach arbitrary non-identifying metadata to objects.

#### Mechanism to Extend Functionality

- **Why**: **Kubernetes** is designed to be extensible, allowing developers to write and register custom controllers or operators that can manage any object.

### Multi-Cluster Deployment

Kubernetes provides capabilities for effective multi-cluster deployment and scaling, be it within a single cloud provider or across hybrid and multi-cloud environments. With tools like **Kubefed**, you can define and manage resource configurations that are shared across clusters.

### Cost and Productivity Considerations

Kubernetes also offers significant cost efficiencies. It optimizes resource utilization, utilizing CPU and memory more effectively, reducing the need for over-provisioning.

On the productivity front, Kubernetes streamlines the development and deployment pipelines, facilitating agility and enabling rapid, consistent updates across different environments.

### Why Choose Kubernetes for Orchestration?

- **Portability**: Kubernetes is portable, running both on-premises and in the cloud.
- **Scalability**: It's proven its mettle in managing massive container workloads efficiently, adapting to varying resource demands and scheduling tasks effectively.
- **Community and Ecosystem**: A vast and active community contributes to the platform, ensuring it stays innovative and adaptable to evolving business needs.
- **Extensive Feature Set**: Kubernetes offers rich capabilities, making it suitable for diverse deployment and operational requirements.
- **Reliability and Fault Tolerance**: From self-healing capabilities to rolling updates, it provides mechanisms for high availability and resilience.
- **Automation and Visibility**: It simplifies operational tasks and offers insights into the state of the system.
- **Security and Compliance**: It provides role-based access control, network policies, and other security measures to meet compliance standards.

### Master-Node Architecture

Kubernetes follows a master-node architecture, divided into:

- **Master Node**: Manages the state and activities of the cluster. It contains essential components like the API server, scheduler, and controller manager.
  - **API Server**: Acts as the entry point for all API interactions.
  - **Scheduler**: Assigns workloads to nodes based on resource availability and specific requirements.
  - **Controller Manager**: Maintains the desired state, handling tasks like node management and endpoint creation.
  - **etcd**: The distributed key-value store that persists cluster state.

- **Worker Nodes**: Also called minions, these are virtual or physical machines that run the actual workloads in the form of containers. Each worker node runs various Kubernetes components like Kubelet, Kube Proxy, and a container runtime (e.g., Docker, containerd).

### Need for Orchestration

Containers, while providing isolation and reproducibility, need a way to be managed, scaled, updated, and networked. This is where **Orchestration platforms** like Kubernetes come in, providing a management layer for your containers, ensuring that they all work together in harmony.
<br>

## 2. Describe the roles of _master_ and _worker nodes_ in the _Kubernetes_ architecture.

In a **Kubernetes** cluster, each **type of node** has a distinct responsibility:

- **Master Node**: 
  - Also known as **Control Plane Node**, it manages the entire cluster.
  - Houses the **API Server**, **Controller Manager**, and **Scheduler**, primarily responsible for cluster orchestration.
  - It initializes and configures workloads, spans across several Master nodes to ensure high availability, and typically doesn't run user applications. 
  - Also, the **etcd** database is often hosted separately or as a clustered data store.

- **Worker Node**:
  - Also called **Minion** or **Ingress Node**.
  - Serves as the **compute** layer of the cluster and by design hosts the containers that make up the actual workloads ('pods') or app services.
  - Acts as a **bridge** to offload the networking and monitoring overhead from the Master node and to ensure the security of the cluster.

- **Other Nodes**:
  - Special-purpose nodes or clusters may introduce other node types, such as dedicated storage or networking elements. However, these are not universal and won't be found in most standard Kubernetes setups.


It is essential to understand the distinct roles of these node types for effectively managing a Kubernetes cluster.

### Kubernetes Cluster Overview

- **Nodes**: Physical or virtual servers that form the worker or master nodes.
  - May also include additional specialized nodes for storage, networking, or other purposes.

- **Kubelet**: An agent installed on each worker node and helps the node connect with the main control panel. It takes `PodSpecs`, which define a group of containers that require to be coordinated, and guarantees that the identified containers are running and healthy.

- **Kube-Proxy**: A network agent on each node for managing network connectivity to local deployments.

- **Control Panel**: A collection of critical processes steamy on a cluster's master nodes to regulate the cluster management and API server.
  - **API Server**: Resembles the front door to the cluster. All external communications cease here.
  - **Scheduler**: Picks the best node for a program to run on.
  - **Controller manager**: Monitors the present state of the cluster and attempts to bring it to the desired state.

- **External Cloud**: Offers the physical hardware where your cluster will run.
<br>

## 3. How do _Pods_ facilitate the management of _containers_ in _Kubernetes_?

In **Kubernetes**, a **Pod** serves as the smallest deployable unit, generally encapsulating a single application. The primary function of the Pod is to provide a cohesive context for executing one or more containers.

### Key Characteristics

- **Context Sharing**: Containers within the same Pod share a common network namespace, hostname, and storage volumes, optimizing their interactions.

- **Logical Bundling**: Designates one of the containers as the primary application with the remaining containers often providing supplementary services or shared resources.

- **Atomicity**: Containers in a single Pod are scheduled onto the same node and execute in close proximity. This ensures that they are either all running or all terminated.

### Core Functionalities

1. **Networking**: Containers within the same Pod are reachable via `localhost` and have a single, shared IP address. This design simplifies internal communication.

2. **Resource Sharing**: All containers in a Pod have identical resource-sharing provisions, ensuring that they have the same CPU and memory limits.

3. **Storage Volumes**: Shared volumes can be mounted across all containers within a Pod to promote data sharing among its constituent containers.

4. **Lifecycle Coordination**: Defines the life cycle for all containers in a Pod, such that if one container exits, the termination affects the entire Pod, consistent with the atomic execution concept.

### Core Relationship with Workloads

- **Dedicated Workloads**: Pods that host a single container are often used for self-sufficient, standalone tasks. These tasks have specific resource requirements and are designed to run independently.

- **Coordinated Workloads**: Pods with multiple containers emphasize complementarity. These containers often tackle related tasks, such as syncing data, logging, or serving as administrative dashboards.

### YAML Configuration

Here is the YAML configuration:

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: my-pod
spec:
  containers:
    - name: primary-container
      image: primary-image
    - name: secondary-container
      image: secondary-image
  volumes:
    - name: shared-data
      emptyDir: {}
```

In this instance, `primary-container` is the main application container, and `secondary-container` is the auxiliary container. Both containers share the same storage volume `shared-data`.

### Management Tools

Popular dev-ops tools, like Helm, provide higher-level abstractions for managing Kubernetes resources, such as the Pod. This simplifies deployment and scaling tasks, allowing for centralized configuration management.


### Benefits of Pod:

- **Resource Efficiency**: Pods refrain from having resource duplication. Each constituent container operates within the same resource environment, reducing wastage.

- **Lifecycle Consistency**: All containers in a Pod are deployed, scaled, and managed as a single unit, promoting consistency in their execution.
<br>

## 4. What types of _Services_ exist in _Kubernetes_, and how do they facilitate _pod communication_?

**Kubernetes** offers different types of **services** for flexible & reliable communication between pods. Let's explore each.

### Service Types

#### ClusterIP
- **Role**: Internal communication within the cluster.
- **How it Works**: Pods communicate with the Service's fixed IP and port. Service distributes traffic among pods based on defined selector.
- **Practical Use**: Ideal for back-end services where direct access from external sources isn't necessary.

#### NodePort
- **Role**: Exposes the Service on a static port on each node of the cluster.
- **How it Works**: In addition to the ClusterIP functionality, it also opens a specific port on each node. This allows external traffic to reach the service through the node's IP address and the chosen port.
- **Practical Use**: Useful for reaching the service from outside the cluster during development and testing phases.

#### LoadBalancer
- **Role**: Makes the Service externally accessible through a cloud provider's load balancer.
- **How it Works**: In addition to NodePort, this service provides a cloud load balancer's IP address, which then distributes traffic across Service Pods.
- **Practical Use**: Especially useful in cloud settings for a public-facing, production deployment.

#### ExternalName
- **Role**: Maps the Service to an external service using a DNS name.
- **How it Works**: When the Service is accessed, Kubernetes points it directly to the external (DNS) name provided.
- **Practical Use**: Useful if you want environments to have a uniform way of accessing external services and don't want to use environment-specific methods like altering /etc/hosts.

#### Headless
- **Role**: Disables the cluster IP, allowing direct access to the Pods selected by the Service.
- **How it Works**: This type of Service doesn't allocate a virtual IP (ClusterIP) to the Service, letting the clients directly access the Pods. It returns the individual Pod IPs.
- **Practical Use**: Useful for specialized use cases when direct access to Pod IPs is necessary.

### Selectors and Endpoints

For many types of Service, the **selector** identifies which Pods to include in the service. The matching Pods are called **endpoints** for the Service.

Kubernetes retrieves these endpoints from the API server, and the Service kube-proxy sets up appropriate iptables rules (or equivalent on other platforms or using `ipvs`) to match the Service behavior.

```yaml
apiVersion: v1
kind: Service
metadata:
  name: my-service
spec:
  selector:
    app: MyApp
    type: Frontend
  ports:
    - protocol: TCP
      port: 80
      targetPort: 9376
```

Here, the Service 'my-service' selects all the Pods with labels `app=MyApp` and `type=Frontend` and forwards the traffic on the port `80` to the `targetPort` `9376`.
<br>

## 5. Explain the functionality of _Namespaces_ in _Kubernetes_.

**Kubernetes** uses **Namespaces** to create separate virtual clusters on top of a physical cluster. This enables multi-tenancy, providing a powerful way to manage diverse workloads efficiently.

### Key Features

- **Resource Isolation**: Namespaces offer a level of separation, ensuring resources like pods, services, and persistent volumes are distinct to a Namespace.

- **Network Isolation**: Each Namespace has its IP, enabling isolated network policies and container-to-container communication.

### Use-Cases

#### Development, Testing, and Staging

- **Multi-Environment Segregation**: Namespaces can delineate development, testing, and staging environments.

#### Multi-Team and Multi-Project Support

- **Multi-Tenancy**: Namespaces allow multiple teams to work independently in the same cluster.

- **Client Isolation**: Service objects in a Namespace are only visible to clients in the same Namespace, providing clear-cut network boundaries.

#### Security and Policy Enforcement

- **Resource Quotas**: Namespaces help establish quotas to govern resource consumption for projects or teams.

- **Limit Ranges**: Namespaces can define minimum and maximum limitations on resource memory and CPU for each container.

#### Network Segmentation and IP Management

- **Ingress Configuration**: Namespaces assist in external-to-internal network mapping and traffic routing.

- **IP Management**: Each Namespace can have distinct IP ranges to uniquely identify services and pods.
<br>

## 6. Differentiate between _Deployments_, _StatefulSets_, and _DaemonSets_.

**Kubernetes** offers various abstractions to manage containerized applications according to specific operational needs. **Deployments**, **StatefulSets**, and **DaemonSets** are all essential controllers in this regard.

### Key Distinctions

- **Deployments** are suitable for stateless, replicated applications, and mainly focus on the management of pods.
  
- **StatefulSets** are designed for stateful applications requiring stable, unique network identities and persistent storage.

- **DaemonSets** are for running agents on each node for system-level tasks.

### Deployments

- **State**: Stateless
  
  These are ideal for microservices that do not store data and can be horizontally scaled.

- **Pod Management**: Managing a replica set of pods.

- **Inter-Pod Communication**: Achieved through services.

- **Storage**: Volatile. Data does not persist beyond the pod's lifecycle.

### StatefulSets

- **State**: Stateful
  
  Suitable for applications that store persistent data, control startup order, and require unique network identities.

- **Pod Management**: Provides sticky identity, persistence, and orderly deployment and scaling. 

- **Inter-Pod Communication**: Managed through stable network identities.

- **Storage**: Provides mechanisms for persistent storage.

### DaemonSets

- **State**: Node-Focused
  
  Ideal for workloads with demon-like functionalities that are needed on each node (e.g. log collection, monitoring).

- **Pod Management**: Ensures one pod per node.

- **Inter-Pod Communication**: Not a primary concern.

- **Storage**: Depends on the specific use case.
<br>

## 7. How do _ReplicaSets_ ensure _pod availability_?

**Kubernetes** provides robust mechanisms, such as **ReplicaSets**, to ensure consistent pod availability. In the context of failure scenarios or manual scaling, it is essential to understand how **ReplicaSets** guarantee pods are up and running according to the defined configuration.

### Key Components

- **Pod Template**: It specifies the required state for individual pods within the configured ReplicaSet.

- **Controller-Reconciler Loop**: This Control Plane component continuously monitors the cluster, compares the observed state against the desired state specified in the ReplicaSet, and takes corrective actions accordingly. If there's a mismatch, this loop is responsible for making the necessary adjustments.

- **Replica Level**: Each ReplicaSet specifies the desired number of replicas. It's the responsibility of the Controller-Reconciler Loop to ensure that this count is maintained.

### Action Workflow

#### Initial Deployment

During the initial setup, the ReplicaSet creates the specified number of pods and ensures they are in an `Up` state.

#### Observation and Feedback Loop

The Controller-Reconciler continuously monitors pods. If the observed state deviates from the Pod Template, corrective action is initiated to bring the system back to the specified configuration.
   -  **Failure Detection**: If a pod is unavailable or not matching the defined template, the Controller-Reconciler identifies the anomaly in the system.
   
   -  **Self-Healing**: The Controller-Reconciler instantiates new pods, replaces unhealthy ones, or ensures the required number of pods is available, maintaining the ReplicaSet's defined state.

#### Horizontal Scaling
The ReplicaSet allows for dynamic scaling of pods in response to changes in demand.

   - **Auto-Scaling**: The Controller-Reconciler automatically scales the number of pods to match the configured thresholds or metrics.

   - **Manual Scaling**: Administrators can manually adjust the number of replicas.

### Code Example: Availability Reporting System

Kubernetes YAML:

```yaml
apiVersion: apps/v1
kind: ReplicaSet
metadata:
  name: av-replicaset
  labels:
    tier: frontend
spec:
  replicas: 2
  selector:
    matchLabels:
      app: availability
  template:
    metadata:
      labels:
        app: availability
    spec:
      containers:
      - name: av-reporter
        image: av-reporter:v1
```

In this example, we ensure that two pods of the `av-reporter:v1` image are continuously available, serving as a live status reporter for an availability system.
<br>

## 8. What are _Jobs_ in _Kubernetes_, and when is it suitable to use them?

In **Kubernetes**, a **job** object is used to run a specific task to completion or a certain number of times. It's ideal for tasks that are rather short and encapsulate work that isn't part of the ongoing application processes.

### Key Characteristics of Jobs

- **Pod Management**: Jobs create one or more Pods and manage their lifecycle, ensuring successful completion.
 
- **Completions**: You can specify the number of successful completions, especially useful for batch tasks.

- **Parallelism**: Control how many Pods run concurrently. This feature allows for efficient management of resources.

- **Pod Cleanup**: After the task has been completed, Jobs ensure that related Pods are terminated. They might also garbage collect completed Jobs, depending on your settings.

- **Auto-Restart**: Jobs do not restart by default if successful. They can be configured to restart on failure.

### The Three Main Job Types

- **Serial Jobs**: Ensure tasks are completed exactly once.

- **Parallel Jobs**: Suitable for tasks where some level of parallel processing can be beneficial for performance.

- **Work Queues**: Suitable for tasks where a specific number of parallel tasks is defined and managed.

### Scenario-Specific Best Fits

1. **Data Processing**: For processing a batch of records or data sets. For example, a tech company might use it in a data pipeline to process thousands of records in chunks.

2. **Clean-up Tasks**: For periodic clean-up, such as an e-commerce site cleaning up expired user data.

3. **Software Compilation**: Useful in CI/CD pipelines to parallelize software builds.

4. **Cron Jobs**: For scheduling recurring batch processes, such as taking database backups nightly.

5. **Metering or Accounting**: Useful for counting or tallying records, possibly in near-real-time.

6. **Health Checks**: Occasionally, when more sophisticated health checks are needed, for tasks perhaps beyond the remit of a Liveness and Readiness check.

7. **Resource Acquisition**: For occasional resource acquisition tasks – imagine a scenario where a system occasionally scales on demand and requires a specific number of resources at run-time.
<br>

## 9. How do _Labels_ and _Selectors_ work together in _Kubernetes_?

In Kubernetes, **Labels** are key-value pairs attached to resources, and **Selectors** are the tools used to manage and interact with these labeled resources.

### Why Use Labels and Selectors?

- **Efficient Grouping**: Labels group resources, simplifying management and configuration.
  
- **Decoupling**: Resources are decoupled from consuming services through selectors, making management flexible and robust.

- **Queries**: Selectors filter resources by matching label sets, enabling focused actions.

### Core Concepts

#### Labels

Kubernetes resources, such as Pods or Services, are associated with labels to indicate attributes. A `spec` section can include labels during resource definition.

Example labels:

```yaml
metadata:
  labels:
    environment: prod
    tier: frontend
```

#### Resources

Select:

- **All** resources without any specific labels are significant when not tagged.
  
- **Specific Resources**: Indicate labels to MATCH (AND condition) for resource retrieval.

Example, to find all "frontend" Pods in the "prod" environment:

```yaml
matchLabels:
  environment: prod
  tier: frontend
```

#### Resources that Supports Labels and Selectors

1. **Services**: Use to route and expose network traffic.
  
2. **ReplicaSets/Horizontally Scaling Controllers**: Helps in load balancing, scaling, and ensuring availability.
  
3. **DaemonSets/Node Level Controllers**: For system-level operations on all or specific nodes.

4. **Deployments/Rolling Updates and Rollbacks**: Manages dynamic scaling and maintains consistent state.

5. **Jobs/CronJobs**: Facilitates task scheduling and execution.

6. **StatefulSets/Persistent Storage**: Ensures stable and unique network identities for stateful applications.

7. **Pods/Microservices**: Basic building blocks for containerized applications.

8. **Ingress**: Routes external HTTP and HTTPS to Services.

9. **Network Policies**: Defines network access policies.

10. **Endpoints**: Populates the subset of backend Pods for a Service.

11. **PersistentVolumeClaims**: Dynamic storage provisioning and management.

12. **Virtual Machines Deployment**: Uses for virtual machines creation and management.

13. **CronJobs**: Facilitates job scheduling.

14. **PodDisruptionBudgets**: Ensures efficient resource allocation during disruptions.

15. **ServiceMonitor/EndpointSlice**: Specific to monitoring and service discovery.
<br>

## 10. What would you consider when performing a _rolling update_ in _Kubernetes_?

In **Kubernetes**, a **rolling update** ensures your application is updated without downtime. This process carefully replaces old pods with new ones, employing a **health check mechanism** for a smooth transition.

A rolling update involves several components, including the **Replica Set**, the **Update Control Loop**, the **Pod Update Process**, and **Health Checking** of pods. Here are their details:

### Upgrade Process Flow

1. **Replica Set**: Initiates the update by altering the pods it supervises.

2. **Pod Lifecycle**: Old pods gradually terminate while new ones start up according to the update strategy, such as `maxUnavailable` and `maxSurge`.

3. **Readiness Probes**: Each new pod is verified against its readiness to serve traffic, ensuring the application is live and responsive before the transition.

4. **Liveness Probes**: These checks confirm the stability of new pods. If a pod fails a liveness probe, it's terminated and replaced with a new version, maintaining application reliability.

5. **Rolling Update Status** `status`: Informs about the progress of the update.

6. **Annotations from Provider**: Kubernetes providers may supply additional insights, such as `spec.rollingUpdate.strategy`.

### Code Example: Rolling Update Process

Here is the Kubernetes YAML code:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: my-app
spec:
  replicas: 3
  strategy:
    rollingUpdate:
      maxUnavailable: 2
      maxSurge: 1
      type: RollingUpdate
  selector:
    matchLabels:
      app: my-app
  template:
    metadata:
      labels:
        app: my-app
    spec:
      containers:
      - name: my-app-container
        image: my-app:v2
        readinessProbe:
          httpGet:
            path: /ready
            port: 8080
        livenessProbe:
          httpGet:
            path: /live
            port: 8080
```
<br>

## 11. How do _Services_ route traffic to _pods_ in _Kubernetes_?

In **Kubernetes**, Services serve as an abstraction layer, enabling consistent access to Pods. Traffic is **routed** to Pods primarily via **Selectors** and **Endpoints**.

### Label Selectors

- **Purpose**: Establish traffic endpoints based on matching labels.
- **Workflow**: Pods are labelled, and Services are configured to match these labels. Upon connectivity, the Service pairs the request to Pods having corresponding labels.
- **Configuration**: Defined in the Service configuration file.

### Endpoints

- **Dynamic Mapping**: Enables fine-grained control over which Pods receive traffic.
- **Workflow**: Automatically managed and updated. When Pods are created or terminated, corresponding Endpoints are adjusted to ensure traffic flow continuity.

### Routing Modes

1. ClusterIP: The default behavior, where each Service is assigned a stable internal IP, accessible only within the cluster.
2. NodePort: Exposes the Service on each Node's IP at a specific port, allowing external access.
3. LoadBalancer: Provisioned by an external cloud provider, creating a load balancer for accessing the Service from outside the cluster.
4. ExternalName: Maps a Service to a DNS name, effectively making the Service accessible from inside the cluster using that DNS name.

### Session Affinity

- **Purpose**: Grants control over the duration for which subsequent requests from the same client are sent to the same Pod.
- **Measured Using Cookies**: When set to `ClientIP`, the user's IP address is used to direct future requests to the same Pod. Using `None` ensures that each request is independently routed.

### Service Types in code and YAML

#### Kubernetes YAML
```yaml
apiVersion: v1
kind: Service
metadata:
  name: my-service
spec:
  selector:
    app: MyApp
  ports:
    - protocol: TCP
      port: 80
      targetPort: 9376
  type: NodePort
```

#### Code Example: Select All Pods with App: MyApp Label

Kubernetes Service:

```yaml
apiVersion: v1
kind: Service
metadata:
  name: my-service
spec:
  selector:
    app: MyApp
  ports:
    - name: http
      protocol: TCP
      port: 80
      targetPort: 9376
  type: LoadBalancer
```

Python code that Accesses the Service:

```python
import requests

response = requests.get('http://my-service/')
print(response.text)
```
<br>

## 12. Describe the purpose of _Ingress_ and its key components.

**Ingress** is a powerful Kubernetes service that acts as an HTTP and HTTPS **load balancer** and provides **routing** for external traffic to services in a cluster. Let's look at its key components:

### Key Components

- **Ingress Resource**: This is a Kubernetes object that acts as a configuration file for managing external access to services in a cluster. It defines rules and settings for how traffic should be routed.

- **Ingress Controller**: The Ingress Controller is in charge of obeying the Ingress Resource's rules and configuring the load balancer and traffic routing. Most often, the Ingress Controller is implemented through a Kubernetes extension or a third-party tool, like NGINX or Traefik. 

- **Load Balancer**: Operating at layer 7 of the OSI model, the Load Balancer directs traffic to available services based on rules. The Ingress Controller configures and manages this Load Balancer on behalf of the cluster.

  **Note**: A cloud provider might offer its own Load Balancer, but the Ingress Controller can also use standard cloud provider tools or its own mechanism if the cluster isn't on a cloud platform.

- **Rules**: These are defined within the Ingress Resource and specify how incoming traffic should be handled. Rules typically match a specific path or domain and define the associated backend Kubernetes service.

### When to Use Ingress

Ingress is an ideal fit for clusters needing to route HTTP or HTTPS traffic. It's especially beneficial in microservices architectures, as it centralizes and simplifies access to services. This, in turn, improves security, ease of management, and facilitates traffic optimizations like caching.

It should be noted that newer configurations, such as gateway API, offer more extensibility and include features such as traffic splitting and enhanced security measures, which might make them a better choice in certain contexts.
<br>

## 13. Explain _Kubernetes DNS_ resolution for _services_.

**Kubernetes** leverages a consistent, centrally managed Domain Name System (DNS) for containers and services, offering ease of discovery and effictive communication within the cluster.

### Core DNS vs. Kubernetes DNS

In early versions of Kubernetes, **SkyDNS** powered service discovery. However, **CoreDNS** has succeeded it as the default DNS server.

- **CoreDNS** is more extendable and easier to manage.
- Its modular nature means you enable specific features through plugins.

### DNS Resolution Workflow

1. Nodes and pods use `kube-dns` or `coredns` as their predefined DNS servers.
2. The DNS server typically resides within a Kubernetes cluster and knows all service names and their IP addresses.
3. On receiving a DNS query, the DNS server tracks IP changes and ensures name-to-IP mapping.

### Example: Query Flow

1. **Pod Initiates DNS Request**: A pod wants to connect to a service inside or outside the cluster.
2. **DNS Query**: The pod sends a DNS query via the specified server (K8s or custom).
3. **DNS Server**: The server processes the query.
4. **Query Results**: Based on pod's namespace, service name, and domain suffix, the DNS server returns the corresponding IP(s).

### DNS Requirements in the Cluster

- **Service Discovery**: Pods need to locate services. DNS offers an effective way, abstracting the complexity of directly handling service discovery.
- **Name Resolution**: Pods and other entities use DNS to get a service's IP address. The DNS server ensures efficient updates, so pods always have the most accurate IP.

### Considerations for Inter-Pod Communication

1. **Direct Cluster IP**: Services communicate via Cluster IP.
2. **Unrestricted or Port-Defined Communication**: Use the service type of "ClusterIP".
3. **Custom Domains**: For custom domains, specify appropriate service names so the DNS server properly resolves their IPs.

### Namespace Segregation

Without namespace information, separate services with the same name and different namespaces might be unreachable. Including namespace info ensures accuracy.

### Key Configuration Parameters

- **pod** / `spec.dnsPolicy`: Set `ClusterFirst` to utilize the default DNS service.
- **pod** / `spec.dnsConfig`: Specify configs for custom DNS.
- **service** : Utilize `spec.clusterIP` for manual IP assignments. This avoids potential IP address reassignment.

### Inter-Kubernetes Cluster Communication

- For multi-cluster communication, several solutions are available, including direct IP endpoint access and Ingress. DNS resolution strategies can consider these factors.

#### Code Example: Access Service from a Pod

Here is the Kubernetes YAML configuration:

```yaml
apiVersion: v1
kind: Service
metadata:
  name: my-service
  namespace: test-ns
spec:
  selector:
    app: MyApp
  ports:
    - protocol: TCP
      port: 80
      targetPort: 9376

---
apiVersion: v1
kind: Pod
metadata:
  name: dns-example
  namespace: test-ns
spec:
  containers:
    - name: test
      image: your-image
      command: [ "nslookup", "my-service" ]
```
<br>

## 14. What is the _Container Network Interface (CNI)_, and how does it integrate with _Kubernetes_?

The **Container Network Interface (CNI)** is a standard for connecting containers and pods in Kubernetes with underlying networking hardware.

### Integrating with Kubernetes

Containers within a Pod share network namespaces, making inter-container communication straightforward. However, Pods require network isolation, which CNI providers, like Calico, address.

Kubernetes initiates and manages the **CNI connectivity process** as follows:

1. **Network Attachment Definitions (NADs)**: Kubernetes 1.18 and later supports a custom resource called Network Attachment Definitions. This allows operators to specify that a Pod should have a particular network interface.

2. **Multus CNI**: This open-source CNI plugin enables Kubernetes Pods to have multiple network interfaces. With Multus, you can use different CNI plugins to assign either overlay or underlay network interfaces to the Pods.

3. **Kubelet Configuration**: Kubelet, an essential Kubernetes component on each node, is responsible for integrating CNI providers. Configuration commands, typically introduced in the kubelet.service file, facilitate this compatibility.

   ```json
   {
      "cniConfigFilePath": "/etc/cni/net.d/",
      "networkPluginName": "cni-type",
      "featureGates": {
         "CSIMigration": true,
         "CSIMigrationAWS": true,
         "CSIMigrationAzureDisk": true,
         "CSIMigrationAzureFile": true,
         "CSIMigrationGCEPD": true,
         "SupportPodPidsLimit": true
      }
   }
   ```

   Here, `cniConfigFilePath` specifies the path for CNI configuration files, while `networkPluginName` names the CNI plugin Kubernetes should use.

4. **API Server Triggers**: By interacting with the pod lifecycle events, the API server can trigger CNI when pods require network interfaces.

5. **CoreDNS Integration**: CoreDNS acts as a plugin for Kubernetes DNS, providing a unified method for service discovery.

6. **Use Throughout the Stack**: After establishing a network, CNI serves an essential role in the Kubernetes stack, including in Service and Ingress controller configurations.

7. **DaemonSets**: Operators use DaemonSets to run CNI plugins on every Kubernetes node. This ensures network configuration consistency across the cluster.

8. **Namespace Segmentation**: Kubernetes frequently utilizes CNI to prevent traffic from leaking between namespaces, ensuring network security. CNI is primarily responsible for performing network segmentation to meet these policy requirements.
<br>

## 15. How do you implement _Network Policies_, and what are their benefits?

**Kubernetes Network Policies** regulate traffic within the cluster, boosting both security and performance.

### Core Elements

- **Network Policy:** The rule-set defining traffic behavior.
- **Selector:** Identifies pods to which the policy applies.
- **Gateways / Egress Points:** Regulate outgoing traffic.
- **Ingress Definition:** Specifies allowed inbound connections.

### Policy Types

- **Allow:** Permits specific traffic types.
- **Deny:** Blocks specified traffic types.
- **Blacklist / Whitelist:** Contradictory to Allow & Deny, they either block everything except what's listed (Whitelist) or block specific things except what's listed (Blacklist).

### Policy Language

Different Kubernetes tools and implementations have their own way of writing policies.

- **Cilium:** Uses rich BPF rule expression to define advanced rules and manage application-level policies.
- **Calico:** Adds its security and network features, with a powerful policy engine allowing you to express complex network rules.

### Best Practices

1. **Start Simple:** Build policies gradually, testing after each rule addition.
2. **Documentation:** Detailed policy design and updates encourage consistent and secure practices.
3. **Review Regularly:** Network requirements can change, necessitating policy updates.
4. **Testing:** Use tools like `kube-router` to verify policy application.

### What It Means for Businesses

By implementing **Network Policies**, businesses can ensure more secure and efficient internal Kubernetes communication, meeting compliance and governance requirements.
<br>



#### Explore all 42 answers here 👉 [Devinterview.io - Kubernetes](https://devinterview.io/questions/software-architecture-and-system-design/kubernetes-interview-questions)

<br>

<a href="https://devinterview.io/questions/software-architecture-and-system-design/">
<img src="https://firebasestorage.googleapis.com/v0/b/dev-stack-app.appspot.com/o/github-blog-img%2Fsoftware-architecture-and-system-design-github-img.jpg?alt=media&token=521fd2a9-0d56-49c0-a723-9bd6ca081893" alt="software-architecture-and-system-design" width="100%">
</a>
</p>

