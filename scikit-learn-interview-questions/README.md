# Top 50 Scikit-Learn Interview Questions

<div>
<p align="center">
<a href="https://devinterview.io/questions/machine-learning-and-data-science/">
<img src="https://firebasestorage.googleapis.com/v0/b/dev-stack-app.appspot.com/o/github-blog-img%2Fmachine-learning-and-data-science-github-img.jpg?alt=media&token=c511359d-cb91-4157-9465-a8e75a0242fe" alt="machine-learning-and-data-science" width="100%">
</a>
</p>

#### You can also find all 50 answers here 👉 [Devinterview.io - Scikit-Learn](https://devinterview.io/questions/machine-learning-and-data-science/scikit-learn-interview-questions)

<br>

## 1. What is _Scikit-Learn_, and why is it popular in the field of _Machine Learning_?

**Scikit-Learn**, an open-source Python library, is a leading solution for machine learning tasks. Its simplicity, versatility, and consistent performance across different ML methods and datasets have earned it tremendous popularity.

### Key Features

- **Straightforward Interface**: Intuitive API design simplifies the implementation of various ML tasks, ranging from data preprocessing to model evaluation.

- **Model Selection and Automation**: Scikit-Learn provides techniques for extensive hyperparameter optimization and model evaluation, reducing the burden on developers in these areas.

- **Consistent Model Objects**: All models and techniques in Scikit-Learn are implemented as unified Python objects, ensuring a standardized approach.

- **Robustness and Flexibility**: Many algorithms and models in Scikit-Learn come with adaptive features, catering to diverse requirements.

- **Versatile Tools**: Apart from standard supervised and unsupervised models, Scikit-Learn offers utilities for feature selection and pipeline construction, allowing for seamless integration of multiple methods.

### Model Consistency

Scikit-Learn maintains a **consistent model interface** adaptable to a plethora of use-cases. This structure sculpts model-training and prediction procedures into recognizable patterns.

  - **Three Basic Techniques**: Users uniformly use `fit()` for model training, `predict()` for data inference, and `score()` for performance evaluation, simplifying interaction with distinct models.

### Versatility and Go-To Algorithms

Scikit-Learn presents an extensive suite of algorithms, especially catering to fundamental ML tasks.

- **Supervised Learning**: Scikit-Learn houses methods for everything from linear and tree-based models to support vector machines and neural networks.

- **Unsupervised Learning**: Clustering and dimensionality reduction are seamlessly achieved using the library's tools.

- **Hyperparameter Tuning**: Feature-rich options for grid search and randomized search streamline the process.

- **Feature Selection**: Employ varied selection techniques to isolate meaningful predictors.
<br>

## 2. Explain the design principles behind _Scikit-Learn's API_.

**Scikit-Learn** aims to provide a consistent and user-friendly interface for various machine learning tasks. Its API design is grounded in several key principles to ensure clarity, modularity, and versatility.

### Core API Principles

- **Consistency**: The API adheres to a consistent design pattern across all its modules.
  
- **Non-Redundancy**: It avoids redundancy by drawing on general routines for common tasks. This keeps the API concise and unified across different algorithms.

### Data Representation

- **Data as Rectangular Arrays**: Scikit-Learn algorithms expect input data to be stored in a two-dimensional array or a matrix-like object. This ensures **data is homogenous** and can be accessed efficiently using NumPy.

- **Encoded Targets**: Categorical target variables are converted to integers or one-hot encodings before feeding them to most estimators.

### Model Fitting and Predictions

- **Fit then Transform**: The API distinguishes between fitting estimators to data and transforming them. In cases where data transformations are involved, pipelines are used to ensure consistency and reusability.

- **Stateless Transforms**: Preprocessing operations like feature scaling and imputation transform data but do not preserve any internal state from one `fit_transform` call to the next.

- **Predict Method**: After fitting, models use the `predict` method to produce predictions or labeling.

### Unsupervised Learning

- **transform Method**: Unsupervised estimators have a `transform` method that modifies inputs as a form of feature extraction, transformation, or clustering—a step distinct from initial fitting.

### Composability and Provenance

- **Make Predictions with Immutable Parts**: A model's prediction phase depends only on its parameters. **Fit state** doesn't influence predictions, ensuring consistency.

- **Pipelines for Chaining Steps**: Pipelines harmonize data processing and modeling stages, providing a single interface for both.

- **Feature and Model Names**: For **interpretability**, Scikit-Learn uses string identifiers for model and feature names.

  Example: In text classification, a feature may be "wordcount" or "tf_idf" instead of the raw text itself.

### Model Evaluation

- **Separation of Concerns**: A distinct set of classes is dedicated to model selection and evaluation, like `GridSearchCV` or `cross_val_score`.

### Task-Specific Estimators

Scikit-Learn features specialized estimators for distinct tasks:

- **Classifier**: For binary or multi-class classification tasks.
- **Regressor**: For continuous target variables in regression problems.
- **Clusterer**: For unsupervised clustering.
- **Transformer**: For data transformation, like dimensionality reduction or feature selection.

This categorization makes it simple to pinpoint the right estimator for a given task.

### The Golden Rules of the Scikit-Learn API

1. **Know the Estimator You Are Using**: There are various supported tasks, but different estimators can't be coerced to accommodate tasks outside their primary wheelhouse.

2. **Be Mindful of Your Data**: Preprocess your data consistently and according to the estimator's requirements using data transformers and pipelines.

3. **Respect the Training-Scoring-Evaluation Discrimination**: Training on one dataset and evaluating on another isn't merely an option; it's a careful protocol that helps prevent overfitting.

4. **Determine a Conveyable and Understandable Feature and Model Identifiers**: Knowing what was used where can sometimes be just as important as knowing the numeric result of a prediction or transformation.

5. **Remember the Task at Hand**: Always keep in mind the specificity of your problem—classification versus regression, supervised versus unsupervised—so you can pick the best tool for the job.
<br>

## 3. How do you handle _missing values_ in a dataset using _Scikit-Learn_?

When handling **missing values** in a dataset, scikit-learn provides several tools and techniques as well. These include:

### Imputation

Imputation replaces missing values with substitutes. Scikit-learn's `SimpleImputer` offers several strategies:

- **Mean, Median, Most Frequent**: Fills in with the mean, median, or mode of the non-missing values in the column.
- **Constant**: Assigns a fixed value to all missing entries.
- **KNN**: Uses the k-Nearest Neighbors algorithm to determine an appropriate value based on other instances' known feature values.

Here is the Python code:

```python
from sklearn.impute import SimpleImputer
import numpy as np

# Example data
X = np.array([[1, 2], [np.nan, 3], [7, 6]])

# Simple imputer
imp_mean = SimpleImputer()
X_mean = imp_mean.fit_transform(X)

print(X_mean)  # Result: [[1. 2.], [4. 3.], [7. 6.]]
```

### K-Means and Missing Values

Using methods that transform data but not handle missing values for example for **K-Means** you can preprocess your data to handle missing values using one of the methods provided by`SimpleImputer` and then use `KMeans` to fit your preprocessed data.
<br>

## 4. Describe the role of _transformers_ and _estimators_ in _Scikit-Learn_.

**Scikit-Learn** employs two primary components for machine learning: **transformers** and **estimators**.

### Transformers

**Transformers** are objects that map data into a new format, usually for feature extraction, scaling, or dimensionality reduction. They perform this transformation using the `.transform()` method.

Some common transformers include the `MinMaxScaler` for feature scaling, `PCA` for dimensionality reduction, and `CountVectorizer` for text preprocessing.

#### Example: MinMaxScaler

Here is the Python code:

```python
from sklearn.preprocessing import MinMaxScaler

# Creating the scaler object
scaler = MinMaxScaler()

# Fitting the data and transforming it
data_transformed = scaler.fit_transform(original_data)
```

In this example, we fit the transformer on the original data and then transform that data into a new format.
  
### Estimators

**Estimators** represent models that learn from data, making predictions or influencing other algorithms. The principal methods used by estimators are `.fit()` to learn from the data and `.predict()` to make predictions on new data.

One example of an estimator is the `RandomForestClassifier`, which is a machine learning model used for classification tasks.

#### Example: RandomForestClassifier

Here is the Python code:

```python
from sklearn.ensemble import RandomForestClassifier

# Creating the classifier object
clf = RandomForestClassifier()

# Fitting the classifier on training data
clf.fit(X_train, y_train)

# Making predictions on the test set
y_pred = clf.predict(X_test)
```

In this example, `X_train` and `y_train` represent the input features and output labels of the training set, respectively. The classifier is trained using these datasets. After training, it can be used to make predictions on new, unseen data represented by `X_test`.
<br>

## 5. What is the typical workflow for building a _predictive model_ using _Scikit-Learn_?

When using **Scikit-Learn** for building predictive models, you'll typically follow these seven steps in a **methodical workflow**:

### Scikit-Learn Workflow Steps

1. **Acquiring** the Data: This step involves obtaining your data from a variety of sources.
2. **Preprocessing** the Data: Data preprocessing includes tasks such as cleaning, transforming, and splitting the data.
3. **Defining** the Model: This step involves choosing the type of model that best fits your data and problem.
4. **Training** the Model: Here, the model is fitted to the training data.
5. **Evaluating** the Model: The model's performance is assessed using testing data or cross-validation techniques.
6. **Fine-Tuning** the Model: Various methods, such as hyperparameter tuning, can improve the model's performance.
7. **Deploying** the Model: The trained and validated model is put to use for making predictions.

### Code Example: Workflow Steps

Here is the Python code:

```python
# Step 1: Acquire the Data
import pandas as pd
from sklearn.datasets import load_iris

# Load the Iris dataset
iris = load_iris()
X, y = iris.data, iris.target
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)

# Step 2: Preprocess the Data
from sklearn.model_selection import train_test_split
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Define the Model
from sklearn.tree import DecisionTreeClassifier
# Initialize the model
model = DecisionTreeClassifier()

# Step 4: Train the Model
# Fit the model to the training data
model.fit(X_train, y_train)

# Step 5: Evaluate the Model
from sklearn.metrics import accuracy_score
# Make predictions
y_pred = model.predict(X_test)
# Assess accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# Step 6: Fine-Tune the Model
from sklearn.model_selection import GridSearchCV
# Define the parameter grid to search
param_grid = {'max_depth': [3, 4, 5]}
# Initialize the grid search
grid_search = GridSearchCV(model, param_grid, cv=5)
# Conduct the grid search
grid_search.fit(X_train, y_train)
# Get the best parameters
best_params = grid_search.best_params_
print(f"Best Parameters: {best_params}")

# Refit the model with the best parameters
best_model = grid_search.best_estimator_
best_model.fit(X_train, y_train)

# Step 7: Deploy the Model
# Use the deployed model to make predictions
new_data = [[5.1, 3.5, 1.4, 0.2], [6.2, 3.4, 5.4, 2.3]]
predictions = best_model.predict(new_data)
print(f"Predicted Classes: {predictions}")
```
<br>

## 6. How can you _scale features_ in a dataset using _Scikit-Learn_?

**Feature scaling** is a crucial step in many machine learning algorithms. It involves transforming numerical features to a "standard" scale, often leading to better model performance. **Scikit-Learn** offers convenient methods for feature scaling.

### Methods for Feature Scaling

1. **Min-Max Scaling**: Rescales data to a specific range using the formula:

   $$X_{\text{scaled}} = \frac{X - X_{\text{min}}}{X_{\text{max}} - X_{\text{min}}}$$

   ```python
   from sklearn.preprocessing import MinMaxScaler
   min_max_scaler = MinMaxScaler()
   X_minmax = min_max_scaler.fit_transform(X)
   ```

2. **Standardization**: Centers the data to have a mean of $0$ and a standard deviation of $1$ using the formula:

   $$X_{\text{standardized}} = \frac{X - \mu}{\sigma}$$

   ```python
   from sklearn.preprocessing import StandardScaler
   std_scaler = StandardScaler()
   X_std = std_scaler.fit_transform(X)
   ```

3. **Robust Scaling**: Scales data based on interquartile range (IQR), making it robust to outliers.

   $$\frac{X - Q_1(X)}{Q_3(X) - Q_1(X)}$$

   ```python
   from sklearn.preprocessing import RobustScaler
   robust_scaler = RobustScaler()
   X_robust = robust_scaler.fit_transform(X)
   ```
<br>

## 7. Explain the concept of a _pipeline_ in _Scikit-Learn_.

A **pipeline** in Scikit-Learn is a way to streamline and automate a sequence of data transformations and model fitting or predicting, all integrated in a single, tidy framework.

### Core Components

1. **Pre-Processors**:  These perform any necessary data transformations, such as imputation of missing values, feature scaling, and feature selection.

2. **Estimators**: These represent any model or algorithm for learning from data. They can be either a classifier or a regressor.

### Benefits of Using Pipelines

- **Streamlined Code**: Piping together several data processing steps makes the code look cleaner and easier to understand.
- **Reduced Data Leakage**: Pipelines apply each step in the sequence to the data, which helps in avoiding common pitfalls like data leakage during transformations and evaluation.
- **Cross-Validation Integration**: Pipelines are supported within cross-validation and grid search, enabling fine-tuning of the entire workflow at once.

### Code Example: Pipelining in Scikit-Learn

Here is the Python code:

```python
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

# Fake or dummy data for illustration.
X = [[-1, 2], [-0.5, 6], [0, 10], [1, 18]]
y = [0, 1, 2, 3]

# Define pipeline components
imputer = SimpleImputer(strategy='mean')
scaler = MinMaxScaler()
classifier = RandomForestClassifier()

# Construct the pipeline
pipeline = make_pipeline(imputer, scaler, classifier)

# Perform cross-validation with the pipeline
scores = cross_val_score(pipeline, X, y, cv=5)
```

In this example, the pipeline consolidates three essential steps:

1. **Data Imputation**: Use mean to fill missing or NaN values.
2. **Data Scaling**: Use Min-Max scaling.
3. **Model Building and Training**: RandomForest's classifier.

Once the pipeline is set up, training or predicting is a one-step process, like so:

```python
pipeline.fit(X_train, y_train)  # Train the pipeline.
predicted = pipeline.predict(X_test)  # Use the pipeline to make predictions.
```
<br>

## 8. What are some of the main categories of _algorithms_ included in _Scikit-Learn_?

**Scikit-Learn** provides a diverse array of algorithms, and here are the main categories for supervised and unsupervised learning.

### Supervised Learning Algorithms

#### Regression

- **Linear Regression**: Establishes linear relationships between features and target.
- **Ridge, Lasso and ElasticNet**: Utilizes regularization methods.

#### Classification

- **Decision Trees & Random Forest**: Uses tree structures for decision-making.
- **SVM (Support Vector Machine)**: Separates data into classes using a hyperplane.
- **K-Nearest Neighbors (K-NN)**: Classifies based on the majority labels in the k-nearest neighbors.

#### Ensembles

- **Adaboost, Gradient Boosting**: Combines multiple weak learners to form a strong model.

#### Neural Networks

- **Multi-layer Perceptron**: A type of feedforward neural network.

### Unsupervised Learning Algorithms

#### Clustering

- **K-Means**: Divides data into k clusters based on centroids.
- **Hierarchical & DBSCAN**: Unsupervised methods that do not require prior specification of clusters.

#### Dimensionality Reduction

- **PCA (Principal Component Analysis)**: Reduces feature dimensionality based on variance.
- **LDA (Linear Discriminant Analysis)**: Reduces dimensions while maintaining class separability.

#### Outlier Detection

- **One Class SVM**: Identifies observations that deviate from the majority.

#### Decomposition and Feature Selection

- **FastICA, NMF, VarianceThreshold**: Feature selection and signal decomposition methods.
<br>

## 9. How do you encode _categorical variables_ using _Scikit-Learn_?

In **Scikit-Learn**, you can use various techniques to encode **Categorical Variables**.

### Categorical Encoding Techniques

- **OrdinalEncoder**: For ordinal categories, assigns a range of numbers to each category. Works well when certain categories have an inherent order.

- **OneHotEncoder**: Creates **Binary** columns representing each category to avoid assuming any ordinal relationship. Ideal for non-binary categories.

- **LabelBinarizer**: A simpler version of OneHotEncoder designed for binary (two-class) categories.

### Example: Using `OneHotEncoder`

Here is the Python code:

```python
from sklearn.preprocessing import OneHotEncoder
import pandas as pd

# Example data
data = pd.DataFrame({'Color': ['Red', 'Green', 'Blue', 'Green', 'Red']})

# Initializing and fitting OneHotEncoder
encoder = OneHotEncoder()
encoded_data = encoder.fit_transform(data[['Color']])

# Converting to DataFrame for visibility
encoded_df = pd.DataFrame(encoded_data.toarray(), columns=encoder.get_feature_names_out(['Color']))

# Displaying encoded DataFrame
print(encoded_df)
```

### Example: Using `LabelBinarizer`

Here is the Python code:

```python
from sklearn.preprocessing import LabelBinarizer
import pandas as pd

# Example data
data = pd.DataFrame({'Color': ['Red', 'Green', 'Blue', 'Green', 'Red']})

# Initializing and fitting LabelBinarizer
binarizer = LabelBinarizer()
encoded_data = binarizer.fit_transform(data['Color'])

# Converting to DataFrame for visibility
encoded_df = pd.DataFrame(encoded_data, columns=binarizer.classes_)

# Displaying encoded DataFrame
print(encoded_df)
```
<br>

## 10. What are the strategies provided by _Scikit-Learn_ to handle _imbalanced datasets_?

**Imbalanced datasets** pose a challenge in machine learning because the frequency of different classes is disproportionate, often leading to biased models.

### Techniques to Handle Imbalance

#### Weighted Loss Function

By assigning different weights to classes, you can make the model prioritize the minority class. For instance, in a binary classification problem with an imbalanced dataset, you can use `class_weight` in classifiers like `LogisticRegression` or `SVC`.

Example with `LogisticRegression`:

```python
from sklearn.linear_model import LogisticRegression

# Set class_weight to 'balanced' or a custom weight
clf = LogisticRegression(class_weight='balanced')  
```

#### Resampling

**Oversampling** involves replicating examples in the minority class, while **undersampling** reduces the number of examples in the majority class. This achieves a better balance for training.

Scikit-Learn doesn't have built-in functions for resampling, but third-party libraries like `imbalanced-learn` offer this capability.

Example using `imbalanced-learn`:

```python
from imblearn.over_sampling import RandomOverSampler

over_sampler = RandomOverSampler()
X_train_resampled, y_train_resampled = over_sampler.fit_resample(X_train, y_train)
```

#### Focused Model Evaluation

The **Area Under the Receiver Operating Characteristic Curve** (AUC-ROC) can be a better evaluation metric than accuracy for imbalanced datasets.

- **Precision-Recall** metrics, which focus on the performance of the minority class.

In Scikit-Learn, you can use `roc_auc_score` and `average_precision_score` for these metrics.

### Key Considerations

- **Resampling** can introduce bias or overfitting. It's essential to validate models carefully.
- **Weighted Loss Functions** are an easy way to address imbalance but may not always be sufficient. Balanced weights are a good starting point, but your problem might require custom weights.
<br>

## 11. How do you split a dataset into _training and testing sets_ using _Scikit-Learn_?

**Train-Test Split** is a fundamental step in machine learning model development for evaluating **model performance**.

Scikit-Learn, through its `model_selection` module, provides a straightforward method for performing this task:

### Code Example: Train-Test Split

Here is the Python code:

```python
from sklearn.model_selection import train_test_split
import numpy as np

# Data
X, y = np.arange(10).reshape((5, 2)), range(5)

# Split with a test size ratio (commonly 80-20 or 70-30)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Or specify a specific number of samples for the test set
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=2)
```
<br>

## 12. Describe the use of `ColumnTransformer` in _Scikit-Learn_.

The **ColumnTransformer** utility in `Scikit-Learn` allows for independent preprocessing of different feature types or subsets (columns) of the input data.

### Key Use Cases

- **Multi-Modal Feature Processing**: For datasets where features are of different types (e.g., text, numerical, categorical), `ColumnTransformer` is particularly useful.
- **Pipelining for Specific Features**: The tool is employed for applies specific transformers to certain subsets of the feature space, allowing for focused pre-processing.
- **Simplifying Transformation Pipelines**: When there are multiple features and multiple steps in the data transformation process, the `ColumnTransformer` methodology can help manage the complexity.

### Core Components and Concepts

- **Transformers**: These translate data from its original format to a format suitable for ML models.
- **Transformations**: These are the operations or `Callables` that the transformers perform on the input data.
- **Feature Groups**: The data features are divided into groups or subsets, and each group is associated with a unique transformation process, defined by different transformers. These feature groups correspond to the columns of the input dataset.

### Code Example: ColumnTransformer

Here is how to use `ColumnTransformer` with multiple pre-processing steps and each active_discovery step tailored to a specific subset of columns:

```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import Normalizer, StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

# Defining the ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['numerical_feature_1', 'numerical_feature_2']),
        ('num2',Normalizer(),['numerical_feature_3']),
        ('cat', OneHotEncoder(), ['categorical_feature_1', 'categorical_feature_2']),
        ('drop_col', 'drop', ['column_to_drop']),
        ('fill_unk', SimpleImputer(strategy='constant', fill_value='Unknown'), ['categorical_feature_with_nan']),
        ('default', 'passthrough', ['remaining_col_1'])  # By default, remaining columns are "passed through"
    ]
)

# Applying the ColumnTransformer
transformed_data = preprocessor.fit_transform(data)
```

In the example above:

- Columns `numerical_feature_1` and `numerical_feature_2` undergo z-score standardization.
- `numerical_feature_3` is normalized.
- We use one-hot encoding for `categorical_feature_1` and `categorical_feature_2`.
- We drop `column_to_drop`.
- For `categorical_feature_with_nan`, we replace `NaN` values with a constant ('Unknown').
- All remaining columns (including `remaining_col_1`) are passed through without any transformations (`'passthrough'`).
<br>

## 13. What _preprocessing steps_ would you take before inputting data into a _machine learning algorithm_?

Before feeding data into a machine learning algorithm, it is crucial to **pre-process** it. This involves several steps: 

### Data Preprocessing Steps

1. **Handling Missing Data**: Remove, impute, or flag missing values.
2. **Handling Categorical Data**: Convert categorical data to numerical form.
3. **Scaling and Normalization**: Rescale numerical data to a similar range.
4. **Splitting Data for Training and Testing**: Split the dataset to evaluate model performance.
5. **Feature Engineering**: Generate new features or transform existing ones for better model performance.

### Scikit-Learn Tools for Data Preprocessing

1. **Imputer**: Fills missing values.
2. **OneHotEncoder**: Encodes categorical data as one-hot vectors.
3. **StandardScaler**: Standardizes numerical data to have zero mean and unit variance.
4. **MinMaxScaler**: Rescales numerical data to a specific range. 
5. **Train-Test Split**: Divides data into training and testing sets. 
6. **PolynomialFeatures**: Generates polynomial features.

### Code Example: Data Preprocessing

Here is the Python code:

```python
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

# Data
X = ...  # Features
y = ...  # Target

# 1. Handling Missing Data
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# 2. Handling Categorical Data
encoder = OneHotEncoder()
X_encoded = encoder.fit_transform(X_imputed)

# 3. Scaling and Normalization
scaler = MinMaxScaler()  # Alternatively, can use StandardScaler
X_scaled = scaler.fit_transform(X_encoded)

# 4. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)
```
<br>

## 14. Explain how `Imputer` works in _Scikit-Learn_ for dealing with _missing data_.

**Imputer**, available in `sklearn.preprocessing`, offers a streamlined solution for handling missing data in your datasets.

### Core Functionality

Using a variety of strategies, `Imputer` takes in your feature matrix and replaces missing values with appropriate data.

The process can be summarized as follows:

1. **Fit**: The Imputer instance estimates the method statistics from the training data. This is done using the `fit` method.
2. **Transform**: The missing values in the training data are then replaced with the learned statistics. This is accomplished using the `transform` method.
3. **Predict/Transform new data**: After training, the imputer can replace missing values in new data in a consistent fashion. For transformation of either training or new data, simply use the `fit_transform` method, which combines the `fit` and `transform` operations.

### Core Methods

- **fit(X)**: Learns the required statistics from the training data.
- **transform(X)**: Uses the learned statistics to replace missing data points in the dataset (self-contained operation, does not modify the imputer itself).
- **fit_transform(X)**: Combines the training and transformation processes for convenience.
- **statistics_**: After fitting, you can access the determined strategy or value from the imputer's `statistics_` attribute.

### Common Strategies for Imputation

- **Mean**: Substitutes missing values with the mean of the feature.
- **Median**: Replaces missing entries with the median of the feature.
- **Most Frequent**: Uses the mode of the feature for imputation.
- **Constant**: Allows you to specify a constant value for filling in missing data.

### Code Example: Using an Imputer

Here is the scikit-learn imputer code:

```python
import numpy as np
from sklearn.impute import SimpleImputer

# Sample data with missing values
X = np.array([[1, 2], [np.nan, 3], [7, 6]])

# Define the imputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

# Fit and transform the data
X_imputed = imputer.fit_transform(X)

# View the imputed data
print(X_imputed)
```
<br>

## 15. How do you _normalize_ or _standardize_ data with _Scikit-Learn_?

When preparing data for a machine learning model, it's often crucial to **normalize** or **standardize** features. Scikit-Learn provides two primary methods for this: `MinMaxScaler` for normalization and `StandardScaler` for standardization.

### Normalization and Min-Max Scaling

*Normalization* allows for rescaling of features within a set range.

The example code demonstrates how to normalize a feature vector using Scikit-Learn's `MinMaxScaler`.

```python
from sklearn.preprocessing import MinMaxScaler

# Feature matrix H lies within 50-80 age and 1.60-1.95 meters height
# Age is generally thousands of days
# Height is generally above 1
H = [[50, 1.90],    
     [80000, 1.95],  
     [45, 1.60],     
     [100000, 1.65]]

scaler = MinMaxScaler()
H_scaled = scaler.fit_transform(H)

print(H_scaled)
```

The output showcases each feature's new normalized range between 0 and 1.

### Z-Score Standardization

*Standardization* transforms data to have a **mean** of 0 and a **standard deviation** of 1.

Here is the Python code to implement Z-Score Standardization using the `StandardScaler` in Scikit-Learn:

```python
from sklearn.preprocessing import StandardScaler

# Feature matrix M representing Mean (mu) and standard deviation (sigma)
# 80 and 1.8 are typical mean and standard deviation for age and height respectively.
M = [[40, 1.60],  # close to average
     [120, 1.95],  # exceptionally tall
     [20, 1.50],   # shorter
     [60, 1.75]]   # slightly above mean

scaler = StandardScaler()
M_scaled = scaler.fit_transform(M)

print(M_scaled)
```
<br>



#### Explore all 50 answers here 👉 [Devinterview.io - Scikit-Learn](https://devinterview.io/questions/machine-learning-and-data-science/scikit-learn-interview-questions)

<br>

<a href="https://devinterview.io/questions/machine-learning-and-data-science/">
<img src="https://firebasestorage.googleapis.com/v0/b/dev-stack-app.appspot.com/o/github-blog-img%2Fmachine-learning-and-data-science-github-img.jpg?alt=media&token=c511359d-cb91-4157-9465-a8e75a0242fe" alt="machine-learning-and-data-science" width="100%">
</a>
</p>

