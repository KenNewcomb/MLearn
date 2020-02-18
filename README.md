# MLearn
A collection of machine learning algorithms, written in Python.

## Supported Algorithms

### Supervised Learning
- Decision Trees [**in progress**]
- Linear Regression
    - L1, L2 Regularization
    - Elastic Net Regularization [**in progress**]
    - Normal Equation Method
- Logistic Regression
    - L1, L2 Regularization
- K-Nearest Neighbors (KNN): regressor, classifier
    - Distance-weighted KNN [**in progress**]
- Multilayer Perceptron [**planned**]
- Naive-Bayes methods
    - Gaussian NB
- Support Vector Machines [**in progress**]
- Ensembles of learners [**planned**]

### Unsupervised Learning
- K-Means Clustering [**planned**]

### Optimizers
- Gradient descent and variants
    - Batch gradient descent
    - Stochastic (mini-batch) gradient descent [**in progress**]

### Data Preprocessing Tools
- Feature normalization/scaling
- Nonlinear feature generation


## Usage
Import the desired learning algorithm from `algorithms`. Then, use the `.fit(X, y)` method to train the algorithm, and `.predict(X)` to make predictions.
