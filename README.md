# MLearn
A collection of machine learning algorithms, written in Python.

## Supported Algorithms

### Supervised Learning
- [Decision Trees](https://en.wikipedia.org/wiki/Decision_tree_learning) [**in progress**]
- [Linear Regression](https://en.wikipedia.org/wiki/Linear_regression)
    - [L1, L2 Regularization](https://en.wikipedia.org/wiki/Regularization_(mathematics))
    - Elastic Net Regularization [**in progress**]
    - Normal Equation Method
- [Logistic Regression](https://en.wikipedia.org/wiki/Logistic_regression)
    - L1, L2 Regularization
- [K-Nearest Neighbors (KNN)](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm): regressor, classifier
    - Distance-weighted KNN [**in progress**]
- [Multilayer Perceptron](https://en.wikipedia.org/wiki/Multilayer_perceptron) [**planned**]
- [Naive-Bayes methods](https://en.wikipedia.org/wiki/Naive_Bayes_classifier)
    - [Gaussian NB](https://en.wikipedia.org/wiki/Naive_Bayes_classifier#Gaussian_na%C3%AFve_Bayes)
- [Support Vector Machines](https://en.wikipedia.org/wiki/Support_vector_machine) [**in progress**]
- [Ensembles of learners](https://en.wikipedia.org/wiki/Ensemble_learning) [**planned**]

### Unsupervised Learning
- [K-Means Clustering](https://en.wikipedia.org/wiki/K-means_clustering) [**planned**]

### Optimizers
- [Gradient descent and variants](https://en.wikipedia.org/wiki/Gradient_descent)
    - Batch gradient descent
    - Stochastic (mini-batch) gradient descent [**in progress**]
- Adam
- RMSProp
- Momentum

### Data Preprocessing Tools
- [Feature normalization/scaling](https://en.wikipedia.org/wiki/Feature_scaling)
- Nonlinear feature generation


## Usage
Import the desired learning algorithm from `algorithms`. Then, use the `.fit(X, y)` method to train the algorithm, and `.predict(X)` to make predictions.
