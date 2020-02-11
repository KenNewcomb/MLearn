'''MNIST.py: A module for importing MNIST data.'''

X_train = []
y_train = []
def load_data():
    data = []
    with open('datasets/MNIST/mnist_train_part1.csv', 'r') as f:
        data.extend(f.readlines())
    with open('datasets/MNIST/mnist_train_part2.csv', 'r') as f:
        data.extend(f.readlines())

    # Extract X and Y
    print("Loading Y training data...")
    y_train = [int(i.split(',')[0]) for i in data]
    print("Loading X training data...")
    X_train = [i.split(',')[1:] for i in data]
    X_train = [[int(j) for j in i] for i in X_train]

    print("Data loaded successfully.")
    
    return X_train[:50], y_train[:50]
