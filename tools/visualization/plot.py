'''plot.py: Plot data using matplotlib.'''
import matplotlib.pyplot as plt

def plot2D(X, y):
    red  = []
    blue = []

    # Split the data into two classes
    for point in range(len(X)):
        if y[point] == -1:
            red.append(X[point])
        elif y[point] == 1:
            blue.append(X[point])

    # Plot it up.
    plt.scatter([r[0] for r in red], [r[1] for r in red], color='red', label='Class 0')
    plt.scatter([b[0] for b in blue], [b[1] for b in blue], color='blue', label='Class 1')
    plt.legend()
    plt.show()

