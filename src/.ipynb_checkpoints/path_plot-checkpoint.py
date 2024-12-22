import numpy as np
import matplotlib.pyplot as plt

def plot_path(true_path, predicted_path):
    plt.figure(figsize=(8, 6))
    plt.plot(true_path[:, 0], true_path[:, 1], 'r--', color='r', label='True Path')
    plt.plot(predicted_path[:, 0], predicted_path[:, 1], 'o', label='Predicted Path')
    plt.scatter(predicted_path[0, 0], predicted_path[0, 1], color='green', label='Start Position', s=100)
    plt.title('True and Predicted Path')
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.legend()
    plt.axis('equal')
    plt.grid(True)
    plt.show()
