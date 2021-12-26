import numpy as np
import matplotlib.pyplot as plt

def plot_dataset(filename):
    color_map = {1. : 'g', 
                 2. : 'm', 
                 3. : 'b', 
                 4. : 'r'}
    
    points = np.empty((0, 3))
    with open(filename) as f:
        for line in f:
            np_line = np.array([float(item) for item in line.strip().split(',')])
            points = np.vstack([points, np_line])

    point_colors = np.array([])
    for item in points[:,2]:
        point_colors = np.append(point_colors, color_map[item])

    plt.scatter(points[:,0], points[:,1], c=point_colors, marker="+", \
        linewidths=0.5)
    plt.show()

if __name__ == '__main__':
    plot_dataset("s1.txt")