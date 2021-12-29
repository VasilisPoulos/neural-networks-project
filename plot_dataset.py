import sys
import numpy as np
from os.path import exists
import matplotlib.pyplot as plt

def plot_dataset(filename):
    if not exists(filename):
        print('{} does not exist.'.format(filename))
        exit(-1)
        
    color_map = {0. : 'y',
                 1. : 'g', 
                 2. : 'm', 
                 3. : 'b', 
                 4. : 'r'}

    f = open(filename, "r")
    number_of_columns = len(f.readline().strip().split(','))
    points = np.empty((0, number_of_columns))

    with open(filename) as f:
        for line in f:
            np_line = np.array([float(item) for item in line.strip().split(',')])
            points = np.vstack([points, np_line])

    if (number_of_columns == 3):
        point_colors = np.array([])

        number_of_clusters = len(set(points[:,2]))
        if number_of_clusters > len(color_map):
            plt.scatter(points[:,0], points[:,1], c=points[:,2], marker="+", \
                linewidths=1)
        else:
            for item in points[:,2]:
                point_colors = np.append(point_colors, color_map[item])
            plt.scatter(points[:,0], points[:,1], c=point_colors, marker="+", \
                linewidths=1)
    else:
        plt.scatter(points[:,0], points[:,1], marker="+", linewidths=1)
    plt.show()

def plot_clusters(filename):
    if not exists(filename):
        print('{} does not exist.'.format(filename))
        exit(-1)
    points = np.empty((0, 3))
    with open(filename) as f:
        for line in f:
            np_line = np.array([float(item) for item in line.strip().split(',')])
            points = np.vstack([points, np_line])
    plt.scatter(points[:,0], points[:,1])
    
if __name__ == '__main__':
    if not len(sys.argv) > 1:
        print('Use: python dataset_generator.py <filename> <kmeans_clusters>')
        exit(0)

    if len(sys.argv) > 2:
        plot_clusters(sys.argv[2])
    plot_dataset(sys.argv[1])