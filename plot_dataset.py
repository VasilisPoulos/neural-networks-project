import sys
import numpy as np
from os.path import exists
import matplotlib.pyplot as plt

def plot_dataset(filename, num_of_columns):
    if not exists(filename):
        print('{} does not exist.'.format(filename))
        exit(-1)
        
    color_map = {1. : 'g', 
                 2. : 'm', 
                 3. : 'b', 
                 4. : 'r'}
    
    points = np.empty((0, num_of_columns))
    with open(filename) as f:
        for line in f:
            np_line = np.array([float(item) for item in line.strip().split(',')])
            points = np.vstack([points, np_line])

    if (num_of_columns == 3):
        point_colors = np.array([])
        for item in points[:,2]:
            point_colors = np.append(point_colors, color_map[item])

        plt.scatter(points[:,0], points[:,1], c=point_colors, marker="+", \
            linewidths=0.5)
    else:
        plt.scatter(points[:,0], points[:,1], marker="+", linewidths=0.5)
    plt.show()

if __name__ == '__main__':
    if not len(sys.argv) > 2:
        print('Use: python dataset_generator.py <filename> <num_of_columns>')
        exit(0)

    plot_dataset(sys.argv[1], int(sys.argv[2]))