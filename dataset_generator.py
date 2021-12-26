import random
import numpy as np
import matplotlib.pyplot as plt

def generate_dataset():
    f = open('s1.txt', 'w')
    for idx in range(0, 4000):
        x1 = random.uniform(-1., 1.)
        x2 = random.uniform(-1., 1.)
        cat = 0
        if (x1 - 0.5)**2 + (x2 - 0.5)**2 < 0.16:
            cat = 1
        elif (x1 + 0.5)**2 + (x2 + 0.5)**2 < 0.16:
            cat = 1
        elif (x1 - 0.5)**2 + (x2 + 0.5)**2 < 0.16:
            cat = 2
        elif (x1 + 0.5)**2 + (x2 - 0.5)**2 < 0.16:
            cat = 2
        elif (x1 < 0 and x2 > 0) or (x1 > 0 and x2 < 0):
            cat = 3
        else:
            cat = 4
            
        f.write('{:.3f}, {:.3f}, {}\n'.format(x1, x2, cat))
    f.close()

def plot_dataset():
    filename = "s1.txt"
    color_map = {1. : 'g', 
                 2. : 'm', 
                 3. : 'b', 
                 4. : 'r'}
    
    points = np.empty((0,3))
    with open(filename) as f:
        for line in f:
            np_line = np.array([float(item) for item in line.strip().split(',')])
            points = np.vstack([points, np_line])

    plt.scatter(points[:,0], points[:,1], c=points[:,2],marker="+", linewidths=0.5)
    plt.show()

if __name__ == '__main__':
    generate_dataset()
    import time 
    start_time = time.time()
    plot_dataset()
    print('{}sec'.format(time.time() - start_time))