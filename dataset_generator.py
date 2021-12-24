import random
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
    f = open("s1.txt", "r")
    color_map = {1. : 'g', 
                 2. : 'm', 
                 3. : 'b', 
                 4. : 'r'}
    
    for line in f:
        x1, x2, cat = [float(item) for item in line.strip().split(',')]
        plt.scatter(x1, x2, c=color_map[cat], marker="+", linewidths=0.5)
    plt.show()

if __name__ == '__main__':
    generate_dataset()
    plot_dataset()