import random

f = open("../data/easy_train.txt", "w")
for i in range(4000):
    label = 0
    x = random.uniform(-1, 1)
    y = random.uniform(-1, 1)
    if x > 0:
        label = 1 
    f.write("{}, {}, {}\n".format(x, y, label))
f.close()

    

f = open("../data/easy_test.txt", "w")
for i in range(4000):
    label = 0
    x = random.uniform(-1, 1)
    y = random.uniform(-1, 1)
    prob = random.uniform(0, 1)
    if x > 0:
        label = 1
        if prob < 0.1:
            label = 0 
    f.write("{}, {}, {}\n".format(x, y, label))
f.close()