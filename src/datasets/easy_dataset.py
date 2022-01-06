import random
train_path = "../data/easy_train.txt"
test_path = "../data/easy_test.txt"
dataset_size = 4000

f = open(train_path, "w")
for i in range(dataset_size):
    label = 0
    x = random.uniform(-1, 1)
    y = random.uniform(-1, 1)
    if x > 0:
        label = 1 
    f.write("{}, {}, {}\n".format(x, y, label))
f.close()

f = open(test_path, "w")
for i in range(dataset_size):
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