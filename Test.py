import tensorflow as tf
import numpy as np
import os

def test():
    cats = []
    label_cats = []
    dogs = []
    label_dogs = []
    for num in range(1000,1100):
        print(str(num)+":")
        if num < 10:
            cats.append(num)
            label_cats.append(0)
        else:
            dogs.append(num)
            label_dogs.append(1)


    print('There are %d cats\nThere are %d dogs' % (len(cats), len(dogs)))

    image_list = np.hstack((cats, dogs))
    label_list = np.hstack((label_cats, label_dogs))

    temp = np.array([image_list, label_list])
    temp = temp.transpose()
    np.random.shuffle(temp)

    image_list = list(temp[:, 0])
    label_list = list(temp[:, 1])
    label_list = [int(i) for i in label_list]

    print(image_list)
    print(label_list)

if __name__ == "__main__":
    test()