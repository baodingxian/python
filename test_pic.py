import os
import numpy as np
import tensorflow as tf
import input_data
import model
from PIL import Image
import matplotlib.pyplot as plt


def get_one_image(train):
    '''Randomly pick one image from training data
    Return: ndarray
    '''
    n = len(train)
    ind = np.random.randint(0, n)
    img_dir = train[ind]

    image = Image.open(img_dir)
    plt.imshow(image)
    image = image.resize([40, 40])
    image = np.array(image)
    return image


def evaluate_one_image():
    '''Test one image against the saved models and parameters
    '''

    # you need to change the directories to yours.
    # train_dir = '/home/kevin/tensorflow/cats_vs_dogs/data/train/'
    # train, train_label = input_data.get_files(train_dir)
    # image_array = get_one_image(train)
    ind = np.random.randint(0, 1000)
    # img_dir ="D:/workspace/python/learn/test/"+str(ind)+".jpg"卷铺盖
    img_dir = "D:/workspace/tmp/IN/tmp/X_162.jpg"
    # print(ind)
    image_array = Image.open(img_dir)
    plt.imshow(image_array)
    image_array = image_array.resize([40, 40])
    image_array = np.array(image_array)
    with tf.Graph().as_default():
        BATCH_SIZE = 1
        N_CLASSES = 62

        image = tf.cast(image_array, tf.float32)
        image = tf.image.per_image_standardization(image)
        image = tf.reshape(image, [1, 40, 40, 3])
        logit = model.inference(image, BATCH_SIZE, N_CLASSES)

        logit = tf.nn.softmax(logit)
        x = tf.placeholder(tf.float32, shape=[40,40, 3])

        # you need to change the directories to yours.
        logs_train_dir = 'D:/workspace/tmp/IN/logs'

        saver = tf.train.Saver()

        with tf.Session() as sess:

            print("Reading checkpoints...")
            ckpt = tf.train.get_checkpoint_state(logs_train_dir)
            if ckpt and ckpt.model_checkpoint_path:
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                saver.restore(sess, ckpt.model_checkpoint_path)
                print('Loading success, global_step is %s' % global_step)
            else:
                print('No checkpoint file found')

            prediction = sess.run(logit, feed_dict={x: image_array})
            max_index = np.argmax(prediction)
            print(prediction[0])
            print(prediction[0][max_index])
            print(get_one_hot(max_index))


def get_one_hot(lable):
    indexs = {
        '10': 'a',
        '11': 'b',
        '12': 'c',
        '13': 'd',
        '14': 'e',
        '15': 'f',
        '16': 'g',
        '17': 'h',
        '18': 'i',
        '19': 'j',
        '20': 'k',
        '21': 'l',
        '22': 'm',
        '23': 'n',
        '24': 'o',
        '25': 'p',
        '26': 'q',
        '27': 'r',
        '28': 's',
        '29': 't',
        '30': 'u',
        '31': 'v',
        '32': 'w',
        '33': 'x',
        '34': 'y',
        '35': 'z',
        '36': 'A',
        '37': 'B',
        '38': 'C',
        '39': 'D',
        '40': 'E',
        '41': 'F',
        '42': 'G',
        '43': 'H',
        '44': 'I',
        '45': 'J',
        '46': 'K',
        '47': 'L',
        '48': 'M',
        '49': 'N',
        '50': 'O',
        '51': 'P',
        '52': 'Q',
        '53': 'R',
        '54': 'S',
        '55': 'T',
        '56': 'U',
        '57': 'V',
        '58': 'W',
        '59': 'X',
        '60': 'Y',
        '61': 'Z',
    }
    if lable <10:
        return lable
    else:
        return indexs[str(lable)]

if __name__ == "__main__":
    evaluate_one_image()