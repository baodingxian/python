#By @Kevin Xu
#kevin28520@gmail.com

# 深度学习QQ群, 1群满): 153032765
# 2群：462661267
#The aim of this project is to use TensorFlow to process our own data.
#    - input_data.py:  read in data and generate batches
#    - model: build the model architecture
#    - training: train

# I used Ubuntu with Python 3.5, TensorFlow 1.0*, other OS should also be good.
# With current settings, 10000 traing steps needed 50 minutes on my laptop.


# data: cats vs. dogs from Kaggle
# Download link: https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition/data
# data size: ~540M

# How to run?
# 1. run the training.py once
# 2. call the run_training() in the console to train the model.

# Note: 
# it is suggested to restart your kenel to train the model multiple times 
#(in order to clear all the variables in the memory)
# Otherwise errors may occur: conv1/weights/biases already exist......


#%%

import tensorflow as tf
import numpy as np
import os

#%%

# you need to change this to your data directory
train_dir = 'D:/workspace/tmp/IN/LaNetDataSet/'

def get_files(file_dir):
    '''
    Args:
        file_dir: file directory
    Returns:
        list of images and labels
    '''
    cats = []
    label_cats = []
    for file in os.listdir(file_dir):
        name = file.split(sep='.')[0].split(sep='_')
        cats.append(file_dir + file)
        label_cats.append(get_one_hot(file[0]))
    print('There are %d cats' %(len(cats)))
    
    image_list = np.hstack((cats))
    label_list = np.hstack((label_cats))
    
    temp = np.array([image_list, label_list])
    temp = temp.transpose()
    np.random.shuffle(temp)
    
    image_list = list(temp[:, 0])
    label_list = list(temp[:, 1])
    label_list = [int(i) for i in label_list]
    
    print(image_list)
    print(label_list)
    return image_list, label_list


def get_one_hot(lable):
    indexs = {
        'a': 10,
        'b': 11,
        'c': 12,
        'd': 13,
        'e': 14,
        'f': 15,
        'g': 16,
        'h': 17,
        'i': 18,
        'j': 19,
        'k': 20,
        'l': 21,
        'm': 22,
        'n': 23,
        'o': 24,
        'p': 25,
        'q': 26,
        'r': 27,
        's': 28,
        't': 29,
        'u': 30,
        'v': 31,
        'w': 32,
        'x': 33,
        'y': 34,
        'z': 35,
        'A': 36,
        'B': 37,
        'C': 38,
        'D': 39,
        'E': 40,
        'F': 41,
        'G': 42,
        'H': 43,
        'I': 44,
        'J': 45,
        'K': 46,
        'L': 47,
        'M': 48,
        'N': 49,
        'O': 50,
        'P': 51,
        'Q': 52,
        'R': 53,
        'S': 54,
        'T': 55,
        'U': 56,
        'V': 57,
        'W': 58,
        'X': 59,
        'Y': 60,
        'Z': 61,
    }
    if lable.isdigit():
        return int(lable)
    else:
        return indexs[lable]

if __name__ == "__main__":
    N_CLASSES = 2
    IMG_W = 40  # resize the image, if the input image is too large, training will be very slow.
    IMG_H = 40
    BATCH_SIZE = 16
    CAPACITY = 60
    train, train_label = get_files(train_dir)
    train_batch, train_label_batch = get_batch(train,train_label, IMG_W, IMG_H, BATCH_SIZE, CAPACITY)

#%%

def get_batch(image, label, image_W, image_H, batch_size, capacity):
    '''
    Args:
        image: list type
        label: list type
        image_W: image width
        image_H: image height
        batch_size: batch size
        capacity: the maximum elements in queue
    Returns:
        image_batch: 4D tensor [batch_size, width, height, 3], dtype=tf.float32
        label_batch: 1D tensor [batch_size], dtype=tf.int32
    '''
    
    image = tf.cast(image, tf.string)
    label = tf.cast(label, tf.int32)

    # make an input queue
    input_queue = tf.train.slice_input_producer([image, label])
    
    label = input_queue[1]
    image_contents = tf.read_file(input_queue[0])
    image = tf.image.decode_jpeg(image_contents, channels=3)
    
    ######################################
    # data argumentation should go to here
    ######################################
    
    image = tf.image.resize_image_with_crop_or_pad(image, image_W, image_H)
    #label = tf.placeholder(tf.float32, [None, 61])
    # if you want to test the generated batches of images, you might want to comment the following line.
    # 如果想看到正常的图片，请注释掉111行（标准化）和 126行（image_batch = tf.cast(image_batch, tf.float32)）
    # 训练时不要注释掉！
    image = tf.image.per_image_standardization(image)
    
    image_batch, label_batch = tf.train.batch([image, label],
                                                batch_size= batch_size,
                                                num_threads= 64, 
                                                capacity = capacity)
    
    #you can also use shuffle_batch 
#    image_batch, label_batch = tf.train.shuffle_batch([image,label],
#                                                      batch_size=BATCH_SIZE,
#                                                      num_threads=64,
#                                                      capacity=CAPACITY,
#                                                      min_after_dequeue=CAPACITY-1)
    
    label_batch = tf.reshape(label_batch, [batch_size])
    image_batch = tf.cast(image_batch, tf.float32)
    
    return image_batch, label_batch


 
#%% TEST
# To test the generated batches of images
# When training the model, DO comment the following codes




#import matplotlib.pyplot as plt
#
#BATCH_SIZE = 2
#CAPACITY = 256
#IMG_W = 208
#IMG_H = 208
#
#train_dir = '/home/kevin/tensorflow/cats_vs_dogs/data/train/'
#
#image_list, label_list = get_files(train_dir)
#image_batch, label_batch = get_batch(image_list, label_list, IMG_W, IMG_H, BATCH_SIZE, CAPACITY)
#
#with tf.Session() as sess:
#    i = 0
#    coord = tf.train.Coordinator()
#    threads = tf.train.start_queue_runners(coord=coord)
#    
#    try:
#        while not coord.should_stop() and i<1:
#            
#            img, label = sess.run([image_batch, label_batch])
#            
#            # just test one batch
#            for j in np.arange(BATCH_SIZE):
#                print('label: %d' %label[j])
#                plt.imshow(img[j,:,:,:])
#                plt.show()
#            i+=1
#            
#    except tf.errors.OutOfRangeError:
#        print('done!')
#    finally:
#        coord.request_stop()
#    coord.join(threads)


#%%





    
