#!/usr/bin/env python
# coding: utf-8

# In[1]:


import keras
import numpy as np
from keras.models import Model
from keras.layers import Input, Flatten, Dense, Dropout, Lambda, Conv2D, MaxPooling2D

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import os
from os import listdir
#from PIL import Image
import matplotlib.pyplot as plt
from numpy import asarray
import random

# In[2]:


# specify input shape
# The omniglot dataset has image size of (105,105) in grayscale
input_shape = (105, 105, 1)

# base_model for the siamese neural network
base_model = keras.models.Sequential([
            Conv2D(64, (10, 10), activation = 'relu', input_shape = input_shape),
            MaxPooling2D(pool_size=(2, 2), strides = 2),

            Conv2D(128, (7, 7), activation = 'relu'),
            MaxPooling2D(pool_size=(2, 2), strides = 2),

            Conv2D(128, (4, 4), activation = 'relu'),
            MaxPooling2D(pool_size=(2, 2), strides = 2),

            Conv2D(256, (4, 4), activation = 'relu'),
            MaxPooling2D(pool_size=(2, 2), strides = 2),

            Flatten(),
            Dense(4096, activation = 'sigmoid')
    ])
base_model.summary()


# In[3]:


#Calculate L1 distance
# "In this paper we use the weighted L1 distance between the twin feature vectors h1 and h2 combined with
# a sigmoid activation, which maps onto the interval [0, 1]."
def L1_dist(vectors):
    a, b = vectors
    c = np.abs(a - b)
    return c


# In[4]:


# Form the Siamese Neural Network Model
input_a = Input(shape = input_shape)
input_b = Input(shape = input_shape)

# Two inputs 'base_a' and 'base_b'
base_a = base_model(input_a)
base_b = base_model(input_b)

# use keras.layers.Lambda to form a layer consisting of functions
L1_layer = Lambda(L1_dist)([base_a, base_b])
# It's a binary classification problem, thus we use 'sigmoid' as the activation
final_layer = Dense(1, activation = 'sigmoid')(L1_layer)

model = Model([input_a, input_b], final_layer)
model.summary()


# In[5]:


# Compile the model.
model.compile(optimizer = 'Adam', loss = keras.losses.binary_crossentropy, metrics = ['accuracy'])


# In[6]:


# load image from folders and preprocess images 
def load_images(directory):
    images = list()

    for filename in listdir(directory):
        path = directory + filename
        # load img as PIL type, specify grayscale and target_size
        image = load_img(path, target_size = (105, 105, 1), grayscale = True)
        # convert img to numpy array
        image = img_to_array(image)
        images.append(image)
    # remember the item returned here is still a list not numpy array. 
    # Need to convert to numpy array later for keras to process
    return images


# In[7]:


# load images from downloaded folder of omniglot
def load_dataset(directory):
    x, y = list(), list()
    # locate subfolders1
    for subdir1 in listdir(directory):
        # path for each alphabeta
        subfolder1 = directory + subdir1 + '/'
        # locate subfolders2
        for subdir2 in listdir(subfolder1):
            # path for each character image
            path = subfolder1 + subdir2 + '/'

            images = load_images(path)
            labels = [subdir1 for _ in range(len(images))] # alphabeta name of the character
            print('>loaded %d examples for class: %s' % (len(images), subdir1))
            
            # append: Adds its argument as a single element to the end of a list.
            # extend: Iterates over its argument and adding each element to the list and extending the list. The 
            # length of the list increases by number of elements in it's argument.
            # Here we use extend is because that images loaded by 'load_images' is a list of images,
            # we want to add each image to x not the list. 
            x.extend(images)
            y.extend(labels)
            
    # here we convert the list to numpy array        
    return asarray(x), asarray(y)


# In[8]:


x, y = load_dataset("images_background/")
x_test, y_test = load_dataset("images_evaluation/")


# In[20]:


print(x.shape, y.shape) 
print(x_test.shape, y_test.shape)
import matplotlib.cm as cm
a = x[0]
a = a.reshape((105, 105))
plt.imshow(a ,cmap=cm.gray, vmin=0, vmax=255)


# In[21]:


x = x.astype('float32')
x_test = x.astype('float32')
x /= 255.0
x_test /= 255.0
num_classes = 30
classes = ['Alphabet_of_the_Magi', 'Anglo-Saxon_Futhorc', 'Arcadian', 'Armenian', 
           'Asomtavruli_(Georgian)', 'Balinese', 'Bengali', 'Blackfoot_(Canadian_Aboriginal_Syllabics)', 
           'Braille', 'Burmese_(Myanmar)', 'Cyrillic', 'Early_Aramaic', 'Futurama', 'Grantha', 'Greek',
           'Gujarati', 'Hebrew', 'Inuktitut_(Canadian_Aboriginal_Syllabics)', 'Japanese_(hiragana)', 
           'Japanese_(katakana)', 'Korean', 'Latin', 'Malay_(Jawi_-_Arabic)', 'Mkhedruli_(Georgian)',
           'N_Ko', 'Ojibwe_(Canadian_Aboriginal_Syllabics)', 'Sanskrit', 'Syriac_(Estrangelo)', 'Tagalog',
           'Tifinagh']

# y is a numpy array with shape (19280,). The return of 'np.where(...)' is a tuple ([index1, index2,...], )
# In order to get the index array, we add '[0]'
# 'category' lists the indexes of training labels from class1 'Alphabet_of_the_Magi' to class30 'Tifinagh'.
# 'category' has size of (num_classes, num_samples)
category = [np.where(y == classes[i])[0] for i in range(num_classes)]

num_classes_test = 20
classes_test = ['Angelic', 'Atemayar_Qelisayer', 'Atlantean', 'Aurek-Besh', 'Avesta', 'Ge_ez', 'Glagolitic', 
                'Gurmukhi', 'Kannada', 'Keble', 'Malayalam', 'Manipuri', 'Mongolian', 
                'Old_Church_Slavonic_(Cyrillic)', 'Oriya', 'Sylheti', 'Syriac_(Serto)', 'Tengwar', 'Tibetan',
                'ULOG']
category_test = [np.where(y_test == classes_test[j])[0] for j in range(num_classes_test)]


# In[22]:


# The model needs pairs input. When the two inputs are from same alphabet, the output is 1, otherwise output 0.
# Thus we need to create both positive pairs and negative pairs for training and testing
# x: images after proprecessing,
# category: indexes of labels from class_1 to class_num.
# num: number of classes
def create_pairs(x, category, num):
    pairs = []
    labels = []
    
    # n: the least number of samples in all classes. (The numbers of samples in each class are different. Thus we use min() to guarantee pairs.)
    # note that category has shape (num_classes, num_samples)
    n = min([len(category[d]) for d in range(num)]) - 1
    for d in range(num):
        for i in range(n):
            # create positive pairs
            z1, z2 = category[d][i], category[d][i + 1] # indexes of the same class
            pairs += [[x[z1], x[z2]]] # '+=' operation is equivalent to .extend([a, b])
            
            # create negative pairs
            inc = random.randrange(1, num) # return a randomly selected element a, 1 <= a < num
            dn = (d + inc) % num # This func guaranteed that dn != d
            z1, z2 = category[d][i], category[dn][i]
            pairs += [[x[z1], x[z2]]] #negative pairs
            labels += [1, 0] # '+=' operation is equivalent to .extend([a, b])
            
    # convert the list to numpy array        
    return np.array(pairs), np.array(labels)


# In[23]:


tr_pairs, tr_y = create_pairs(x, category, num_classes)
te_pairs, te_y = create_pairs(x_test, category_test, num_classes_test)


# In[24]:


print(tr_pairs.shape)
print(te_pairs.shape)


# In[ ]:


model.fit([tr_pairs[:, 0], tr_pairs[:, 1]], tr_y,
         batch_size = 128, epochs = 10, validation_data = ([te_pairs[:, 0], te_pairs[:,1]], te_y), verbose = 1)
#Epoch 8/20, loss: 0.0183, acc: 0.9976, evaluation loss: 0.2958, evaluation acc: 0.9134 


# In[ ]:


model.save('Siamese_NN.h5')


# In[ ]:


model.save_weights('Siamese_NN_weights.h5')


# In[ ]:




