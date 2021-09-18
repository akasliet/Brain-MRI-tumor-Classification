import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np

import os
import cv2

import random

import skimage.io
import skimage.filters
import sys

from sklearn.model_selection import train_test_split

no_of_epoches = 30
image_X = 320
image_Y = 240
data_augmentation_flag = 0
# Path of dataset with grade0 and grade 1 skin burn images

#yesf = 'C:\\Users\\admin\\Desktop\\Thermal\\G1'
#nof = 'C:\\Users\\admin\\Desktop\\Thermal\\G0'

yesf = 'C:\\Users\\admin\\Desktop\\Thermal\\Yes'
nof = 'C:\\Users\\admin\\Desktop\\Thermal\\No'

sigma1 = 5.0
sigma2 = 10.0
sigma3 = 15.0
sigma4 = 20.0

# Read tumor images and mark class label as 1  

woai = []
woac = []
woni = 0
wonyi = 0

wiai = []
wiac = []
wini = 0
winyi = 0
for filename in os.listdir(yesf):
        wonyi = wonyi + 1
        winyi = winyi + 5
        fn1 = os.path.join(yesf,filename)
        temp1 = cv2.imread(fn1)
        tempo= cv2.resize(temp1,dsize=(image_X,image_Y),interpolation=cv2.INTER_CUBIC)
        tempb1 = skimage.filters.gaussian(tempo, sigma=(sigma1, sigma1), truncate=3.5, multichannel=True)
        tempb2 = skimage.filters.gaussian(tempo, sigma=(sigma2, sigma2), truncate=3.5, multichannel=True)  
        tempb3 = skimage.filters.gaussian(tempo, sigma=(sigma3, sigma3), truncate=3.5, multichannel=True)
        tempb4 = skimage.filters.gaussian(tempo, sigma=(sigma4, sigma4), truncate=3.5, multichannel=True)
        
        woai.append(tempo)
        woac.append(1)
        
        wiai.append(tempo)
        wiai.append(tempb1)
        wiai.append(tempb2)
        wiai.append(tempb3)
        wiai.append(tempb4)
        
        wiac.append(1)
        wiac.append(1)
        wiac.append(1)
        wiac.append(1)
        wiac.append(1)


# Read no tumor images and mark class label as 0        
wonni = 0
winni = 0
for filename in os.listdir(nof):    
        wonni = wonni +1
        winni = winni + 5
        temp2 = cv2.imread(os.path.join(nof,filename))
        tempo = cv2.resize(temp2,dsize=(image_X,image_Y),interpolation=cv2.INTER_CUBIC)        
        tempb1 = skimage.filters.gaussian(tempo, sigma=(sigma1, sigma1), truncate=3.5, multichannel=True)
        tempb2 = skimage.filters.gaussian(tempo, sigma=(sigma2, sigma2), truncate=3.5, multichannel=True)  
        tempb3 = skimage.filters.gaussian(tempo, sigma=(sigma3, sigma3), truncate=3.5, multichannel=True)
        tempb4 = skimage.filters.gaussian(tempo, sigma=(sigma4, sigma4), truncate=3.5, multichannel=True)
       
        woai.append(tempo)
        woac.append(0)
        
        wiai.append(tempo)
        wiai.append(tempb1)
        wiai.append(tempb2)
        wiai.append(tempb3)
        wiai.append(tempb4)
       
        wiac.append(0)
        wiac.append(0)
        wiac.append(0)
        wiac.append(0)
        wiac.append(0)
        

wonti = wonyi + wonni
winti = winyi + winni

# woai woac wiai wiac
print('no. of yes, no and total images without augnetation= ', wonyi, wonni, wonti)
print('no. of yes, no and total images with augnetation= ', winyi, winni, winti)
 

# Plot training images
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(wiai[i])
plt.show()

n = winti
ai = wiai
ac = wiac


# Prepare indices for training and testing images and their classes
ntr = round(n*0.8)
ntt = n - ntr
tri = [random.randint(1,n) for i in range(0,ntr)]
tot = np.arange(n)
tes = np.setdiff1d(tot,tri)

# Split into training and testing sets.
train_images, test_images, train_labels, test_labels = train_test_split(ai, ac, test_size=0.2, random_state=0)


for i in range(len(train_images)):
    train_images[i] = train_images[i]/255.0

for i in range(len(test_images)):
    test_images[i] = test_images[i]/255.0
    
    
# Normalize images in 0-1
#train_images, test_images = train_images / 255.0, test_images / 255.0

# Class names
class_names = ['grade 0', 'grade 1']

# Plot training images
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i])
    plt.xlabel(class_names[train_labels[i]])
plt.show()
