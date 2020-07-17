import shutil
import cv2
import os
from keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as mpimg
base_dir = './'
img_dir = './images/'
f_dic = {}
index = 0
X = []
Y = []
def img_proc(filename):
    #img = cv2.imread(img_dir+filename+'.png')
    img = cv2.imread(img_dir+filename+'.png', flags=cv2.IMREAD_GRAYSCALE)
    if img.shape[1] > img.shape[0]:
        percent = 256/img.shape[1]
    else:
        percent = 256/img.shape[0]
    img = cv2.resize(img, None, fx=percent, fy=percent, interpolation=cv2.INTER_AREA)
    h, w = img.shape[:2]
    w_x = (256-w)/2
    h_y = (256-h)/2
    M = np.float32([[1, 0, w_x], [0, 1, h_y]])
    img_re = cv2.warpAffine(img, M, (256, 256))
    img_re = img_re/255.
    return img_re

for dirpath, dirnames, filenames in os.walk(img_dir):
    for filename in filenames:
        if filename[0] == '.': continue
        filename = filename.replace('.png', '')
        if filename[-1] != 'L':
            f_dic[filename] = index
            index += 1
    #diction 만들기

for i in f_dic:
    x = img_proc(i)
    y = img_proc(i+'L')
    X.append(x)
    Y.append(y)
X = np.array(X)
Y = np.array(Y)
X = np.expand_dims(X, axis=-1)
Y = np.expand_dims(Y, axis=-1)
print(X.shape)
print(Y.shape)
np.save(base_dir+'/dataset/x_data', X)
np.save(base_dir+'/dataset/y_data', Y)