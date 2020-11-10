# CT Femoral Bone Segmentation

2020 KNU Capstone Design Project

TEAM: Bone to be Deep Learning

※ **Unable to upload processed images due to security by medical ethics.**

|OS|Program Language|Additional|
|:---:|:---:|:---:|
|WINDOWS 10|Python|OpenCV 2, tensorflow 2, CUDA 10, KERAS, matplotlib|


| CPU | GPU | RAM |
|:---:|:---:|:---:|
|Intel i7-9700K|NDVIA RTX2080Ti (x2, Parallel programming)|32GB|


* * *
## 1. Pre - Image Process
Image processing program, for using images for deep learning.
Image is processed as array by using **numpy**. Image processing functions are from **Open CV 2**.

### 1) Rotate.py
Read image as open cv image. Rotate the image counterclockwise as much as _'degree'_ based on the **center** of the image.

And Save png file.

### 2) auto_resizing.py
Original Labeling, X-ray images are too big to process Deep Learning.

Change images size into (900 x _X_) or (_X_ x 900) pixels. (_X_ is equal or smaller than 900) 

ex) 2700 x 2700 -> 900 * 900  /  2700 * 1800 -> 900 * 600  /  1800 * 2700 -> 600 * 900

### 3) preprocessing.py
Make the item 2) more advanced and save the entire image inside the folder.

(Maximum pixel size : 256)
* * *
## 2. Deep Learning
X-ray Image Segmentation program using Deep Learning

### 1) train.py
Segmentation program structured by CNN using tensorflow and keras.
* * *
## 3. Post Image Process


### 1) division ...

* * *
## 4. Similarity


* * *
## 5. 3D Model Viewer



* * *
###### Copyright 2020. BornToBeDeeplearning All Rights Reserved
