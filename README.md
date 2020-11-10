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

Each function is used in combination according to a given case.

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
After segmentation, we have to image process for check similarity and open the 3D viewer.

For calculating area & finding points that have features, I used __'contours'__ in open cv.

And for evaluating angle, I used __'minAreaRect'__ in open cv.

### 1) expand.py
When rotating the image, make it 1.5 times the pixel size starting from the center and fill the gap with black to prevent image loss (cut) at both ends.

|Input|Output|
|:---:|:---:|
|OpenCV2 image|OpenCV2 image(expanded image)|

### 2) angle.py
Evaluate the angle at which the bone is rotated around the y-axis in the image.

|Input|Output|
|:---:|:---:|
|OpenCV2 image|integer(angle)|

### 3) rotate.py
Rotate the image as expanded by the input angle.

|Input|Output|
|:---:|:---:|
|OpenCV2 image, integer(angle)|OpenCV2 image(rotated image)|

### 4) division.py
Divide the area into two parts, and remove any unwanted parts from deep learning.

|Input|Output|
|:---:|:---:|
|OpenCV2 image|OpenCV2 image(divided & removed unwanted part image)|


### 5) fracWhere.py
A function for finding features in an image in which a fractured bone overlaps, and for pre-processing to separate a fractured bone from feature points into two images.

|Input|Output|
|:---:|:---:|
|OpenCV2 image|OpenCV2 image(that representing points with features)|

* * *
## 4. Similarity
Comparing similarity to determine the angle rotated around the z-axis. Pre-processed in 3 and use the straight line image with the z-axis.

Z-axis rotated standard images are already saved in directory.

* * *
## 5. 3D Model Viewer
Rotating 3D model(bone, * .STL).

Input : angle (x, y, z)


### 1) Load STL
<img src="https://user-images.githubusercontent.com/58382336/98698584-8ad3f280-23b9-11eb-9055-3bfbb126cde9.png"  width="700" height="382">

### 2) Rotation
<img src="https://user-images.githubusercontent.com/58382336/98698681-aa6b1b00-23b9-11eb-9547-a6f6d66ea951.png"  width="700" height="382">

* * *
###### Copyright 2020. BornToBeDeeplearning All Rights Reserved
