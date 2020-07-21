import os
import cv2
import numpy as np

# Main Function
def main() :
    path = "./"
    file_list = os.listdir(path)
    img_list_png = [file for file in file_list if file.endswith(".png")]

    for i in img_list_png :
        resizing(i)

#Resizing Function
def resizing(fname) :
    # Read image
    orig = cv2.imread(fname, cv2.IMREAD_UNCHANGED)

    h, w, c = orig.shape    # height, width, channel of original image

    # Maintain the portion
    if h >= w:
        min = int(900*w/h)
        state = 1   # vertical image
        edit = cv2.resize(orig, dsize=(min, 900), interpolation=cv2.INTER_AREA)  # Resizing
    else:
        min = int(900*h/w)
        state = 2   # horizontal image
        edit = cv2.resize(orig, dsize=(900, min), interpolation=cv2.INTER_AREA)  # Resizing

    fin = np.zeros((900, 900, c), np.uint8)  # png = 8bits, 900*900 black image

    b1 = int((900-min)/2)  # for calculate staring point

    if state == 1:  # vertical image
        pos = fin[0:900, b1:b1+min] = edit
    else:  # horizontal image
        pos = fin[b1:b1+min, 0:900] = edit

    cv2.imwrite("./new/"+fname, fin)    # Save result image
    print(fname + " Store Complete.")


#Call main()
main()