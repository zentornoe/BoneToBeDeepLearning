import cv2
import numpy as np

def _cutting(cv_image, pt1, pt2):
    h, w, _ = cv_image.shape        # height, width, channel

    img0 = cv_image
    img1 = np.copy(img0)

    if pt2 < pt1:
        fst = pt1
        snd = pt2
    else:
        fst = pt2
        snd = pt1

    cv2.rectangle(img0, (0, snd), (w, h), (0, 0, 0), -1)      # Fill the lower part
    cv2.rectangle(img1, (0, 0), (w, fst), (0, 0, 0), -1)      # Fill the upper part

    return img0, img1
