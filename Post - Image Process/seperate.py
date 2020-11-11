import cv2
import numpy as np

def _seperate(cv_image):
    maxArea = 0
    maxNum = 0
    maxNum2 = 0

    img = cv_image                                  # read image
    img1 = np.copy(img)                             # copied image
    # convert color file to gray scale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thr = cv2.threshold(gray, 127, 255, 0)     # get threshold
    cnt, _ = cv2.findContours(
        thr, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)  # find contours

    for i in range(len(cnt)):       # calculate contours which has maximum area
        area = cv2.contourArea(cnt[i])
        if maxArea < area:
            maxArea = area
            maxNum2 = maxNum
            maxNum = i

    cv2.fillPoly(img, pts=[cnt[maxNum2]], color=(0, 0, 0))  # 2nd Contour filling
    cv2.fillPoly(img1, pts=[cnt[maxNum]], color=(0, 0, 0))  # 1st Contour filling

    return img, img1
