import numpy as np
import cv2

degree = 3
img = cv2.imread("edit_miu.png", cv2.IMREAD_UNCHANGED)
h, w = img.shape[:2]

R1 = cv2.getRotationMatrix2D((w/2, h/2), degree, 1)  # central point, rotate angle [degree], scale

fin = cv2.warpAffine(img, R1, (w,h)) # Overwrite the rotated image

cv2.imwrite(str(degree)+'_degree_edit_miu.png', fin)     # Save image
cv2.imshow(str(degree)+'-Rotated', fin)
cv2.waitKey(0)
cv2.destroyAllWindows()