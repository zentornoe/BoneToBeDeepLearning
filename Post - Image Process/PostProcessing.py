import cv2
import numpy as np

#file path of image
FILE_PATH = './ImageProcessing/images/b_02.png'


# *** Interner Functions ***

#evaluate angle
def _angle(cv_image):  # INPUT : cv2 image (cv2.imread(file_path))
    maxArea = 0
    maxNum = 0

    img = cv_image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thr = cv2.threshold(gray, 127, 255, 0)
    cnt, _ = cv2.findContours(thr, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for i in range(len(cnt)):       # calculate contours which has maximum area
        area = cv2.contourArea(cnt[i])
        if maxArea < area:
            maxArea = area
            maxNum = i

    rect = cv2.minAreaRect(cnt[maxNum])     # Calculate minimum area rectangle
    box = cv2.boxPoints(rect)

    box = np.int0(box)
    # Draw contour of  mimAreaRectangle
    cv2.drawContours(img, [box], 0, (255, 255, 0), 1)

    # Rotated angle
    if((pow(box[0][0]-box[1][0], 2)+pow(box[0][1]-box[1][1], 2)) >= (pow(box[1][0]-box[2][0], 2)+pow(box[1][1]-box[2][1], 2))):
        return -90+rect[2]
    else:
        return rect[2]


#expand image w/o resizing
def _expand(cv_image):     # INPUT : cv2 image (cv2.imread(file_path))
    img = cv_image                  # read image
    h, w, c = img.shape             # get height, width, channel of image
    # root(2) = 1.41 ... -> resizing ratio = 1.5
    nh = int(h*1.5)
    nw = int(w*1.5)

    blank_img = np.zeros((nh, nw, c), np.uint8)  # generate blank resized image
    blank_img[:, :] = (0, 0, 0)

    #calculate offset to center image
    result = blank_img.copy()
    x_offset = int((nh-h)/2)
    y_offset = int((nw-w)/2)

    # overwrite the orignial image to blank image
    result[y_offset:y_offset+h, x_offset:x_offset+w] = img.copy()

    return result  # open_cv image


#rotate image
def _rotate(cv_image, angle):  # INPUT : cv2 image (cv2.imread(file_path))
    img = cv_image                  # read image
    h, w, c = img.shape             # Get the height, width, channel of image

    # Set the rotation axis, angle, and scale
    mat = cv2.getRotationMatrix2D((w/2, h/2), 90+angle, 1)
    # result(rotated image) : img -> (angle) rotated
    rotated = cv2.warpAffine(img, mat, (w, h))

    return rotated  # cv2 image


#Area division & filling
def _division(cv_image):       # INPUT : cv2 image (cv2.imread(file_path))
    maxArea = 0
    maxNum = 0
    maxNum2 = 0

    img = cv_image                                 # read image
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

    #cv2.drawContours(img, [cnt[maxNum2]], 0, (255,0,255), 5)    #Magenta: 2nd maxArea
    #cv2.drawContours(img, [cnt[maxNum]], 0, (255,255,0), 5)     #Cyan   : 1st maxArea
    cv2.fillPoly(img, pts=[cnt[maxNum2]], color=(
        0, 0, 0))  # 2nd Contour filling

    #cv2.imwrite('Result.png', img)

    cv2.imshow('result1', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return img


# Fractured bone w/o overlap
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

    #cv2.drawContours(img, [cnt[maxNum2]], 0, (255,0,255), 5)    #Magenta: 2nd maxArea
    #cv2.drawContours(img, [cnt[maxNum]], 0, (255,255,0), 5)     #Cyan   : 1st maxArea
    cv2.fillPoly(img, pts=[cnt[maxNum2]], color=(0, 0, 0))  # 2nd Contour filling
    cv2.fillPoly(img1, pts=[cnt[maxNum]], color=(0, 0, 0))  # 1st Contour filling

    return img, img1


# Fractured bone with overlap
def _fracWhere(cv_image):      # INPUT : cv2 image
    maxArea = 0
    maxNum = 0
    clst = 0
    clst_ = 0
    clst1 = 0
    clst2 = 0

    img = cv_image                                 # read image
    img1 = np.copy(img)
    # convert color file to gray scale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thr = cv2.threshold(gray, 127, 255, 0)     # get threshold
    cnt, _ = cv2.findContours(
        thr, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # find contours
    h, w, _ = img.shape
    center = int(h/2)
    clst = h
    clst_ = h
    print('Center : '+str(center))

    for i in range(len(cnt)):       # calculate contours which has maximum area
        area = cv2.contourArea(cnt[i])
        if maxArea < area:
            maxArea = area
            maxNum = i
    ct = cnt[maxNum]

    cv2.drawContours(img, [cnt[maxNum]], 0, (125, 125, 0), 3)

    hull = cv2.convexHull(cnt[maxNum], returnPoints=False)
    defects = cv2.convexityDefects(cnt[maxNum], hull)

    for i in range(defects.shape[0]):
        sp, ep, fp, dist = defects[i, 0]
        start = tuple(ct[sp][0])
        end = tuple(ct[ep][0])
        farthest = tuple(ct[fp][0])

        if(clst_ > abs(farthest[1]-center)):
            if(clst > abs(farthest[1]-center)):
                clst_ = clst
                clst = abs(farthest[1]-center)
                clst2 = clst1
                clst1 = farthest[1]
            else:
                clst_ = abs(farthest[1]-center)
                clst2 = farthest[1]

        cv2.circle(img1, farthest, 5, (0, 255, 255), -1)

    cv2.line(img1, (0, center), (w, center), (0, 255, 0), 5)
    cv2.line(img1, (0, clst1), (w, clst1), (255, 255, 0), 3)
    cv2.line(img1, (0, clst2), (w, clst2), (255, 255, 0), 3)

    return clst1, clst2


# Seperate overlapped images
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






## *** Functions for Cases ***

# Fracture X, Area division X
def case1(cv_image):
    return _rotate(_expand(cv_image), _angle(cv_image))


# Fracture X, Area division O
def case2(cv_image):
    image = _division(cv_image)
    return _rotate(_expand(image), _angle(image))


# Fracture O, Overlap X
def case3(cv_image):
    img0, img1 = _seperate(cv_image)
    angle0 = _angle(img0)
    angle1 = _angle(img1)

    # OUTPUT : cv2_image0, cv2_image1, angle0(image0), angle1(image1)
    return _rotate(_expand(img0), angle0), _rotate(_expand(img1), angle1), angle0, angle1


# Fracture O, Overlap O
def case4(cv_image):
    image = _rotate(_expand(cv_image), _angle(cv_image))
    cv_image1 = np.copy(image)
    n0, n1 = _fracWhere(image)
    img0, img1 = _cutting(cv_image1, n0, n1)
    ang0 = _angle(img0)
    ang1 = _angle(img1)
    return _rotate(_expand(img0), ang0), _rotate(_expand(img1), ang1), ang0, ang1
