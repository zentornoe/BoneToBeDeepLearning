def _angle(cv_image):  # INPUT : OpenCV2 image (cv2.imread(file_path))
    maxArea = 0
    maxNum = 0

    img =  cv_image                                 # read image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)    # convert color file to gray scale
    ret, thr = cv2.threshold(gray, 127, 255, 0)     # get threshold
    cnt, _ = cv2.findContours(thr, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # find contours

    for i in range(len(cnt)):       # calculate contours which has maximum area
        area = cv2.contourArea(cnt[i])
        if maxArea < area:
            maxArea = area
            maxNum = i

    rect = cv2.minAreaRect(cnt[maxNum])     # Calculate minimum area rectangle
    
    # drawing contours to show
    
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    cv2.drawContours(img, [box], 0, (255, 255, 0), 1)    # Draw contour of  mimAreaRectangle
    
    print('\nRotated angle : '+str(90-rect[2])+'\n\n')  # for vertical image

    return rect[2]  # rotated angle
