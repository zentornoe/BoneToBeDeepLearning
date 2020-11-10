def _division(cv_image) :       # INPUT : cv2 image (cv2.imread(file_path))
    maxArea = 0
    maxNum = 0
    maxNum2 = 0

    img = cv_image                                 # read image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)    # convert color file to gray scale
    ret, thr = cv2.threshold(gray, 127, 255, 0)     # get threshold
    cnt, _ = cv2.findContours(thr, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)  # find contours

    for i in range(len(cnt)):       # calculate contours which has maximum area
        area = cv2.contourArea(cnt[i])
        if maxArea < area:
            maxArea = area
            maxNum2 = maxNum
            maxNum = i
    
    cv2.drawContours(img, [cnt[maxNum]], 0, (255,255,0), 5)     #Cyan   : 1st maxArea
    cv2.fillPoly(img, pts = [cnt[maxNum2]], color = (0,0,0))    #2nd Contour filling

    cv2.imwrite('result.png', img)

    cv2.imshow('result2', img)
    cv2.imshow('result1', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return img      # OUTPUT : OpenCV2 image
