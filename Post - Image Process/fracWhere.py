def _fracWhere(cv_image) :      # INPUT : cv2 image
    maxArea = 0
    maxNum = 0
    clst = 0
    clst_ = 0
    clst1 = 0
    clst2 = 0

    img =  cv_image                                 # read image
    img1 = np.copy(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)    # convert color file to gray scale
    ret, thr = cv2.threshold(gray, 127, 255, 0)     # get threshold
    cnt, _ = cv2.findContours(thr, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # find contours
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
    
    cv2.drawContours(img, [cnt[maxNum]], 0, (125,125,0), 3)

    hull = cv2.convexHull(cnt[maxNum], returnPoints=False)
    defects = cv2.convexityDefects(cnt[maxNum], hull)
    
    
    for i in range(defects.shape[0]):
        sp, ep, fp, dist = defects[i, 0]
        start = tuple(ct[sp][0])
        end = tuple(ct[ep][0])
        farthest = tuple(ct[fp][0])
        print(farthest[1])
        
        if(clst_ > abs(farthest[1]-center)):
            if(clst > abs(farthest[1]-center)):
                clst_ = clst
                clst = abs(farthest[1]-center)
                clst2 = clst1
                clst1 = farthest[1]  
            else :
                clst_ = abs(farthest[1]-center)
                clst2 = farthest[1]

        cv2.circle(img1, farthest, 5, (0, 255,255), -1)
        
    cv2.line(img1, (0, center), (w, center), (0, 255, 0), 5)
    cv2.line(img1, (0, clst1), (w, clst1), (255, 255, 0), 3)
    cv2.line(img1, (0, clst2), (w, clst2), (255, 255, 0), 3)
    print('Closest Value')
    print(clst1)
    print(clst2)
   
    cv2.imshow('result1', img1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
