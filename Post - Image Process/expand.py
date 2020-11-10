def _expand(cv_image):     # INPUT : cv2 image (cv2.imread(file_path))
    img = cv_image                  # read image
    h, w, c = img.shape             # get height, width, channel of image
    nh = int(h*1.5)                 # root(2) = 1.41 ... -> resizing ratio = 1.5 
    nw = int(w*1.5)

    blank_img = np.zeros((nh, nw, c), np.uint8) # generate blank resized image
    blank_img[:, :] = (0, 0, 0)

    #calculate offset to center image
    result = blank_img.copy()
    x_offset = int((nh-h)/2)
    y_offset = int((nw-w)/2)

    result[y_offset:y_offset+h, x_offset:x_offset+w] = img.copy()   # overwrite the orignial image to blank image

    return result  # open_cv image
