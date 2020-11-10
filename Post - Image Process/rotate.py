def _rotate(cv_image, angle):  # INPUT : cv2 image (cv2.imread(file_path)), angle (integer)
    img = cv_image                  # read image
    h, w, c = img.shape             # Get the height, width, channel of image

    mat = cv2.getRotationMatrix2D((w/2, h/2), 90+angle, 1)  # Set the rotation axis, angle, and scale
    rotated = cv2.warpAffine(img, mat, (w, h))              # result(rotated image) : img -> (angle) rotated
    
    return rotated  # cv2 image
