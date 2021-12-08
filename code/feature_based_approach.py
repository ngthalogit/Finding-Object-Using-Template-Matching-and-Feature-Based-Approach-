import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

if __name__ == '__main__':
    dir_img = "../Inputs/input1.png"
    dir_glr = "../Gallery/gal2.jpeg"

    input = cv.imread(dir_img, 0)
    glr = cv.imread(dir_glr, 0)

    sift = cv.SIFT_create()

    keypoints_input, descriptor_input = sift.detectAndCompute(input, None)
    keypoints_glr, descriptor_glr = sift.detectAndCompute(glr, None)

    flannMatcher = cv.FlannBasedMatcher(
        {'algorithm': 1, 'trees': 5},
        {'checks': 50}
    )

    matches = flannMatcher.knnMatch(descriptor_input, descriptor_glr, k=2)

    good = []
    for m, n in matches:
        if m.distance < n.distance:
            good.append(m)

    if len(good) > 10:
        src_pts = np.float32([keypoints_input[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints_glr[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()
        h, w = input.shape
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        dst = cv.perspectiveTransform(pts, M)
        glr = cv.polylines(glr, [np.int32(dst)], True, 0, 3, cv.LINE_AA)
    else:
        print("Not enough matches are found - {}/{}".format(len(good), 10))
        matchesMask = None

    draw_params = dict(matchColor=(0, 156, 0),
                       singlePointColor=None,
                       matchesMask=matchesMask,  # draw only inliers
                       flags=2)
    glr = cv.drawMatches(input, keypoints_input, glr, keypoints_glr, good, None, **draw_params)
    #plt.imshow(output, 'gray'), plt.show()
    cv.imwrite("ft.png", glr)
