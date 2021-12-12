import os
import numpy as np
import cv2 as cv
import time

if __name__ == '__main__':
    start = time.time()

    # read data
    path = os.getcwd()
    input = cv.imread(path + "/Inputs/input2.png")
    glr = cv.imread(path + "/Gallery/gal1.png")

    # SIFT algorithm object
    sift = cv.SIFT_create()

    # get keypoints and descriptor
    keypoints_input, descriptor_input = sift.detectAndCompute(input, None)
    keypoints_glr, descriptor_glr = sift.detectAndCompute(glr, None)

    # set up parameter for FlannBasedMatcher
    flannMatcher = cv.FlannBasedMatcher(
        {'algorithm': 1, 'trees': 5},
        {'checks': 50}
    )

    # take the k=2 best match
    matches = flannMatcher.knnMatch(descriptor_input, descriptor_glr, k=2)

    # using ratio test to eliminate false matches - Lowe's proposal
    good = []
    for m, n in matches:
        if m.distance < 0.9 * n.distance:
            good.append(m)

    if len(good) > 20:
        print("good matches:", len(good))

        # find homography
        src_pts = np.float32([keypoints_input[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints_glr[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()

        # inliers
        print("inliers: ", matchesMask.count(1))

        # inliers
        print("outliers: ", matchesMask.count(0))

        # draw bouding box with homography M
        h, w = input.shape[0], input.shape[1]
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        dst = cv.perspectiveTransform(pts, M)
        glr = cv.polylines(glr, [np.int32(dst)], True, (0, 0, 255), 3, cv.LINE_AA)
    else:
        print("Not enough matches are found - {}/{}".format(len(good), 5))
        matchesMask = None

    draw_params = dict(matchColor=(0, 156, 0),
                       singlePointColor=None,
                       matchesMask=matchesMask,  # draw only inliers
                       flags=2)
    glr = cv.drawMatches(input, keypoints_input, glr, keypoints_glr, good, None, **draw_params)

    folder = os.listdir(path + "/Outputs")
    if "feature_based_approach" not in folder:
        os.mkdir(path + "/Outputs/" + "feature_based_approach")
    os.chdir(path + "/Outputs/" + "feature_based_approach")

    # save results
    cv.imwrite("output12.png", glr)

    print("--- %s seconds ---" % (time.time() - start))
