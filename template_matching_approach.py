# lib
import numpy as np
import os
import cv2 as cv
import time


print("Hello")
# method for template matching
METHODS = ['cv.TM_CCOEFF', 'cv.TM_CCOEFF_NORMED', 'cv.TM_CCORR',
           'cv.TM_CCORR_NORMED', 'cv.TM_SQDIFF', 'cv.TM_SQDIFF_NORMED']


# scale up weight and height values to integer
def round(size):
    (w, h) = (int(size[0]), int(size[1]))
    dw = np.absolute(size[0] - w)
    dh = np.absolute(size[1] - h)
    if dw >= 0.5:
        w += 1
    if dh > 0.5:
        h += 1
    return (w, h)


# main
if __name__ == '__main__':
    start = time.time()
    # read data
    path = os.getcwd()
    input = cv.imread(path + "/Inputs/input1.png")
    gallary = cv.imread(path + "/Gallery/gal1.png")
    gallary_copy = np.copy(gallary)

    # gray scale
    input = cv.cvtColor(input, cv.COLOR_BGR2GRAY)
    gallary = cv.cvtColor(gallary, cv.COLOR_BGR2GRAY)

    # width and height scale values
    scale_w = np.arange(0.015, 0.03, 0.001)
    scale_h = np.arange(0.07, 0.121, 0.001)

    # method { 0: 'cv.TM_CCOEFF', 1: 'cv.TM_CCOEFF_NORMED', 2:'cv.TM_CCORR',
    #            3: 'cv.TM_CCORR_NORMED', 4: 'cv.TM_SQDIFF', 5: 'cv.TM_SQDIFF_NORMED' }
    m = METHODS[0]
    med = eval(m)

    # save the results
    performances = []
    # find the region taking the best results with resized template
    for sh in scale_h:
        for sw in scale_w:
            # get the size of template by scaling gallary's size
            size = round(tuple((gallary.shape[1] * sw, gallary.shape[0] * sh)))

            # resize input
            template = cv.resize(input, size, interpolation=cv.INTER_AREA)

            # slide the template image over image source  with methods in METHODS
            matches = cv.matchTemplate(template, gallary, method=med)

            # we take the smallest value base on  distance calculations
            if med in [cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]:
                minVal = cv.minMaxLoc(matches)[0]
                minLoc = cv.minMaxLoc(matches)[-2]
                performances.append([minVal, minLoc, (sw, sh), template.shape])
            # we take the largest value bases on correlation coefficient
            else:
                maxVal = cv.minMaxLoc(matches)[1]
                maxLoc = cv.minMaxLoc(matches)[-1]
                performances.append([maxVal, maxLoc, (sw, sh), template.shape])

    index = 0 if med in [cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED] else -1
    performances.sort(key=lambda x: x[0])

    print(str(m), end='\n')
    print(performances[index])

    h, w = performances[index][-1]
    loc = performances[index][-3]

    # draw bouding box for object
    cv.rectangle(gallary_copy, loc, (loc[0] + w, loc[1] + h), color=(0, 0, 255), thickness=3)
    folder = os.listdir(path + "/Outputs")

    # save results
    if m not in folder:
        os.mkdir(path + "/Outputs/" + m)
    #cv.imwrite(path + "/Outputs/" + m + "/output11.png", gallary_copy)

    print("--- %s seconds ---" % (time.time() - start))
