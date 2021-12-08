import os
import numpy as np
import cv2 as cv
import sys



def gray_scale(img):
    return cv.cvtColor(img, cv.COLOR_BGR2GRAY)


def round(SIZE):
    (w, h) = (int(SIZE[0]), int(SIZE[1]))
    dw = np.absolute(SIZE[0] - w)
    dh = np.absolute(SIZE[1] - h)
    if dw >= 0.5:
        w += 1
    if dh > 0.5:
        h += 1
    return (w, h)


def findeTemplate(template, img_source, method=cv.TM_CCOEFF_NORMED):
    scale_array_h = np.arange(0.074, 0.121, 0.001)
    scale_array_w = np.arange(0.015, 0.03, 0.001)

    template_gray = gray_scale(template)
    img_source_gray = gray_scale(img_source)

    performances = []
    for scale_h in scale_array_h:
        for scale_w in scale_array_w:
            SIZE = (img_source.shape[1] * scale_w, img_source.shape[0] * scale_h)
            SIZE = round(SIZE)
            resized_template = cv.resize(template_gray, SIZE, interpolation=cv.INTER_AREA)
            h, w = resized_template.shape
            matched = cv.matchTemplate(img_source_gray, resized_template, method=cv.TM_CCOEFF_NORMED)
            minVal, maxVal, minLoc, maxLoc = cv.minMaxLoc(matched)
            performances.append([maxVal, maxLoc, scale_w, scale_h, w, h])
    performances.sort(key=lambda x: x[0], reverse=True)
    return performances


if __name__ == '__main__':
    path = os.getcwd()
    input = []
    gallary = []
    for img in os.listdir(path + "/Inputs"):
        input.append(cv.imread((cv.samples.findFile(path + "/Inputs/" + img))))
    for glr in os.listdir(path + "/Gallery"):
        gallary.append(cv.imread((cv.samples.findFile(path + "/Gallery/" + glr))))
    j = 0
    for glr in gallary:
        for template in input:
            print(template.shape)
            tmp_glr = np.copy(glr)
            performances = findeTemplate(template, tmp_glr)
            print(performances[0])
            top_left = performances[0][1]
            bottom_right = (top_left[0] + performances[0][-2], top_left[1] + performances[0][-1])
            cv.rectangle(glr, top_left, bottom_right, color=(0, 0, 255), thickness=3)
        j += 1
        dir = path + "/Outputs/glr_tmpl_matching" + str(j) + ".png"
        cv.imwrite(dir, glr)
