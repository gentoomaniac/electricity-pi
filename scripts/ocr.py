#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import sys

import click

from imutils.perspective import four_point_transform
from imutils import contours
import imutils
import numpy as np
import cv2

log = logging.getLogger(__file__)

DEBUG = True


def _configure_logging(verbosity):
    loglevel = max(3 - verbosity, 0) * 10
    logging.basicConfig(level=loglevel, format='[%(asctime)s] %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    if loglevel >= logging.DEBUG:
        # Disable debugging logging for external libraries
        for loggername in 'urllib3', 'google.auth.transport.requests':
            logging.getLogger(loggername).setLevel(logging.CRITICAL)


def test_image(img):
    cv2.imshow('', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# define the dictionary of digit segments so we can identify
# each digit on the thermostat
DIGITS_LOOKUP = {
    (1, 1, 0, 0, 0, 0, 0): 100,
    (1, 1, 1, 0, 1, 1, 1): 0,
    (0, 0, 1, 0, 0, 1, 0): 1,
    (1, 0, 1, 1, 1, 1, 0): 2,
    (1, 0, 1, 1, 0, 1, 1): 3,
    (0, 1, 1, 1, 0, 1, 0): 4,
    (1, 1, 0, 1, 0, 1, 1): 5,
    (1, 1, 0, 1, 1, 1, 1): 6,
    (1, 0, 1, 0, 0, 1, 0): 7,
    (1, 1, 1, 1, 1, 1, 1): 8,
    (1, 1, 1, 1, 0, 1, 1): 9
}


def grabcut_algorithm(original_image, bounding_box):

    image = original_image.copy()
    segment = np.zeros(image.shape[:2], np.uint8)

    x, y, width, height = bounding_box
    segment[y:y + height, x:x + width] = 1

    background_mdl = np.zeros((1, 65), np.float64)
    foreground_mdl = np.zeros((1, 65), np.float64)

    cv2.grabCut(image, segment, bounding_box, background_mdl, foreground_mdl, 5, cv2.GC_INIT_WITH_RECT)

    new_mask = np.where((segment == 2) | (segment == 0), 0, 1).astype('uint8')

    return image * new_mask[:, :, np.newaxis]


def get_edges(imagepath: str):
    # load the example image
    image = cv2.imread(imagepath)

    # pre-process the image by resizing it, converting it to
    # graycale, blurring it, and computing an edge map
    image = imutils.resize(image, height=500)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 200, 255)
    test_image(edged)

    # find contours in the edge map, then sort them by their
    # size in descending order
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    displayCnt = None
    # loop over the contours
    for c in cnts:
        # approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        # if the contour has four vertices, then we have found
        # the thermostat display
        if len(approx) == 4:
            displayCnt = approx
            break
    # extract the thermostat display, apply a perspective transform
    # to it
    warped = four_point_transform(gray, displayCnt.reshape(4, 2))
    output = four_point_transform(image, displayCnt.reshape(4, 2))

    test_image(output)

    # threshold the warped image, then apply a series of morphological
    # operations to cleanup the thresholded image
    thresh = cv2.threshold(warped, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    test_image(thresh)

    # find contours in the thresholded image, then initialize the
    # digit contours lists
    dilate = cv2.dilate(thresh, None, iterations=4)
    # test_image(dilate)
    cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    digitCnts = []
    # loop over the digit area candidates
    for c in cnts:
        # compute the bounding box of the contour
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(output, (x, y), (x + w, y + h), (255, 0, 0), 1)

        # if the contour is sufficiently large, it must be a digit
        if w >= 60 and (h >= 100 and h <= 200):
            digitCnts.append(c)
            # cv2.putText(output, "({w},{h})".format(w=w, h=h), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
            #             (0, 255, 0), 2)

    test_image(output)

    # sort the contours from left-to-right, then initialize the
    # actual digits themselves
    digitCnts = contours.sort_contours(digitCnts, method="left-to-right")[0]
    digits = []

    if DEBUG:
        img = cv2.cvtColor(thresh, cv2.COLOR_RGBA2RGB)

    # loop over each of the digits
    for c in digitCnts:
        # extract the digit ROI
        (x, y, w, h) = cv2.boundingRect(c)
        roi = thresh[y:y + h, x:x + w]
        # compute the width and height of each of the 7 segments
        # we are going to examine
        (roiH, roiW) = roi.shape
        (dW, dH) = (int(roiW * 0.25), int(roiH * 0.15))
        dHC = int(roiH * 0.15)
        dHS = h // 2  # Height of side segments
        # define the set of 7 segments
        segments = [
            ((0, 0), (w, dH)),  # top
            ((0, 0), (dW, dHS)),  # top-left
            ((w - dW, 0), (dW, h // 2)),  # top-right
            (((w // 4), (dHS) - (dHC // 2)), ((w // 2), dHC)),  # center
            ((0, dHS), (dW, dHS)),  # bottom-left
            ((w - dW, dHS), (dW, dHS)),  # bottom-right
            ((0, h - dH), (w, dH))  # bottom
        ]
        on = [0] * len(segments)  # create array with len(segments length)

        if DEBUG:
            for segment in segments:
                cv2.rectangle(img, (x + segment[0][0], y + segment[0][1]),
                              (x + segment[0][0] + segment[1][0], y + segment[0][1] + segment[1][1]), (0, 0, 255), 1)
            test_image(img)


# ToDo: here is the error

# loop over the segments
        for (i, ((xA, yA), (xB, yB))) in enumerate(segments):
            # extract the segment ROI, count the total number of
            # thresholded pixels in the segment, and then compute
            # the area of the segment
            segROI = roi[yA:yB, xA:xB]
            total = cv2.countNonZero(segROI)
            area = (xB - xA) * (yB - yA)
            print(area)
            # if the total number of non-zero pixels is greater than
            # 50% of the area, mark the segment as "on"
            if area:
                if total / float(area) > 0.1:
                    on[i] = 1
        # lookup the digit and draw it on the image
        digit = DIGITS_LOOKUP[tuple(on)]
        digits.append(digit)
        cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 1)
        cv2.putText(output, str(digit), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

        test_image(output)
    return


@click.command()
@click.option('-v', '--verbosity', help='Verbosity', default=0, count=True)
@click.argument('imagepath')
def cli(verbosity: int, imagepath: str):
    """ main program
    """
    _configure_logging(verbosity)

    get_edges(imagepath)

    return 0


if __name__ == '__main__':
    # pylint: disable=E1120
    sys.exit(cli())
