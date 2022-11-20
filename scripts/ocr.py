#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import sys

import click

import cv2
import numpy as np
import imutils
import pytesseract

log = logging.getLogger(__file__)

options = "outputbase digits"


def _configure_logging(verbosity):
    loglevel = max(3 - verbosity, 0) * 10
    logging.basicConfig(level=loglevel, format='[%(asctime)s] %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    if loglevel >= logging.DEBUG:
        # Disable debugging logging for external libraries
        for loggername in 'urllib3', 'google.auth.transport.requests':
            logging.getLogger(loggername).setLevel(logging.CRITICAL)


def test_image(img):
    log.debug(pytesseract.image_to_string(img, config=options))
    cv2.imshow('', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def ocr(imagepath: str):
    # Webcamera no 0 is used to capture the frames
    #cap = cv2.VideoCapture(0)

    # This drives the program into an infinite loop.
    # Captures the live stream frame-by-frame
    #, frame = cap.read()

    frame = cv2.imread(imagepath, 0)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
    # Converts images from BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_red = np.array([0, 0, 10])
    upper_red = np.array([0, 0, 85])

    # Here we are defining range of bluecolor in HSV
    # This creates a mask of white coloured
    # objects found in the frame.
    mask = cv2.inRange(hsv, lower_red, upper_red)

    # The bitwise and of the frame and mask is done so
    # that only the blue coloured objects are highlighted
    # and stored in res
    res = cv2.bitwise_and(frame, frame, mask=mask)

    # Blur the image
    res = cv2.GaussianBlur(res, (13, 13), 0)
    test_image(res)
    # Edge detection
    #edged = cv2.Canny(res, 100, 200)
    edged = res
    # Dilate it , number of iterations will depend on the image
    dilate = cv2.dilate(edged, None, iterations=4)
    test_image(dilate)
    # perform erosion
    erode = cv2.erode(dilate, None, iterations=4)
    test_image(erode)

    # make an empty mask
    mask2 = np.ones(frame.shape[:2], dtype="uint8") * 255

    # find contours
    cnts = cv2.findContours(erode.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # cnts = cnts[0] if imutils.is_cv2() else cnts[1]

    # orig = frame.copy()
    # for c in cnts:
    #     # if the contour is not sufficiently large, ignore it
    #     if cv2.contourArea(c) < 600:
    #         cv2.drawContours(mask2, [c], -1, 0, -1)
    #         continue

    # # Remove ignored contours
    # newimage = cv2.bitwise_and(erode.copy(), dilate.copy(), mask=mask2)
    # # Again perform dilation and erosion
    # newimage = cv2.dilate(newimage, None, iterations=7)
    # newimage = cv2.erode(newimage, None, iterations=5)
    # ret, newimage = cv2.threshold(newimage, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    # Tesseract OCR
    text = pytesseract.image_to_string(erode)

    # release the captured frame
    #cap.release()


@click.command()
@click.option('-v', '--verbosity', help='Verbosity', default=0, count=True)
@click.argument('imagepath')
def cli(verbosity: int, imagepath: str):
    """ main program
    """
    _configure_logging(verbosity)

    ocr(imagepath)

    return 0


if __name__ == '__main__':
    # pylint: disable=E1120
    sys.exit(cli())
