import cv2
import datetime
import numpy as np
import os
import json
import torch
from PIL import Image
from PIL import ImageDraw , ImageFont
from torchvision import transforms
import matplotlib.pyplot as plt
import matplotlib


def mask_extraction(frame1, kernel_size, iteration):
    h, l = frame1.shape[:2]
    frame2 = cv2.resize(frame1, (int(l / 3), int(h / 3)))
    frame2 = cv2.GaussianBlur(frame2, (15, 15), 0)

    l_g = np.array([55, 80, 0])
    u_g = np.array([75, 255, 255])
    # it is important to add gaussian filter first to reduce the image
    hsv = cv2.cvtColor(frame2, cv2.COLOR_BGR2HSV)
    mask_res = cv2.inRange(hsv, l_g, u_g)
    mask = cv2.bitwise_not(mask_res)
    res = cv2.bitwise_and(frame2, frame2, mask=mask)
    # adjust value of Canny here
    can = cv2.Canny(res, 20, 0)
    k_s = kernel_size
    it = iteration
    half_padding = int(2 * it * (k_s - 1) / 2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_s, k_s))
    inter = np.zeros((can.shape[0] + 2 * half_padding, can.shape[1] + 2 * half_padding), dtype=np.uint8)
    inter[half_padding:can.shape[0] + half_padding, half_padding:can.shape[1] + half_padding] = can
    dilated = cv2.dilate(inter, kernel, iterations=it)
    rdy_contour = dilated[half_padding:can.shape[0] + half_padding, half_padding:can.shape[1] + half_padding]

    # create the mask
    contours, hierarchy = cv2.findContours(rdy_contour, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    length = len(contours)
    hier = np.array(hierarchy)
    mask = []
    location = []
    ycord = []
    xcord = []
    for i in range(length):
        if hier[0, i, 3] == 0:
            arc = cv2.arcLength(contours[i], True)
            epsilon = max(3, int(arc * 0.02))
            approx = cv2.approxPolyDP(contours[i], epsilon, True)
            app = np.array(approx)
            if app.shape[0] != 4:
                i = i + 1
            else:
                submask = np.zeros(can.shape, dtype=np.uint8)
                cv2.drawContours(submask, contours, i, 120, -1)

                # use four points to draw a rectangle, and fill it with white color to generate a mask
                x_cord = int((app[0, 0, 0] + app[1, 0, 0] + app[2, 0, 0] + app[3, 0, 0]) / 4)
                y_cord = int((app[0, 0, 1] + app[1, 0, 1] + app[2, 0, 1] + app[3, 0, 1]) / 4)
                ycord.append(y_cord)
                xcord.append(x_cord)

                rectangle = np.array([[app[0, 0, 0], app[0, 0, 1]], [app[1, 0, 0], app[1, 0, 1]],
                                                                     [app[2, 0, 0], app[2, 0, 1]], [app[3, 0, 0], app[3, 0, 1]]])
                cv2.fillConvexPoly(submask, rectangle, (255, 255, 255))

                mask.append(submask)
                location.append(rectangle)

    return mask, location, ycord, xcord