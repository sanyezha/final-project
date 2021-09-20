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


def makedir(path):
    # remove ' '
    path = path.strip()
    # remove '\'
    path = path.rstrip("\\")
    isExists = os.path.exists(path)
    print(isExists)
    if not isExists:
        os.makedirs(path)
        print(1)
    else:
        print(2)


def count(dir):
    return int(len(os.listdir(dir))/2)