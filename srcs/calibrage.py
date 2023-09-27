import numpy as np
import cv2
from matplotlib import pyplot as plt
from typing import List, Union, Dict, Tuple
from PIL import Image

from srcs.utils.utils import read_image
from srcs.utils.constants import CHESSBOARD_SIZE


SAVE_IMAGES = False

def calibrate():
    image = read_image('../data/mire_2.png')

    _, chessboard_corners = cv2.findChessboardCorners(image, patternSize=CHESSBOARD_SIZE)
    cv2.drawChessboardCorners(image, patternSize=CHESSBOARD_SIZE, corners=chessboard_corners, patternWasFound=_)
    if SAVE_IMAGES:
        cv2.imwrite('../data/chessboard_corners.png', image)
    coord_px = np.flip(np.squeeze(chessboard_corners))
    coord_mm_x, coord_mm_y = np.meshgrid(20*np.arange(0, CHESSBOARD_SIZE[0]), 20*np.arange(0, CHESSBOARD_SIZE[1]))
    coord_mm_x = np.ravel(coord_mm_x)
    a=1
if __name__ == "__main__":
    calibrate()
