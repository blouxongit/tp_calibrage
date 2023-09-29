import numpy as np
import cv2
from matplotlib import pyplot as plt
from typing import List, Union, Dict, Tuple
from PIL import Image

from srcs.utils.utils import read_image, get_A_matrix, get_parameters_from_L
from srcs.utils.constants import CHESSBOARD_SIZE, P_POINT_X, P_POINT_Y

SAVE_IMAGES = True


def calibrate():
    image_1 = read_image('../data/mire_1.png')
    image_2 = read_image('../data/mire_2.png')

    _1, chessboard_corners_1 = cv2.findChessboardCorners(image_1, patternSize=CHESSBOARD_SIZE)
    _2, chessboard_corners_2 = cv2.findChessboardCorners(image_2, patternSize=CHESSBOARD_SIZE)
    cv2.drawChessboardCorners(image_1, patternSize=CHESSBOARD_SIZE, corners=chessboard_corners_1, patternWasFound=_1)
    cv2.drawChessboardCorners(image_2, patternSize=CHESSBOARD_SIZE, corners=chessboard_corners_2, patternWasFound=_2)
    if SAVE_IMAGES:
        cv2.imwrite('../data/chessboard_corners_1.png', image_1)
        cv2.imwrite('../data/chessboard_corners_2.png', image_2)
    coord_px_1 = np.flip(np.squeeze(chessboard_corners_1))
    coord_px_2 = np.flip(np.squeeze(chessboard_corners_2))
    coord_px = np.concatenate((coord_px_1, coord_px_2), axis=0)


    coord_mm_y_1, coord_mm_x_1 = np.meshgrid(20 * np.arange(0, CHESSBOARD_SIZE[0]), 20 * np.arange(0, CHESSBOARD_SIZE[1]))
    coord_mm_1 = np.array([np.ravel(coord_mm_x_1), np.ravel(coord_mm_y_1), np.ravel(coord_mm_x_1)]).T

    coord_mm_y_2, coord_mm_x_2 = np.meshgrid(20 * np.arange(0, CHESSBOARD_SIZE[0]), 20 * np.arange(0, CHESSBOARD_SIZE[1]))
    coord_mm_2 = np.array([np.ravel(coord_mm_x_2), np.ravel(coord_mm_y_2), -100*np.ones_like(np.ravel(coord_mm_x_2))]).T

    coord_mm = np.concatenate((coord_mm_1, coord_mm_2), axis=0)

    A = get_A_matrix(coord_px, coord_mm, P_POINT_X, P_POINT_Y)
    U = np.expand_dims(coord_px[:,0] - P_POINT_X, axis=0)
    A_inv = np.linalg.pinv(A)

    L = A_inv@U.T
    params = get_parameters_from_L(np.ravel(L))

    debug = True


if __name__ == "__main__":
    calibrate()
