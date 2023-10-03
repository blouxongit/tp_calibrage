import numpy as np
import cv2
from matplotlib import pyplot as plt
from typing import List, Union, Dict, Tuple
from PIL import Image

from srcs.utils.utils import (
    read_image,
    get_A_matrix,
    get_parameters_from_L,
    get_angles_from_matrix,
    get_B_matrix,
    get_R_matrix,
    get_intrinsics,
    get_extrinsics, get_transformation_matrix,
)
from srcs.utils.constants import CHESSBOARD_SIZE, P_POINT_X, P_POINT_Y, FOCAL_LENGTH

SAVE_IMAGES = False


def calibrate():
    image_1 = read_image("../data/mire_1.png")
    image_2 = read_image("../data/mire_2.png")

    _1, chessboard_corners_1 = cv2.findChessboardCorners(image_1, patternSize=CHESSBOARD_SIZE)
    _2, chessboard_corners_2 = cv2.findChessboardCorners(image_2, patternSize=CHESSBOARD_SIZE)
    cv2.drawChessboardCorners(image_1, patternSize=CHESSBOARD_SIZE, corners=chessboard_corners_1, patternWasFound=_1)
    cv2.drawChessboardCorners(image_2, patternSize=CHESSBOARD_SIZE, corners=chessboard_corners_2, patternWasFound=_2)
    if SAVE_IMAGES:
        cv2.imwrite("../data/chessboard_corners_1.png", image_1)
        cv2.imwrite("../data/chessboard_corners_2.png", image_2)

    coord_px_1 = np.flip(np.squeeze(chessboard_corners_1))
    coord_px_2 = np.flip(np.squeeze(chessboard_corners_2))
    coord_px = np.concatenate((coord_px_1, coord_px_2), axis=0)

    coord_mm_y_1, coord_mm_x_1 = np.meshgrid(
        20 * np.arange(0, CHESSBOARD_SIZE[0]), 20 * np.arange(0, CHESSBOARD_SIZE[1])
    )
    coord_mm_1 = np.array([np.ravel(coord_mm_x_1), np.ravel(coord_mm_y_1), np.zeros_like(np.ravel(coord_mm_x_1))]).T

    coord_mm_y_2, coord_mm_x_2 = np.meshgrid(
        20 * np.arange(0, CHESSBOARD_SIZE[0]), 20 * np.arange(0, CHESSBOARD_SIZE[1])
    )
    coord_mm_2 = np.array(
        [np.ravel(coord_mm_x_2), np.ravel(coord_mm_y_2), -100 * np.ones_like(np.ravel(coord_mm_x_2))]
    ).T

    coord_mm = np.concatenate((coord_mm_1, coord_mm_2), axis=0)

    A = get_A_matrix(coord_px, coord_mm, P_POINT_X, P_POINT_Y)
    U = np.expand_dims(coord_px[:, 0] - P_POINT_X, axis=0)
    A_pseudoinv = np.linalg.pinv(A)

    L = A_pseudoinv @ U.T
    o2c, beta, o1c, rotation_matrix = get_parameters_from_L(np.ravel(L))
    yaw, pitch, roll = get_angles_from_matrix(rotation_matrix)

    B = get_B_matrix(
        coord_px,
        coord_mm,
        P_POINT_Y,
        r_21=rotation_matrix[1, 0],
        r_22=rotation_matrix[1, 1],
        r_23=rotation_matrix[1, 2],
        o2c=o2c,
    )

    R = get_R_matrix(
        coord_px,
        coord_mm,
        P_POINT_Y,
        r_31=rotation_matrix[2, 0],
        r_32=rotation_matrix[2, 1],
        r_33=rotation_matrix[2, 2],
    )

    o3c, f_2 = np.linalg.pinv(B) @ R
    s_2 = FOCAL_LENGTH / f_2
    s_1 = s_2 / beta

    M_int = get_intrinsics(FOCAL_LENGTH, s_1, s_2, P_POINT_X, P_POINT_Y)
    M_ext = get_extrinsics(rotation_matrix=rotation_matrix, translation_vector=np.array([o1c, o2c, o3c]))
    M = get_transformation_matrix(M_intrinsics=M_int, M_extrinsics=M_ext)

    coords_to_check_mm = np.c_[coord_mm[:88, :], np.ones((88,1))]
    coords = (M@coords_to_check_mm.T).T

    for i in range(88):
        coords[i] = coords[i]/coords[i,-1]

    X = coords[:,0]
    Y = coords[:,1]

    im = plt.imread("../data/mire_1.png")
    plt.plot(X,Y, 's')
    plt.imshow(im)
    plt.show()
    debug = True


if __name__ == "__main__":
    calibrate()
