import cv2
import numpy as np
from matplotlib import pyplot as plt

from srcs.utils.constants import CHESSBOARD_SIZE, FOCAL_LENGTH, P_POINT_X, P_POINT_Y, SAVE_IMAGES
from srcs.utils.utils import (
    get_A_matrix,
    get_angles_from_matrix,
    get_B_matrix,
    get_coords_from_world_2_image,
    get_extrinsics,
    get_intrinsics,
    get_parameters_from_L,
    get_R_matrix,
    get_transformation_matrix,
    read_image,
)


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

    coord_px_1 = np.fliplr(np.flip(np.squeeze(chessboard_corners_1)))
    coord_px_2 = np.fliplr(np.flip(np.squeeze(chessboard_corners_2)))
    coord_px = np.concatenate((coord_px_1, coord_px_2), axis=0)

    coord_mm_y_1, coord_mm_x_1 = np.meshgrid(
        20 * np.arange(0, CHESSBOARD_SIZE[0]), 20 * np.arange(0, CHESSBOARD_SIZE[1])
    )
    coord_mm_1 = np.array([np.ravel(coord_mm_x_1), np.ravel(coord_mm_y_1), np.zeros_like(np.ravel(coord_mm_x_1))]).T
    coord_mm_1 = np.c_[coord_mm_1[:, 1], coord_mm_1[:, 0], coord_mm_1[:, 2]]

    coord_mm_y_2, coord_mm_x_2 = np.meshgrid(
        20 * np.arange(0, CHESSBOARD_SIZE[0]), 20 * np.arange(0, CHESSBOARD_SIZE[1])
    )
    coord_mm_2 = np.array(
        [np.ravel(coord_mm_x_2), np.ravel(coord_mm_y_2), 100 * np.ones_like(np.ravel(coord_mm_x_2))]
    ).T
    coord_mm_2 = np.c_[coord_mm_2[:, 1], coord_mm_2[:, 0], coord_mm_2[:, 2]]

    coord_mm = np.concatenate((coord_mm_1, coord_mm_2), axis=0)

    A = get_A_matrix(coord_px, coord_mm, P_POINT_X, P_POINT_Y)
    U = np.expand_dims(coord_px[:, 0] - P_POINT_X, axis=0)
    A_pseudoinv = np.linalg.pinv(A)

    L = A_pseudoinv @ U.T
    o2c, beta, o1c, rotation_matrix = get_parameters_from_L(np.ravel(L))
    yaw, pitch, roll = np.degrees(get_angles_from_matrix(rotation_matrix))

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
    M_ext = get_extrinsics(rotation_matrix=rotation_matrix, translation_vector=np.array([o1c, o2c, o3c], dtype=object))
    M_transfo = get_transformation_matrix(M_intrinsics=M_int, M_extrinsics=M_ext)

    coords_to_check = get_coords_from_world_2_image(coord_mm_2, M_transfo)
    X_im = coords_to_check[:, 0]
    Y_im = coords_to_check[:, 1]

    coord_px_true = coord_px[88:, :]
    coord_px_tested = np.c_[X_im, Y_im]

    MSE = np.mean(np.sum((coord_px_tested - coord_px_true) ** 2, axis=1))

    im = plt.imread("../data/mire_2.png")
    plt.plot(X_im, Y_im, "r.", markersize=3)
    plt.plot(coord_px_true[:,0], coord_px_true[:,1], "b.", markersize=3)
    plt.imshow(im)
    plt.show()


if __name__ == "__main__":
    calibrate()
