from pathlib import Path
import cv2
import numpy as np


def read_image(path_image):
    assert Path(path_image).exists(), "Image not found"
    return cv2.imread(path_image)


def get_A_matrix(u: np.array, x: np.array, pp_x: float, pp_y) -> np.array:
    U = np.array([
        u[:, 1] - pp_y,
        u[:, 1] - pp_y,
        u[:, 1] - pp_y,
        u[:, 1] - pp_y,
        -(u[:, 0] - pp_x),
        -(u[:, 0] - pp_x),
        -(u[:, 0] - pp_x),
    ]).T

    X = np.array([
        x[:, 0],
        x[:, 1],
        x[:, 2],
        np.ones_like(x[:, 0]),
        x[:, 0],
        x[:, 1],
        x[:, 2],
    ]).T

    A = np.multiply(U, X)
    return A


def get_parameters_from_L(L):

    u = 1000*L
    o2c_abs = 1/np.linalg.norm(L[-3:])
    beta = o2c_abs*np.linalg.norm(L[:3])
    o1c = L[3]*o2c_abs/beta
    r_11 = L[0]*o2c_abs/beta
    r_12 = L[1]*o2c_abs/beta
    r_13 = L[2]*o2c_abs/beta
    r_21 = L[4]*o2c_abs
    r_22 = L[5]*o2c_abs
    r_23 = L[6]*o2c_abs

    debug = True