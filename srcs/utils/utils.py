from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np


def read_image(path_image):
    assert Path(path_image).exists(), "Image not found"
    return cv2.imread(path_image)


def get_A_matrix(u: np.array, x: np.array, pp_x: float, pp_y) -> np.array:
    U = np.array(
        [
            u[:, 1] - pp_y,
            u[:, 1] - pp_y,
            u[:, 1] - pp_y,
            u[:, 1] - pp_y,
            -(u[:, 0] - pp_x),
            -(u[:, 0] - pp_x),
            -(u[:, 0] - pp_x),
        ]
    ).T

    X = np.array(
        [
            x[:, 0],
            x[:, 1],
            x[:, 2],
            np.ones_like(x[:, 0]),
            x[:, 0],
            x[:, 1],
            x[:, 2],
        ]
    ).T

    return np.multiply(U, X)


def get_parameters_from_L(L: np.array) -> Tuple[float, float, float, np.array]:
    o2c = -1 / np.linalg.norm(L[-3:])
    beta = abs(o2c) * np.linalg.norm(L[:3])
    o1c = L[3] * o2c / beta

    r_11 = L[0] * o2c / beta
    r_12 = L[1] * o2c / beta
    r_13 = L[2] * o2c / beta
    r_21 = L[4] * o2c
    r_22 = L[5] * o2c
    r_23 = L[6] * o2c
    r_31, r_32, r_33 = np.cross(np.array([r_21, r_22, r_23]), np.array([r_11, r_12, r_13]))

    rotation_matrix = np.array(
        [
            [r_11, r_12, r_13],
            [r_21, r_22, r_23],
            [r_31, r_32, r_33],
        ]
    )
    return o2c, beta, o1c, rotation_matrix


def get_angles_from_matrix(M: np.array) -> Tuple[float, float, float]:
    yaw = -np.arctan(M[1, 2] / M[2, 2])
    pitch = -np.arctan(M[0, 1] / M[0, 0])
    roll = np.arctan(M[0, 2] / (-M[1, 2] * np.sin(yaw) + M[2, 2] * np.cos(yaw)))
    return yaw, pitch, roll


def get_B_matrix(u: np.array, x: np.array, pp_y: float, r_21: float, r_22: float, r_23: float, o2c: float) -> np.array:
    U = np.array(
        [
            u[:, 1] - pp_y,
        ]
    ).T

    construct = -(r_21 * x[:, 0] + r_22 * x[:, 1] + r_23 * x[:, 2] + o2c)
    construct = np.expand_dims(construct, axis=1)
    test = np.c_[U, construct]
    return np.c_[U, construct]


def get_R_matrix(u: np.array, x: np.array, pp_y: float, r_31: float, r_32: float, r_33: float) -> np.array:
    U = np.array(
        [
            -(u[:, 1] - pp_y),
        ]
    ).T
    construct = r_31 * x[:, 0] + r_32 * x[:, 1] + r_33 * x[:, 2]
    construct = np.expand_dims(construct, axis=1)
    return np.multiply(U, construct)


def get_extrinsics(rotation_matrix: np.array, translation_vector: np.array) -> np.array:
    M = np.eye(4)
    M[:3, :3] = rotation_matrix
    M[:3, -1] = translation_vector
    return M


def get_intrinsics(f: float, s_1: float, s_2: float, i_1: float, i_2: float) -> np.array:
    return np.array(
        [
            [f / s_1, 0, i_1, 0],
            [0, f / s_2, i_2, 0],
            [0, 0, 1, 0],
        ],
        dtype=object,
    )


def get_transformation_matrix(M_intrinsics: np.array, M_extrinsics: np.array) -> np.array:
    return M_intrinsics @ M_extrinsics


def get_world_2_camera_coords(coords_world: np.array, M_transfo: np.array) -> np.array:
    return (M_transfo @ (np.c_[coords_world, np.ones((np.shape(coords_world)[0], 1))]).T).T


def get_camera_2_image_coords(camera_coords: np.array) -> np.array:
    alpha_matrix = np.c_[camera_coords[:, -1], camera_coords[:, -1], camera_coords[:, -1]]
    return np.divide(camera_coords, alpha_matrix)


def get_coords_from_world_2_image(coords_mm: np.array, M_transfo: np.array) -> np.array:
    world_2_camera_coords = get_world_2_camera_coords(coords_mm, M_transfo)
    return get_camera_2_image_coords(world_2_camera_coords)
