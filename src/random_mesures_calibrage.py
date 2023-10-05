import random

def multiple_mesures_calibrage_rand(n_mesures):
    import numpy as np
    import cv2
    from matplotlib import pyplot as plt

    images = [cv2.imread("../mire_1.png"), cv2.imread("../mire_2.png")]
    ny = 11
    nx = 8
    coord_px = []
    for image in images:
        ret, corners = cv2.findChessboardCorners(image, (ny, nx))
        corners_fliped = np.fliplr(np.flip(np.squeeze(corners)))
        coord_px.append(corners_fliped)
        cv2.drawChessboardCorners(image, (ny, nx), corners_fliped, ret)

    coord_px = np.concatenate((coord_px[0], coord_px[1]))

    ecart_corners_y = np.linspace(0, 20*nx - 20, nx)
    ecart_corners_x = np.linspace(0, 20*ny - 20, ny)
    ecart_corners_z_picture_1 = np.linspace(0, 0, ny)
    ecart_corners_z_picture_2 = np.linspace(100, 100, ny)

    xv, yv = np.meshgrid(ecart_corners_x, ecart_corners_y)
    zv_1, yv = np.meshgrid(ecart_corners_z_picture_1, ecart_corners_y)
    zv_2, yv = np.meshgrid(ecart_corners_z_picture_2, ecart_corners_y)

    coord_mm = np.stack((xv, yv, zv_1), axis=2)
    coord_mm = np.concatenate((coord_mm, np.stack((xv, yv, zv_2), axis=2)), axis=0)
    coord_mm = coord_mm.reshape(176, 3)

    f = 4

    i2, i1 = images[0].shape[0]/2, images[0].shape[1]/2

    u_1_tilde = coord_px[:,0] - i1
    u_2_tilde = coord_px[:,1] - i2

    lignes = 176
    mesures_index = random.sample(range(lignes), n_mesures)

    A = np.zeros((n_mesures, 7))
    U1 = [u_1_tilde[index] for index in mesures_index]

    for i, val in enumerate(mesures_index):
        A[i] = np.array([u_2_tilde[val]*coord_mm[val,0], u_2_tilde[val]*coord_mm[val,1], u_2_tilde[val]*coord_mm[val,2], u_2_tilde[val], 
                        -u_1_tilde[val]*coord_mm[val,0], -u_1_tilde[val]*coord_mm[val,1], -u_1_tilde[val]*coord_mm[val,2]])

    L = np.linalg.pinv(A.T @ A) @ A.T @ U1

    o_2_c_abs = 1/(np.sqrt(L[4]**2+L[5]**2+L[6]**2))
    o_2_c = - o_2_c_abs
    beta = o_2_c_abs * np.sqrt(L[0]**2+L[1]**2+L[2]**2)
    o_1_c = L[3] * o_2_c / beta
    r11 = L[0] * o_2_c / beta
    r12 = L[1] * o_2_c / beta
    r13 = L[2] * o_2_c / beta
    r21 = L[4] * o_2_c
    r22 = L[5] * o_2_c
    r23 = L[6] * o_2_c

    R1 = np.array([r11, r12, r13])
    R2 = np.array([r21, r22, r23])
    R3 = np.cross(R1, R2)
    r31, r32, r33 = R3[0], R3[1], R3[2]

    phi = - np.arctan(r23/r33)
    gamma = - np.arctan(r12/r11)
    omega = np.arctan(r13/(-r23*np.sin(phi)+r33*np.cos(phi)))

    B = np.zeros((n_mesures, 2))
    R = np.zeros((n_mesures, 1))
    for i, val in enumerate(mesures_index):
        R[i] = -u_2_tilde[val] * (r31*coord_mm[val, 0]+r32*coord_mm[val, 1]+r33*coord_mm[val, 2])
        B[i] = np.array([u_2_tilde[val], -(r21*coord_mm[val, 0] + r22*coord_mm[val, 1] + r23*coord_mm[val, 2] + o_2_c)])

    F = np.squeeze(np.linalg.pinv(B.T @ B) @ B.T @ R)
    o_3_c = F[0]
    f2 = F[1]

    s2 = f/f2
    s1 = s2 / beta
    print(s1)
    print(s2)

    M_int = np.array(
        [[f/s1, 0, i1, 0],
        [0, f/s2, i2, 0],
        [0, 0, 1, 0]]
    )
    M_ext = np.array(
        [[r11, r12, r13, o_1_c],
        [r21, r22, r23, o_2_c],
        [r31, r32, r33, o_3_c],
        [0, 0, 0, 1]]
    )
    M = M_int @ M_ext

    im = plt.imread("../mire_1.png")
    fig, ax = plt.subplots()
    U = np.ndarray((int(lignes/2),1))
    V = np.ndarray((int(lignes/2),1))
    for i in range(int(lignes/2)):
        alpha_U = M @ np.append(coord_mm[i, :], 1)
        alpha = alpha_U[2]
        U[i] = alpha_U[0] / alpha
        V[i] = alpha_U[1] / alpha

    U = np.squeeze(U)
    V = np.squeeze(V)
    plt.plot(U, V, 's')
    plt.imshow(im)
    plt.savefig(f"../test_mesures_calibrage/calibrage_mire1_{n_mesures}_mesures_random.png")

n_mesures = 176
print(f"Nombre de mesures random : {n_mesures}")
multiple_mesures_calibrage_rand(n_mesures)