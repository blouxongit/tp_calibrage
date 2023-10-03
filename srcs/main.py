# TP Calibrage module du même nom - Raphaël COEZ

# Imports
import cv2
import numpy as np
from matplotlib import pyplot as plt


# Parameters
img_list = ['./data/mire_1.png', './data/mire_2.png']
Nx = 8
Ny = 11
box_size = 20
i1 = 480/2
i2 = 640/2
val = [0,-100]


# main fct
def main() :
    # Containers
    list_coord_px = []
    list_coord_mm = []

    # Construction de list_coords
    for i in range(len(img_list)):
        img = open_img(img_list[i])
        corners_pos = detection(img, Nx, Ny)
        draw_corners(img, corners_pos, Nx, Ny)
        list_coord_px.append(np.flip(corners_pos))
        list_coord_mm.append(build_coord_mm(Nx,Ny,box_size, val[i]))
    coord_px = np.concatenate(list_coord_px)
    coord_mm = np.concatenate(list_coord_mm)
    A = np.zeros([2*Ny*Nx,7])

    # Matrice A
    for i in range(2*Ny*Nx):
        for j in range(7):
            A[i,j] = get_formula(i,j,coord_px, coord_mm)
    
    # Matrice U1
    U1 = np.zeros([176,1])
    for i in range(Nx*Ny*2):
        U1[i,0] = coord_px[i,0] - i1

    # Transposer matrice A
    A_t = np.transpose(A)

    # Calcul de L
    L = np.linalg.inv(A_t @ A) @ A_t @ U1 
    L = np.linalg.pinv(A)@U1
    
    # Déduction de la valeur des paramètres
    abs_oc2 = 1/np.sqrt(L[4]**2 + L[5]**2 + L[6]**2)
    oc2 = - abs_oc2
    beta = abs_oc2 * np.sqrt(L[0]**2 + L[1]**2 + L[2]**2)
    oc1 = L[3] * oc2 / beta
    r11 = L[0] * oc2 / beta
    r12 = L[1] * oc2 / beta
    r13 = L[2] * oc2 / beta
    r21 = L[4] * oc2
    r22 = L[5] * oc2
    r23 = L[6] * oc2
    R1 = np.array([r11,r12,r13])
    R2 = np.array([r21,r22,r23])
    #R3 = np.cross(R1,R2)
    print(oc2)

# Open img
def open_img(path_img) :
    img = cv2.imread(path_img)
    return img

# Display img
def display_img(img) :
    cv2.imshow("Test",img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Construction de coord_mm 
def build_coord_mm(Nx, Ny, box_size, val) :
    coord_mm = np.zeros([Nx*Ny,3])
    x = 0
    for i in range(Nx*Ny) :
        coord_mm[i,1] = (x) * box_size
        coord_mm[i,0] = (i//11) * box_size
        coord_mm[i,2] = val
        x += 1 
        if x >= 11 :
            x = 0
    return coord_mm
    
# Draw corners
def draw_corners(img, img_corners, Nx, Ny) :
    cv2.drawChessboardCorners(img,patternSize = [Ny,Nx], corners = img_corners, patternWasFound=True)

# Return the positions of the chessboard corners
def detection(img, Nx, Ny) :
    ret = np.squeeze(cv2.findChessboardCorners(img,patternSize=[Ny,Nx])[1])
    return ret
    
# Get Formula
def get_formula(i, j, coord_px, coord_mm) :
    formula = {0 : (coord_px[i,1]-i2)*coord_mm[i,0], 1 : (coord_px[i,1]-i2)*coord_mm[i,1], 
               2 : (coord_px[i,1]-i2)*coord_mm[i,2], 3 : (coord_px[i,1]-i2), 
               4 : -(coord_px[i,0]-i1)*coord_mm[i,0], 5 : -(coord_px[i,0]-i1)*coord_mm[i,1], 
               6 : -(coord_px[i,0]-i1)*coord_mm[i,2]}
    return formula[j]

main()