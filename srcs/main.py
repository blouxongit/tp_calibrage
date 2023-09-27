# TP Calibrage module du même nom // Raphaël COEZ

# Imports
import cv2
import numpy as np
from matplotlib import pyplot as plt

# main fct
def main() :
    # Parameters
    img_list = ['./data/mire_1.png', './data/mire_2.png']
    Nx = 8
    Ny = 11
    box_size = 20
    for i in img_list :
        img = open_img(i)
        corners_pos = detection(img, Nx, Ny)
        draw_corners(img, corners_pos, Nx, Ny)
        coord_px = np.flip(corners_pos)
        coord_mm = np.empty(shape = [np.shape(coord_px)[0],3])
        for x in range(np.shape(coord_mm)[0]):
            coord_mm[x][0] = box_size * x
        print(coord_mm)
        display_img(img)


# Open img
def open_img(path_img) :
    img = cv2.imread(path_img)
    return img

# Display img
def display_img(img) :
    cv2.imshow("Test",img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Draw corners
def draw_corners(img, img_corners, Nx, Ny) :
    cv2.drawChessboardCorners(img,patternSize = [Ny,Nx], corners = img_corners, patternWasFound=True)

# Return the positions of the chessboard corners
def detection(img, Nx, Ny) :
    ret = np.squeeze(cv2.findChessboardCorners(img,patternSize=[Ny,Nx])[1])
    return ret
    

main()