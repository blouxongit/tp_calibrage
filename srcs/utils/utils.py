from pathlib import Path
import cv2

def read_image(path_image):

    assert Path(path_image).exists(), "Image not found"
    return cv2.imread(path_image)

