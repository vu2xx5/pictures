import cv2
import numpy as np
def open_image(upload_image):
    file_bytes = np.asarray(bytearray(upload_image.read()), dtype=np.uint8)
    cv2_image = cv2.imdecode(file_bytes, 1)
    return cv2_image