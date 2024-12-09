import numpy as np
import cv2
import streamlit as st

class Edit:
    def __init__(self, image: np.ndarray):
        self.image = image


    def rotate_image(self, angle = int) -> np.ndarray:
        # Xoay ảnh 
        if angle == 90:
            return cv2.rotate(self.image, cv2.ROTATE_90_CLOCKWISE)
        elif angle == 180:
            return cv2.rotate(self.image, cv2.ROTATE_180)
        elif angle == 270:
            return cv2.rotate(self.image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        else:
            return self.image


    def brightness_image(self, brightness: float) -> np.ndarray:
        # Điều chỉnh độ sáng của ảnh
        brightness_img = cv2.convertScaleAbs(self.image, alpha=brightness, beta=0)
        return brightness_img
    

    def blur_image(self, ksize: int = 5) -> np.ndarray:
        # Làm mờ ảnh bằng bộ lọc trung vị
        blurred_img = cv2.medianBlur(self.image, ksize)
        return blurred_img
