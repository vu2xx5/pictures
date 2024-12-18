import numpy as np
import cv2
import streamlit as st

class Edit:
    def __init__(self, image: np.ndarray) -> np.ndarray:
        self.image = image

    def rotate_image(self, angle = int) -> np.ndarray:
        '''
            This function rotates the image
            
            Args:
                angle: angle of rotation in degrees

            Retrurn:
                image is rotated
        ''' 
        if angle == 90:
            return cv2.rotate(self.image, cv2.ROTATE_90_CLOCKWISE)
        elif angle == 180:
            return cv2.rotate(self.image, cv2.ROTATE_180)
        elif angle == 270:
            return cv2.rotate(self.image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        else:
            return self.image

    def brightness_image(self, brightness: float) -> np.ndarray:
        '''
            This function brightness the image

            Args:
                brightness: brightness of image
            Returns:
                image is brightened
        '''
        brightness_img = cv2.convertScaleAbs(self.image, alpha=brightness, beta=0)
        return brightness_img 

    def blur_image(self, ksize: int = 5) -> np.ndarray:
        '''
            This function blurs the image

            Args:
                ksize: kernel size for blurring
            Returns:
                image is blurred
        '''
        blurred_img = cv2.GaussianBlur(self.image, (ksize, ksize), 0)
        return blurred_img
