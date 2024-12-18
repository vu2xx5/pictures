import cv2
import numpy as np
from skimage.feature import hog

class GenderClassifier:
    def __init__(self, target_size=(128, 128)):
        self.target_size = target_size

    def prepocess_image(self, image):
        '''
            This function is used to resize and gray the image

            Args:
                image
            Returns:
                resized image and gray image
        '''
        if isinstance(image, np.ndarray):
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            resized_image = cv2.resize(gray_image, (128, 128))  
            return resized_image
        else:
            raise ValueError("Input image must be a NumPy array.")

    def extract_hog_features(self, image):
        '''
            This function extracts HOG features from the image

            Args:
                image
            Return: 
                features of image
        '''
        features, hog_image = hog(
            image,
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            block_norm="L2-Hys",
            visualize=True,
        )
        return features