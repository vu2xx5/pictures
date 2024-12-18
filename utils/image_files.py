import cv2
import numpy as np
def open_image(upload_image):
    '''
        This function opens an image from a file upload

        Args:
            upload_image: image file to be opened
        returns:
            image: OpenCV image object    
    '''
    file_bytes = np.asarray(bytearray(upload_image.read()), dtype=np.uint8)
    cv2_image = cv2.imdecode(file_bytes, 1)
    return cv2_image

def adjust_canvas_size(image):
    '''
        This function adjusts the size of the canvas to fit the image

        Args:
            image
        returns:
            new_width, new_height
    '''
    max_canvas_width = 700  
    img_width, img_height = image.size
    aspect_ratio = img_height / img_width

    if img_width > max_canvas_width:
        new_width = max_canvas_width
        new_height = int(new_width * aspect_ratio)
    else:
        new_width, new_height = img_width, img_height

    return new_width, new_height
