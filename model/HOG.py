import cv2
from skimage.feature import hog

class GenderClassifier:

    def __init__(self, target_size=(128, 128)):
        '''
        Khởi tạo kích thước của ảnh
        '''
        self.target_size = target_size


    def prepocess_image(self, image_path):
        '''
        Xử lý ảnh đầu vào, chuyển về kích thước tương ứng và chuyển sang ảnh xám
        '''
        image = cv2.imread(image_path)
        image_resized = cv2.resize(image, self.target_size)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return gray_image
    

    def extract_hog_features(self, image):
        '''
        Trích xuất dặc trưng từ ảnh'''
        features, hog_image = hog(
            image,
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            block_norm="L2-Hys",
            visualize=True,
        )
        return features