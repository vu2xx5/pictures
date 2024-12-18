import os
import numpy as np


class PredictModel:
    def __init__(self, weights = None, bias: np.ndarray =  None):
        self.weights = weights
        self.bias = bias
        
    def load_weights_bias(self, weights_dir: str) -> np.ndarray:
        '''
            This function loads the weights and bias from weights_dir

            Args:
                weights_dir: directory path of weights and bias

            Returns:
                weigths of the weights
                bias of the weights        
        '''
        weights_path = os.path.join(weights_dir, "weights.txt")
        bias_path = os.path.join(weights_dir, "bias.txt")

        with open(weights_path, "r") as f:
            data = f.read().strip()
            self.weights = np.array([float(w) for w in data.split()])

        with open(bias_path, "r") as f:
            self.bias = float(f.read().strip())

    def predict(self, features: np.ndarray) -> int:
        '''
            This function is to predict gender of an image

            Args:
                features: HOG features of an image

            Returns:
                probability: prediction probability
        ''' 
        probability = 1 / (1 + np.exp(-(np.dot(features, self.weights) + self.bias)))
        return 1 if probability >= 0.5 else 0
    
    def pre_probality(self, features: np.ndarray) -> int:
        '''
            This function is return probability

            Args:
                features: HOG features of an image

            Returns:
                probability
        ''' 
        probability = 1 / (1 + np.exp(-(np.dot(features, self.weights) + self.bias)))
        probability * 100
        return probability 

#     def test(self, image_path: str, weights_dir: str) -> int:
#         '''
#             This function is to test a image

#             Args:
#                 image_path: image path
            
#             Returns:
#                 Results about classification image is male or female
#         '''
#         gender = GenderClassifier()
#         features = gender.extract_hog_features(gender.prepocess_image(image_path))
#         self.load_weights_bias(weights_dir=weights_dir)
#         result = self.predict(features)

#         return result


# if __name__ == "__main__":
# #     dir_path = "archive/Training"
# #     weights_dir = "weights" 
# #     model = PredictModel()
# #     female_path = os.path.join(dir_path, "female")    
# #     male_path = os.path.join(dir_path, "male")  
# #     y_pred_female = []
# #     y_pred_male = []

# #     for image_name in os.listdir(female_path):
# #         image_path = os.path.join(female_path, image_name)
# #         y_pred_female.append(model.test(image_path, weights_dir))
# #     y_pred_female = np.array(y_pred_female)

# #     for image_name in os.listdir(male_path):
# #         image_path = os.path.join(male_path, image_name)
# #         y_pred_male.append(model.test(image_path, weights_dir))
# #     y_pred_male = np.array(y_pred_male)

# #     y_trues_female = np.ones(y_pred_female.shape)
# #     y_trues_male = np.zeros(y_pred_male.shape)
# #     print(accuracy_score(y_trues_female, y_pred_female))
# #     print(accuracy_score(y_trues_male, y_pred_male))

#     image_path = "archive/Validation/female/112950.jpg.jpg"
#     gender = GenderClassifier()
#     features = gender.extract_hog_features(gender.prepocess_image(image_path))
#     pred = PredictModel()
#     pred.load_weights_bias("weights")
#     result = pred.predict(features)
#     print(result)
