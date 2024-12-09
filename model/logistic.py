from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import os
import numpy as np
from HOG import GenderClassifier

train_dir = os.path.join("archive", "Training")
val_dir = os.path.join("archive", "Validation")
label_map = {"male": 0, "female": 1}
gender_classifier = GenderClassifier()


class LogisticRegressionModel:
    def __init__(self, max_iter = 200):
        self.max_iter = max_iter
        self.model = LogisticRegression(max_iter=self.max_iter)
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None

    def load_data(self, directory, labal_map):
        X = []
        y = []
        for label_name, 