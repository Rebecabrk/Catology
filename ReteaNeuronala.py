import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

class Perceptron:
    def __init__(self, df):
        y = df['Race']
        X = df.drop(columns='Race')
        self.train_x, self.test_x, self.train_y, self.test_y = train_test_split(X, y, test_size=0.2, random_state=42)
        self.train_x, self.test_x, self.train_y, self.test_y = np.array(self.train_x), np.array(self.test_x), np.array(self.train_y), np.array(self.test_y)
        self.__one_hot_encode__()

    def __one_hot_encode__(self):
        encoder = OneHotEncoder(sparse_output=False)
        self.train_y = encoder.fit_transform(self.train_y.reshape(-1, 1))
        self.test_y = encoder.fit_transform(self.test_y.reshape(-1, 1))

    def __weights_init_Xavier_Uniform__(self):
        low_bound = -math.sqrt(6 / (self.train_x.shape[1] + 10)) # 10 neuroni pe stratul ascuns
        upper_bound = math.sqrt(6 / (self.train_x.shape[1] + 10))

        self.W1 = np.random.uniform(low=low_bound, high=upper_bound, size=(self.train_x.shape[1], 10))
        self.b1 = np.zeros((1, 10))

        low_bound = -math.sqrt(6 / (10 + self.train_y.shape[1])) # 10 neuroni pe stratul ascuns
        upper_bound = math.sqrt(6 / (10 + self.train_y.shape[1]))

        self.W2 = np.random.uniform(low=low_bound, high=upper_bound, size=(10, self.train_y.shape[1]))  
        self.b2 = np.zeros((1, self.train_y.shape[1]))

    def antreneaza(self, rata_de_invatare=0.01, epoci=1000, batch_size=100):
        self.__weights_init_Xavier_Uniform__()

p = Perceptron(pd.read_excel('cat_data_preprocesat.xlsx'))
print(p.train_y[1])