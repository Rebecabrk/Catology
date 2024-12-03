import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from scipy.stats import bernoulli

class Perceptron:
    """
    Clasa pentru antrenarea si testarea unei retele neuronale cu un singur strat ascuns

    Date membru:
    - train_x: datele de antrenare
    - test_x: datele de testare
    - train_y: etichetele de antrenare
    - test_y: etichetele de testare
    - W1, b1: ponderile si bias-urile pentru stratul ascuns
    - W2, b2: ponderile si bias-urile pentru stratul de iesire
    """
    def __init__(self, df):
        y = df['Race']
        X = df.drop(columns='Race')
        self.train_x, self.test_x, self.train_y, self.test_y = train_test_split(X, y, test_size=0.1, random_state=42)
        self.train_x, self.test_x, self.train_y, self.test_y = np.array(self.train_x), np.array(self.test_x), np.array(self.train_y), np.array(self.test_y)
        self.__one_hot_encode__()

    def __one_hot_encode__(self):
        encoder = OneHotEncoder(sparse_output=False)
        self.train_y = encoder.fit_transform(self.train_y.reshape(-1, 1))
        # self.test_y = encoder.fit_transform(self.test_y.reshape(-1, 1))

    def __weights_init_Xavier_Uniform__(self, nr_neuroni_strat_ascuns):
        low_bound = -math.sqrt(6 / (self.train_x.shape[1] + nr_neuroni_strat_ascuns)) 
        upper_bound = math.sqrt(6 / (self.train_x.shape[1] + nr_neuroni_strat_ascuns))

        self.W1 = np.random.uniform(low=low_bound, high=upper_bound, size=(self.train_x.shape[1], nr_neuroni_strat_ascuns))
        self.b1 = np.zeros((1, nr_neuroni_strat_ascuns))

        low_bound = -math.sqrt(6 / (nr_neuroni_strat_ascuns + self.train_y.shape[1])) 
        upper_bound = math.sqrt(6 / (nr_neuroni_strat_ascuns + self.train_y.shape[1]))

        self.W2 = np.random.uniform(low=low_bound, high=upper_bound, size=(nr_neuroni_strat_ascuns, self.train_y.shape[1]))  
        self.b2 = np.zeros((1, self.train_y.shape[1]))

    def __sigmoid__(self, Z):
        return 1/(1 + np.exp(-Z))

    def __softmax__(self, Z):
        expZ = np.exp(Z - np.max(Z, axis=1, keepdims=True))
        return expZ / np.sum(expZ, axis=1, keepdims=True)
    
    def __forward_pass__(self, X, W1, b1, W2, b2, dropout_rate=0):
        Z1 = np.dot(X, W1) + b1
        A1 = self.__sigmoid__(Z1) # functia de activare pentru stratul ascuns

        dropouts = bernoulli.rvs(1 - dropout_rate, size=A1.shape)
        A1 = A1 * dropouts / (1 - dropout_rate)

        Z2 = np.dot(A1, W2) + b2
        A2 = self.__softmax__(Z2) # functia de activare pentru stratul de iesire

        return A1, A2, dropouts
    
    def __backward_pass__(self, X, y, A1, A2, W2, dropouts=None, dropout_rate=0):
        m = y.shape[0]

        dZ2 = A2 - y  # Derivative of softmax + cross-entropy
        dW2 = np.dot(A1.T, dZ2) / m  # Gradient w.r.t. weights (output layer)
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m  # Gradient w.r.t. biases (output layer)
    
        # Hidden layer error term (dZ1)
        dZ1 = np.dot(dZ2, W2.T) * A1 * (1 - A1)  # For sigmoid activation (derivative of sigmoid)

        dZ1 = dZ1 * dropouts
        dZ1 /= (1 - dropout_rate)

        dW1 = np.dot(X.T, dZ1) / m  # Gradient w.r.t. weights (hidden layer)
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m  # Gradient w.r.t. biases (hidden layer)
    
        return dW1, db1, dW2, db2

    
    def __compute_loss_cross_entropy__(self, y, A2):
        m = y.shape[0]
        log_likelihood = -np.log(A2[range(m), y.argmax(axis=1)])
        loss = np.sum(log_likelihood) / m
        return loss
    
    def __update_weights__(self, W1, b1, W2, b2, dW1, db1, dW2, db2, learning_rate):
        W1 = W1 - learning_rate * dW1
        b1 = b1 - learning_rate * db1
        W2 = W2 - learning_rate * dW2
        b2 = b2 - learning_rate * db2
        return W1, b1, W2, b2
    
    def antreneaza(self, nr_neuroni_strat_ascuns=100, rata_de_invatare=0.01, epoci=1000, batch_size=100):
        self.__weights_init_Xavier_Uniform__(nr_neuroni_strat_ascuns)

        for epoca in range(epoci):
            perm = np.random.permutation(self.train_x.shape[0])
            X = self.train_x[perm]
            y = self.train_y[perm]

            for i in range (0, X.shape[0], batch_size):
                X_batch = X[i:i+batch_size]
                y_batch = y[i:i+batch_size]

                # Forward pass
                A1, A2, dropouts = self.__forward_pass__(X_batch, self.W1, self.b1, self.W2, self.b2, dropout_rate=0.5)

                # Compute loss
                loss = self.__compute_loss_cross_entropy__(y_batch, A2)

                # Backward pass
                dw1, db1, dw2, db2 = self.__backward_pass__(X_batch, y_batch, A1, A2, self.W2, dropouts, dropout_rate=0.5)

                # Update weights
                self.W1, self.b1, self.W2, self.b2 = self.__update_weights__(self.W1, self.b1, self.W2, self.b2, dw1, db1, dw2, db2, rata_de_invatare)

            print(f'Epoch {epoca+1}/{epoci}, Loss: {loss:.4f}')

        return self.W1, self.b1, self.W2, self.b2
    
    def predict(self, X, W1, b1, W2, b2):
        _, A2, _ = self.__forward_pass__(X, W1, b1, W2, b2)
        return np.argmax(A2, axis=1)
        
p = Perceptron(pd.read_excel('cat_data_preprocesat.xlsx'))
W1, b1, W2, b2 = p.antreneaza()

train_predictions = p.predict(p.train_x, W1, b1, W2, b2)
test_predictions = p.predict(p.test_x, W1, b1, W2, b2)

print(f'Training Accuracy: {accuracy_score(np.argmax(p.train_y, axis=1), train_predictions) * 100:.2f}%')
print(f'Test Accuracy: {accuracy_score(p.test_y, test_predictions) * 100:.2f}%')