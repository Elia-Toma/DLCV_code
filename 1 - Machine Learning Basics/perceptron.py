import numpy as np

class Perceptron(object):

    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        #initializer method (constructor)Parameters
        #----------
        # eta (float): Learning rate (between 0.0 and 1.0)
        # n_iter (int): passes over the training dataset
        # random_state (int): seed for random weight initialization

        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        #Fit training data

        #Parameters
        # X : {array-like}, shape = [n_examples , n_features], Training vectors , where n_examples is the number of examples and n_features is the number of features
        # y : array-like, shape = [n_examples], Target values.

        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        self.errors_ = []
        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self
    
    def net_input(self, X):
        #Calculate the linear combination
        # between input and weights

        #Calculate a_j input
        return np.dot(X, self.w_[1:]) + self.w_[0]
    
    def predict(self, X):
        #Return class label after unit step
        return np.where(self.net_input(X) >= 0.0, 1, -1)