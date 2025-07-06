from sklearn.linear_model import Perceptron


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
# Example dataset: Iris (first 100 samples for binary classification)
iris = load_iris()
data = pd.DataFrame(iris.data, columns=iris.feature_names)
data['target'] = iris.target

# Use only two classes for perceptron (e.g., class 0 and 1)
data = data[data['target'] < 2]

X_train, X_test = train_test_split(data, test_size=0.2, random_state=42)


X = X_train.values[:, 0:4]
y = X_train.values[:, 4]

model = Perceptron()
model.fit(X, y)

X = X_test.values[:, 0:4]
y = X_test.values[:, 4]
print("%0.3f" % model.score(X, y))