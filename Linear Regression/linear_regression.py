import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt

class LinearRegression:
  def __init__(self, lr = 0.001, n_iters = 1000):
    self.lr = lr
    self.n_iters = n_iters
    self.weights = None
    self.bias = None

  def fit(self, X, y):
    n_samples, n_features = X.shape
    self.weights = np.zeros(n_features)
    self.bias = 0

    for _ in range(self.n_iters):
      y_pred = np.dot(X, self.weights) + self.bias

      dw = (1/n_samples) * np.dot(X.T, (y_pred - y))
      db = (1/n_samples) * np.sum(y_pred - y)

      self.weights = self.weights - self.lr * dw
      self.bias = self.bias - self.lr * db

  def predict(self, X):
    return np.dot(X, self.weights) + self.bias

if __name__ == '__main__':
  X, y = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=4)
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

  model = LinearRegression(lr = 0.1, n_iters = 100)
  model.fit(X_train, y_train)
  y_pred = model.predict(X_test)
  
  mse = np.mean((y_test - y_pred)**2)
  print('Mean squared error : ', mse)

  fig = plt.figure(figsize=(4, 4))
  plt.scatter(X_train, y_train, color = 'blue')
  plt.scatter(X_test, y_test, color = 'green')
  plt.plot(X_test, y_pred, color='black')
  plt.show()