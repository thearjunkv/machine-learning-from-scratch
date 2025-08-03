import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets

def sigmoid(y):
  return 1 / (1 + np.exp(-y))

def accuracy(y_pred, y):
  return np.sum(y_pred == y) / len(y)

class LogisticRegression:
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
      z = np.dot(X, self.weights) + self.bias
      y_pred = sigmoid(z)

      dw = (1/n_samples) *  np.dot(X.T, (y_pred - y))
      db = (1/n_samples) *  np.sum(y_pred - y)

      self.weights = self.weights - self.lr * dw
      self.bias = self.bias - self.lr * db

  def predict(self, X):
    z = np.dot(X, self.weights) + self.bias
    y_pred = sigmoid(z)
    class_pred = [0 if y <= 0.5 else 1 for y in y_pred]
    return class_pred

if __name__ == '__main__':
  bc = datasets.load_breast_cancer()
  X, y = bc.data, bc.target
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

  model = LogisticRegression(lr = 0.01, n_iters = 1000)
  model.fit(X_train, y_train)
  
  y_pred = model.predict(X_test)
  acc = accuracy(y_pred, y_test)
  print('Test accuracy : ', acc)
  
  y_train_pred = model.predict(X_train)
  acc = accuracy(y_train_pred, y_train)
  print('Training accuracy : ', acc)