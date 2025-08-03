import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from collections import Counter

def euclidean_distance(x1, x2):
  return np.sqrt(np.sum((x1 - x2) ** 2))

class KNN:
  def __init__(self, k=3):
    self.k = k

  def fit(self, X, y):
    self.X_train = X
    self.y_train = y

  def predict(self, X):
    predictions = [self._predict(x) for x in X]
    return predictions

  def _predict(self, x):
    distances = [euclidean_distance(x, x_train) for x_train in self.X_train]

    k_indices = np.argsort(distances)[:self.k]
    k_nearest_labels = [self.y_train[i] for i in k_indices]

    most_common = Counter(k_nearest_labels).most_common()
    return most_common[0][0]

if __name__ == '__main__':
  iris = datasets.load_iris()
  X, y = iris.data, iris.target
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=142)

  model = KNN()
  model.fit(X_train, y_train)
  y_pred = model.predict(X_test)

  acc = np.sum(y_pred == y_test) / len(y_test)
  print('Test accuracy : ', acc)