import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

class PCA:
  def __init__(self, n_components):
    self.n_components = n_components
    self.mean = None
    self.components = None

  def fit(self, X):
    self.mean = np.mean(X, axis=0)
    centered_X = X - self.mean
    cov = np.cov(centered_X.T)

    eigenvalues, eigenvectors = np.linalg.eig(cov)

    idxs = np.argsort(eigenvalues)[::-1]
    eigenvalues, eigenvectors = eigenvalues[idxs], eigenvectors[idxs]

    self.components = eigenvectors[:self.n_components]

  def transform(self, X):
    X = X - self.mean
    return np.dot(X, self.components.T)

if __name__ == '__main__':
  iris = datasets.load_iris()
  X, y = iris.data, iris.target

  model = PCA(n_components=2)
  model.fit(X)
  projected_X = model.transform(X)

  print('Shape of X: ', X.shape)
  print('Shape of projected X: ', projected_X.shape)

  x1 = projected_X[:,0]
  x2 = projected_X[:,1]

  plt.scatter(x1, x2, c=y)
  plt.xlabel("Principal Component 1")
  plt.ylabel("Principal Component 2")
  plt.colorbar()
  plt.show()