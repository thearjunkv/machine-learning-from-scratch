import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split

rgen = np.random.default_rng(seed=42)

class SVM:
  def __init__(self, lr=0.01, n_iters=100, lambda_param=0.1, w=None, b=None):
    self.lr = lr
    self.n_iters = n_iters
    self.lambda_param = lambda_param
    self.w = w
    self.b = b

  def fit(self, X, y):
    n_samples, n_features = X.shape

    self.w = rgen.normal(loc=0.0, scale=0.01, size=n_features) 
    self.b = 0

    y_ = np.where(y <= 0, -1, 1)

    for _ in range(self.n_iters):
      for idx, x_i in enumerate(X):
        condition = y_[idx] * (np.dot(self.w, x_i) + self.b) >= 1
        regularization = 2 * self.lambda_param * self.w

        if condition:
          self.w -= self.lr * regularization
        else:
           self.w -= self.lr * ( regularization - np.dot(x_i, y_[idx]) )
           self.b -= self.lr * -y_[idx]

  def predict(self, X):
    return np.sign(np.dot(X, self.w) + self.b)

if __name__ == '__main__':
  X, y = make_blobs(centers=2, n_samples=500, n_features=2, cluster_std=1, shuffle=True, random_state=40)
  y = np.where(y == 0, -1, 1)
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

  model = SVM(lambda_param=0.009)
  model.fit(X_train, y_train)

  y_pred = model.predict(X_test)
  acc = np.sum(y_pred == y_test) / len(y_test)

  print('Test accurary : ', acc)

  def visualise():
    def get_value(x1, w, b, offset):
       x2 = (-(w[0] * x1) - b + offset) / w[1]
       return x2

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    plt.scatter(x=X[:,0], y=X[:,1], c=y)

    x_1_start = np.amin(X[:, 0])
    x_1_end = np.amax(X[:, 0])

    x_2_start = get_value(x_1_start, model.w, model.b, 0)
    x_2_end = get_value(x_1_end, model.w, model.b, 0)

    x_2_start_m1 = get_value(x_1_start, model.w, model.b, 1)
    x_2_end_m1 = get_value(x_1_end, model.w, model.b, 1)

    x_2_start_m2 = get_value(x_1_start, model.w, model.b, -1)
    x_2_end_m2 = get_value(x_1_end, model.w, model.b, -1)

    ax.plot([x_1_start, x_1_end], [x_2_start, x_2_end], 'k')
    ax.plot([x_1_start, x_1_end], [x_2_start_m1, x_2_end_m1], 'r--')
    ax.plot([x_1_start, x_1_end], [x_2_start_m2, x_2_end_m2], 'r--')

    x_2_min = np.amin(X[:, 1])
    x_2_max = np.amax(X[:, 1])
    ax.set_ylim([x_2_min - 3, x_2_max + 3])

    plt.show()

  visualise()