import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets

class GaussianNB:
  def fit(self, X, y):
    n_samples, n_features = X.shape
    self.classes = np.unique(y)
    n_classes = len(self.classes)

    self.mean = np.zeros((n_classes, n_features), dtype=np.float64)
    self.var = np.zeros((n_classes, n_features), dtype=np.float64)
    self.priors = np.zeros(n_classes, dtype=np.float64)

    for idx, c in enumerate(self.classes):
      X_c = X[y == c]
      self.mean[idx, :] = np.mean(X_c, axis=0)
      self.var[idx, :] = np.var(X_c, axis=0)
      self.priors[idx] = X_c.shape[0] / float(n_samples)

  def predict(self, X):
    y_pred = [self._predict(x) for x in X]
    return np.array(y_pred)

  def _predict(self, x):
    posteriors = []

    for idx, c in enumerate(self.classes):
      prior = np.log(self.priors[idx])
      evidence = np.sum(np.log(self._pdf(idx, x)))
      posterior = prior + evidence
      posteriors.append(posterior)

    return self.classes[np.argmax(posteriors)]

  def _pdf(self, class_idx, x):
    mean = self.mean[class_idx]
    var = self.var[class_idx]
    numerator = np.exp(-((x - mean) ** 2) / (2 * var))
    denominator = np.sqrt(2 * np.pi * var)
    return numerator / denominator

if __name__ == '__main__':
  X, y = datasets.make_classification(
      n_samples=1000,
      n_features=10,
      n_classes=2,
      random_state=123
  )
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

  model = GaussianNB()
  model.fit(X_train, y_train)
  y_pred = model.predict(X_test)
  acc = np.sum(y_pred == y_test) / len(y_test)

  print('Test accuracy : ', acc)