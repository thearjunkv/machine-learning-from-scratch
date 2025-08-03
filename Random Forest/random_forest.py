import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from collections import Counter
from decision_tree import DecisionTree

class RandomForest:
  def __init__(self, n_trees=10, n_features=None, min_samples_split=3, max_depth=20):
    self.n_trees = n_trees
    self.n_features = n_features
    self.min_samples_split = min_samples_split
    self.max_depth = max_depth
    self.trees = []

  def fit(self, X, y):
    self.trees = []

    for _ in range(self.n_trees):
      X_sample, y_sample = self._bootstrap_samples(X, y)

      tree = DecisionTree(
          min_samples_split=self.min_samples_split,
          max_depth=self.max_depth,
          n_features=self.n_features
      )
      tree.fit(X_sample, y_sample)
      self.trees.append(tree)

  def _bootstrap_samples(self, X, y):
    n_samples = X.shape[0]
    idxs = np.random.choice(n_samples, n_samples)
    return X[idxs], y[idxs]

  def predict(self, X):
    predictions = [tree.predict(X) for tree in self.trees]
    tree_preds = np.swapaxes(predictions, 0, 1)
    return np.array([self._most_common_label(pred) for pred in tree_preds])

  def _most_common_label(self, y):
    counter = Counter(y)
    return counter.most_common(1)[0][0]

if __name__ == '__main__':
  bc = datasets.load_breast_cancer()
  X, y = bc.data, bc.target

  X_train, X_test, y_train, y_test = train_test_split(
      X, y, test_size=0.2, random_state=123
  )

  model = RandomForest(n_trees=15, n_features=15)
  model.fit(X_train, y_train)

  y_pred = model.predict(X_test)
  acc = np.sum(y_pred == y_test) / len(y_test)

  print('Test accuracy : ', acc)