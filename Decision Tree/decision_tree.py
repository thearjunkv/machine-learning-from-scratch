import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from collections import Counter

class Node:
  def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
    self.feature = feature
    self.threshold = threshold
    self.left = left
    self.right =  right
    self.value = value

  def is_leaf_node(self):
    return self.value is not None

class DecisionTree:
  def __init__(self, min_samples_split=3, max_depth=20, n_features=None):
    self.min_samples_split = min_samples_split
    self.max_depth = max_depth
    self.n_features = n_features
    self.root = None

  def fit(self, X, y):
    self.n_features = X.shape[1] if not self.n_features else min(X.shape[1], self.n_features)
    self.root = self._grow_tree(X, y)

  def _grow_tree(self, X, y, depth=0):
    n_samples, n_feats = X.shape
    n_labels = len(np.unique(y))

    # stopping criteria
    if (n_labels == 1 or depth >= self.max_depth or n_samples < self.min_samples_split):
      leaf_value = self._most_common_label(y)
      return Node(value=leaf_value)

    feat_idxs = np.random.choice(n_feats, self.n_features, replace=False)

    # best split
    best_feature, best_threshold = self._best_split(X, y, feat_idxs)

    # create child nodes
    left_idxs, right_idxs = self._split(X[:, best_feature], best_threshold)
    left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth+1)
    right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth+1)

    return Node(best_feature, best_threshold, left, right)

  def _best_split(self, X, y, feat_idxs):
    best_gain = -1
    split_idx, split_threshold = None, None

    for feat_idx in feat_idxs:
      X_column = X[:, feat_idx]
      thresholds = np.unique(X_column)

      for thr in thresholds:
        gain = self._information_gain(y, X_column, thr)

        if (gain > best_gain):
          best_gain = gain
          split_idx = feat_idx
          split_threshold = thr

    return split_idx, split_threshold

  def _information_gain(self, y, X_column, threshold):
    parent_entropy = self._entropy(y)

    left_idxs, right_idxs = self._split(X_column, threshold)
    if len(left_idxs) == 0 or len(right_idxs) == 0:
      return 0

    # weighted children entropy
    n = len(y)
    n_l, n_r = len(left_idxs), len(right_idxs)
    e_l, e_r = self._entropy(y[left_idxs]), self._entropy(y[right_idxs])

    child_entropy = (n_l/n) * e_l + (n_r/n) * e_r
    info_gain = parent_entropy - child_entropy
    return info_gain

  def _split(self, X_column, thr):
    left = np.argwhere(X_column <= thr).flatten()
    right = np.argwhere(X_column > thr).flatten()
    return left, right

  def _entropy(self, X):
    pX = np.bincount(X) / len(X)
    return -np.sum([p*np.log2(p) for p in pX if p > 0])

  def _most_common_label(self, y):
    counter = Counter(y)
    value = counter.most_common(1)[0][0]
    return value

  def predict(self, X):
    return np.array([self._traverse(x, self.root) for x in X])

  def _traverse(self, x, node):
    if node.is_leaf_node():
      return node.value

    if x[node.feature] <= node.threshold:
      return self._traverse(x, node.left)
    return self._traverse(x, node.right)

if __name__ == '__main__':
  bc = datasets.load_breast_cancer()
  X, y = bc.data, bc.target

  X_train, X_test, y_train, y_test = train_test_split(
      X, y, test_size=0.2, random_state=123
  )

  model = DecisionTree()
  model.fit(X_train, y_train)

  y_pred = model.predict(X_test)
  acc = np.sum(y_pred == y_test) / len(y_test)
  print('Test accuracy : ', acc)