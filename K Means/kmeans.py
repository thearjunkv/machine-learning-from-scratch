import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

def euclidean_distance(x1, x2):
  return np.sqrt(np.sum((x1-x2)**2))

def distortion_error(X, clusters, centroids):
  error = 0
  for cluster_idx, cluster in enumerate(clusters):
    samples = X[cluster]
    diffs = samples - centroids[cluster_idx]
    error += np.sum(diffs ** 2)
  return error

class KMeans:
  def __init__(self, K=3, n_iters=100):
    self.K = K
    self.n_iters = n_iters

    self.clusters = []
    self.centroids = []

    self.distortion_errors = []

  def predict(self, X):
    self.X = X
    n_samples, n_features = X.shape

    random_idxs = np.random.choice(n_samples, self.K, replace=False)
    self.centroids = [X[idx] for idx in random_idxs]

    prev_dist_error = None
    self.distortion_errors = []

    for i_iter in range(self.n_iters):
      clusters = [[] for _ in range(self.K)]

      for idx, sample in enumerate(X):
        centroid_idx = self._get_closest_centroid(sample)
        clusters[centroid_idx].append(idx)
      self.clusters = clusters

      self.centroids = self._get_centroids()
      dist_error = distortion_error(X, self.clusters, self.centroids)
      self.distortion_errors.append([i_iter + 1, dist_error])

      if prev_dist_error is not None:
        if prev_dist_error == dist_error:
          print(f'Converged at {i_iter}th iteration')
          break

      prev_dist_error = dist_error
    return self._get_cluster_labels()

  def _get_cluster_labels(self):
    labels = np.zeros(self.X.shape[0])
    for cluster_idx, cluster in enumerate(self.clusters):
      for sample_idx in cluster:
        labels[sample_idx] = cluster_idx
    return labels

  def _get_centroids(self):
    centroids = np.zeros((self.K, self.X.shape[1]))
    for cluster_idx, cluster in enumerate(self.clusters):
      cluster_mean = np.mean(self.X[cluster], axis=0)
      centroids[cluster_idx] = cluster_mean
    return centroids

  def _get_closest_centroid(self, sample):
    distances = [euclidean_distance(sample, centroid) for centroid in self.centroids]
    return np.argmin(distances)

  def plot_clusters(self):
    fig, ax = plt.subplots(figsize=(8, 6))

    for cluster in self.clusters:
      point = self.X[cluster].T
      ax.scatter(*point)

    for centroid in self.centroids:
      ax.scatter(*centroid, marker="x", color="black", linewidth=2)

    plt.show()

  def plot_distortion_errors(self):
    dist_errors = np.array(self.distortion_errors)
    iters = dist_errors[:, 0]
    errors = dist_errors[:, 1]
    plt.plot(iters, errors)
    plt.show()

if __name__ == '__main__':
  X, y = make_blobs(
    centers=3, n_samples=500, n_features=2, shuffle=True, random_state=40
  )
  clusters = len(np.unique(y))
  print(clusters)

  model = KMeans(K=clusters, n_iters=50)
  y_pred = model.predict(X)

  model.plot_clusters()
  model.plot_distortion_errors()