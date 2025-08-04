import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from collections import Counter
import re

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

class MultinomialNB:
  def fit(self, X, y):
    n_samples = X.shape[0]
    X = np.array([self._preprocess(x) for x in X])

    all_tokens = [token for text in X for token in text.split()]
    self.vocab = sorted(set(all_tokens))
    # self.word2idx = {word:idx for idx, word in enumerate(self.vocab)}
    self.vocab_size = len(self.vocab)

    self.classes = np.unique(y)
    n_classes = len(self.classes)

    self.priors = np.zeros(n_classes, dtype=np.float64)
    self.likelihoods = [None] * n_classes

    for class_idx, c in enumerate(self.classes):
      self.priors[class_idx] = len(y[y == c]) / float(n_samples)
      X_c = X[y == c]
      words = [word for text in X_c for word in text.split()]
      word_counts = Counter(words)
      n_words = sum(word_counts.values())
      self.likelihoods[class_idx] = (word_counts, n_words)

  def _preprocess(self, text):
    text = text.lower() # lowercase
    text = re.sub(r'[^a-z0-9\s]', '', text) # remove punctuations
    tokens = text.split() # tokenise
    tokens = np.array([word for word in tokens if word not in stop_words]) # remove stopwords
    return ' '.join(tokens)

  def _word_prod(self, word, class_idx):
    word_counts, n_words = self.likelihoods[class_idx]
    return (word_counts.get(word, 0) + 1) / (n_words + self.vocab_size)

  def predict(self, X):
    X = [self._preprocess(x) for x in X]
    pred = np.array([self._predict(x) for x in X])
    return pred

  def _predict(self, x):
    posteriors = []
    for class_idx, c in enumerate(self.classes):
      prior = np.log(self.priors[class_idx])
      words = x.split()
      evidence = [self._word_prod(word, class_idx) for word in words]

      posterior = prior + np.sum(np.log(evidence))
      posteriors.append(posterior)

    return self.classes[np.argmax(posteriors)]

if __name__ == '__main__':
  df = pd.read_csv('./SMSSpamCollection', sep='\t', names=['label', 'text'])

  dataset = df.to_numpy()
  X, y = dataset[:, 1], dataset[:, 0]

  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

  model = MultinomialNB()
  model.fit(X_train, y_train)
  y_pred = model.predict(X_test)

  acc = np.sum(y_pred == y_test) / len(y_test)
  print(f"Test accuracy: {acc:.4f}")
  print(classification_report(y_test, y_pred))