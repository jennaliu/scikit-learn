import argparse

import numpy as np
from scipy import sparse
from scipy.sparse.csr import csr_matrix
from sklearn.linear_model.stochastic_gradient import SGDRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

np.random.seed(42)

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--density', type=float, default=0.01)
parser.add_argument('-i', '--intercept', type=float, default=100)
result = parser.parse_args()

density = result.density
# SGD on csr_matrix fits poorly when absolute value of itercept is big
intercept = result.intercept

n_samples, n_features = 5000, 300

print("input data density: %f" % density)
print("intercept: %f" % intercept)

X = sparse.random(n_samples, n_features, density=density).A
coef = 3 * np.random.randn(n_features)
y = np.dot(X, coef) + intercept

# add noise
y += 0.01 * np.random.randn()

# Split data in train set and test set
n_samples = X.shape[0]
X_train, y_train = X[:n_samples // 2], y[:n_samples // 2]
X_test, y_test = X[n_samples // 2:], y[n_samples // 2:]


###############################################################################
def benchmark(clf, X_train, X_test, case):
    print("-" * 50)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_train)
    r2 = r2_score(y_train, y_pred)
    print("r^2 on train data (%s) : %f" % (case, r2))

    y_pred = clf.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    print("r^2 on test data (%s) : %f" % (case, r2))


clf = SGDRegressor(max_iter=2000, tol=None)
benchmark(clf, X_train, X_test, "SGDRegressor numpy array")

clf = SGDRegressor(max_iter=2000, tol=None)
benchmark(clf, csr_matrix(X_train), csr_matrix(X_test),
          "SGDRegressor csr_matrix")

clf = Ridge()
benchmark(clf, X_train, X_test, "Ridge numpy array")

clf = Ridge()
benchmark(clf, csr_matrix(X_train), csr_matrix(X_test), "Ridge csr_matrix")
