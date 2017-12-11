%matplotlib inline
import numpy as np
import cvxopt.solvers
import logging
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import itertools

class Kernel(object):
    @staticmethod
    def linear():
        return lambda x, y: np.inner(x, y)

    @staticmethod
    def gaussian(sigma):
        return lambda x, y: np.exp(-(np.linalg.norm(x-y)**2/(sigma**2)))
     
        
MIN_SUPPORT_VECTOR_MULTIPLIER = 1e-5


class SVMTrainer(object):
    def __init__(self, kernel, c=0.1):
        self._kernel = kernel
        self._c = c


    def train(self, X, y):
        lagrange_multipliers = self._compute_lagrange_multipliers(X, y)
        return self._create_predictor(X, y, lagrange_multipliers)


    def _kernel_matrix(self, X):
        n_samples = X.shape[0]

        K = np.zeros((n_samples, n_samples))

        print(X)

        for i, x_i in enumerate(X):
            for j, x_j in enumerate(X):
                K[i, j] = self._kernel(x_i, x_j)

        return K


    def _create_predictor(self, X, y, lagrange_multipliers):

        support_vector_indices = lagrange_multipliers > MIN_SUPPORT_VECTOR_MULTIPLIER

        support_multipliers = lagrange_multipliers[support_vector_indices]

        support_vectors = X[support_vector_indices]

        support_vector_labels = y[support_vector_indices]

        bias = np.mean(
            [y_k - SVMPredictor(
                    kernel=self._kernel,
                    bias=0.0,
                    weights=support_multipliers,
                    support_vectors=support_vectors,
                    support_vector_labels=support_vector_labels
                ).predict(x_k) for (y_k, x_k) in zip(support_vector_labels, support_vectors)
            ]
        )

        return SVMPredictor(
            kernel=self._kernel,
            bias=0.0,
            weights=support_multipliers,
            support_vectors=support_vectors,
            support_vector_labels=support_vector_labels
        )


    def _compute_lagrange_multipliers(self, X, y):
        n_samples = X.shape[0]

        K = self._kernel_matrix(X)

        P = cvxopt.matrix(np.outer(y, y) * K)

        q = cvxopt.matrix(-1 * np.ones(n_samples))

        G = cvxopt.matrix(np.diag(np.ones(n_samples) * -1))

        h = cvxopt.matrix(np.zeros(n_samples))

        A = cvxopt.matrix(y, (1, n_samples))

        b = cvxopt.matrix(0.0)


        solution = cvxopt.solvers.qp(P, q, G, h, A, b)

        return np.ravel(solution['x'])


class SVMPredictor(object):
    def __init__(
                self,
                kernel,
                bias,
                weights,
                support_vectors,
                support_vector_labels
            ):
        
        self._kernel = kernel
        self._bias = bias
        self._weights = weights
        self._support_vectors = support_vectors
        self._support_vector_labels = support_vector_labels


        assert len(support_vectors) == len(support_vector_labels)
        assert len(weights) == len(support_vector_labels)


        logging.info("Bias: %s", self._bias)
        logging.info("Weights: %s", self._weights)
        logging.info("Support vectors: %s", self._support_vectors)
        logging.info("Support vector labels: %s", self._support_vector_labels)

    def predict(self, x):
        """
        Computes the SVM prediction on the given features x.
        """
        result = self._bias
        for w_i, x_i, y_i in zip(self._weights,
                                 self._support_vectors,
                                 self._support_vector_labels):
            result += w_i * y_i * self._kernel(x_i, x)

        return np.sign(result).item()

def example(num_samples=20, num_features=2, grid_size=30):
    
    samples = np.matrix(np.random.normal(size=num_samples * num_features)
                        .reshape(num_samples, num_features))
    
    labels = 2 * (samples.sum(axis=1) > 0) - 1.0
    
    trainer = SVMTrainer(Kernel.gaussian(1))
    
    predictor = trainer.train(samples, labels)

    plot(predictor, samples, labels, grid_size)


def plot(predictor, X, y, grid_size):
    
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, grid_size),
        np.linspace(y_min, y_max, grid_size),
        indexing='ij'
    )
    
    flatten = lambda m: np.array(m).reshape(-1,)

    result = []
    
    for (i, j) in itertools.product(range(grid_size), range(grid_size)):
        point = np.array([xx[i, j], yy[i, j]]).reshape(1, 2)
        result.append(predictor.predict(point))

    Z = np.array(result).reshape(xx.shape)
    
    plt.contourf(
        xx, yy, Z,
        cmap=cm.Paired,
        levels=[-0.01, 0.01],
        extend='both',
        alpha=0.8
    )
    
    
    plt.scatter(
        flatten(X[:, 0]),
        flatten(X[:, 1]),
        c=flatten(y),
        cmap=cm.Paired
    )
    
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    