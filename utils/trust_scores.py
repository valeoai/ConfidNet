# Code referenced from https://github.com/google/TrustScore
import numpy as np
from sklearn.neighbors import KDTree
from sklearn.neighbors import KNeighborsClassifier
import multiprocessing
import time
import utils.logger
LOGGER = utils.logger.get_logger(__name__, level='DEBUG')

class TrustScore:
    """
    Trust Score: a measure of classifier uncertainty based on nearest neighbors.
    """
    def __init__(self, k=10, alpha=0., filtering="none", min_dist=1e-12, num_workers=1):
        """
            k and alpha are the tuning parameters for the filtering,
            filtering: method of filtering. option are "none", "density",
            "uncertainty"
            min_dist: some small number to mitigate possible division by 0.
        """
        self.k = k
        self.filtering = filtering
        self.alpha = alpha
        self.min_dist = min_dist
        self.num_workers = num_workers 

    def filter_by_density(self, X):
        """Filter out points with low kNN density.
        Args:
        X: an array of sample points.
        Returns:
        A subset of the array without points in the bottom alpha-fraction of
        original points of kNN density.
        """
        kdtree = KDTree(X)
        knn_radii = kdtree.query(X, k=self.k)[0][:, -1]
        eps = np.percentile(knn_radii, (1 - self.alpha) * 100)
        return X[np.where(knn_radii <= eps)[0], :]

    def filter_by_uncertainty(self, X, y):
        """Filter out points with high label disagreement amongst its kNN neighbors.
        Args:
        X: an array of sample points.
        Returns:
        A subset of the array without points in the bottom alpha-fraction of
        samples with highest disagreement amongst its k nearest neighbors.
        """
        neigh = KNeighborsClassifier(n_neighbors=self.k)
        neigh.fit(X, y)
        confidence = neigh.predict_proba(X)
        cutoff = np.percentile(confidence, self.alpha * 100)
        unfiltered_idxs = np.where(confidence >= cutoff)[0]
        return X[unfiltered_idxs, :], y[unfiltered_idxs]

    def fit(self, X, y):
        """Initialize trust score precomputations with training data.
        WARNING: assumes that the labels are 0-indexed (i.e.
        0, 1,..., n_labels-1).
        Args:
        X: an array of sample points.
        y: corresponding labels.
        """
        self.n_labels = np.max(y) + 1
        self.kdtrees = [None] * self.n_labels
        if self.filtering == "uncertainty":
            X_filtered, y_filtered = self.filter_by_uncertainty(X, y)
        for label in range(self.n_labels):
            if self.filtering == "none":
                X_to_use = X[np.where(y == label)[0]]
                self.kdtrees[label] = KDTree(X_to_use)
            elif self.filtering == "density":
                X_to_use = self.filter_by_density(X[np.where(y == label)[0]])
                self.kdtrees[label] = KDTree(X_to_use)
            elif self.filtering == "uncertainty":
                X_to_use = X_filtered[np.where(y_filtered == label)[0]]
                self.kdtrees[label] = KDTree(X_to_use)

            if len(X_to_use) == 0:
                print("Filtered too much or missing examples from a label! Please lower alpha or check data.")

    def get_score(self, X, y_pred):
        """Compute the trust scores.
        Given a set of points, determines the distance to each class.
        Args:
        X: an array of sample points.
        y_pred: The predicted labels for these points.
        Returns:
        The trust score, which is ratio of distance to closest class that was not
        the predicted class to the distance to the predicted class.
        """
        d = np.tile(None, (X.shape[0], self.n_labels))
        
        if self.num_workers<2:
            LOGGER.warning('Using single processing for computing trust scores')
            for label_idx in range(self.n_labels):
                t0 = time.time()
                d[:, label_idx] = self.kdtrees[label_idx].query(X, k=2)[0][:, -1]
                print('KDTree for label %d run in %ls secondes' % (label_idx, time.time() - t0))
        else:
            LOGGER.warning('Enabling multiprocessing with {} workers for computing trust scores'.format(self.num_workers))
            self.current_X = X
            pool = multiprocessing.Pool(self.num_workers)    
            d = pool.map(self.process_d, range(self.n_labels))
            d = np.array(d).T

        sorted_d = np.sort(d, axis=1)
        d_to_pred = d[range(d.shape[0]), y_pred]
        d_to_closest_not_pred = np.where(sorted_d[:, 0] != d_to_pred,
                                         sorted_d[:, 0], sorted_d[:, 1])
        return d_to_closest_not_pred / (d_to_pred + self.min_dist)

    
    def process_d(self, label_idx):
        t0 = time.time()
        a = self.kdtrees[label_idx].query(self.current_X, k=2)[0][:, -1]
        print('KDTree for label %d run in %ls secondes' % (label_idx, time.time() - t0))
        return a
        
    class KNNConfidence:
        """Baseline which uses disagreement to kNN classifier.
        """
        def __init__(self, k=10):
            self.k = k

        def fit(self, X, y):
            self.kdtree = KDTree(X)
            self.y = y

        def get_score(self, X, y_pred):
            knn_idxs = self.kdtree.query(X, k=self.k)[1]
            knn_outputs = self.y[knn_idxs]
            return np.mean(knn_outputs == np.transpose(np.tile(y_pred, (self.k, 1))), axis=1)