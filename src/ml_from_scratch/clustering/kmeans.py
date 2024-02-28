from typing import Callable, Union

import numpy as np
import pandas as pd

from ml_from_scratch import Transformer
from ml_from_scratch.statistics.distances import euclidean_distance


class KMeans(Transformer):
    """
    It performs k-means clustering on the dataset.
    It groups samples into k clusters by trying to minimize the distance between samples and their closest centroid.
    It returns the centroids and the indexes of the closest centroid for each point.

    Parameters
    ----------
    k: int
        Number of clusters.
    max_iter: int
        Maximum number of iterations.
    distance: Callable
        Distance function.

    Attributes
    ----------
    centroids: np.array
        Centroids of the clusters.
    labels: np.array
        Labels of the clusters.
    """

    def __init__(self, k: int, max_iter: int = 1000, distance: Callable = euclidean_distance):
        """
        K-means clustering algorithm.

        Parameters
        ----------
        k: int
            Number of clusters.
        max_iter: int
            Maximum number of iterations.
        distance: Callable
            Distance function.
        """
        # parameters
        self.k = k
        self.max_iter = max_iter
        self.distance = distance

        # attributes
        self.centroids = None
        self.labels = None

    def _init_centroids(self, x: pd.DataFrame) -> None:
        """
        It generates initial k centroids.

        Parameters
        ----------
        x: pd.DataFrame
            Dataset.
        """
        seeds = np.random.permutation(x.shape[0])[:self.k]
        self.centroids = x.iloc[seeds].values

    def _get_closest_centroid(self, sample: np.ndarray) -> np.ndarray:
        """
        Get the closest centroid to each data point.

        Parameters
        ----------
        sample : np.ndarray, shape=(n_features,)
            A sample.

        Returns
        -------
        np.ndarray
            The closest centroid to each data point.
        """
        centroids_distances = self.distance(sample, self.centroids)
        closest_centroid_index = np.argmin(centroids_distances, axis=0)
        return closest_centroid_index

    def _fit(self, x: pd.DataFrame, y: Union[pd.DataFrame, pd.Series] = None) -> 'KMeans':
        """
        It fits k-means clustering on the dataset.
        The k-means algorithm initializes the centroids and then iteratively updates them until convergence or max_iter.
        Convergence is reached when the centroids do not change anymore.

        Parameters
        ----------
        x: pd.DataFrame
            Dataset.
        y: Union[pd.DataFrame, pd.Series], optional
            Target variable.
            The default is None.

        Returns
        -------
        KMeans
            KMeans object.
        """
        # generate initial centroids
        self._init_centroids(x)

        # fitting the k-means
        convergence = False
        i = 0
        labels = np.zeros(x.shape[0])
        while not convergence and i < self.max_iter:

            # get closest centroid
            new_labels = np.apply_along_axis(self._get_closest_centroid, axis=1, arr=x.values)

            # compute the new centroids
            centroids = []
            for j in range(self.k):
                centroid = np.mean(x[new_labels == j], axis=0)
                centroids.append(centroid)

            self.centroids = np.array(centroids)

            # check if the centroids have changed
            convergence = not np.any(new_labels != labels)

            # replace labels
            labels = new_labels

            # increment counting
            i += 1

        self.labels = labels
        return self

    def _get_distances(self, sample: np.ndarray) -> np.ndarray:
        """
        It computes the distance between each sample and the closest centroid.

        Parameters
        ----------
        sample : np.ndarray, shape=(n_features,)
            A sample.

        Returns
        -------
        np.ndarray
            Distances between each sample and the closest centroid.
        """
        return self.distance(sample, self.centroids)

    def _transform(self, x: pd.DataFrame, y: Union[pd.DataFrame, pd.Series] = None) -> pd.DataFrame:
        """
        It transforms the dataset.
        It computes the distance between each sample and the closest centroid.

        Parameters
        ----------
        x: pd.DataFrame
            Dataset.
        y: Union[pd.DataFrame, pd.Series], optional
            Target variable.
            The default is None.

        Returns
        -------
        np.ndarray
            Transformed dataset.
        """
        centroids_distances = np.apply_along_axis(self._get_distances, axis=1, arr=x.values)
        centroids_distances = pd.DataFrame(centroids_distances, columns=[f'centroid_{i}' for i in range(self.k)])
        return centroids_distances

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        """
        It predicts the labels of the dataset.

        Parameters
        ----------
        x: pd.DataFrame
            Dataset.

        Returns
        -------
        np.ndarray
            Predicted labels.
        """
        return np.apply_along_axis(self._get_closest_centroid, axis=1, arr=x)

    def fit_predict(self, x: pd.DataFrame) -> np.ndarray:
        """
        It fits and predicts the labels of the dataset.

        Parameters
        ----------
        x: pd.DataFrame
            Dataset.

        Returns
        -------
        np.ndarray
            Predicted labels.
        """
        self.fit(x)
        return self.predict(x)


if __name__ == '__main__':
    dataset_ = pd.DataFrame(np.random.rand(100, 2))

    k_ = 3
    kmeans = KMeans(k_)
    res = kmeans.fit_transform(dataset_)
    predictions = kmeans.predict(dataset_)
    print(res.shape)
    print(predictions.shape)
    print(kmeans.centroids)
    print(kmeans.labels)
    # plot
    import matplotlib.pyplot as plt
    plt.scatter(dataset_[0], dataset_[1], c=predictions)
    plt.scatter(kmeans.centroids[:, 0], kmeans.centroids[:, 1], c='red')
    plt.show()
