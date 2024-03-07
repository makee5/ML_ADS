from typing import Callable, Union

import numpy as np
import pandas as pd

from ml_from_scratch.models import Model
from ml_from_scratch.statistics.distances import euclidean_distance
from ml_from_scratch.statistics.metrics import accuracy


class KNNClassifier(Model):
    """
    KNN Classifier
    The k-Nearst Neighbors classifier is a machine learning model that classifies new samples based on
    a similarity measure (e.g., distance functions). This algorithm predicts the classes of new samples by
    looking at the classes of the k-nearest samples in the training data.

    Parameters
    ----------
    k: int
        The number of nearest neighbors to use
    distance: Callable
        The distance function to use

    Attributes
    ----------
    dataset: np.ndarray
        The training data
    """
    def __init__(self, k: int = 1, distance: Callable = euclidean_distance):
        """
        Initialize the KNN classifier

        Parameters
        ----------
        k: int
            The number of nearest neighbors to use
        distance: Callable
            The distance function to use
        """
        # parameters
        self.k = k
        self.distance = distance

        # attributes
        self.dataset = None

    def _fit(self, x: pd.DataFrame, y: Union[pd.DataFrame, pd.Series]) -> None:
        """
        It fits the model to the given dataset

        Parameters
        ----------
        x: Dataset
            The dataset to fit the model to
        y: Union[pd.DataFrame, pd.Series]
            The target variable

        Returns
        -------
        self: KNNClassifier
            The fitted model
        """
        self.x = x
        self.y = y
        return self

    def _get_closest_label(self, sample: np.ndarray) -> Union[int, str]:
        """
        It returns the closest label of the given sample

        Parameters
        ----------
        sample: np.ndarray
            The sample to get the closest label of

        Returns
        -------
        label: str or int
            The closest label
        """
        # compute the distance between the sample and the dataset
        distances = self.distance(sample, self.x)

        # get the k nearest neighbors
        k_nearest_neighbors = np.argsort(distances)[:self.k]

        # get the labels of the k nearest neighbors
        k_nearest_neighbors_labels = self.y[k_nearest_neighbors]

        # get the most common label
        labels, counts = np.unique(k_nearest_neighbors_labels, return_counts=True)
        return labels[np.argmax(counts)]

    def _predict(self, x: pd.DataFrame) -> np.ndarray:
        """
        It predicts the classes of the given dataset

        Parameters
        ----------
        x: Dataset
            The dataset to predict the classes of

        Returns
        -------
        predictions: np.ndarray
            The predictions of the model
        """
        return np.apply_along_axis(self._get_closest_label, axis=1, arr=x)

    def score(self, x: pd.DataFrame, y: Union[pd.DataFrame, pd.Series]) -> float:
        """
        It returns the accuracy of the model on the given dataset

        Parameters
        ----------
        x: Dataset
            The dataset to evaluate the model on
        y: Union[pd.DataFrame, pd.Series]
            The target variable

        Returns
        -------
        accuracy: float
            The accuracy of the model
        """
        predictions = self.predict(x)
        return accuracy(y, predictions)


if __name__ == '__main__':
    dataset_train = pd.DataFrame({
        'x1': np.random.rand(100),
        'x2': np.random.rand(100),
        'y': np.random.choice([0, 1], 100)
    })
    dataset_test = pd.DataFrame({
        'x1': np.random.rand(100),
        'x2': np.random.rand(100),
        'y': np.random.choice([0, 1], 100)
    })

    X_train = dataset_train[['x1', 'x2']].values
    y_train = dataset_train['y'].values
    X_test = dataset_test[['x1', 'x2']].values
    y_test = dataset_test['y'].values

    # initialize the KNN classifier
    knn = KNNClassifier(k=3)

    # fit the model to the train dataset
    knn.fit(X_train, y_train)

    # evaluate the model on the test dataset
    score = knn.score(X_test, y_test)
    print(f'The accuracy of the model is: {score}')