from typing import Union

import numpy as np
import pandas as pd

from ml_from_scratch.models import Model
from ml_from_scratch.statistics.metrics import accuracy
from ml_from_scratch.statistics.sigmoid import sigmoid_function


class LogisticRegression(Model):
    """
    The LogisticRegression is a logistic model using the L2 regularization.
    This model solves the logistic regression problem using an adapted Gradient Descent technique

    Parameters
    ----------
    l2_penalty: float
        The L2 regularization parameter
    alpha: float
        The learning rate
    max_iter: int
        The maximum number of iterations

    Attributes
    ----------
    theta: np.array
        The model parameters, namely the coefficients of the logistic model.
        For example, sigmoid(x0 * theta[0] + x1 * theta[1] + ...)
    theta_zero: float
        The intercept of the logistic model
    """
    def __init__(self, l2_penalty: float = 1, alpha: float = 0.001, max_iter: int = 1000,
                 patience: int = 5, scale: bool = True):
        """

        Parameters
        ----------
        l2_penalty: float
            The L2 regularization parameter
        alpha: float
            The learning rate
        max_iter: int
            The maximum number of iterations
        patience: int
            The number of iterations without improvement before stopping the training
        scale: bool
            Whether to scale the dataset or not
        """
        # parameters
        self.l2_penalty = l2_penalty
        self.alpha = alpha
        self.max_iter = max_iter
        self.patience = patience
        self.scale = scale

        # attributes
        self.theta = None
        self.theta_zero = None
        self.mean = None
        self.std = None
        self.cost_history = {}

    def _fit(self, x: pd.DataFrame, y: Union[pd.DataFrame, pd.Series]) -> 'LogisticRegression':
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
        if self.scale:
            # compute mean and std
            self.mean = np.nanmean(x, axis=0)
            self.std = np.nanstd(x, axis=0)
            # scale the dataset
            X = (x - self.mean) / self.std
        else:
            X = x

        m, n = x.shape

        # initialize the model parameters
        self.theta = np.zeros(n)
        self.theta_zero = 0

        i = 0
        early_stopping = 0
        # gradient descent
        while i < self.max_iter and early_stopping < self.patience:
            # predicted y
            y_pred = np.dot(X, self.theta) + self.theta_zero

            # apply sigmoid function
            y_pred = sigmoid_function(y_pred)

            # compute the gradient using the learning rate
            gradient = (self.alpha / m) * np.dot(y_pred - y, X)

            # compute the penalty
            penalization_term = self.theta * (1 - self.alpha * (self.l2_penalty / m))

            # update the model parameters
            self.theta = penalization_term - gradient
            self.theta_zero = self.theta_zero - (self.alpha * (1 / m)) * np.sum(y_pred - y)

            # compute the cost
            self.cost_history[i] = self.cost(x, y)
            if i > 0 and self.cost_history[i] > self.cost_history[i - 1]:
                early_stopping += 1
            else:
                early_stopping = 0
            i += 1

        return self

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
        X = (x - self.mean) / self.std if self.scale else x
        predictions = sigmoid_function(np.dot(X, self.theta) + self.theta_zero)

        # convert the predictions to 0 or 1 (binarization)
        mask = predictions >= 0.5
        predictions[mask] = 1
        predictions[~mask] = 0
        return predictions

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
        y_pred = self.predict(x)
        return accuracy(y, y_pred)

    def cost(self, x: pd.DataFrame, y: Union[pd.DataFrame, pd.Series]) -> float:
        """
        Compute the cost function (J function) of the model on the dataset using L2 regularization

        Parameters
        ----------
        x: Dataset
            The dataset to compute the cost on
        y: Union[pd.DataFrame, pd.Series]
            The target variable

        Returns
        -------
        cost: float
            The cost function of the model
        """
        predictions = sigmoid_function(np.dot(x, self.theta) + self.theta_zero)
        cost = (y * np.log(predictions)) + (1 - y) * np.log(1 - predictions)
        cost = np.sum(cost) * (-1 / x.shape[0])
        cost = cost + (self.l2_penalty * np.sum(self.theta ** 2) / (2 * x.shape[0]))
        return cost


if __name__ == '__main__':
    from sklearn.model_selection import train_test_split

    # load and split the dataset
    dataset_ = pd.DataFrame({
        'x1': np.random.rand(100),
        'x2': np.random.rand(100),
        'y': np.random.choice([0, 1], 100)
    })
    x_train, x_test, y_train, y_test = train_test_split(dataset_[['x1', 'x2']], dataset_['y'], test_size=0.3)

    # fit the model
    model = LogisticRegression(l2_penalty=1, alpha=0.001, max_iter=1000)
    model.fit(x_train, y_train)

    print(model.theta)
    print(model.theta_zero)

    print(model.predict(x_test))

    # compute the score
    score = model.score(x_test, y_test)
    print(f"Score: {score}")

    # plot the cost history
    import matplotlib.pyplot as plt

    plt.plot(list(model.cost_history.keys()), list(model.cost_history.values()))
    plt.show()
