from abc import ABC, abstractmethod
from typing import Union

import numpy as np
import pandas as pd


class Model(ABC):
    """
    Abstract class for all models.
    """
    _fitted: bool = False

    def fit(self, x: pd.DataFrame, y: Union[pd.DataFrame, pd.Series]) -> 'Model':
        """
        Fit the model to the data.

        Parameters
        ----------
        x : pd.DataFrame
            The data to fit the model.
        y : Union[pd.DataFrame, pd.Series]
            The target variable to try to predict.

        Returns
        -------
        self : object
            The fitted model.
        """
        self._fit(x, y)
        self._fitted = True
        return self

    @abstractmethod
    def _fit(self, x: pd.DataFrame, y: Union[pd.DataFrame, pd.Series]) -> None:
        """
        Fit the model to the data.

        Parameters
        ----------
        x : pd.DataFrame
            The data to fit the model.
        y : Union[pd.DataFrame, pd.Series]
            The target variable to try to predict.
        """
        pass

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        """
        Predict the target variable.

        Parameters
        ----------
        x : pd.DataFrame
            The data to predict the target variable.

        Returns
        -------
        np.ndarray
            The predicted target variable.
        """
        if not self.fitted:
            raise ValueError("The model must be fitted before making predictions.")
        return self._predict(x)

    @abstractmethod
    def _predict(self, x: pd.DataFrame) -> np.ndarray:
        """
        Predict the target variable.

        Parameters
        ----------
        x : pd.DataFrame
            The data to predict the target variable.
        """
        pass

    @abstractmethod
    def score(self, x: pd.DataFrame, y: Union[pd.DataFrame, pd.Series]) -> float:
        """
        Return the score of the model.

        Parameters
        ----------
        x : pd.DataFrame
            The data to score the model.
        y : pd.DataFrame
            The target variable to try to predict.

        Returns
        -------
        float
            The score of the model.
        """
        pass

    @property
    def fitted(self) -> bool:
        """
        Return True if the model is fitted, False otherwise.
        """
        return self._fitted
