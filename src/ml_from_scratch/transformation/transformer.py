from abc import ABC, abstractmethod
from typing import Union

import pandas as pd


class Transformer(ABC):
    """
    Abstract class for all transformers.
    """
    _fitted: bool = False

    def fit(self, x: pd.DataFrame, y: Union[pd.DataFrame, pd.Series] = None) -> 'Transformer':
        """
        Fit the transformer to the data.

        Parameters
        ----------
        x : pd.DataFrame
            The data to fit the transformer.
        y : Union[pd.DataFrame, pd.Series], optional
            The target variable to try to predict.
            The default is None.

        Returns
        -------
        self : object
            The fitted transformer.
        """
        self._fit(x, y)
        self._fitted = True
        return self

    @abstractmethod
    def _fit(self, x: pd.DataFrame, y: Union[pd.DataFrame, pd.Series] = None) -> None:
        """
        Fit the transformer to the data.

        Parameters
        ----------
        x : pd.DataFrame
            The data to fit the transformer.
        y : Union[pd.DataFrame, pd.Series], optional
            The target variable to try to predict.
            The default is None.
        """
        pass

    def transform(self, x: pd.DataFrame, y: Union[pd.DataFrame, pd.Series] = None) -> pd.DataFrame:
        """
        Transform the data.

        Parameters
        ----------
        x : pd.DataFrame
            The data to transform.
        y : pd.DataFrame, optional
            The target variable to try to predict.
            The default is None.

        Returns
        -------
        pd.DataFrame
            The transformed data.
        """
        if not self.fitted:
            raise ValueError("The transformer must be fitted before transforming the data.")
        return self._transform(x, y)

    @abstractmethod
    def _transform(self, x: pd.DataFrame, y: Union[pd.DataFrame, pd.Series] = None) -> pd.DataFrame:
        """
        Transform the data.

        Parameters
        ----------
        x : pd.DataFrame
            The data to transform.
        y : pd.DataFrame, optional
            The target variable to try to predict.
            The default is None.
        """
        pass

    def fit_transform(self, x: pd.DataFrame, y: Union[pd.DataFrame, pd.Series] = None) -> pd.DataFrame:
        """
        Fit the transformer to the data and transform the data.

        Parameters
        ----------
        x : pd.DataFrame
            The data to fit and transform.
        y : pd.DataFrame, optional
            The target variable to try to predict.
            The default is None.

        Returns
        -------
        pd.DataFrame
            The transformed data.
        """
        self.fit(x, y)
        return self.transform(x, y)

    @property
    def fitted(self) -> bool:
        """
        Return True if the transformer is fitted, False otherwise.
        """
        return self._fitted
