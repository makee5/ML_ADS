from typing import Union

import numpy as np
import pandas as pd

from ml_from_scratch.statistics import f_classif
from ml_from_scratch.transformation import Transformer


class SelectKBest(Transformer):
    """
    Select features according to the k highest scores.

    Parameters
    ----------
    k : int, optional
        The number of features to select.
    score_func : callable, optional
        The scoring function to use.
        The default is f_classif.

    Attributes
    ----------
    scores : pd.Series
        The scores of the features.
    selected_features : pd.Index
        The selected features.

    Example usage:
    ```python
    # Example usage
    X = pd.DataFrame({'a': [1, 2, 3, 4, 5], 'b': [1, 2, 3, 4, 5], 'c': [1, 1, 1, 1, 1]})
    selector = SelectKBest(k=2)
    X = selector.fit_transform(X, y)
    ```
    """

    def __init__(self, k: int = 10, score_func: callable = f_classif) -> None:
        """
        Select features according to the k highest scores.
        """
        self.k = k
        self.score_func = score_func
        self.scores = None
        self.selected_features = None

    def _fit(self, x: pd.DataFrame, y: Union[pd.DataFrame, pd.Series]) -> 'SelectKBest':
        """
        Fit the transformer to the data.

        Parameters
        ----------
        x: pd.DataFrame
            The data to fit the transformer to.
        y: pd.DataFrame or pd.Series
            The target variable.
        """
        self.scores = self.score_func(x, y)[0]  # f_classif returns f-values and p-values we only need f-values
        self.selected_features = np.argsort(self.scores)[-self.k:]
        return self

    def _transform(self, x: pd.DataFrame, y: Union[pd.DataFrame, pd.Series]) -> pd.DataFrame:
        """
        Transform the data.

        Parameters
        ----------
        x: pd.DataFrame
            The data to transform.
        y: pd.DataFrame or pd.Series
            The target variable.

        Returns
        -------
        pd.DataFrame
            The transformed data.
        """
        return x.iloc[:, self.selected_features]


if __name__ == '__main__':
    # Example usage
    X = pd.DataFrame({'a': [0, 0, 0, 1], 'b': [2, 1, 1, 1], 'c': [0, 4, 1, 1], 'd': [3, 3, 2, 3]})
    y = pd.Series([0, 1, 0, 1])
    selector = SelectKBest(k=2)
    X = selector.fit_transform(X, y)
    print(X)
