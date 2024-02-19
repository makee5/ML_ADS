from typing import Union

import pandas as pd

from ml_from_scratch.transformation import Transformer


class VarianceThreshold(Transformer):
    """
    Remove features with variance below a certain threshold.

    Parameters
    ----------
    threshold : float, optional
        The threshold below which to remove features.
        The default is 0.0.

    Attributes
    ----------
    variances : pd.Series
        The variances of the columns.

    Example usage:
    ```python
    # Example usage
    X = pd.DataFrame({'a': [1, 2, 3, 4, 5], 'b': [1, 2, 3, 4, 5], 'c': [1, 1, 1, 1, 1]})
    selector = VarianceThreshold(threshold=0.1)
    X = selector.fit_transform(X)
    """

    def __init__(self, threshold: float = 0.0):
        """
        Remove features with variance below a certain threshold.
        """
        self.threshold = threshold
        self.variances = None

    def _fit(self, x: pd.DataFrame, y: Union[pd.DataFrame, pd.Series] = None) -> 'VarianceThreshold':
        """
        Fit the transformer to the data.
        Fit the transformer to the data.
        Parameters
        ----------
        x : pd.DataFrame
            The data to fit the transformer to.
        y : pd.DataFrame or pd.Series, optional
            Ignored.
        """
        self.variances = x.var()
        return self

    def _transform(self, x: pd.DataFrame, y: Union[pd.DataFrame, pd.Series] = None) -> pd.DataFrame:
        """
        Transform the data.
        Parameters
        ----------
        x : pd.DataFrame
            The data to transform.
        y : pd.DataFrame or pd.Series, optional
            Ignored.
        Returns
        -------
        pd.DataFrame
            The transformed data.
        """
        return x.loc[:, self.variances > self.threshold]


if __name__ == '__main__':
    # Example usage
    X = pd.DataFrame({'a': [1, 2, 3, 4, 5], 'b': [1, 2, 3, 4, 5], 'c': [1, 1, 1, 1, 1]})
    selector = VarianceThreshold(threshold=0.1)
    selector.fit(X)
    print(selector.variances)
    X = selector.transform(X)
    print(X)
