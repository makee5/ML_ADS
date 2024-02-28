from typing import Union

import pandas as pd

from ml_from_scratch.transformation import Transformer


class Normalization(Transformer):
    """
    Normalize the data to have a minimum of 0 and a maximum of 1.

    Attributes
    ----------
    min : pd.Series
        The minimums of the columns.
    max : pd.Series
        The maximums of the columns.

    Example usage:
    ```python
    # Example usage
    X = pd.DataFrame({'a': [1, 2, 3, 4, 5], 'b': [2, 4, 6, 8, 10], 'c': [1, 2, 3, 4, 5]})
    scaler = Normalization()
    X = scaler.fit_transform(X)
    ```
    """
    min = None
    max = None

    def _fit(self, X: pd.DataFrame, y: Union[pd.DataFrame, pd.Series] = None) -> 'Normalization':
        """
        Fit the scaler to the data.

        Parameters
        ----------
        X : pd.DataFrame
            The data to scale.
        y : pd.DataFrame, optional
            Ignored.
        """
        self.min = X.min()
        self.max = X.max()
        return self

    def _transform(self, X: pd.DataFrame, y: Union[pd.DataFrame, pd.Series] = None) -> pd.DataFrame:
        """
        Scale the data.

        Parameters
        ----------
        X: pd.DataFrame
            The data to scale.
        y: pd.DataFrame, optional
            Ignored.

        Returns
        -------
        pd.DataFrame
            The scaled data.
        """
        return (X - self.min) / (self.max - self.min)

    def inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Inverse scale the data.

        Parameters
        ----------
        X: pd.DataFrame
            The data to inverse scale.

        Returns
        -------
        pd.DataFrame
            The inverse scaled data.
        """
        return X * (self.max - self.min) + self.min


if __name__ == '__main__':
    # Example usage
    X = pd.DataFrame({'a': [1, 2, 3, 4, 5], 'b': [2, 4, 6, 8, 10], 'c': [0.1, 2, 3.3, 4, 5]})
    scaler = Normalization()
    scaler.fit(X)
    print(scaler.min, scaler.max)
    X = scaler.transform(X)
    print(X)
    X = scaler.inverse_transform(X)
    print(X)
