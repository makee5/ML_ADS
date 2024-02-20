from typing import Union

import numpy as np
import pandas as pd

from ml_from_scratch.transformation import Transformer


class Standardization(Transformer):
    """
    Standardize the data to have a mean of 0 and a standard deviation of 1.

    Attributes
    ----------
    mean : pd.Series
        The means of the columns.
    std : pd.Series
        The standard deviations of the columns.

    Example usage:
    ```python
    # Example usage
    X = pd.DataFrame({'a': [1, 2, 3, 4, 5], 'b': [1, 2, 3, 4, 5], 'c': [1, 2, 3, 4, 5]})
    scaler = Standardization()
    X = scaler.fit_transform(X)
    ```
    """
    mean = None
    std = None

    def _fit(self, X: pd.DataFrame, y: Union[pd.DataFrame, pd.Series] = None) -> 'Standardization':
        """
        Fit the scaler to the data.

        Parameters
        ----------
        X : pd.DataFrame
            The data to scale.
        y : pd.DataFrame or pd.Series, optional
            Ignored.
        """
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        return self

    def _transform(self, X: pd.DataFrame, y: Union[pd.DataFrame, pd.Series] = None) -> pd.DataFrame:
        """
        Scale the data.

        Parameters
        ----------
        X: pd.DataFrame
            The data to scale.
        y: pd.DataFrame or pd.Series, optional
            Ignored.

        Returns
        -------
        pd.DataFrame
            The scaled data.
        """
        return (X - self.mean) / self.std

    def inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Inverse scale the data.

        Parameters
        ----------
        X: pd.DataFrame
            The data to inverse scale.
        y: pd.DataFrame, optional
            Ignored.

        Returns
        -------
        pd.DataFrame
            The inverse scaled data.
        """
        return X * self.std + self.mean


if __name__ == '__main__':
    # Example usage
    X = pd.DataFrame({'a': [1, 2, 3, 4, 5], 'b': [1, 2, 3, 4, 5], 'c': [1, 2, 3, 4, 5]})
    print(X)
    scaler = Standardization()
    scaler.fit(X)
    print(scaler.mean, scaler.std)
    print(scaler.transform(X))
    print(scaler.inverse_transform(scaler.transform(X)))
