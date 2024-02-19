from typing import Union

import numpy as np
import pandas as pd

from ml_from_scratch.transformation import Transformer


class MeanImputer(Transformer):
    """
    Impute missing values with the mean of the column.

    Attributes
    ----------
    means : pd.Series
        The means of the columns.

    Example usage:
    ```python
    # Example usage
    X = pd.DataFrame({'a': [np.nan, 2, 3, 4, 5], 'b': [1, np.nan, 3, 4, 5], 'c': [1, 2, np.nan, 4, 5]})
    imputer = MeanImputer()
    X = imputer.fit_transform(X)
    ```
    """
    means = None

    def _fit(self, X: pd.DataFrame, y: Union[pd.DataFrame, pd.Series] = None) -> 'MeanImputer':
        """
        Fit the imputer to the data.

        Parameters
        ----------
        X : pd.DataFrame
            The data to impute.
        y : pd.DataFrame or pd.Series, optional
            Ignored.
        """
        self.means = X.mean()
        return self

    def _transform(self, X: pd.DataFrame, y: Union[pd.DataFrame, pd.Series] = None) -> pd.DataFrame:
        """
        Impute the data.

        Parameters
        ----------
        X : pd.DataFrame
            The data to impute.
        y : pd.DataFrame or pd.Series, optional
            Ignored.

        Returns
        -------
        pd.DataFrame
            The imputed data.
        """
        return X.fillna(self.means)


if __name__ == '__main__':
    # Example usage
    X = pd.DataFrame({'a': [np.nan, 2, 3, 4, 5], 'b': [1, np.nan, 3, 4, 5], 'c': [1, 2, np.nan, 4, 5]})
    print(X)
    imputer = MeanImputer()
    imputer.fit(X)
    print(f"Means:\n{imputer.means}")
    print(imputer.transform(X))
