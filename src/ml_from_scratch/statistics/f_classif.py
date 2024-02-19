from typing import Tuple

import numpy as np
import pandas as pd
from scipy.stats import f_oneway


def f_classif(X: pd.DataFrame, y: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the ANOVA F-value for the provided data.

    Parameters
    ----------
    X : pd.DataFrame
        The data to test.
    y : pd.DataFrame
        The target variable.

    Returns
    -------
    np.ndarray
        The F-values.
    np.ndarray
        The p-values.
    """
    classes = y.unique()
    group_data = [X[y == c] for c in classes]
    f_values, p_values = f_oneway(*group_data)
    return f_values, p_values
