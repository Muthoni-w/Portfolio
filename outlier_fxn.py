#import the necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Read in the data
df = pd.read_csv('titanic_train.csv')
df
x = df['Fare']


def detect_outliers(x, iqr_multiplier = 1.5, how = "both"):
    """
    Used to detect outliers using the 1.5 IQR (Inner Quartile Range) Method. 

    Args:
        x (Pandas Series): 
            A numeric pandas series. 
        
        iqr_multiplier (int, float, optional): 
            A multiplier used to modify the IQR sensitivity. 
            Must be positive. Lower values will add more outliers. 
            Larger values will add fewer outliers. Defaults to 1.5.

        how (str, optional): 
            One of "both", "upper" or "lower". Defaults to "both".
            - "both": flags both upper and lower outliers.
            - "lower": flags lower outliers only.
            - "upper": flags upper outliers only. 

    Returns:
        [Pandas Series]: A Boolean Series that flags outliers as True/False.
    """

    # CHECKS
    if type(x) is not pd.Series:
        raise Exception("`x` must be a Pandas Series.")


    if not isinstance(iqr_multiplier, (float, int)):
        raise Exception("`iqr_multiplier` must be an int or float.")
    if iqr_multiplier <= 0:
        raise Exception("`iqr_multiplier` must be a positive value.")

    
    how_options = ['both', 'upper', 'lower']
    if how not in how_options:
        raise Exception(
            f"Invalid `how`. Expected one of {how_options}"
        )

    # IQR LOGIC 

    q75 = np.quantile(x, 0.75)
    q25 = np.quantile(x, 0.25)
    iqr = q75 - q25

    lower_limit = q25 - iqr_multiplier * iqr
    upper_limit = q75 + iqr_multiplier * iqr

    outliers_upper = x >= upper_limit
    outliers_lower = x <= lower_limit

    if how == "both":
        outliers = outliers_upper | outliers_lower
    elif how == "lower":
        outliers = outliers_lower
    else:
        outliers = outliers_upper

    return outliers

# detect_outliers and filter the dataframe to retain records that lie in the permissible range.
upper = df[ detect_outliers(df['Fare'], iqr_multiplier=0.3,how = "upper")]