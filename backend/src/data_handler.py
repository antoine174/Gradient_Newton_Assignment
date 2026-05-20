import pandas as pd
import numpy as np

class DataHandler:
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.x = None # GrLivArea
        self.y = None # SalePrice
        self.x_mean, self.x_std = 0, 1
        self.y_mean, self.y_std = 0, 1

    def load_and_preprocess(self):
        # TODO (Antoine): Load the CSV using pandas.
        # TODO (Antoine): Extract 'GrLivArea' as x and 'SalePrice' as y.
        # TODO (Antoine): Implement Z-score normalization: (val - mean) / std.
        # Store the means and stds so we can un-normalize later for the report if needed.
        # Note: Do not use sklearn's StandardScaler, write the basic math.
        pass

    def get_data(self):
        # TODO (Antoine): Return the normalized x and y as numpy arrays.
        return self.x, self.y