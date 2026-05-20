import pandas as pd
import numpy as np

class DataHandler:
    def __init__(self, file_path="data/AmesHousing.csv"):
        self.file_path = file_path
        self.x = None
        self.y = None

    def load_and_preprocess(self):
        df = pd.read_csv(self.file_path)
        x_raw = df['Gr Liv Area'].values 
        y_raw = df['SalePrice'].values
        
        self.x = (x_raw - np.mean(x_raw)) / np.std(x_raw)
        self.y = (y_raw - np.mean(y_raw)) / np.std(y_raw)
        
        return self.x, self.y