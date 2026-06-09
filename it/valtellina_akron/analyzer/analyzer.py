import pandas as pd

class Analyzer:
    def __init__(self, train_path, test_path):
        self.train = pd.read_csv(train_path)
        self.test = pd.read_csv(test_path)