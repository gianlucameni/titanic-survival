from sklearn.preprocessing import RobustScaler

class Preprocessor:

    def __init__(self):
        self.scaler = RobustScaler()

    def fit_transform(self, X_train):
        cols = ['Age', 'Fare']

        X_train = X_train.copy()
        X_train[cols] = self.scaler.fit_transform(X_train[cols])

        return X_train

    def transform(self, X_test):
        cols = ['Age', 'Fare']

        X_test = X_test.copy()
        X_test[cols] = self.scaler.transform(X_test[cols])

        return X_test