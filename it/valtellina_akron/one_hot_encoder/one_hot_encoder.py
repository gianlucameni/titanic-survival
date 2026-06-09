from sklearn.preprocessing import OneHotEncoder
import pandas as pd

class OHEncoder:
    def __init__(self, X_train, X_test):
        self.X_train = X_train
        self.X_test = X_test
        
    def encode(self, col):
        # Encoder
        encoder = OneHotEncoder(
            drop='first',  #elimina una dummy per ogni variabile categorica
                            # evitando multicollinearità nella regressione logistica.
            sparse_output=False,
            handle_unknown='ignore' # evita errori se nel test compare
                                    # una categoria non presente nel training set.
        )

        # Fit sull'intero train, transform su entrambi
        encoded_train = encoder.fit_transform(self.X_train[[col]])
        encoded_test = encoder.transform(self.X_test[[col]])

        # Estraiamo i nomi delle nuove colonne ('Title_Miss', 'Title_Mr')
        feature_names = encoder.get_feature_names_out([col])

        df_train_enc = pd.DataFrame(encoded_train, columns=feature_names, index=self.X_train.index)
        df_test_enc = pd.DataFrame(encoded_test, columns=feature_names, index=self.X_test.index)

        # Concateniamo i nuovi dati e rimuoviamo la colonna originale
        self.X_train = pd.concat([self.X_train.drop(columns=[col]), df_train_enc], axis=1)
        self.X_test = pd.concat([self.X_test.drop(columns=[col]), df_test_enc], axis=1)

        return self.X_train, self.X_test


    def stampa(self, df):
        print(df.head())