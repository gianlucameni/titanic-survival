from sklearn.preprocessing import OneHotEncoder
import pandas as pd

class OHEncoder:
    def __init__(self, X_train, X_test):
        self.X_train = X_train
        self.X_test = X_test
        self.encoders = {}

    def encode(self, col):
        # Encoder
        if col not in self.encoders:
            self.encoders[col] = OneHotEncoder(
                drop='first',
                sparse_output=False,
                handle_unknown='ignore'
            )
            # 2. Fit SOLO sul training set
            self.encoders[col].fit(self.X_train[[col]])

        # Trasformazione (usiamo l'encoder già 'fittato')
        encoded_train = self.encoders[col].transform(self.X_train[[col]])
        encoded_test = self.encoders[col].transform(self.X_test[[col]])

        # Estraiamo i nomi delle nuove colonne ('Title_Miss', 'Title_Mr')
        feature_names = self.encoders[col].get_feature_names_out([col])

        df_train_enc = pd.DataFrame(encoded_train, columns=feature_names, index=self.X_train.index)
        df_test_enc = pd.DataFrame(encoded_test, columns=feature_names, index=self.X_test.index)

        # Concateniamo i nuovi dati e rimuoviamo la colonna originale
        self.X_train = pd.concat([self.X_train.drop(columns=[col]), df_train_enc], axis=1)
        self.X_test = pd.concat([self.X_test.drop(columns=[col]), df_test_enc], axis=1)

        return self.X_train, self.X_test

    def transform_only(self, col_name, new_data):
        """
        Applica l'encoding a nuovi dati usando l'encoder già fittato sul train.
        """
        if col_name not in self.encoders:
            raise ValueError(f"L'encoder per {col_name} non è stato ancora fittato!")

        # Trasforma solo i nuovi dati
        transformed_data = self.encoders[col_name].transform(new_data[[col_name]])

        # Crea il DataFrame con gli stessi nomi di colonne del train
        feature_names = self.encoders[col_name].get_feature_names_out([col_name])
        df_encoded = pd.DataFrame(
            transformed_data,
            columns=feature_names,
            index=new_data.index
        )

        # Concatenazione e rimozione della colonna originale
        new_data = pd.concat([new_data.drop(columns=[col_name]), df_encoded], axis=1)

        return new_data




    def stampa(self, df):
        print(df.head())