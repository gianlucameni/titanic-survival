import pandas as pd
from scipy import stats

class Analyzer:
    def __init__(self, train_path, test_path):
        train = pd.read_csv(train_path)
        test = pd.read_csv(test_path)
        self.df = train
        self.test = test


    def overview(self):
        print("Shape:", self.df.shape)
        print("\nInfo:")
        print(self.df.info())

    def drop_columns(self,columns):
        for df in [self.df, self.test]:
            for col in columns:
                df.drop(col, axis=1, inplace=True)

    def remove_duplicates(self):
        before = self.df.shape[0]
        self.df = self.df.drop_duplicates()
        after = self.df.shape[0]
        print(f"Duplicati trovati e rimossi: {before - after}")

    # Missing values
    def missing_values(self):
        #missing = self.df.isnull().sum()
        #total = len(self.df)
        #return {
        #    "missing_values": missing.to_dict(),
        #    "missing_percentage": (missing / total * 100).to_dict()
        #}
        print(self.df.isnull().sum())


    # test normalita
    def dagostino_noramlity(self, col):
        # estraiamo solo i valori validi
        data = self.df[col].dropna()

        # stat: la statistica K^2
        # p: il p-value
        stat, p = stats.normaltest(data)

        # 3. Interpretazione dei risultati
        print(f'Statistica K^2: {stat:.3f}')
        print(f'p-value: {p:.3e}')
        alpha = 0.05
        if p > alpha:
            print("Il campione sembra seguire una distribuzione normale (non possiamo rifiutare H0)")
        else:
            print("Il campione NON segue una distribuzione normale (rifiutiamo H0)")


    def count_outliers(self):
            # seleziono solo le colonne numeriche
            columns = self.df.select_dtypes(include=['number']).columns

            outlier_counts = {}

            for col in columns:
                # Calcolo dei parametri IQR
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1

                # Calcolo dei limiti
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                # Conteggio dei valori che stanno al di fuori dei limiti
                outliers = (self.df[col] < lower_bound) | (self.df[col] > upper_bound)
                outlier_counts[col] = outliers.sum()

            return outlier_counts



    def detect_outliers(self, column):
        Q1 = self.df[column].quantile(0.25)
        Q3 = self.df[column].quantile(0.75)
        IQR = Q3 - Q1

        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        outliers = self.df[
            (self.df[column] < lower) |
            (self.df[column] > upper)
            ]

        print(f"Outliers in {column}: {len(outliers)}")
        count_outliers = len(outliers)
        total = len(self.df)
        perc_outliers = (count_outliers / total) * 100
        print(f"Outliers percentage: {perc_outliers:.2f}%")


    def title_extraction(self):
        for df in [self.df, self.test]:
            df['Title'] = df['Name'].str.extract(r',\s*([^.]+)\.', expand=False).str.strip()

        return self.df['Title'].value_counts()

        print('Train title counts:')
        print(train['Title'].value_counts())

    # Consolidate rare titles into 4 groups
    def rare_groups(self):
        rare_titles = ['Lady', 'Countess', 'the Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona']

        for df in [self.df, self.test]:
            df['Title'] = df['Title'].replace(rare_titles, 'Rare')
            df['Title'] = df['Title'].replace({'Mlle': 'Miss', 'Ms': 'Miss', 'Mme': 'Mrs'}) #Mlle: mademoiselle, Mme: madame

        return self.df['Title'].value_counts(), self.df.groupby('Title')['Survived'].mean().round(2)


    # compute mediana eta raggruppando per classe  e titolo
    def impute_age_medians(self):
        # Calcoliamo le mediane basandoci sul train
        train_medians = self.df.groupby(['Pclass', 'Title'])['Age'].median()
        global_median = self.df['Age'].median()

        # Imputazione sul Train
        self.df['Age'] = self.df['Age'].fillna(
            self.df.groupby(['Pclass', 'Title'])['Age'].transform('median')
        )
        # Se avanzano NaN (combinazioni rare nel train), usiamo la mediana globale
        self.df['Age'] = self.df['Age'].fillna(global_median)

        # Imputazione sul Test
        imputed_values = self.test.set_index(['Pclass', 'Title']).index.map(train_medians)

        self.test['Age'] = self.test['Age'].fillna(pd.Series(imputed_values, index=self.test.index))

        self.test['Age'] = self.test['Age'].fillna(global_median)

    def sex_encoding(self):
        mapping = {'male': 0, 'female': 1}
        for df in [self.df, self.test]:
            df['Sex'] = df['Sex'].map(mapping)

    def family_size(self):
        for df in [self.df, self.test]:
            df['FamilySize'] = df['SibSp'] + df['Parch'] + 1





