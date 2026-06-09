'''
XGBoost costruisce alberi in sequenza (serie), non in parallelo
Ogni nuovo albero non rifà la previsione, ma corregge gli errori dei precedenti
Il modello finale è la somma delle correzioni di tutti gli alberi
Il contributo di ogni albero è controllato dal learning rate  -> la nuova predict = old_predict + learning_rate*correzione
Risultato: modello progressivamente più preciso ad ogni iterazione
'''

from xgboost import XGBClassifier
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, \
    classification_report


class XGBoostModel:

    def __init__(self, n_estimators=200, max_depth=4, learning_rate=0.1, random_state=42):

        self.model = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=random_state,
            eval_metric='logloss'
        )

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)


    def metrics(self, X, y):

        y_pred = self.predict(X)

        #return {
        #    "accuracy": accuracy_score(y, y_pred),
        #    "precision": precision_score(y, y_pred),
        #    "recall": recall_score(y, y_pred),
        #    "f1_score": f1_score(y, y_pred)
        #}

        print("--- Risultati del Modello ---")
        print(f"Accuracy:  {accuracy_score(y, y_pred):.4f}")
        print(f"Precision: {precision_score(y, y_pred):.4f}")
        print(f"Recall:    {recall_score(y, y_pred):.4f}")
        print(f"F1 Score:  {f1_score(y, y_pred):.4f}")
        print("-----------------------------")
