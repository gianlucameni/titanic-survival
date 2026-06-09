import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class LogRegression:
    def __init__(self, C=1.0, max_iter=1000, random_state=0):

        self.C = C # gestisce regolarizzazione dei pesi
        self.max_iter = max_iter # numero iterazioni
        self.random_state = random_state # serve per riproducibilità

        # modello interno sklearn
        self.model = LogisticRegression(
            C=self.C,
            max_iter=self.max_iter,
            random_state=self.random_state
        )

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_probs(self, X):
        return self.model.predict_proba(X) # restituisce il valore (es. 0.92)

    def metrics(self, X, y):
        # predizioni
        y_pred = self.predict(X) # restituisce la classe, 0 o 1

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

