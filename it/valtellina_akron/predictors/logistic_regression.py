import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV
import warnings
from sklearn.exceptions import FitFailedWarning

warnings.filterwarnings("ignore", category=FitFailedWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


class LogRegression:
    def __init__(self, C=1.0, max_iter=1000, random_state=0):

        self.C = C # gestisce regolarizzazione dei pesi
        self.max_iter = max_iter # numero iterazioni
        self.random_state = random_state # serve per riproducibilità

        # modello interno sklearn
        self.model = LogisticRegression(
            C=self.C,
            max_iter=self.max_iter,
            random_state=self.random_state,
            solver='saga'
        )

        self.best_model = None
        self.grid = None

    def fit(self, X, y):
        self.model.fit(X, y)
        return self


    # GridSearch
    def grid_search(self, X, y, cv=5, scoring='f1'):
        param_grid = [
            # L1 (NESSUN l1_ratio)
            {
                "solver": ["saga"], # aggiorna pesi gradiente, supporta l1, l2, elasticnet
                "penalty": ["l1"],
                "C": [0.01, 0.1, 1, 10, 100]
            },

            # L2 (NESSUN l1_ratio)
            {
                "solver": ["saga"],
                "penalty": ["l2"],
                "C": [0.01, 0.1, 1, 10, 100]
            },

            # ElasticNet (SOLO QUI l1_ratio)
            {
                "solver": ["saga"],
                "penalty": ["elasticnet"],
                "l1_ratio": [0.1, 0.5, 0.9],
                "C": [0.01, 0.1, 1, 10, 100]
            }
        ]

        self.grid = GridSearchCV(
            LogisticRegression(max_iter=self.max_iter, random_state=self.random_state, solver="saga"),
            param_grid=param_grid,
            cv=cv,
            scoring=scoring,
            n_jobs=-1
        )

        self.grid.fit(X, y)

        self.best_model = self.grid.best_estimator_

        return self.grid.best_params_, self.grid.best_score_


    def predict(self, X):
        return self.best_model.predict(X)

    def predict_probs(self, X):
        return self.best_model.predict_proba(X) # restituisce il valore (es. 0.92)

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

