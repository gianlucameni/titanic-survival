from flask import Flask, jsonify, request, send_file
from it.valtellina_akron.analyzer.analyzer import Analyzer
from it.valtellina_akron.graphs.graphs import Graphs
from it.valtellina_akron.preprocessor.preprocessor import Preprocessor
from it.valtellina_akron.splitter_train_test.splitter_train_test import SplitterTrainTest
from it.valtellina_akron.one_hot_encoder.one_hot_encoder import OHEncoder
from it.valtellina_akron.predictors.logistic_regression import LogRegression
from it.valtellina_akron.predictors.xgboost_model import XGBoostModel

app = Flask(__name__)

@app.route('/')
def intro():
    return """
        <h1>Titanic Survival Analyzer API</h1>

        <p>
            REST API per l'analisi del dataset Titanic e la valutazione
            di modelli di Machine Learning per la previsione della sopravvivenza.
        </p>

        <h2>Available Endpoints</h2>

        <h3>Dataset Analysis</h3>
        <ul>
            <li>GET /api/missing-values</li>
            <li>GET /api/outliers</li>
            <li>GET /api/cleaning-data</li>
        </ul>

        <h3>Plots</h3>
        <ul>
            <li>GET /api/plot-survival-stats</li>
            <li>GET /api/plot-distribution/&lt;feature&gt;</li>
            <li>GET /api/plot-outliers/&lt;feature&gt;</li>
            <li>GET /api/correlation-matrix</li>
        </ul>

        <h3>Machine Learning Models</h3>
        <ul>
            <li>GET /api/logistic-regression-naive</li>
            <li>GET /api/logistic-regression-grid</li>
            <li>GET /api/XGBoost</li>
            <li>GET /api/XGBoost-encoded</li>
        </ul>

        <h2>Examples</h2>
        <pre>
    GET /api/plot-distribution/Age

    GET /api/plot-outliers/Fare

    GET /api/logistic-regression-grid
        </pre>
        """

@app.route('/api/missing-values')
def missing_values():
    tit = Analyzer("data/train.csv",    "data/test.csv")
    return jsonify({"missing values:": tit.missing_values()})

@app.route('/api/outliers')
def outliers():
    tit = Analyzer("data/train.csv", "data/test.csv")

    return jsonify({
        "Age": tit.detect_outliers("Age"),
        "Fare": tit.detect_outliers("Fare")
    })

@app.route('/api/cleaning-data')
def cleaning_data():
    tit = Analyzer("data/train.csv",    "data/test.csv")

    # uso la mediana e i titoli delle persone per imputare l'età e il costo del biglietto
    tit.title_extraction()
    tit.rare_groups()
    tit.impute_age_medians()

    tit.impute_fare()
    # rende fare meno asimmetrica (normalization tra 0 e 1)
    tit.normalization("Fare")

    # drop colonne non utili per predictions
    tit.drop_columns(["PassengerId", "Name", "Ticket", "Cabin", "Embarked"])

    #encoding feature
    # label encoding per il sesso
    tit.sex_encoding()

    # creazione family_size che combina fratelli/spose(SibSp) e relazioni familiari(Parch)
    tit.family_size()
    tit.drop_columns(['SibSp', 'Parch'])

    result = {
        "status": "success",
        "train_shape": tit.df.shape,
        "test_shape": tit.test.shape,
        "missing_train_after": tit.df.isnull().sum().to_dict(),
        "missing_test_after": tit.test.isnull().sum().to_dict()
    }

    return jsonify(result)

@app.route('/api/plot-survival-stats')
def plot_survival_stats():
    tit = Analyzer("data/train.csv", "data/test.csv")
    grf = Graphs(tit.df)
    img = grf.survival_by_feature()
    return send_file(img, mimetype='image/png')

@app.route('/api/plot-distribution/<feature>')
def plot_distribution(feature): # distribuzione dopo la manipolazione del dataset
    tit = Analyzer("data/train.csv", "data/test.csv")

    tit.drop_columns(["PassengerId", "Name", "Ticket", "Cabin", "Embarked"])
    tit.sex_encoding()
    tit.family_size()
    tit.drop_columns(['SibSp', 'Parch'])

    if feature not in tit.df.columns:
        return jsonify({"message": "feature non presente"})

    grf = Graphs(tit.df)
    img = grf.plot_distribution(feature)

    return send_file(img, mimetype='image/png')

@app.route('/api/plot-outliers/<feature>')
def plot_outliers(feature):
    tit = Analyzer("data/train.csv", "data/test.csv")

    tit.drop_columns(["PassengerId", "Name", "Ticket", "Cabin", "Embarked"])
    tit.sex_encoding()
    tit.family_size()
    tit.drop_columns(['SibSp', 'Parch'])

    if feature not in tit.df.columns:
        return jsonify({"message": "feature non presente"})

    grf = Graphs(tit.df)
    img = grf.plot_outliers(feature)

    return send_file(img, mimetype='image/png')

@app.route('/api/correlation-matrix')
def correlation_matrix():
    tit = Analyzer("data/train.csv", "data/test.csv")

    grf = Graphs(tit.df)
    img = grf.correlation_matrix()

    return send_file(img, mimetype='image/png')


@app.route('/api/logistic-regression-naive')
def logistic_regression_naive():
    tit = Analyzer("data/train.csv", "data/test.csv")

    tit.title_extraction()
    tit.rare_groups()
    tit.impute_age_medians()

    tit.impute_fare()
    tit.normalization("Fare")
    print("prima")
    print(tit.df.columns)
    tit.drop_columns(["PassengerId", "Name", "Ticket", "Cabin", "Embarked"])
    tit.sex_encoding()
    tit.family_size()
    tit.drop_columns(['SibSp', 'Parch'])

    print("dopo")
    print(tit.df.columns)

    splitter = SplitterTrainTest(tit.df, "Survived")
    X_train, X_test, y_train, y_test = splitter.split()

    # OHE per il titolo
    ohe = OHEncoder(X_train, X_test)
    X_train_encoded, X_test_encoded = ohe.encode('Title')
    # la colonna Master non è codificata perche si ricava dalle altre
    # evita multicollinearità perfetta

    test_econded = ohe.transform_only('Title', tit.test)

    # Robust Scaler su AGE e FARE perchè contengono outliers (necessario per la regressione)
    preprocessor = Preprocessor()

    X_train_processed = preprocessor.fit_transform(X_train_encoded)
    X_test_processed = preprocessor.transform(X_test_encoded)
    test_processed = preprocessor.transform(test_econded)

    # applichiamo la regressione logistica
    log_reg_model = LogRegression()

    log_reg_model.fit(X_train_processed, y_train)

    y_pred = log_reg_model.predict(X_train_processed)

    metrics = log_reg_model.metrics(X_test_processed, y_test)

    return jsonify({"metriche": metrics})


@app.route('/api/logistic-regression-grid')
def logistic_regression_grid():
    tit = Analyzer("data/train.csv", "data/test.csv")

    tit.title_extraction()
    tit.rare_groups()
    tit.impute_age_medians()

    tit.impute_fare()
    tit.normalization("Fare")

    tit.drop_columns(["PassengerId", "Name", "Ticket", "Cabin", "Embarked"])
    tit.sex_encoding()
    tit.family_size()
    tit.drop_columns(['SibSp', 'Parch'])


    splitter = SplitterTrainTest(tit.df, "Survived")
    X_train, X_test, y_train, y_test = splitter.split()

    # OHE per il titolo
    ohe = OHEncoder(X_train, X_test)
    X_train_encoded, X_test_encoded = ohe.encode('Title')
    # la colonna Master non è codificata perche si ricava dalle altre
    # evita multicollinearità perfetta

    test_econded = ohe.transform_only('Title', tit.test)

    # Robust Scaler su AGE e FARE perchè contengono outliers (necessario per la regressione)
    preprocessor = Preprocessor()

    X_train_processed = preprocessor.fit_transform(X_train_encoded)
    X_test_processed = preprocessor.transform(X_test_encoded)
    test_processed = preprocessor.transform(test_econded)

    # applichiamo la regressione logistica
    log_reg_model = LogRegression(max_iter=5000)

    # uso gridsearch
    log_reg_model.grid_search(X_train_processed, y_train)

    y_pred = log_reg_model.predict(X_train_processed)

    metrics = log_reg_model.metrics(X_test_processed, y_test)

    return jsonify({"metriche": metrics})


@app.route('/api/XGBoost')
def xgboost():
    tit = Analyzer("data/train.csv", "data/test.csv")

    tit.title_extraction()
    tit.rare_groups()
    tit.impute_age_medians()

    tit.impute_fare()
    tit.normalization("Fare")

    tit.drop_columns(["PassengerId", "Name", "Ticket", "Cabin", "Embarked"])
    tit.sex_encoding()
    tit.family_size()
    tit.drop_columns(['SibSp', 'Parch'])

    splitter = SplitterTrainTest(tit.df, "Survived")
    X_train, X_test, y_train, y_test = splitter.split()

    # OHE per il titolo
    ohe = OHEncoder(X_train, X_test)
    X_train_encoded, X_test_encoded = ohe.encode('Title')
    # la colonna Master non è codificata perche si ricava dalle altre
    # evita multicollinearità perfetta

    test_econded = ohe.transform_only('Title', tit.test)

    # applichiamo XGBoost con OHE
    xgb_model = XGBoostModel()

    xgb_model.fit(X_train_encoded, y_train)

    y_pred_xgb = xgb_model.predict(X_test_encoded)

    metrics_xgb = xgb_model.metrics(X_test_encoded, y_test)

    return jsonify({"metriche": metrics_xgb})


@app.route('/api/XGBoost-encoded')
def xgboost_encoded():
    tit = Analyzer("data/train.csv", "data/test.csv")

    tit.title_extraction()
    tit.rare_groups()
    tit.impute_age_medians()

    tit.impute_fare()
    tit.normalization("Fare")

    tit.drop_columns(["PassengerId", "Name", "Ticket", "Cabin", "Embarked"])
    tit.sex_encoding()
    tit.family_size()
    tit.drop_columns(['SibSp', 'Parch'])

    splitter = SplitterTrainTest(tit.df, "Survived")
    X_train, X_test, y_train, y_test = splitter.split()

    # OHE per il titolo
    ohe = OHEncoder(X_train, X_test)
    X_train_encoded, X_test_encoded = ohe.encode('Title')
    # la colonna Master non è codificata perche si ricava dalle altre
    # evita multicollinearità perfetta

    test_econded = ohe.transform_only('Title', tit.test)

    # Robust Scaler su AGE e FARE perchè contengono outliers (necessario per la regressione)
    preprocessor = Preprocessor()

    X_train_processed = preprocessor.fit_transform(X_train_encoded)
    X_test_processed = preprocessor.transform(X_test_encoded)
    test_processed = preprocessor.transform(test_econded)

    # applichiamo XGBoost con OHE
    xgb_model = XGBoostModel()

    xgb_model.fit(X_train_processed, y_train)

    y_pred_xgb = xgb_model.predict(X_test_processed)

    metrics_xgb = xgb_model.metrics(X_test_processed, y_test)

    return jsonify({"metriche": metrics_xgb})


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)