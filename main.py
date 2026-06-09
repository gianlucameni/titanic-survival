from flask import Flask, jsonify, request, send_file
from it.valtellina_akron.analyzer.analyzer import Analyzer
from it.valtellina_akron.graphs.graphs import Graphs
from it.valtellina_akron.splitter_train_test.splitter_train_test import SplitterTrainTest
from it.valtellina_akron.one_hot_encoder.one_hot_encoder import OHEncoder
from it.valtellina_akron.predictors.logistic_regression import LogRegression
from it.valtellina_akron.predictors.xgboost_model import XGBoostModel

app = Flask(__name__)

@app.route('/')
def intro():
    return """
    <h1>Titanic Survival Analyzer API</h1>

    <h2>Available Endpoints</h2>
    
    # TO-DO
    """

@app.route('/api/missing-values')
def missing_values():
    tit = Analyzer("data/train.csv",    "data/test.csv")
    return jsonify({"missing values:": tit.missing_values()})

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





    return jsonify({
        "missing values": msr.missing_values(),
        "duplicates": msr.remove_duplicates()
    })



if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)