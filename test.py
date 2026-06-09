from it.valtellina_akron.analyzer.analyzer import Analyzer

# creiamo oggetto che contiene il df
tit = Analyzer("data/train.csv",    "data/test.csv")
df = tit.train

