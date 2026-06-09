from it.valtellina_akron.analyzer.analyzer import Analyzer
from it.valtellina_akron.graphs.graphs import Graphs
from it.valtellina_akron.one_hot_encoder.one_hot_encoder import OHEncoder

# creiamo oggetto che contiene il df
tit = Analyzer("data/train.csv",    "data/test.csv")
df = tit.df #df -> train
tit.overview() #size: 891*12
#tit.missing_values() # 177 age, 687 cabin, 2 embarked
tit.remove_duplicates() # no duplicati

# grafici distribuzione
grf = Graphs(tit.df)
#grf.survival_by_feature()
#grf.plot_distribution("Age") #--> no simmetrica
#grf.plot_outliers("Age")

# rimozione features non significative: id, ticket number, cabin (troppi null), embarked
# name solo dopo aver estratto i titoli
tit.drop_columns(["PassengerId", "Ticket", "Cabin", "Embarked"])


tit.count_outliers()
tit.detect_outliers("Age") # 11 outliers, 1.23%, ma valori reali. quindi si mantengono

tit.dagostino_noramlity("Age") #--> conferma non normalita allora imputo mediana

# imputiamo l'eta calcolando la mediana raggruppando per titolo e classe


# 1. Prima estrai i titoli (crea la colonna 'Title')
tit.title_extraction()

# 2. Poi gestisci i titoli rari (opzionale, ma consigliato)
tit.rare_groups()

# 3. Infine, imputa l'età (che ora troverà la colonna 'Title')
tit.impute_age_medians()
#-------- AGE PRONTA ---------
# ora possiamo droppare anche il nome
tit.drop_columns(['Name'])

# label encoding per il sesso
tit.sex_encoding()

# creazione family_size che combina fratelli/spose(SibSp) e relazioni familiari(Parch)
tit.family_size()

# OHE per il titolo
ohe = OHEncoder(tit.df, tit.test)
tit.df, tit.test = ohe.encode('Title')

print("Nuove colonne nel train:")
print(tit.df.head())


#tit.impute_age_medians()

#title = tit.title_extraction()
#print(title)

#def_title, survival_by_title = tit.rare_groups()
#print(def_title)
#print(survival_by_title)



