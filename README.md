# Titanic Survival

Titanic Survival è un'API in Flask 
progettata per analizzare il [dataset Titanic](https://www.kaggle.com/competitions/titanic/data) e 
prevedere se un passeggero del Titanic è sopravvissuto o meno alla tragedia
utilizzando modelli di Machine Learning e fine-tuning.

Il lavoro include una pipeline completa di preprocessing dei dati (One-Hot Encoding e Label Encoding), 
analisi esplorativa del dataset,
training di modelli supervisionati per la classificazione binaria (Logistic Regression, XGBoost),
e tuning degli iperparametri (GridSearchCV).

## Funzionalità Principali

- **Analisi del dataset**: 
Esplorazione delle feature numeriche e categoriche del dataset Titanic, 
analisi delle distribuzioni e delle frequenze delle classi, 
studio delle relazioni tra variabili (EDA)
- **Preprocessing dei dati**: 
Gestione dei valori mancanti,
One-Hot e Label Encoding delle variabili categoriche,
preparazione del dataset per modelli ML
- **Utilizzo di modelli di Machine Learning**:
Addestramento di modelli di Logistic Regression e XGBoost,
con valutazione delle performance tramite metriche standard (Accuracy, Precision, Recall, F1-score)
- **Tuning degli Ipermparametri**: 
Utiizzo di GridSearchCV per trovare il valore di C, la penalty e il solver. 
Visualizzazione delle metriche standard migliorare con il processo di fine-tuning

## Avvio di Titanic Survival

### Requisiti

Prima di avviare Titanic Survival, assicurati di avere installato tutte le dipendenze necessarie. Puoi trovarle nel file `requirements.txt`.

### Avvio in locale

Per visualizzare l'API in locale è necessario recarsi al seguente indirizzo:

   ```
   http://127.0.0.1:5000/
   ```
## Avvio con Docker

### Prerequisiti

Assicurati di avere installato Docker. Puoi scaricare l'applicazione [qui](https://www.docker.com/products/docker-desktop/).

Verifica l’installazione con:

```bash
docker --version
```

### Utilizzo di Docker

1. Costruisci l'immagine Docker da linea di comando:
   ```bash
   docker build -t titanic-survival .
   ```
   
2. Si possono controllare le informazioni dell'immagine appena creata con il comando:
   ```bash
   docker image ls
   ```

3. Costruisci e avvia il container da linea di comando (copiando questo comando, verrà chiamato "titanic"):
   ```bash
   docker run -d --name titanic -p 5000:5000 titanic-survival
   ```
   Se l'operazione è andata a buon fine è possibile vedere lo stavo attivo del container tramite il comando:
   ```bash
   docker ps
   ```

4. Accedi all'applicazione (nella sua route home) tramite il tuo browser all'indirizzo:
   ```
   http://127.0.0.1:5000/
   ```
## Utilizzo dell'API

L'API è consultabile direttamente da browser.

Se dovesse servire, è possibile installare dei plug-in, tra cui:

- Rest-Client (Chrome): [download](https://chromewebstore.google.com/detail/rest-client/oienkoejnhkbcibhdnpjoemdnmiokgah)
- Rested (Firefox): [download](https://addons.mozilla.org/en-US/firefox/addon/rested/)

L'utilizzo dei plug-in non permette però la restituzione delle immagini, facendo risultare "strana" la risposta di alcuni endpoint

## API Endpoints di Titanic Survival

TS è provvisto di diversi endpoint GET, consultabili nella route **home**.
```
http://127.0.0.1:5000/
```
Le funzionalità dell'API sono le seguenti:

### Missing Values

Endpoint che permette di visualizzare la quantità di missing values presenti nel dataset
```
http://127.0.0.1:5000/api/missing-values
```

### Outliers

Endpoint che restituisce la quantità e la percentuale di outliers presenti nel dataset
```
http://127.0.0.1:5000/api/outliers
```

### Cleaning Data

Endpoint che permette di pulire il dataset riempiendo i valori nulli e droppando le colonne in eccesso per la creazione dei modelli
```
http://127.0.0.1:5000/api/cleaning-data
```

### Plot Survival Stats

Endpoint che restituisce i grafici relativi alla sopravvienza (o meno) degli ospiti sul titanic
```
http://127.0.0.1:5000/api/plot-survival-stats
```

### Plot Distribution

Endpoint che restituisce la distribuzione della feature passata in input dopo l'ultimo " / "
```
http://127.0.0.1:5000/api/plot-distribution/&lt;feature&gt;
```

Un esempio corretto è il seguente:
```
http://127.0.0.1:5000/api/plot-distribution/Age
```

### Plot Outliers

Endpoint che restituisce il boxplot della feature passata in input dopo l'ultimo " / "
```
http://127.0.0.1:5000/api/plot-outliers/&lt;feature&gt;
```

Un esempio corretto è il seguente:
```
http://127.0.0.1:5000/api/plot-outliers/Fare
```

### Correlation Matrix

Endpoint che restituisce la matrice di correlazione delle features
```
http://127.0.0.1:5000/api/correlation-matrix
```

### Logistic Regression Naive

Endpoint che stampa a video le metriche della logistic regression eseguita sul dataset ripulito
```
http://127.0.0.1:5000/api/logistic-regression-naive
```
### Logistic Regression Grid

Endpoint che stampa a video le metriche della logistic regression con GridSearchCV eseguita sul dataset ripulito
```
http://127.0.0.1:5000/api/logistic-regression-grid
```

### XGBoost

Endpoint che stampa a video le metriche di XGBoost eseguito sul dataset ripulito
```
http://127.0.0.1:5000/api/XGBoost
```

### XGBoost Encoded

Endpoint che potenzia quello precedente, andando ad applicare robust scaling sugli outliers
```
http://127.0.0.1:5000/api/XGBoost-encoded
```