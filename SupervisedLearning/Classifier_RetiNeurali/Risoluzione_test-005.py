#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# in un line di excel  trovi delle dichiarazioni e nell’altro trovi le righe delle dichiarazioni.
# In pratica si tratta di trovare quali dati possono non essere attendibili

# In[3]:


# IMPORT DEI FILES IN FORMATO CSV
dichiarazioni = pd.read_csv('Projects/Multitraccia/CSV/Ricevute_AEE.csv')
righe = pd.read_csv('Projects/Multitraccia/CSV/Righe_AEE.csv')


# ID MANCANTE



# print('Dimensione', ricevute.shape)
# print('Numero di NaN per ogni colonna:\n', ricevute.isnull().sum())
# print('\nControllo tipo dato \n', ricevute.dtypes)

# print('\nRIGHE \n')
# print('Dimensione', righe.shape)
# print('\nNumero di NaN per ogni colonna: \n', righe.isnull().sum())
# print('\nControllo tipo dato \n', righe.dtypes)


# In[3]:


# INFO DICHIARAZIONI
# print(dichiarazioni.info())
dichiarazioni.head()
# print('DICHIARAZIONI --> DIMENSIONI {} \n'.format(dichiarazioni.shape))
# print('dtypes\n{} \n'.format(dichiarazioni.dtypes))
# print('Numero di records', dichiarazioni.count())
# print('Numero di NaN per ogni colonna:\n', dichiarazioni.isnull().sum())


# In[4]:


# Il numero dei ID_DICHIARAZIONE nel file delle dichiarazioni corrispondono al numero di records (6273) e non si ripetono
print('Numero totale di ID_DICHIARAZIONE: ', dichiarazioni.ID_DICHIARAZIONE.count())
print(any(dichiarazioni.ID_DICHIARAZIONE.value_counts()>1))


# In[5]:


# INFO RIGHE DICHIARAZIONI
print(righe.info())
# print('RIGHE DICHIARAZIONI --> DIMENSIONI {} \n'.format(righe.shape))
# print('dtypes\n{} \n'.format(righe.dtypes))
# print('Numero di records\n', righe.count())
print('\nNumero di NaN per ogni colonna:\n', righe.isnull().sum())
print(righe.ID_DICHIARAZIONE.value_counts())
print(righe.a_imponibile.sort_values().unique())


# In[6]:


# Il numero dei ID_DICHIARAZIONE nel file RIGHE NON corrisponde al numero di records (6273) e molti si ripetono ma non tutti
print('Numero totale di ID_DICHIARAZIONE: ', righe.ID_DICHIARAZIONE.count())
print('Numero ID_DICHIARAZIONE unici: ', righe.ID_DICHIARAZIONE.nunique())
print('ID_DICHIARAZIONE si ripete ? -->', any(righe.ID_DICHIARAZIONE.value_counts()>1))
print('ID_DICHIARAZIONE si ripete con tutti ? -->', all(righe.ID_DICHIARAZIONE.value_counts()>1))


# In[7]:


# ID NON CONSIDERATO --> ID = 30154 --> FAI LIST(SET(df.ID_DICHIARAZIONE) - SET(ldf.where(df.ID_DICHIARAZIONE.notna())))
# lst = list(df.a_descrizione.unique()) ESEMPIO
ldf = pd.merge(dichiarazioni, righe, how='left', on='ID_DICHIARAZIONE')
# ldf.where(df.ID_DICHIARAZIONE.notna())
ldf.info()


# In[5]:


df = pd.merge(dichiarazioni, righe, on='ID_DICHIARAZIONE')


# In[6]:


# CONTROLLO RIDONDANZA COLONNE
print(np.all(df.a_anno_x ==df.a_anno_y))
print(np.all(df.a_mese_x ==df.a_mese_y))
print(np.all(df.a_trimestre_x ==df.a_trimestre_y))
print(np.all(df.a_cod_comparto_x ==df.a_cod_comparto_y))
# ELIMINO COLONNE SUPERFLUE
delete_col = ['a_anno_y', 'a_mese_y', 'a_trimestre_y', 'a_cod_comparto_y']
rename_col = {'a_anno_x': 'a_anno', 'a_mese_x': 'a_mese', 'a_trimestre_x': 'a_trimestre', 'a_cod_comparto_x': 'a_cod_comparto'}
df = df.drop(delete_col, axis=1)
df = df.rename(columns=rename_col)

# df[['a_cod_categoria', 'a_descrizione']].head()
df.info()


# In[31]:


# istogramma CODICI CON MAGGIOR NUMERO ORDINI
plt.figure(figsize=(10, 6))

# soglia minima vendite analizzate
cutoff = 1e6
mask = df['a_quantita'] > cutoff
# stat = df[mask].groupby(['a_codice_ordine']).agg({'a_quantita': 'sum'})
stat = df[mask].groupby(['a_codice_ordine'])['a_quantita'].sum()
stat.plot.bar()
plt.title('CODICI CON MAGGIOR NUMERO ORDINI')
plt.xlabel('')
plt.ylabel('QUANTITA ORDINI')
plt.legend('')
plt.show()


# In[11]:


# TOTALE ORDINI PER ANNO
explode = (0, 0, 0, 0.1, 0)
stat = df.groupby(['a_anno'])['a_quantita'].sum()
plt.pie(stat, explode=explode, labels=stat.index, autopct='%1.1f%%', shadow=True, startangle=90)
plt.show()

# FAI PLOT TOT ORDINI VS ANNO


# In[12]:


# NUMERO TOTALE ORDINI PER CATEGORIE
stat = df.groupby(['a_cod_categoria'])['a_quantita'].sum()
plt.figure(figsize=(10, 10))
plt.bar(stat.index, stat, width=0.8)
plt.xticks(np.arange(1,11), ('FORNI-\nFRIGO', 'TV-\nSCHERMI', 'PC-\nTELEFONIA',
                           'INVERTER-\nBATTERIE', 'LAMPADE', 'TORNIO-\nFRESA',
                          'SPORT', 'LAB', 'TENNOSTATI', 'DISTRIB'))

plt.title('NUMERO TOTALE ORDINI PER CATEGORIE ')
plt.show()


# # ALGORITMI MACHINE LEARNING PER MODELLI PREDITTIVI

# ## CLASSIFICAZIONE CATEGORIA OGGETTO A PARTIRE DALLA DESCRIZIONE PRODOTTO

# In[114]:


from nltk.corpus import stopwords
import re
import string
import sqlite3
import json

categorie = " CREATE TABLE if not exists categorie(id INTEGER PRIMARY KEY AUTOINCREMENT, categoria VARCHAR(50) NOT NULL);"

descrizioni = 'CREATE TABLE if not exists descrizioni(id INTEGER PRIMARY KEY AUTOINCREMENT, descrizione VARCHAR(255) NOT NULL,id_categoria INTEGER NOT NULL,FOREIGN KEY (id_categoria) REFERENCES categorie (id));'

# creo il db e mi ci connetto
database = "test_MT.db"
conn = sqlite3.connect(database)
cursor = conn.cursor()


# creo le tabelle
for tabella in (categorie, descrizioni):
    cursor.execute(tabella)

# scrivo i dati

categories = ['FORNI-FRIGO', 'TV-SCHERMI', 'PC-TELEFONIA','INVERTER-BATTERIE', 'LAMPADE', 'TORNIO-FRESA',
             'SPORT', 'LABORATORIO', 'TERMOSTATI', 'DISTRIB']

for el in categories:
    # aggiungo la categoria se non esiste
    q = "INSERT INTO categorie (categoria) VALUES ('{}')"
    cursor.execute(q.format(el))

# aggiungo le descrizioni

stopwords = set(stopwords.words('italian'))

regex = re.compile('[%s]' % re.escape(string.punctuation))

def bagOfWords(elem):
    """
    Partiziona la stringa in un elenco di parole.
    rimuove spazi, caratteri di punteggiatura,
    stop words, parole specifiche e caratteri speciali.
    Trasforma tutte le maiuscole in minuscole.
    """
    stringa = str(elem)
    words = stringa.lower().strip().split(' ')
    for word in words:
        word = word.rstrip().lstrip()
        if not re.match(':\/\/.*[\r\n]*', word)         and not re.match('^÷*[kg]', word)         and not re.match("\d", word)         and not re.match('^@.*""', word)         and re.match('^[a-z]', word)         and not re.match('\s', word)         and word not in stopwords:
            word = regex.sub("", word)
            if word:
                word.replace('"', '')
                yield word


# lista array (descrizione, categoria)
lmap = list(df[['a_descrizione', 'a_cod_categoria']].values)

desc = [(list(bagOfWords(desc)), cat) for desc, cat in lmap]

for descrizione, categoria in desc:
    q = """
        INSERT INTO descrizioni (descrizione, id_categoria)
        VALUES
            ("{0}", "{1}")
    """.format(descrizione, categoria)
    cursor.execute(q)


# # Ricavare le parole chiave dalle descrizioni (temi)

# In[126]:


from nltk.stem.snowball import ItalianStemmer
from nltk import word_tokenize

stemmer = ItalianStemmer()

def elabora_corpus(corpus):
    """
    corpus sarà una lista di tuple, formata da:
    [
        ("una descrizione", "categoria1"),
        ("un'altra descrizione", "categoria2")
    ]
    """
    temi = set()
    categorie = set()
    documenti = []

    for descrizione, categoria in corpus:

        parole = [
            p.replace("'", '') for p in word_tokenize(descrizione)
            if p not in stopwords
            and p not in string.punctuation
        ]

        temi.update(parole)
        documenti.append((parole, categoria))
        categorie.add(categoria)

    temi = list(set(parola for parola in temi))
    categorie = list(categorie)

    return temi, categorie, documenti


# In[74]:


elabora_corpus([("Ciao, mi chiamo Luca", "pippo")])


# In[127]:


q = "SELECT  descrizione, categoria FROM descrizioni INNER JOIN categorie ON (id_categoria = categorie.id)"

documenti = cursor.execute(q).fetchall()
temi, categorie, documenti = elabora_corpus(documenti)


# In[80]:


q = """
    SELECT categoria
    FROM categorie
"""
print(cursor.execute(q).fetchall())


# In[146]:


print("Documenti = {}".format(documenti[8]))
print(len(documenti))
print("Categorie = {}".format(categorie))


# # Da un testo al numero

# In[170]:


import random

def crea_training_set(documenti, categorie):
    """
    Metodo che ritorna una tupla di due valori:
        - l'array degli input (train_x)
        - l'array degli output (train_y)

    I due array hanno lungezza fissa:
     - len(train_x) == len(temi)
     - len(train_y) == len(categorie)
    """
    training = []
    output_vuota = [0] * len(categorie)
    categorie = list(categorie)

    for parole, categoria in documenti:

        temi_descrizione = [parola for parola in parole]

        # riempio la lista di input
        riga_input = [1 if t in temi_descrizione else 0 for t in temi]

        # riempio la lista di output
        riga_output = output_vuota[:]
        riga_output[categorie.index(categoria)] = 1

        training.append([riga_input, riga_output])

    # mischio il mazzo
    random.shuffle(training)
    # trasformo in un array
    training = np.array(training)

    # e creo il training set
    train_x = list(training[:, 0])
    train_y = list(training[:, 1])
    return train_x, train_y


# In[171]:


# TRAINING CON 70% delle 18115 descrizioni a disposizione
X, y = crea_training_set(documenti[:12681], categorie)


# In[142]:


q = """
    SELECT  categoria
    FROM descrizioni
    INNER JOIN categorie ON (id_categoria = categorie.id)
"""
doc_test = cursor.execute(q).fetchall()

Xtest, ytest = crea_training_set(documenti[6:8], categorie)
print("Categorie = {}".format(categorie))
print(doc_test[6:8])
print(ytest)


# # Il cervello del BOT
#
# #### Rete neurale con 1 livello di input, 2 livelli nascosti, 1 livello di output

# In[172]:


import tensorflow as tf
from tflearn import input_data, fully_connected, regression, DNN

def BotANN(X, y):
    """
    Questo metodo definisce e istruisce una
    ANN (Artificial Neural Network), di tipo
    DNN (Deep Neural Network) composta da:
        - un livello di input,
        - due hidden layer,
        - uno di output.
    Utilizza softmax come funzione di attivazione.

    I parametri sono:
       - X: array bidimensionale con i dati di input
       - y: array bidimensionale con i dati di output

    Una volta definita la struttura della rete neurale,
    ne viene fatto il training, e il modello viene
    salvato in un file, chiamato "rete.tflearn".
    """
    # resetto i dati del grafo
    tf.compat.v1.reset_default_graph()

    # Definire la Rete Neurale
    rete = input_data(shape=[None, len(X[0])])
    rete = fully_connected(rete, 8)
    rete = fully_connected(rete, 8)
    rete = fully_connected(rete, len(y[0]), activation='softmax')
    rete = regression(rete)

    # Faccio il training
    model = DNN(rete, tensorboard_dir='logs')
    model.fit(X, y, n_epoch=1000, batch_size=8, show_metric=True)
    return model


# In[ ]:


modello = BotANN(X, y)


# In[150]:


# SALVA PESO DEI NEURONI IN FILE DI TESTO
modello.save('rete_70perc_ep1000 ')


# In[151]:


def genera_temi(testo_descrizione):
    stop = set(stopwords.words('italian'))
    lista_parole = word_tokenize(testo_descrizione)
    temi = [
        p.lower() for p in lista_parole
        if p not in stop and p not in string.punctuation
    ]
    return temi

def genera_input(lista_temi):
    lista_input = [0]*len(temi)
    for tema in lista_temi:
        for i, t in enumerate(temi):
            if t == tema:
                lista_input[i] = 1
    return(np.array(lista_input))


# In[152]:


from nltk.corpus import stopwords
temi_frase = genera_temi("televisori")
X = genera_input(temi_frase)


# In[153]:


SOGLIA_ERRORE = 0.25

def classifica(modello, array):
    # genera le probabilità
    prob = modello.predict([array])[0]  # lista delle probabilità associate ad ognuna delle 10 categorie
    # filtro quelle che superano la soglia
    prob_d = {v: prob[i]  for i, v in enumerate(categorie)}
    print('mappa categoria - probabilita:\n{}'.format(prob_d))
    risultati = [
        [i,p] for i,p in enumerate(prob)
        if p > SOGLIA_ERRORE
    ]
    # ordino per le categorie più probabili in ordine decrescente
    risultati.sort(key=lambda x: x[1], reverse=True)
    lista_categorie = []
    for r in risultati:
        lista_categorie.append((list(categorie)[r[0]], r[1]))
    return lista_categorie


# In[154]:


def trova_categoria(modello, frase_descrizione):
    temi_frase = genera_temi(frase_descrizione)
    X = genera_input(temi_frase)
    # genero la classifica delle categorie predette inserendo i temi di input nel modello di rete neurale
    categorie_predette = classifica(modello, X)

    if categorie_predette:
        # SELEZIONO CATEGORIA CON MAGGIORE PROBABILITA'
        categoria_predetta = categorie_predette[0]
        return categoria_predetta


# In[155]:


categoria_predetta = trova_categoria(modello, "telefoni")
print('\nLa categoria predetta dal modello è : {}'.format(categoria_predetta))


# # TEST FUNZIONAMENTO

# In[165]:


parola = "luce"

q = """
    SELECT categoria, descrizione, count(categoria)
    FROM descrizioni
    INNER JOIN categorie ON (id_categoria = categorie.id)
    WHERE descrizione like "%{}%"
    GROUP BY categoria;
""".format(parola)

rows = cursor.execute(q).fetchall()

lst = [(c, n) for c, d, n in rows]

print(lst)

trova_categoria(modello, parola)


# In[ ]:





# ### INSERIMENTO CATEGORIA SU SET 1000 DESCRIZIONI

# In[ ]:





# In[ ]:





# In[29]:


# CLASSIFICAZIONE ORIGINALE
cursor.execute('SELECT id_categoria, count(id_categoria) FROM (select * from descrizioni limit 1000) group by id_categoria ')
rows = cursor.fetchall()
print('\n', len(rows), rows)


# ## ALGORITMO ML CON REGRESSIONE

# In[30]:


# MODELLO PREDITTIVO TRAMITE REGRESSIONE ([INPUT], OUTPUT) --> ([PESO, QUANTITA, CATEGORIA], IMPONIBILE)

def histogram_intersection(a, b):
    v = np.minimum(a, b).sum().round(decimals=1)
    return v

raggruppamenti = ['a_cod_categoria', 'a_cod_sottocategoria']
agg = df.groupby(raggruppamenti).agg('mean')
df_agg = agg.reset_index()
mask = df_agg['a_anno'] == 2016
df_agg

# hypothesis function h(x) = theta0 + theta1*x1 + theta2*x2 + ...     where x0 = 1
# we normalize the 2nd and 3rd columns

# nuova_casa = [1, (Size - np.mean(df['Size'])) / np.std(df['Size']),
#               (Beds - np.mean(df['#Beds'])) / np.std(df['#Beds']) ]

# print("\n Price for a house of 1650 sq-ft with 3 beds using Gradient Descent : ", stima_regressione(nuova_casa, thetaGD))
# print("\n Price for a house of 1650 sq-ft with 3 beds using Normal Equation : ", stima_regressione(nuova_casa, thetaNE))
