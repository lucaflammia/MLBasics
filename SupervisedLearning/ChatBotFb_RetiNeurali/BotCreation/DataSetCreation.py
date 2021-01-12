#!/usr/bin/env python
# coding: utf-8

# # Creazione del database

# In[1]:


import sqlite3
import json

classi = """
    CREATE TABLE if not exists classi(id INTEGER PRIMARY KEY AUTOINCREMENT, classe VARCHAR(50) NOT NULL);
"""

domande = """
CREATE TABLE if not exists domande(
       id INTEGER PRIMARY KEY AUTOINCREMENT,
       domanda VARCHAR(255) NOT NULL,
       id_classe INTEGER NOT NULL,
       
       FOREIGN KEY (id_classe) REFERENCES classi (id)
    );
"""

risposte = """
    CREATE TABLE if not exists risposte(
       id INTEGER PRIMARY KEY AUTOINCREMENT,
       risposta VARCHAR(255) NOT NULL,
       id_classe INTEGER NOT NULL,
       
       FOREIGN KEY (id_classe) REFERENCES classi (id)
    );
"""


# creo il db e mi ci connetto
database = "bot.db"
conn = sqlite3.connect(database)
cursor = conn.cursor()

# creo le tabelle
for tabella in (classi, domande, risposte):
    cursor.execute(tabella)

    
# scrivo i dati del file json
with open("data.json") as f:
    data = json.load(f)
    for elemento in data:
        # aggiungo la classe se non esiste
        q = "INSERT INTO classi (classe) VALUES ('{}')"
        cursor.execute(q.format(elemento['classe']))
        id_classe = cursor.lastrowid
            
        # aggiungo le domande
        for domanda in elemento['domande']:
            q = """
                INSERT INTO domande (domanda, id_classe)
                VALUES
                    ("{0}", "{1}")
            """.format(domanda, id_classe)
            cursor.execute(q)
            
        # e le risposte
        for risposta in elemento['risposte']:
            q = """
                INSERT INTO risposte (risposta, id_classe)
                VALUES
                    ("{0}", "{1}")
            """.format(risposta, id_classe)
            cursor.execute(q)


# # Ricavare il significato di una frase

# In[2]:


from nltk.stem.snowball import ItalianStemmer
stemmer = ItalianStemmer()

parole = [
    "sviluppare", 
    "sviluppavo", 
    "sviluppa", 
    "sviluppate", 
    "sviluppiamo"
]

temi = [stemmer.stem(parola) for parola in parole]
print(temi)


# In[3]:


from nltk.corpus import stopwords
stop = set(stopwords.words('italian'))
frase = "per questo ebbero più paura " +        "dei nostri discorsi " +        "che delle nostre facce"
lista_parole = frase.split(" ")
rimosse = [w for w in lista_parole if w in stop]
rimaste = [w for w in lista_parole if w not in stop]
print("Le stopwords sono: \n{}\n".format(rimosse))
print("I termini significativi sono: \n{}".format(rimaste))


# In[4]:


from nltk.corpus import stopwords
from nltk.stem.snowball import ItalianStemmer
from nltk import word_tokenize
import nltk
nltk.download('punkt')
import string

stemmer = ItalianStemmer()
stop = set(stopwords.words('italian'))

def elabora_corpus(corpus):
    """
    corpus sarà una lista di tuple, formata da:
    [
        ("una frase", "classe1"),
        ("un'altra frase", "classe2")
    ]
    """
    temi = set()
    classi = set()
    documenti = []
    
    stop = set(stopwords.words('italian'))
    
    for frase, classe in corpus:
        # rimuovo le stopwords
        parole = [
            p.replace("?", "").lower() for p in word_tokenize(frase) 
            if p not in stop
            and p not in string.punctuation
        ]
        
        temi.update(parole)
        documenti.append((parole, classe))
        classi.add(classe)

    # creo i temi
    temi = list(set(stemmer.stem(parola) for parola in temi))
    classi = list(classi)
    return temi, classi, documenti


# In[5]:


elabora_corpus([("Ciao, mi chiamo Luca", "pippo")]) 


# In[6]:


q = """
    SELECT domanda, classe
    FROM domande
    INNER JOIN classi ON (id_classe = classi.id)
"""

domande = cursor.execute(q).fetchall()
temi, classi, documenti = elabora_corpus(domande)


# In[7]:


print("Numero di classi: {}".format(len(classi)))
print("Numero di documenti: {}".format(len(documenti)))
print("Temi: \n{}".format(temi))


# In[8]:


import tensorflow as tf

x1 = tf.constant([1,2,3,4])
x2 = tf.constant([5,6,7,8])

risultato = tf.multiply(x1, x2)

print(risultato)


# In[9]:


with tf.Session() as sessione:
    writer = tf.summary.FileWriter('logs', sessione.graph)
    print(sessione.run(risultato))


# # Da un testo al numero

# In[10]:


import random
import numpy as np

def crea_training_set(documenti, classi):
    """
    Metodo che ritorna una tupla di due valori:
        - l'array degli input (train_x)
        - l'array degli output (train_y)
        
    I due array hanno lungezza fissa:
     - len(train_x) == len(temi)
     - len(train_y) == len(classi) 
    """
    training = []
    output_vuota = [0] * len(classi)
    classi = list(classi)

    for parole, classe in documenti:
        
        temi_frase = [stemmer.stem(parola) for parola in parole]
        
        # riempio la lista di input
        riga_input = [1 if t in temi_frase else 0 for t in temi]

        # riempio la lista di output
        riga_output = output_vuota[:]
        riga_output[classi.index(classe)] = 1

        training.append([riga_input, riga_output])

    # mischio il mazzo
    random.shuffle(training)
    # trasformo in un array
    training = np.array(training)

    # e creo il training set
    train_x = list(training[:,0])
    train_y = list(training[:,1])
    return train_x, train_y


# In[11]:


print("Temi = {}".format(temi))
print("Classi = {}".format(classi))


# In[12]:


print("Parole Documento = {}".format(documenti[-1][0]))
print("Classe Documento = {}".format(documenti[-1][1]))


# In[13]:


print(crea_training_set([documenti[-1]], classi))


# In[14]:


X, y = crea_training_set(documenti, classi)
print("X = {}".format(X))
print("y = {}".format(y))


# # Il cervello del BOT
# 
# #### Rete neurale con 1 livello di input, 2 livelli nascosti, 1 livello di output 

# In[15]:


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
    tf.reset_default_graph()
    
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


# In[16]:


modello = BotANN(X, y) 


# In[30]:


def genera_temi(testo):
    stop = set(stopwords.words('italian'))
    lista_parole = word_tokenize(testo)
    temi = [
        stemmer.stem(p.lower()) for p in lista_parole
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


# In[31]:


temi_frase = genera_temi("Ciao")
X = genera_input(temi_frase)


# In[32]:


print(X)


# In[20]:


SOGLIA_ERRORE = 0.25

def classifica(modello, array):
    # genera le probabilità 
    prob = modello.predict([array])[0]
    # filtro quelle che superano la soglia
    risultati = [
        [i,p] for i,p in enumerate(prob) 
        if p > SOGLIA_ERRORE
    ]
    # ordino per le classi più probabili
    risultati.sort(key=lambda x: x[1], reverse=True)
    lista_classi = []
    for r in risultati:
        lista_classi.append((list(classi)[r[0]], r[1]))
    return lista_classi


# In[21]:


def rispondi(modello, frase):
    temi_frase = genera_temi(frase)
    X = genera_input(temi_frase)
    print(X)
    classi_predette = classifica(modello, X[0])
    print(classi_predette)
    
    if classi_predette:
        # leggo le risposte
        q = """
            SELECT risposta 
            FROM risposte
            INNER JOIN classi ON (risposte.id_classe = classi.id)
            WHERE classe = '{0}'
        """.format(classi_predette[0][0])
        risposte = [r[0] for r in cursor.execute(q).fetchall()]
        return np.random.choice(risposte)


# In[22]:


rispondi(modello, "Salve?")


# # AGGIUNTA DEL CONTESTO
# 
# ### Capire in quale contesto appartiene la domanda

# In[23]:


contesti = """
CREATE TABLE if not exists contesti(
       id INTEGER PRIMARY KEY AUTOINCREMENT,
       contesto VARCHAR(255) NOT NULL,
       id_classe INTEGER NOT NULL,
       
       FOREIGN KEY (id_classe) REFERENCES classi (id)
    );
"""

# aggiungo la tabella al db
cursor.execute(contesti)

# rileggo il file e aggiungo i contesti
q_select = """
    SELECT id FROM classi
    WHERE classe = '{}'
"""

q_insert = """
    INSERT INTO contesti (contesto, id_classe)
    VALUES ("{0}", {1})
"""

with open("data.json") as f:
    data = json.load(f)

# aggiungo l'ultima classe inserita
pagamento_escursioni = data[-1]
q = """
INSERT INTO classi (classe)
VALUES ("{}")
""".format(pagamento_escursioni['classe'])
cursor.execute(q)

id_classe = cursor.lastrowid
for domanda in pagamento_escursioni['domande']:
    cursor.execute(
        """
        INSERT INTO domande (domanda, id_classe)
        VALUES ("{}", {})
        """.format(domanda, id_classe)
    )
for risposta in pagamento_escursioni['risposte']:
    cursor.execute(
        """
        INSERT INTO risposte (risposta, id_classe)
        VALUES ("{}", {})
        """.format(risposta, id_classe)
    )
    
# aggiungo tutti i contesti
for elemento in data:
    if elemento.get('contesto'):
        id_classe = cursor.execute(
            q_select.format(elemento['classe'])
        ).fetchone()[0]
        cursor.execute(
            q_insert.format(elemento['contesto'], id_classe)
        )


# In[24]:


contesti = {}

def rispondi(modello, frase, utente="utente_prova"):
    temi_frase = genera_temi(frase)
    X = genera_input(temi_frase)
    classi_predette = classifica(modello, X[0])
    print('Classi predette e probabilità: {}'.format(classi_predette))
    # tolgo le probabilità
    classi_predette = [c[0] for c in classi_predette]
    
    if classi_predette:
        # ho un contesto settato?
        if contesti.get(utente):
            contesto = contesti[utente]
            
            # quali classi hanno questo contesto?
            q = """
                SELECT classe FROM classi
                INNER JOIN contesti ON (classi.id = contesti.id_classe)
                WHERE classe IN ({})
            """.format(",".join(
                "'{}'".format(classe) for classe in classi_predette
                )
            )
            filtro_classi = [c[0] for c in cursor.execute(q).fetchall()]
            if filtro_classi:
                # ho almeno una classe predetta che usa un contesto
                classi_predette = [c for c in classi_predette]
                
        # leggo le risposte
        q = """
            SELECT risposta 
            FROM risposte
            INNER JOIN classi ON (risposte.id_classe = classi.id)
            WHERE classe = '{0}'
        """.format(classi_predette[0])
        
        risposte = [r[0] for r in cursor.execute(q).fetchall()]
        
        # scelgo una risposta
        risposta = np.random.choice(risposte)
        
        # imposto il contesto, se c'è
        q = """
            SELECT contesto from contesti
            INNER JOIN classi ON (contesti.id_classe = classi.id)
            INNER JOIN risposte ON (risposte.id_classe = classi.id)
            WHERE risposta = "{}"
        """.format(risposta)
        contesto = cursor.execute(q).fetchone()
        contesti[utente] = contesto[0] if contesto else None
        print('Contesto del messaggio: {}'.format(contesto))
            
        return risposta


# In[25]:


print(rispondi(modello, "Quali ristoranti?"))
# print(rispondi(modello, "Quale ristorante"))
# print(rispondi(modello, "Avete un pos anche durante le escursioni?"))
# print(contesti)


# In[26]:


import pickle

d = {
    'temi': temi,
    'classi': classi,
    'documenti': documenti
}

pickle.dump(d, open("corpus.p", "wb"))


# In[27]:


modello.save("rete")
conn.commit()
conn.close()


# In[28]:


modello.load("./rete")
conn = sqlite3.connect(database)
cursor = conn.cursor()
rispondi(modello, "Posso pagare con carta di credito?")


# In[29]:


rispondi(modello, "Ciao")


# In[ ]:




