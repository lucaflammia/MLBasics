#!/usr/bin/env python
# coding: utf-8

# In[1]:


# -*- coding: utf-8 -*-# -*- c 

import math
from collections import defaultdict
from nltk.corpus import stopwords
import re
import string
import json

stopwords = set(stopwords.words('english') + stopwords.words('italian'))
regex = re.compile('[%s]' % re.escape(string.punctuation))

def bagOfWords(stringa):
    """
    Partiziona la stringa in un elenco di parole.
    rimuove spazi, caratteri di punteggiatura,
    url, stop words, menzioni (@) e 'RT'.
    Trasforma tutte le maiuscole in minuscole.
    """
    words = stringa.lower().strip().split(' ')
    for word in words:
        word = word.rstrip().lstrip()
        if not re.match(r'^https?:\/\/.*[\r\n]*', word)         and not re.match(r'^http?:\/\/.*[\r\n]*', word)         and not re.match('^@.*', word)         and not re.match('\s', word)         and word not in stopwords         and word != 'rt':
            word = regex.sub("", word)
            if word:
                yield word

class NaiveBayesClassifier(object):

    def __init__(self, k=1):
        self.diz_parole = {}
        self.prob_classi = {}
        self.k = k

    def train(self, X, Y):
        """
        In questa funzione, X e Y saranno due array (liste).
        - X contiene tutte le bag of words
          di tutti i tweet già classificati.
          Es. X = [ 
                    ['casa', 'bellissima', 'comprare'], 
                    ['agenzia', 'costi'], ...
                  ]
        - Y invece contiene le classi a cui appartengono i tweet: 
          - 0 = Sentiment Negativo
          - 1 = Sentiment Positivo
          Es. Y = [1, 0, 0, 1, 0, 1, 1, ...]

        Questa funzione non ritorna niente, serve a creare
        il dizionario di parole con il numero delle occorrenze
        relative alla classe. 
        """
        classi_distinte = set(Y)

        for lista, classe in zip(X, Y):
            for parola in lista:
                self.diz_parole.setdefault(parola, defaultdict(int))[classe] += 1

        tot_generale = sum(x[0]+x[1] for x in 
                           self.diz_parole.values())

        for classe in classi_distinte:
            tot_classe = sum(lista[classe] for lista in
                             self.diz_parole.values())
            self.prob_classi[classe] = tot_classe / float(tot_generale)

    def classify(self, X):
        """
        Questa funzione sarà quella usata per classificare
        i nuovi tweet. 

        X è una lista di liste come per la funzione di addestramento.
        Questo ci permetterà di catalogare più di un tweet
        per volta

        Ritorna una lista con le classi di appartenza dei tweet, 
        ovvero: 1 se il tweet esprime un sentimento positivo,
        0 altrimenti.
        """
        risultato = []
        for tweet in X:
            prob_per_classe = {}
            for classe in self.prob_classi:
                tot_classe = sum(lista[classe] for lista in
                             self.diz_parole.values())
                
                prob = math.exp(
                    sum([math.log(self.prob_classi[classe])] + 
                        [math.log(((self.k + self.diz_parole.get(w, {}).get(classe, 0)) / 
                            (2*self.k + float(tot_classe)))) for w in tweet])
                    )
                prob_per_classe[classe] = prob
            classe_max = next(k for k, v in prob_per_classe.items() 
                            if v == max(prob for prob in 
                                prob_per_classe.values()))
            risultato.append(classe_max)
        return risultato
    

with open('data.json') as f:
    data = json.loads(f.read())
    
nb = NaiveBayesClassifier()
X = [list(bagOfWords(t)) for t in data['X']]
nb.train(X, data['Y'])

tweets = [
    "Oggi sarà una bellissima giornata, me lo sento!",
    "Questo piatto non mi piace, è troppo salato",
    "Il lavoro è interessante, credo accetterò la loro offerta",
    "Mi sarebbe piaciuto andare al concerto, ma non ho trovato i biglietti"
]

tweets = [list(bagOfWords(t)) for t in tweets]
print(nb.classify(tweets))


# In[ ]:




