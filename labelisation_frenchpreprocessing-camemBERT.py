# -*- coding: utf-8 -*-


############################## camemBERT ##############################

# Vérifier que le module french_preprocessing installé est bien celui avec le StanfordPOSTagger
# https://github.com/anaishoareau/french_preprocessing


""" Imports """

import pandas as pd
import time

from french_preprocessing.french_preprocessing import FrenchPreprocessing

"""Définition de variables"""

file_text = []

""" Chargement du dataset à évaluer"""

df = pd.read_csv('./data/sequoia.csv')

""""Chargement des outils de preprocessing"""

# FrenchPreprocessing
t0 = time.time()
f = FrenchPreprocessing()
t1 = time.time()
print("Temps de chargement de l'outil FrenchPreprocessing (camemBERT) : ", t1-t0)
file_text.append("Temps de chargement de l'outil FrenchPreprocessing (camemBERT) : "+str(t1-t0)+"\n")

"""Fonctions pour l'execution"""
    
# Fonction réalisant le FrenchPreprocessing sans simplification (tokenisation, tagging, lemmatisation)
def preprocess(x):
    frenchpreprocessing = []
    f_tokens = f.pretokenize(x)
    f_tag = f.tag(f_tokens)
    f_lemma = f.lemmatize(f_tag)
    for i in range(len(f_tokens)):
        frenchpreprocessing.append((f_tokens[i], f_tag[i][1], f_lemma[i]))
    return frenchpreprocessing


""" Execution """

# Application du FrenchPreprocessing
t0 = time.time()
df['frenchpreprocessing_camemBERT'] = df['sent'].map(lambda x: preprocess(x))
t1 = time.time()
print("Temps d'execution de l'opération pour FrenchPreprocessing (camemBERT) : ", t1-t0)
file_text.append("Temps d'execution de l'opération pour FrenchPreprocessing (camemBERT) : "+str(t1-t0)+"\n")


""" Ecriture dans un csv """

# Ecriture du fichier sequoia.csv à étudier
df.to_csv('./data/sequoia.csv', index = False)

file1 = open("./data/time_camemBERT.txt","w")
file1.writelines(file_text) 
file1.close()