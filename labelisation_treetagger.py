# -*- coding: utf-8 -*-


############################## TreeTagger ##############################

# Vérifier que le module french_preprocessing installé est bien celui avec le StanfordPOSTagger
# https://github.com/anaishoareau/french_preprocessing-stanfordpostagger


""" Imports """

import pandas as pd
import re
import time

import treetaggerwrapper

"""Définition de variables"""

file_text = []

""" Chargement du dataset à évaluer"""

df = pd.read_csv('./data/sequoia.csv')

""""Chargement des outils de preprocessing"""

# TreeTagger
t0 = time.time()
tagger = treetaggerwrapper.TreeTagger(TAGDIR = "./TreeTagger", TAGLANG='fr')
t1 = time.time()
print("Temps de chargement de l'outil TreeTagger : ", t1-t0)
file_text.append("Temps de chargement de l'outil TreeTagger : "+str(t1-t0)+"\n")

"""Fonctions pour l'execution"""

# Fonction de réduction des xtags et utags de sequoia
# convert
# u = {'PONCT', 'CLR', 'P+PRO', 'CC', 'ADV', 'CLS', 'I', 'VIMP', 'VPP', 'P+D', 'CLO', 'NPP', 'ADJWH', 'CS', 'VPR', 'V', 'DETWH', 'DET', 'PROREL', 'PRO', 'ET', 'PROWH', 'VINF', 'P', 'PREF', 'NC', 'ADJ', 'VS', 'ADVWH'}
# x = {'V', 'ADV', 'I', 'A', 'P', 'PONCT', 'PREF', 'N', 'PRO', 'D', 'C', 'CL', 'P+PRO', 'P+D', 'ET'}
# to 'v', 'nc', 'adj', 'c', 'npp', 'adv', 'det', 'pro', 'prep', 'i', 'ponct', 'cl', 'et'
def ux_tag_reduction(tag):
    tag_reduc = ''
    if tag in ['VS', 'VINF', 'VPP', 'VPR', 'VIMP']:
        tag_reduc = 'V'
    elif tag in ['N','PREF']:
        tag_reduc = 'NC'
    elif tag in ['CS', 'CC']:
        tag_reduc = 'C'
    elif tag in ['CLS', 'CLO', 'CLR']:
        tag_reduc = 'CL'
    elif tag in ['ADJWH', 'A']:
        tag_reduc = 'ADJ'
    elif tag == 'ADVWH':
        tag_reduc = 'ADV'
    elif tag in ['PROREL', 'PROWH']:
        tag_reduc = 'PRO'
    elif tag in ['DETWH', 'D']:
        tag_reduc = 'DET'
    elif tag in ['P','P+PRO', 'P+D']:
        tag_reduc = 'PREP'
    else:
        tag_reduc = tag
        
    return(tag_reduc.lower())

# Fonction d'extraction des informations utiles générées par TreeTagger
def notag_sup(string):        
    notag = re.compile(r'<.+ text="(?P<g3>.+)" />')
    return re.sub(notag, r'\g<g3>', string)

def lemma_recup(tags):
    word_lemma = []
    #lemmas = []
    # Récupération des lemmes
    for i in range(len(tags)):
        try:
            tags[i].lemma
            if tags[i].pos == 'NUM':
                word_lemma.append((tags[i].word, tags[i].word))
                #lemmas.append(tags[i].word)
            else:
                word_lemma.append((tags[i].word, tags[i].lemma))
                #lemmas.append(tags[i].lemma)
        except:
            #del lemmas[-1]
            del word_lemma[-1]
            word_lemma.append((tags[i].what, tags[i].what))
            #lemmas.append(tags[i].what)
    word_lemma_2 = []
    # Suppression des Tags inutiles et variantes
    for i in range(len(word_lemma)):
        if '|' in word_lemma[i][1]:
            word_lemma_2.append((tags[i].word, word_lemma[i][1].split('|')[0]))
            #lemmas2.append(e.split('|')[0])
        else:
            word_lemma_2.append((notag_sup(word_lemma[i][0]), notag_sup(word_lemma[i][1])))
            #lemmas2.append(notag_sup(e))
            
    return word_lemma_2


""" Execution """

# Application de TreeTagger
t0 = time.time()
df['treetagger'] = df['sent'].map(lambda x: lemma_recup(treetaggerwrapper.make_tags(tagger.tag_text(x))))
t1 = time.time()
print("Temps d'execution de l'opération pour TreeTagger : ", t1-t0)
file_text.append("Temps d'execution de l'opération pour TreeTagger : "+str(t1-t0)+"\n")


""" Ecriture dans un csv """

# Ecriture du fichier sequoia.csv à étudier
df.to_csv('./data/sequoia.csv', index = False)

file1 = open("./data/time_treetagger.txt","a")
file1.writelines(file_text) 
file1.close()