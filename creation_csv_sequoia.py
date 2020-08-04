# -*- coding: utf-8 -*-


############################## TreeTagger ##############################

# Vérifier que le module french_preprocessing installé est bien celui avec le StanfordPOSTagger
# https://github.com/anaishoareau/french_preprocessing-stanfordpostagger


""" Imports """

import pandas as pd
import pyconll
#https://pyconll.readthedocs.io/en/stable/pyconll/unit/token.html
#Attributs des objets pyconll : id, form, lemma, upos, xpos, feats, head, deprel, deps, misc

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

# Fonction de récupération des objets conll pour les inscrire dans un dataset
def dataset_sequoia(file_conll):
    f = pyconll.load_from_file(file_conll)
    df = pd.DataFrame(columns = ['sent_id', 'sent', 'sequoia'])
    i = 0
    for sentence in f:
        s_id = sentence.id
        s = sentence.text
        sequoia = []
        for token in sentence:
            sequoia.append((token.form, ux_tag_reduction(token.xpos), token.lemma))
        df.loc[i] = [s_id, s, sequoia]
        i+=1
    return df

""" Execution """

# Création du dataset
df = dataset_sequoia("./data/sequoia.deep.conll")

""" Ecriture dans un csv """

# Ecriture du fichier sequoia.csv à étudier
df.to_csv('./data/sequoia.csv', index = False)

