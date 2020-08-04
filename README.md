# Test de performance du FrenchPreprocessing (StanfordPOSTagger vs CamemBERT)

1. **Objectif :**

Etudier les performances de FrenchPreprocessing (avec deux taggers différents : StanfordPOSTagger et modèle CamemBERT) sur un dataset labélisé et les comparer à celles de TreeTagger.

2. **Corpus utilisé :**
    - Deep sequoia (version 9.1) : https://gitlab.inria.fr/sequoia/deep-sequoia/-/tree/master/tags/sequoia-9.1
    - README deep Sequoia : https://gitlab.inria.fr/sequoia/deep-sequoia/-/blob/master/tags/sequoia-9.1/README-distrib.md
    - Fichier utilisé dans l'étude : https://gitlab.inria.fr/sequoia/deep-sequoia/-/blob/master/tags/sequoia-9.1/sequoia.deep.conll  
    
 
3. **Fichers d'exécution pour l'étude :**

    - creation_csv_sequoia.py : Permet de créer un dataset pour l'étude à partir des données du corpus Sequoia dans le fichier sequoia.conll
    - labelisation_frenchpreprocessing-stanford.py : Réalise la lemmatisation des données brutes de sequoia.csv avec le FrenchPreprocessing (StanfordPOSTagger)
    - labelisation_frenchpreprocessing-camemBERT.py : Réalise la lemmatisation des données brutes de sequoia.csv avec le FrenchPreprocessing (camemBERT)
    - labelisation_treetagger.py : Réalise la lemmatisation des données brutes de sequoia.csv avec TreeTagger
    
    
4. **Aide à l'installation :**


* Pour installer TreeTagger à utiliser avec Python :

    - Installer le wrapper python : https://pypi.org/project/treetaggerwrapper/ 
    - Télécharger TreeTagger et ajouter à /lib le fichier nécessaire pour le fonctionnement du wrapper : https://www.cis.uni-muenchen.de/~schmid/tools/TreeTagger/
    - Tagset de TreeTagger : https://www.cis.uni-muenchen.de/~schmid/tools/TreeTagger/data/french-tagset.html


* Pour installer les FrenchPreprocessing :

    - Installation depuis github (version camemBERT) : https://github.com/anaishoareau/french_preprocessing   
    - Installation depuis github (version StanfordPOSTagger) : https://github.com/anaishoareau/french_preprocessing-stanfordpostagger  


<h1>Table of Contents<span class="tocSkip"></span></h1>
<div class="toc"><ul class="toc-item"><li><span><a href="#Imports" data-toc-modified-id="Imports-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Imports</a></span></li><li><span><a href="#Chargement-du-dataset" data-toc-modified-id="Chargement-du-dataset-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Chargement du dataset</a></span></li><li><span><a href="#Réduction-du-dataset" data-toc-modified-id="Réduction-du-dataset-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Réduction du dataset</a></span></li><li><span><a href="#Exemple-de-lemmatisation" data-toc-modified-id="Exemple-de-lemmatisation-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Exemple de lemmatisation</a></span></li><li><span><a href="#Performance-de-la-lemmatisation" data-toc-modified-id="Performance-de-la-lemmatisation-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>Performance de la lemmatisation</a></span><ul class="toc-item"><li><span><a href="#Récupération-des-doublets-(mot,lemme)-labélisés" data-toc-modified-id="Récupération-des-doublets-(mot,lemme)-labélisés-5.1"><span class="toc-item-num">5.1&nbsp;&nbsp;</span>Récupération des doublets (mot,lemme) labélisés</a></span></li><li><span><a href="#Calcul-du-nombre-de-doublets-(mot,lemme)-identiques" data-toc-modified-id="Calcul-du-nombre-de-doublets-(mot,lemme)-identiques-5.2"><span class="toc-item-num">5.2&nbsp;&nbsp;</span>Calcul du nombre de doublets (mot,lemme) identiques</a></span></li><li><span><a href="#Affichage-des-pourcentages" data-toc-modified-id="Affichage-des-pourcentages-5.3"><span class="toc-item-num">5.3&nbsp;&nbsp;</span>Affichage des pourcentages</a></span></li></ul></li><li><span><a href="#Conclusion" data-toc-modified-id="Conclusion-6"><span class="toc-item-num">6&nbsp;&nbsp;</span>Conclusion</a></span></li></ul></div>

## Imports


```python
from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
```


<style>.container { width:100% !important; }</style>


## Chargement du dataset 


```python
df = pd.read_csv('./data/sequoia.csv')
```


```python
#Conversion du texte des colonnes "sequoia", "frenchpreprocessing", "treetagger" en objets list
df['sequoia'] = df['sequoia'].map(lambda x: list(eval(x)))
df['frenchpreprocessing_stanford'] = df['frenchpreprocessing_stanford'].map(lambda x: list(eval(x)))
df['frenchpreprocessing_camemBERT'] = df['frenchpreprocessing_camemBERT'].map(lambda x: list(eval(x)))
df['treetagger'] = df['treetagger'].map(lambda x: list(eval(x)))
```


```python
# Suppresion des majuscules dans les lemmes (non vecteur de sens dans la lemmatisation)
def lemma_maj_to_min(x):
    l = []
    for e in x:
        if len(e)==3:
            l.append((e[0], e[1], e[2].lower()))
        else:
            l.append((e[0], e[1].lower()))   
    return l

df['sequoia'] = df['sequoia'].map(lambda x: lemma_maj_to_min(x))
df['frenchpreprocessing_stanford'] = df['frenchpreprocessing_stanford'].map(lambda x: lemma_maj_to_min(x))
df['frenchpreprocessing_camemBERT'] = df['frenchpreprocessing_camemBERT'].map(lambda x: lemma_maj_to_min(x))
df['treetagger'] = df['treetagger'].map(lambda x: lemma_maj_to_min(x))
```


```python
#Colonne comptant le nombre de tokens proposés par sequoia
df['sequoia_nb_tokens'] = df['sequoia'].map(lambda x: len(x))
```


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sent_id</th>
      <th>sent</th>
      <th>sequoia</th>
      <th>treetagger</th>
      <th>frenchpreprocessing_camemBERT</th>
      <th>frenchpreprocessing_stanford</th>
      <th>sequoia_nb_tokens</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>annodis.er_00001</td>
      <td>Gutenberg</td>
      <td>[(Gutenberg, npp, gutenberg)]</td>
      <td>[(Gutenberg, gutenberg)]</td>
      <td>[(Gutenberg, npp, gutenberg)]</td>
      <td>[(Gutenberg, npp, gutenberg)]</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>annodis.er_00002</td>
      <td>Cette exposition nous apprend que dès le XIIe ...</td>
      <td>[(Cette, det, ce), (exposition, nc, exposition...</td>
      <td>[(Cette, ce), (exposition, exposition), (nous,...</td>
      <td>[(Cette, det, ce), (exposition, nc, exposition...</td>
      <td>[(Cette, det, ce), (exposition, nc, exposition...</td>
      <td>22</td>
    </tr>
    <tr>
      <th>2</th>
      <td>annodis.er_00003</td>
      <td>à peu près au même moment que Gutenberg invent...</td>
      <td>[(à, prep, à), (peu, adv, peu), (près, adv, pr...</td>
      <td>[(à, à), (peu, peu), (près, près), (au, au), (...</td>
      <td>[(à, prep, à), (peu, adv, peu), (près, adv, pr...</td>
      <td>[(à, prep, à), (peu, adv, peu), (près, adv, pr...</td>
      <td>30</td>
    </tr>
    <tr>
      <th>3</th>
      <td>annodis.er_00004</td>
      <td>Ensuite, fut installée une autre forge à la Va...</td>
      <td>[(Ensuite, adv, ensuite), (,, ponct, ,), (fut,...</td>
      <td>[(Ensuite, ensuite), (,, ,), (fut, être), (ins...</td>
      <td>[(Ensuite, adv, ensuite), (,, nc, ,), (fut, v,...</td>
      <td>[(Ensuite, adv, ensuite), (,, ponct, ,), (fut,...</td>
      <td>19</td>
    </tr>
    <tr>
      <th>4</th>
      <td>annodis.er_00005</td>
      <td>En 1953, les hauts fourneaux et fonderies de C...</td>
      <td>[(En, prep, en), (1953, nc, 1953), (,, ponct, ...</td>
      <td>[(En, en), (1953, 1953), (,, ,), (les, le), (h...</td>
      <td>[(En, prep, en), (1953, nc, 1953), (,, nc, ,),...</td>
      <td>[(En, prep, en), (1953, nc, 1953), (,, ponct, ...</td>
      <td>62</td>
    </tr>
  </tbody>
</table>
</div>



# Temps d'execution des outils


```python
# readlines function 
file = open("./data/time_treetagger.txt","r+") 
l = file.readlines()
for e in l:
    print(e)
file.close() 
```

    Temps de chargement de l'outil TreeTagger : 0.005977630615234375
    
    Temps d'execution de l'opération pour TreeTagger : 10.856043577194214
    
    


```python
# readlines function 
file = open("./data/time_stanford.txt","r+") 
l = file.readlines()
for e in l:
    print(e)
file.close() 
```

    Temps de chargement de l'outil FrenchPreprocessing (StanfordPOSTagger) : 5.900420665740967
    
    Temps d'execution de l'opération pour FrenchPreprocessing (StanfordPOSTagger) : 1734.35395693779
    
    


```python
# readlines function 
file = open("./data/time_camembert.txt","r+") 
l = file.readlines()
for e in l:
    print(e)
file.close() 
```

    Temps de chargement de l'outil FrenchPreprocessing (camemBERT) : 10.079216241836548
    
    Temps d'execution de l'opération pour FrenchPreprocessing (camemBERT) : 328.23137068748474
    
    

## Réduction du dataset

On ne conserve que les phrases pour lesquelles la tokenisation est identique entre sequoia, les FrenchPreprocessing et TreeTagger pour avoir exactement les mêmes phrases étudiées. On sélectionne donc 2492 phrases pour un total de 50654 tokens.


```python
# Fonction testant l'égalité entre les tokens effectués entre deux colonnes
def tokens_equals(x,y):
    if len(x)==len(y):
        ind = True
        for i in range(len(x)):
            if x[i][0]!=y[i][0]:
                ind = False
                break
    else:
        ind = False
    return ind

# Colonnes indiquant l'égalité entre deux tokenisation
df['sequoia_frenchpreprocessing_stanford_tokens_equals'] = df.apply(lambda x: tokens_equals(x['sequoia'],x['frenchpreprocessing_stanford']), axis = 1)
df['sequoia_frenchpreprocessing_camemBERT_tokens_equals'] = df.apply(lambda x: tokens_equals(x['sequoia'],x['frenchpreprocessing_camemBERT']), axis = 1)
df['sequoia_treetagger_tokens_equals'] = df.apply(lambda x: tokens_equals(x['sequoia'],x['treetagger']), axis = 1)
df['sequoia_frenchpreprocessing_treetagger_tokens_equals'] = df['sequoia_treetagger_tokens_equals']&df['sequoia_frenchpreprocessing_stanford_tokens_equals']&df['sequoia_frenchpreprocessing_camemBERT_tokens_equals']==True
```


```python
print("Nombre de phrases tokenisé de la même manière que dans le dataset Sequoia avec le FrenchPreprocessing :", df[df['sequoia_frenchpreprocessing_stanford_tokens_equals']==True]['sequoia_frenchpreprocessing_stanford_tokens_equals'].count())
print("Nombre de phrases tokenisé de la même manière que dans le dataset Sequoia avec le FrenchPreprocessing :", df[df['sequoia_frenchpreprocessing_camemBERT_tokens_equals']==True]['sequoia_frenchpreprocessing_camemBERT_tokens_equals'].count())
print("Nombre de phrases tokenisé de la même manière que dans le dataset Sequoia avec le TreeTagger :", df[df['sequoia_treetagger_tokens_equals']==True]['sequoia_treetagger_tokens_equals'].count())
print("Nombre de phrases tokenisé de la même manière que dans le dataset Sequoia avec les FrenchPreprocessing et le TreeTagger :", df[df['sequoia_frenchpreprocessing_treetagger_tokens_equals']==True]['sequoia_frenchpreprocessing_treetagger_tokens_equals'].count())
```

    Nombre de phrases tokenisé de la même manière que dans le dataset Sequoia avec le FrenchPreprocessing : 2630
    Nombre de phrases tokenisé de la même manière que dans le dataset Sequoia avec le FrenchPreprocessing : 2630
    Nombre de phrases tokenisé de la même manière que dans le dataset Sequoia avec le TreeTagger : 2600
    Nombre de phrases tokenisé de la même manière que dans le dataset Sequoia avec les FrenchPreprocessing et le TreeTagger : 2492
    


```python
# On crée un nouveau dataset avec seulement les phrases ayant la même tokenisation
df_equal = df[df['sequoia_frenchpreprocessing_treetagger_tokens_equals'] == True]
df_equal.reset_index(inplace = True)
print("Nombre de phrases sélectionnées pour l'étude :", df_equal['sent_id'].count())
print("Nombre de tokens sélectionnés pour l'étude :", df_equal['sequoia_nb_tokens'].sum())
```

    Nombre de phrases sélectionnées pour l'étude : 2492
    Nombre de tokens sélectionnés pour l'étude : 50654
    

## Exemple de lemmatisation


```python
i = 6
n = len(df_equal['sequoia'][i])

l_sequoia = []
l_treetagger = []
l_frenchpreprocessing_camemBERT = []
l_frenchpreprocessing_stanford = []

for k in range(n):
    l_sequoia.append(df_equal['sequoia'][i][k][2])
    l_treetagger.append(df_equal['treetagger'][i][k][1])
    l_frenchpreprocessing_camemBERT.append(df_equal['frenchpreprocessing_camemBERT'][i][k][2])
    l_frenchpreprocessing_stanford.append(df_equal['frenchpreprocessing_stanford'][i][k][2])

print("Sequoia :")
print(" ".join(l_sequoia))

print("TreeTagger :")
print(" ".join(l_treetagger))

print("FrenchPreprocessing (StanfordPOSTagger) :")
print(" ".join(l_frenchpreprocessing_stanford))

print("FrenchPreprocessing (camemBERT) :")
print(" ".join(l_frenchpreprocessing_camemBERT))
```

    Sequoia :
    le pose de un panneau stop paraître être le formule le mieux adapté pour assurer le sécurité de usager .
    TreeTagger :
    le pose de un panneau stop paraître être le formule le mieux adapter pour assurer le sécurité du usager .
    FrenchPreprocessing (StanfordPOSTagger) :
    le pose de un panneau stop paraître être le formule le mieux adapté pour assurer le sécurité des usager .
    FrenchPreprocessing (camemBERT) :
    le pose de un panneau stop paraître être le formule le mieux adapter pour assurer le sécurité des usager .
    

## Performance de la lemmatisation

### Récupération des doublets (mot,lemme) labélisés


```python
def word_lemma(x):
    l = []
    for e in x:
        l.append((e[0],e[2]))
    return l
```


```python
df_equal['sequoia_word_lemma'] = df_equal['sequoia'].map(lambda x: word_lemma(x))
df_equal['frenchpreprocessing_stanford_word_lemma'] = df_equal['frenchpreprocessing_stanford'].map(lambda x: word_lemma(x))
df_equal['frenchpreprocessing_camemBERT_word_lemma'] = df_equal['frenchpreprocessing_camemBERT'].map(lambda x: word_lemma(x))
```


```python
def compare_sequoia(x,y):
    cp = 0
    for i in range(len(x)):
        if x[i][1] == y[i][1]:
                cp += 1
    return cp
```

### Calcul du nombre de doublets (mot,lemme) identiques


```python
df_equal['sequoia_frenchpreprocessing_stanford_word_lemma'] = df_equal.apply(lambda x: compare_sequoia(x['sequoia_word_lemma'],x['frenchpreprocessing_stanford_word_lemma']), axis = 1)
df_equal['sequoia_frenchpreprocessing_camemBERT_word_lemma'] = df_equal.apply(lambda x: compare_sequoia(x['sequoia_word_lemma'],x['frenchpreprocessing_camemBERT_word_lemma']), axis = 1) 
df_equal['sequoia_treetagger_word_lemma'] = df_equal.apply(lambda x: compare_sequoia(x['sequoia_word_lemma'],x['treetagger']), axis = 1)
```

### Affichage des pourcentages


```python
nb = df_equal['sequoia_treetagger_word_lemma'].sum()/df_equal['sequoia_nb_tokens'].sum()
print("Pourcentage de doublets (mot,lemme) identiques entre les labélisations de sequoia et TreeTagger : "+"{:.2%}".format(nb))

nb = df_equal['sequoia_frenchpreprocessing_stanford_word_lemma'].sum()/df_equal['sequoia_nb_tokens'].sum()
print("Pourcentage de doublets (mot,lemme) identiques entre les labélisations de sequoia et FrenchPreprocessing (StanfordPOSTagger) : "+"{:.2%}".format(nb))

nb = df_equal['sequoia_frenchpreprocessing_camemBERT_word_lemma'].sum()/df_equal['sequoia_nb_tokens'].sum()
print("Pourcentage de doublets (mot,lemme) identiques entre les labélisations de sequoia et FrenchPreprocessing (camemBERT) : "+"{:.2%}".format(nb))
```

    Pourcentage de doublets (mot,lemme) identiques entre les labélisations de sequoia et TreeTagger : 92.88%
    Pourcentage de doublets (mot,lemme) identiques entre les labélisations de sequoia et FrenchPreprocessing (StanfordPOSTagger) : 92.85%
    Pourcentage de doublets (mot,lemme) identiques entre les labélisations de sequoia et FrenchPreprocessing (camemBERT) : 93.12%
    

## Conclusion

L'outil FrenchPreprocessing (camemBERT) est beaucoup plus lent que TreeTagger, mais il peut servir pour de faibles volumes de données, ou bien pour du traitement en continu. Il est cependant légèrement plus performant (93.12% contre 92.88%).

**Attention :** Le StanfordPOSTagger résiste mieux lorsque la séquence donnée est plus longue qu'une phrase. FrenchPreprocessing (camemBERT) mets environ 0,1 seconde par phrase pour la lemmatisation contre 0,7 seconde pour celui avec le StanfordPOSTagger, mais camemBERT est un modèle à complexité temporelle quadratique sur la longueur de la séquence donnée. Il est donc préférable, pour lemmatiser du texte, de découper celui-ci par phrases avant de le lemmatiser. Ensuite, il faudrait paralléliser les calculs réalisés par FrenchPreprocessing, car cela permettrait de diviser le temps d'execution par le nombre de coeurs du processeur utilisé. 


```python

```
