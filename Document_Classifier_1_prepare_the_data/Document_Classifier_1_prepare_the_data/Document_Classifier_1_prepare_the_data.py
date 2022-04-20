
# Document Classifier 1 prepare the data
# The dataset is made up of three unbalanced classes (0 = 3984, 1 = 1935, 2 = 238).
# The goal is to balance the classes by increasing the minority class and subsequently the creation of the corpus by cleaning the texts

import pandas as pd 
import numpy as np 
import re, spacy, nlpaug, nltk, random, time 
from nlpaug.augmenter.word import SynonymAug 
from collections import Counter 
from  sklearn.utils import shuffle

nlp = spacy.load('it_core_news_lg')

# Text preprocessing: the documents are messages from different sources. I delete the headers of the emails, pec, letters ...
def  cleanText ( text ):
    text = text.lower()
    text = text.replace('"', ' ')
    text = text.replace('- -', '--')
    text = re.sub( r'\-{4,10}', '|$|', text)
    tl = list(text.split('|$|'))
    good = ''    
    for w in tl:        
        w = '$$ '+w+' $$ '
        if 'from' in w and 'subject' in w:
            r1 = re.search( r'from', w)
            n1 = r1.span()[0]
            r2 = re.search( r'subject', w)
            n2 = r2.span()[1]            
            a = w[:n1]
            b = w[n2:]
            good = good+' '+a+' '+b
        elif 'version' in w and 'oggetto' in w :
            r1 = re.search( r'version', w)
            n1 = r1.span()[0]
            r2 = re.search( r'oggetto', w)
            n2 = r2.span()[1]            
            a = w[:n1]
            b = w[n2:]
            good = good+' '+a+' '+b
        else:
            good = good+' '+w
    
    good = good.replace('$', ' ')
    good = re.sub( r' +', ' ', good) 
    return good 


# Augment data:  I replace a part of the words of the text with their synonyms in order to create new documents for the minority class  
def augmentText ( df ):
    num = df.name
    if num%100 == 0 : print(" aumento ",num) 
    rx = random.choice( nr )    
    td = str(df['testi_doc'])
    k = int( len( td.split('.'))/3)
    aug = SynonymAug( lang='ita', aug_min=k)
    if rx == 1 or rx == 3 :
        tx = str(df['testi_doc'])        
        ag = aug.augment( tx)
        nl = bigDic['testi_doc']
        nl.append( ag[0])
        bigDic['testi_doc'] = nl
        for c in anList:
            w = str(df[c])
            nl = bigDic[c]
            nl.append( w )
            bigDic[c] = nl        
    if rx == 2 or rx == 4 or rx == 5 :
        tx = str( df['testi_doc'])        
        la = aug.augment( tx, n=2)
        for w in la:
            nl = bigDic['testi_doc']
            nl.append( w)
            bigDic['testi_doc'] = nl
        for i in range(0, 2):
            for c in anList:
                w = str( df[c])
                nl = bigDic[c]
                nl.append( w)
                bigDic[c] = nl

    for c in cnList:
        w = str( df[c])
        nl = bigDic[c]
        nl.append( w)
        bigDic[c] = nl
    return bigDic


# Union the augmented set for the minor class with the other classes
def unionData( df ):
    w = str(df['Ramo'])
    if 'AUTO' not in w:
        for c in cnList:
            u = str(df[c])
            nl = bigDic[c]
            nl.append(u)
            bigDic[c] = nl
    return bigDic


# make the Corpus: delete punctuation, stop words and transform words into their lemmas
def makeCorpus ( df ):            
    num = df.name 
    if num%500 == 0: print("spacy ",num)
    text = str(df['testi_doc'])        
    text = text.lower()
    text = text.replace("'", " ")
    text = text.replace('\n', ' ')
    text = text.replace('\t', ' ')    
    text = re.sub( r'\S+@\S+', ' ', text)
    text = re.sub( '\w*\d\w*', ' ', text)    
    text = re.sub( r' +', ' ', text)
    if len(text) <= 4 :
        text = 'vuoto' 
    else:   
        doc = nlp(text)
        good = ''
        for tk in doc:        
            if( tk.is_stop == False and tk.is_punct == False) or ( tk.text in keywords ): # and tk.is_space == False:
              good = good+tk.lemma_+' '
        text = good 
    text = re.sub( r'[ () {} \[\] ;: °_ \- <> º€\+\*\$# \| \_ ]', ' ', text)
    text = re.sub( r'[^\s\w]', ' ', text)    
    text  = re.sub(r' +', ' ', text)                               
    procesList.append( text)
    return   procesList 

print(" go ")
df = pd.read_excel('...\\df_testi_and_labels.xlsx')
print(" df size ",df.shape)
print(list(df))
print(" labels",Counter(df['Ramo']))

df['testi_doc'] = df['testi_originali'].apply( cleanText  )
df['dim'] = df['testi_doc'].apply(lambda x: len( x.split('.'))) 
print(" mean of  sentences in a text ",df['dim'].mean())

# select the  minority class 
df_a = df[ df['Ramo'] == 'AUTO']
print(" small df size :",df_a.shape)
random.seed(42 )
nr = [0, 1, 2] # 3, 4, 5
keywords = ['auto', 'motoveicolo', 'veicolo', 'dispositivo', 'satellitare', 'vettura', 'automobile', 'strade', 'viaggiare', 'viaggiarsicuri', 'condominio', 'immobile', 'cat', 'catastale', 'mutuo', 'ipotetica', 'casa', 'rc', 'obiettivo', 'salute', 'vita', 'rendita', 'fondo', 'pensione', 'vitalizio', 'trf', 'previdenza', 'genera',  'immagina', 'futuro']
cnList = list(df)
print(" cnList ",cnList)
anList = list(df) 
del anList[len(anList)-2]
print(" anList ",anList)
bigDic = {}
for c in cnList:
    bigDic[c] = []

df_a.apply( augmentText, axis=1)
df.apply( unionData, axis=1)
df_big = pd.DataFrame( bigDic )
df_big = df_big.drop('dim', axis=1)
print(" big df size: ",df_big.shape)
print(" labels big  ",Counter(df_big['Ramo']))

procesList = []
df_big.apply( makeCorpus, axis=1)
df_big['corpus'] = procesList
print(" corpus ",df_big.shape)
for i in range(0, 3):
    df_big = shuffle(df_big, random_state=33)

ds = pd.ExcelWriter('...\\df_ramo_corpus_small_with_keywords.xlsx')
df_big.to_excel( ds, index=False)
ds.save()

print("end ")
