# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 20:56:17 2020

@author: Rodrigo
"""

from textblob.classifiers import NaiveBayesClassifier
import pickle


with open('news.csv', errors='ignore') as fp:
    cl = NaiveBayesClassifier(fp, format="csv")
    
filename = 'modelo_treinado.sav'
pickle.dump(cl, open(filename, 'wb'))
    
