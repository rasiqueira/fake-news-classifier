# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 20:56:17 2020

@author: Rodrigo
"""

from urllib.request import urlopen
from bs4 import BeautifulSoup
import pickle
import streamlit as st

def main(site):
    url = site
    html = urlopen(url).read()
    soup = BeautifulSoup(html)
    
    # kill all script and style elements
    for script in soup(["script", "style"]):
        script.extract()    # rip it out
    
    # get text
    text = soup.get_text()
    
    # break into lines and remove leading and trailing space on each
    lines = (line.strip() for line in text.splitlines())
    # break multi-headlines into a line each
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    # drop blank lines
    text = '\n'.join(chunk for chunk in chunks if chunk)
    print(text)
    loaded_model = pickle.load(open('model_trainned.sav', 'rb'))
    resultado = loaded_model.predict([text])
    resultado1 = loaded_model.predict_proba([text])
    print(resultado)
    print(100*round(resultado1[0][1],2),'%')
    
st.title('Fake News Classifiers')    
st.header('Introducing Sentiment Analysis')
st.markdown('''Also known as "Opinion Mining", Sentiment Analysis refers to the use of Natural Language Processing to determine the attitude, opinions and emotions of a speaker, writer, or other subject within an online mention.\n
**Essentially, it is the process of determining whether a piece of writing is positive or negative. This is also called the Polarity of the content.**\n
As humans, we are able to classify text into positive/negative subconsciously. For example, the sentence "The kid had a gorgeous smile on his face", will most likely give us a positive sentiment. In layman’s terms, we kind of arrive to such conclusion by examining the words and averaging out the positives and the negatives. For instance, the words "gorgeous" and "smile" are more likely to be positive, while words like “the”, “kid” and “face” are really neutral. Therefore, the overall sentiment of the sentence is likely to be positive.\n
A common use for this technology comes from its deployment in the social media space to discover how people feel about certain topics, particularly through users’ word-of-mouth in textual posts, or in the context of Twitter, their tweets.
''')
st.header('Demo Application')
st.markdown('''
Below you enter a term and the amount of tweets passed to be analyzed. The application applies sentiment analysis and brings several insisghts about the term.''')
termo = st.text_input('What term you want to verify?')
numero = st.text_input('How many past tweets do you want to analyse?')
st.button('Verify the term')
if termo and numero:
    main(termo,numero)
