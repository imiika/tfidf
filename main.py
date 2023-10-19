import streamlit as st
import numpy as np
import os
from streamlit_option_menu import option_menu
import string # for removing punctuation
import nltk #natural language toolkit
from nltk import word_tokenize, download, stem #preprocessing
from nltk.corpus import stopwords, words #remove stopwords

#nltk.download('punkt')
#nltk.download('stopwords')
#nltk.download('words')

with st.sidebar :
    choose = option_menu("Menu", ["Home", "Dictionaries", "TF-IDF", "Contact", "Help"],
                         icons=['house', 'table 2 columns', 'table', 'stars','person lines fill'],
                         menu_icon="menu-app",
                         styles={
        "container": {"padding": "5!important", "background-color": "#fafafa"},
        "icon": {"color": "orange", "font-size": "25px"}, 
        "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
        "nav-link-selected": {"background-color": "#02ab21"},
    }
    )

def read_data():
    doc_names = os.listdir( 'Documents/' )
    st.write(doc_names)
    docs = []
    for doc in doc_names :
        with open( 'Documents/'+doc, 'r' ) as file :
            docs.append( file.read() )
    return docs

def clean_preprocess(docs):
    stemmer = stem.PorterStemmer()
    for i in range( len(docs) ) :
        docs[i] = docs[i].lower() # cleaning : lower case
        docs[i] = ' '.join( docs[i].split() ) # cleaning: remove non words
        
        tokens = word_tokenize( docs[i] ) # preprocessing : tokenization
        stop_words = nltk.corpus.stopwords.words( 'english' ) # preprocessing : stop words
        docs[i] = ' '.join( [token for token in tokens if token not in stop_words] ) # preprocessing : stop words removal

        tokens = word_tokenize( docs[i] ) # preprocessing : tokenization
        docs[i] = ' '.join( [stemmer.stem(token) for token in tokens] ) # preprocessing : stemming
        return docs

if choose == "Home" :
    st.title( "Reconnaissance des mots arabes manuscrits pris de la base de données IFN/ENIT" )

elif choose == "Dictionaries" :
    docs = read_data()
    st.write(docs[0])
        
    docs = clean_preprocess(docs)
    st.write('then')
    st.write(docs[0])
